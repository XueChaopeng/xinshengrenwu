"""Task 3 -- graph classification (TUDataset) and graph regression (ZINC).

Mini-batching
-------------
Graphs are combined into a disjoint union with
:class:`torch_geometric.loader.DataLoader` (standard *graph batching*): every
batch holds ``batch_size`` whole graphs, ``batch.batch`` records which graph each
node belongs to, and the readout pools nodes per graph.  Neighbour sampling --
the technique compared against full-graph training in Task 1 -- does **not** apply
to graph-level tasks (it would break graphs apart at every layer), so
``--num-neighbors`` is accepted only for CLI parity with Task 1 and is ignored.

Metrics
-------
* TUDataset -> **accuracy** (percent, higher is better); macro-F1 is recorded too.
* ZINC      -> **MAE**, lower is better.  ``y`` is PyG's ZINC target field
  ``logP_SA_cycle_normalized`` (penalized logP); MAE is reported **in those
  native units** (this is also the scale the *Benchmarking GNNs* literature
  uses).  ``test_mae_std_units = test_mae / train_target_std`` is recorded as
  well so the number can be re-expressed on a unit-variance scale; the training
  target std of the standard 12k ZINC subset is stored in
  ``data/ZINC_processed/stats.json``.  Passing ``--standardize`` instead trains
  on ``(y - mean) / std`` and *de-standardises predictions before* computing
  MAE, so the reported MAE stays in the same native units either way.

Protocol
--------
Default = **stratified 80 / 10 / 10 split, repeated over seeds 0/1/2**, so every
number carries a mean ± std.  ``--folds K`` switches to stratified K-fold
cross-validation (test = fold, val = 10 % of the remainder), which is the
literature protocol for TUDataset; the two protocols are never mixed in one
table.

Examples
--------
>>> python train.py --dataset MUTAG --model gin --pool mean --seeds 0 1 2
>>> python train.py --dataset IMDB-BINARY --model gcn --pool min --hidden 128
>>> python train.py --dataset ZINC --model gin --pool sum --lr 1e-3
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (  # noqa: E402
    DATASET_DEFAULTS,
    DATASETS,
    TU_DATASETS,
    ZINC_DATASET,
    default_root,
    describe_tu,
    describe_zinc,
    is_regression,
    load_tu_dataset,
    load_zinc,
    stratified_split,
)
from models import POOL_NAMES, build_model  # noqa: E402
from utils import (  # noqa: E402
    Timer,
    accuracy,
    append_csv,
    count_parameters,
    ensure_dir,
    fmt_mean_std,
    get_device,
    mae,
    macro_f1,
    mean_std,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    set_seed,
    write_json,
)

TASK_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

CSV_FIELDS = [
    "experiment", "dataset", "model", "pool", "task", "fold",
    "hidden", "num_layers", "heads", "dropout",
    "lr", "weight_decay", "batch_size", "standardize",
    "batchnorm", "residual", "params",
    "val_metric_mean", "test_acc_mean", "test_acc_std",
    "test_f1_mean", "test_f1_std",
    "test_mae_mean", "test_mae_std", "test_mae_std_units_mean",
    "test_loss_mean",
    "fit_time_mean", "fit_time_std", "inference_time_mean", "inference_time_std",
    "epochs_mean", "epochs_run_mean", "peak_mem_mb",
    "num_graphs", "num_train", "num_val", "num_test",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
#: Knobs that fall back to :data:`GLOBAL_DEFAULTS` (and then to the per-dataset
#: defaults) when left as ``None``.  Keeping them ``None`` by default is what
#: lets ``run_experiments.py`` override a *single* knob without the per-dataset
#: defaults silently clobbering the values it passed explicitly.
TUNABLE = ("hidden", "num_layers", "heads", "dropout", "lr", "weight_decay",
           "epochs", "patience", "batch_size", "eval_every")

GLOBAL_DEFAULTS = {
    "hidden": 64, "num_layers": 3, "heads": 8, "dropout": 0.5,
    "lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 50,
    "batch_size": 32, "eval_every": 1,
}


@dataclass
class Config:
    dataset: str = "MUTAG"
    model: str = "gcn"
    pool: str = "mean"

    # --- tunable knobs: ``None`` means "inherit the dataset default" --------
    hidden: Optional[int] = None
    num_layers: Optional[int] = None
    heads: Optional[int] = None
    dropout: Optional[float] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    epochs: Optional[int] = None
    patience: Optional[int] = None
    batch_size: Optional[int] = None
    eval_every: Optional[int] = None

    # --- structural / bookkeeping options -----------------------------------
    residual: bool = False
    batchnorm: Optional[bool] = None
    standardize: bool = False          # ZINC only: standardise the target
    folds: int = 1                     # 1 = 80/10/10 split; K>1 = stratified K-fold
    experiment: str = "main"
    seeds: Sequence[int] = (0, 1, 2)
    device: str = "auto"
    data_root: str = field(default_factory=default_root)
    results_dir: str = field(default_factory=lambda: os.path.join(TASK_DIR, "results"))
    tag: str = ""
    log_every: int = 20

    def resolved(self) -> "Config":
        """Fill every ``None`` knob from the dataset defaults, then the globals."""
        cfg = copy.deepcopy(self)
        merged = dict(GLOBAL_DEFAULTS)
        merged.update(DATASET_DEFAULTS.get(cfg.dataset, {}))
        for key in TUNABLE:
            if getattr(cfg, key) is None:
                setattr(cfg, key, merged[key])
        return cfg

    @property
    def key(self) -> str:
        bits = [self.dataset, self.model, f"pool-{self.pool}"]
        if self.folds > 1:
            bits.append(f"{self.folds}fold")
        if self.tag:
            bits.append(self.tag)
        return "_".join(bits)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
class Task:
    """Everything the training loop needs: splits, loaders and the target scale."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regression = is_regression(cfg.dataset)
        self.stats: Dict[str, float] = {}
        self._zinc = None

        if self.regression:
            splits, stats = load_zinc(cfg.data_root)
            self._zinc = splits
            self.stats = stats
            self.num_features = 1
            self.num_classes = 1
            self.target_mean = float(stats.get("train_mean", stats.get("mean", 0.0)))
            self.target_std = float(stats.get("train_std", stats.get("std", 1.0)))
            for d in splits["train"]:
                d.y = d.y.view(-1)
        else:
            graphs, nf, nc = load_tu_dataset(cfg.dataset, cfg.data_root)
            self.dataset = graphs          # list[Data] with materialised x
            self.num_features = nf
            self.num_classes = nc
            self.stats = describe_tu(graphs, nf, nc)
            self.y_all = [int(d.y.item()) for d in graphs]
            self.target_mean, self.target_std = 0.0, 1.0

    # -- splits -------------------------------------------------------------
    def folds(self) -> List[Tuple[List[int], List[int], List[int]]]:
        """Return ``(train_idx, val_idx, test_idx)`` per fold."""
        if self.regression:
            n = {k: len(v) for k, v in self._zinc.items()}
            tr = list(range(n["train"]))
            va = list(range(n["val"]))
            te = list(range(n["test"]))
            return [(tr, va, te)]

        out = []
        for fold in range(self.cfg.folds):
            if self.cfg.folds == 1:
                seed = self.cfg.seeds[0] if self.cfg.seeds else 0
                out.append(stratified_split(self.y_all, (0.8, 0.1, 0.1), seed))
            else:
                # Stratified K-fold: test = 1/K of the data, and 10 % of what is
                # left over becomes the validation set (the remainder trains).
                n = len(self.y_all)
                test_idx, _, _ = stratified_split(
                    self.y_all, (1 - 1.0 / self.cfg.folds, 0.0, 1.0 / self.cfg.folds), fold)
                keep = np.setdiff1d(np.arange(n), np.asarray(test_idx, dtype=int))
                sub_y = [self.y_all[i] for i in keep]
                tr_sub, va_sub, _ = stratified_split(sub_y, (0.9, 0.1, 0.0), fold)
                out.append(([int(keep[i]) for i in tr_sub],
                            [int(keep[i]) for i in va_sub],
                            [int(i) for i in test_idx]))
        return out

    def loaders(self, split_idx: Tuple[List[int], List[int], List[int]], seed: int
                ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        tr, va, te = split_idx
        bs = self.cfg.batch_size
        if self.regression:
            z = self._zinc
            make = lambda idx, ds, shuffle: DataLoader(  # noqa: E731
                Subset(ds, idx), batch_size=bs, shuffle=shuffle, num_workers=0)
            return (make(tr, z["train"], True), make(va, z["val"], False),
                    make(te, z["test"], False))
        make = lambda idx, shuffle: DataLoader(  # noqa: E731
            Subset(self.dataset, idx), batch_size=bs, shuffle=shuffle, num_workers=0)
        return make(tr, True), make(va, False), make(te, False)

    def describe_split(self, split_idx) -> dict:
        tr, va, te = split_idx
        if self.regression:
            return {"num_train": len(tr), "num_val": len(va), "num_test": len(te)}
        return {"num_train": len(tr), "num_val": len(va), "num_test": len(te)}


# --------------------------------------------------------------------------- #
# Loss / metric helpers
# --------------------------------------------------------------------------- #
def _targets(batch, regression: bool, target_mean: float, target_std: float,
             standardize: bool):
    """Return ``(model_target, raw_target)`` for a batch.

    ``model_target`` is what the loss consumes: class indices (long) for
    classification, float values for regression -- optionally standardised.
    ``raw_target`` is always on the dataset's native scale, so every reported
    metric stays comparable across the ``--standardize`` setting.
    """
    raw = batch.y.view(-1)
    if not regression:
        return raw.long(), raw.long()
    raw = raw.float()
    if standardize:
        return (raw - target_mean) / target_std, raw
    return raw, raw


def _loss_fn(regression: bool):
    return F.mse_loss if regression else F.cross_entropy


@torch.no_grad()
def evaluate(model, loader, cfg: Config, task: Task, device) -> dict:
    """Mean loss plus task metric(s) over a loader (lower loss is better)."""
    model.eval()
    loss_fn = _loss_fn(task.regression)
    total_loss, total_n = 0.0, 0
    preds, raws = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch.x, batch.edge_index, batch.batch)
        target, raw = _targets(batch, task.regression, task.target_mean,
                               task.target_std, cfg.standardize)
        loss = loss_fn(out, target)
        total_loss += float(loss) * batch.num_graphs
        total_n += batch.num_graphs
        if task.regression:
            pred = out * task.target_std + task.target_mean if cfg.standardize else out
            preds.append(pred.detach().cpu())
            raws.append(raw.detach().cpu())
        else:
            preds.append(out.detach().cpu())
            raws.append(raw.detach().cpu())
    if total_n == 0:
        return {"loss": float("nan"), "acc": float("nan"),
                "f1": float("nan"), "mae": float("nan")}
    pred = torch.cat(preds)
    raw = torch.cat(raws)
    return {
        "loss": total_loss / total_n,
        "acc": float("nan") if task.regression else accuracy(pred, raw),
        "f1": float("nan") if task.regression else macro_f1(pred, raw, task.num_classes),
        "mae": mae(pred, raw) if task.regression else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Single run (one seed, one fold)
# --------------------------------------------------------------------------- #
def run_single_full(cfg: Config, task: Task, split_idx, seed: int, fold: int, device) -> dict:
    """Train one model for one seed on one split.

    Early stopping watches the **validation** metric (accuracy for
    classification, MAE for regression); the best checkpoint is restored and
    only then is the test set evaluated once, together with timings, parameter
    count and peak GPU memory.
    """
    set_seed(seed)
    reset_peak_gpu_memory(device)

    model = build_model(
        cfg.model,
        in_channels=task.num_features,
        hidden_channels=cfg.hidden,
        out_channels=task.num_classes,
        pool=cfg.pool,
        regression=task.regression,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        heads=cfg.heads,
        batchnorm=cfg.batchnorm,
        residual=cfg.residual,
    ).to(device)

    train_loader, val_loader, test_loader = task.loaders(split_idx, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = _loss_fn(task.regression)
    sign = -1.0 if task.regression else 1.0
    best_score, best_state, best_epoch = -float("inf"), None, 0
    best_val: Dict[str, float] = {"loss": float("nan"), "acc": float("nan"),
                                  "f1": float("nan"), "mae": float("nan")}
    epochs_run = 0

    timer = Timer()
    timer.start()
    for epoch in range(1, cfg.epochs + 1):
        epochs_run = epoch
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            target, _ = _targets(batch, task.regression, task.target_mean,
                                 task.target_std, cfg.standardize)
            loss_fn(out, target).backward()
            optimizer.step()

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val = evaluate(model, val_loader, cfg, task, device)
            score = sign * (val["mae"] if task.regression else val["acc"])
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()})
                best_val = val
            if cfg.log_every and (epoch % cfg.log_every == 0):
                metric = (f"val_mae={val['mae']:.4f}" if task.regression
                          else f"val_acc={val['acc']:.2f}")
                print(f"      epoch {epoch:3d}  val_loss={val['loss']:.4f}  "
                      f"{metric}  best@{best_epoch}", flush=True)
        if epoch - best_epoch >= cfg.patience:
            break
    timer.stop()

    if best_state is not None:
        model.load_state_dict(best_state)

    infer = Timer()
    infer.start()
    test = evaluate(model, test_loader, cfg, task, device)
    infer.stop()

    return {
        "seed": seed,
        "fold": fold,
        "val_loss": float(best_val["loss"]),
        "val_acc": float(best_val["acc"]),
        "val_mae": float(best_val["mae"]),
        "val_metric": float(best_score),
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_f1": test["f1"],
        "test_mae": test["mae"],
        "fit_time": timer.elapsed,
        "inference_time": infer.elapsed,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "params": count_parameters(model),
        "peak_mem_mb": peak_gpu_memory_mb(device),
    }


# --------------------------------------------------------------------------- #
# Multi-seed driver
# --------------------------------------------------------------------------- #
def run(cfg: Config, verbose: bool = True, write: bool = True) -> dict:
    cfg = cfg.resolved()
    device = get_device(cfg.device)
    task = Task(cfg)

    runs: List[dict] = []
    for fold, split_idx in enumerate(task.folds()):
        for seed in cfg.seeds:
            runs.append(run_single_full(cfg, task, split_idx, seed, fold, device))

    def col(name):
        return [r[name] for r in runs]

    mae_mean, mae_std = mean_std(col("test_mae"))
    std_units = mae_mean / task.target_std if task.target_std else float("nan")

    result = {
        "config": {**asdict(cfg), "regression": task.regression},
        "dataset_stats": task.stats,
        "target_std": task.target_std,
        "params": runs[0]["params"],
        "runs": runs,
        "val_metric_mean": mean_std(col("val_metric"))[0],
        "test_acc_mean": mean_std(col("test_acc"))[0],
        "test_acc_std": mean_std(col("test_acc"))[1],
        "test_f1_mean": mean_std(col("test_f1"))[0],
        "test_f1_std": mean_std(col("test_f1"))[1],
        "test_mae_mean": mae_mean,
        "test_mae_std": mae_std,
        "test_mae_std_units_mean": std_units,
        "test_loss_mean": mean_std(col("test_loss"))[0],
        "fit_time_mean": mean_std(col("fit_time"))[0],
        "fit_time_std": mean_std(col("fit_time"))[1],
        "inference_time_mean": mean_std(col("inference_time"))[0],
        "inference_time_std": mean_std(col("inference_time"))[1],
        "epochs_mean": mean_std(col("best_epoch"))[0],
        "epochs_run_mean": mean_std(col("epochs_run"))[0],
        "peak_mem_mb": max(col("peak_mem_mb")),
    }

    if verbose:
        metric = (f"MAE={fmt_mean_std(col('test_mae'), 4)}"
                  if task.regression else
                  f"test={fmt_mean_std(col('test_acc'))}")
        print(
            f"[{cfg.key}] params={result['params']:,}  "
            f"{metric}  "
            f"fit={result['fit_time_mean']:.2f}s  "
            f"infer={result['inference_time_mean'] * 1000:.1f}ms  "
            f"epochs={result['epochs_mean']:.0f}  "
            f"mem={result['peak_mem_mb']:.0f}MB",
            flush=True,
        )

    if write:
        detail_dir = ensure_dir(os.path.join(cfg.results_dir, "detail"))
        write_json(os.path.join(detail_dir, f"{cfg.key}.json"), result)
        append_csv(os.path.join(cfg.results_dir, "all_results.csv"),
                   _csv_row(cfg, result, task), CSV_FIELDS)
    return result


def _split_sizes(stats: dict) -> dict:
    """Graph counts for the results CSV, tolerant of both dataset flavours.

    TUDataset reports ``num_graphs`` and has a *seed dependent* split, so the
    training count is the mean (80 %).  ZINC reports ``splits`` with the fixed
    official sizes.  Getting this wrong used to raise ``KeyError`` and silently
    drop every ZINC row from ``all_results.csv`` (the per-config ``detail/*.json``
    was still written, which is why the loss was easy to miss).
    """
    if stats.get("num_train") is not None:
        return {"num_graphs": stats.get("num_graphs", stats["num_train"]),
                "num_train": stats["num_train"],
                "num_val": stats.get("num_val", "-"),
                "num_test": stats.get("num_test", "-")}

    splits = stats.get("splits") or {}
    if splits:
        total = splits.get("train", 0) + splits.get("val", 0) + splits.get("test", 0)
        return {"num_graphs": stats.get("count", total),
                "num_train": splits.get("train", "-"),
                "num_val": splits.get("val", "-"),
                "num_test": splits.get("test", "-")}

    total = stats.get("num_graphs", stats.get("count"))
    return {"num_graphs": total if total is not None else "-",
            "num_train": int(round(0.8 * total)) if total else "-",
            "num_val": "-", "num_test": "-"}


def _csv_row(cfg: Config, result: dict, task: Task) -> dict:
    stats = task.stats
    sizes = _split_sizes(stats)
    return {
        "experiment": cfg.experiment,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "pool": cfg.pool,
        "task": "regression" if task.regression else "classification",
        "fold": cfg.folds,
        "hidden": cfg.hidden,
        "num_layers": cfg.num_layers,
        "heads": cfg.heads,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "standardize": cfg.standardize,
        "batchnorm": cfg.batchnorm if cfg.batchnorm is not None else (cfg.model == "gin"),
        "residual": cfg.residual,
        "params": result["params"],
        "val_metric_mean": round(result["val_metric_mean"], 4),
        "test_acc_mean": round(result["test_acc_mean"], 3),
        "test_acc_std": round(result["test_acc_std"], 3),
        "test_f1_mean": round(result["test_f1_mean"], 3),
        "test_f1_std": round(result["test_f1_std"], 3),
        "test_mae_mean": round(result["test_mae_mean"], 5),
        "test_mae_std": round(result["test_mae_std"], 5),
        "test_mae_std_units_mean": round(result["test_mae_std_units_mean"], 5),
        "test_loss_mean": round(result["test_loss_mean"], 5),
        "fit_time_mean": round(result["fit_time_mean"], 3),
        "fit_time_std": round(result["fit_time_std"], 3),
        "inference_time_mean": round(result["inference_time_mean"], 4),
        "inference_time_std": round(result["inference_time_std"], 4),
        "epochs_mean": round(result["epochs_mean"], 1),
        "epochs_run_mean": round(result["epochs_run_mean"], 1),
        "peak_mem_mb": round(result["peak_mem_mb"], 1),
        "num_graphs": sizes["num_graphs"],
        "num_train": sizes["num_train"],
        "num_val": sizes["num_val"],
        "num_test": sizes["num_test"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ds = " (default: dataset specific -- see code/dataset.py)"
    p = argparse.ArgumentParser(
        description="Task 3: graph classification (TUDataset) / regression (ZINC)",
    )
    p.add_argument("--dataset", default=Config.dataset, choices=list(DATASETS))
    p.add_argument("--model", default=Config.model, choices=["gcn", "gat", "sage", "gin"])
    p.add_argument("--pool", default=Config.pool, choices=list(POOL_NAMES),
                   help="graph readout: mean (AvgPooling), max, min, sum/add")
    p.add_argument("--hidden", type=int, default=None, help="hidden width" + ds)
    p.add_argument("--num-layers", type=int, default=None,
                   help="number of message-passing layers" + ds)
    p.add_argument("--heads", type=int, default=None, help="GAT attention heads" + ds)
    p.add_argument("--dropout", type=float, default=None, help="dropout rate" + ds)
    p.add_argument("--lr", type=float, default=None, help="Adam learning rate" + ds)
    p.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay" + ds)
    p.add_argument("--epochs", type=int, default=None, help="max epochs" + ds)
    p.add_argument("--patience", type=int, default=None,
                   help="early stopping patience on the validation metric" + ds)
    p.add_argument("--batch-size", type=int, default=None,
                   help="number of whole graphs per batch" + ds)
    p.add_argument("--eval-every", type=int, default=None,
                   help="validate every N epochs" + ds)
    p.add_argument("--num-neighbors", type=int, nargs="+", default=None,
                   help="UNUSED: neighbour sampling does not apply to graph-level "
                        "batching; accepted for CLI parity with Task 1 only")
    p.add_argument("--residual", action="store_true",
                   help="add linear skip connections around hidden layers")
    p.add_argument("--batchnorm", dest="batchnorm", action="store_true", default=None,
                   help="force batch normalisation on (default: only for GIN)")
    p.add_argument("--no-batchnorm", dest="batchnorm", action="store_false")
    p.add_argument("--standardize", action="store_true",
                   help="ZINC only: train on (y-mean)/std, still report MAE in "
                        "native units (default: train on raw y, as PyG's own "
                        "ZINC example does)")
    p.add_argument("--folds", type=int, default=1,
                   help="1 = stratified 80/10/10 per seed (default); K>1 = "
                        "stratified K-fold cross-validation")
    p.add_argument("--log-every", type=int, default=Config.log_every,
                   help="print a progress line every N epochs (0 = silent)")
    p.add_argument("--experiment", default=Config.experiment,
                   help="label used to group rows in results/all_results.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=list(Config.seeds))
    p.add_argument("--device", default=Config.device)
    p.add_argument("--data-root", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--no-write", action="store_true", help="do not persist results to disk")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(dataset=args.dataset, model=args.model, pool=args.pool)
    for key in TUNABLE:
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            setattr(cfg, key, value)
    cfg.residual = args.residual
    cfg.batchnorm = args.batchnorm
    cfg.standardize = args.standardize
    cfg.folds = args.folds
    cfg.log_every = args.log_every
    cfg.experiment = args.experiment
    cfg.seeds = tuple(args.seeds)
    cfg.device = args.device
    cfg.tag = args.tag
    if args.data_root:
        cfg.data_root = args.data_root
    if args.results_dir:
        cfg.results_dir = args.results_dir
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    t0 = time.perf_counter()
    run(cfg, write=not args.no_write)
    print(f"  (wall clock {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
