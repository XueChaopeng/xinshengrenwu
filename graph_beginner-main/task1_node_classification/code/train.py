"""Task 1 -- node classification with GCN / GAT / GraphSAGE / GIN.

Two training regimes are supported so that they can be compared directly:

* ``--mode full``   : full-graph (full-batch) training, every epoch sees the
  whole graph.
* ``--mode sample`` : mini-batch training.  A ``NeighborLoader`` (the sampler
  shipped with PyTorch Geometric) draws a fixed-size neighbour sub-graph around
  each seed node; gradients are computed only on the seed nodes of the batch.
  Evaluation and early-stopping model selection are always performed on the
  **full** graph, which is the convention used in the GraphSAINT / neighbour
  sampling literature and makes the two regimes comparable.

Examples
--------
>>> # full-graph GCN on Cora
>>> python train.py --dataset Cora --model gcn --mode full
>>> # mini-batch GraphSAGE on Flickr with 2-hop fanout [15, 10]
>>> python train.py --dataset Flickr --model sage --mode sample --num-neighbors 15 10
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DATASET_DEFAULTS, default_root, describe, load_dataset  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    Timer,
    accuracy,
    append_csv,
    count_parameters,
    ensure_dir,
    fmt_mean_std,
    get_device,
    macro_f1,
    mean_std,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    set_seed,
    write_json,
)

TASK_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

CSV_FIELDS = [
    "experiment", "dataset", "model", "mode", "hidden", "num_layers", "heads", "dropout",
    "lr", "weight_decay", "batch_size", "num_neighbors", "aggr",
    "batchnorm", "residual", "params",
    "val_acc_mean", "test_acc_mean", "test_acc_std",
    "test_f1_mean", "test_f1_std",
    "fit_time_mean", "fit_time_std", "inference_time_mean",
    "epochs_mean", "epochs_run_mean", "peak_mem_mb",
    "num_nodes", "num_edges", "num_train", "num_test",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
#: Knobs that fall back to :data:`GLOBAL_DEFAULTS` (and then to the per-dataset
#: defaults) when left as ``None``.  Keeping them ``None`` by default is what
#: lets ``run_experiments.py`` override a *single* knob without the per-dataset
#: defaults silently clobbering the values it passed explicitly.
TUNABLE = ("hidden", "num_layers", "heads", "dropout", "aggr", "lr",
           "weight_decay", "epochs", "patience", "batch_size",
           "num_neighbors", "eval_every")

GLOBAL_DEFAULTS = {
    "hidden": 64, "num_layers": 2, "heads": 8, "dropout": 0.5, "aggr": "mean",
    "lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 100,
    "batch_size": 64, "num_neighbors": [10, 10], "eval_every": 1,
}


@dataclass
class Config:
    dataset: str = "Cora"
    model: str = "gcn"
    mode: str = "full"                     # "full" | "sample"

    # --- tunable knobs: ``None`` means "inherit the dataset default" --------
    hidden: Optional[int] = None
    num_layers: Optional[int] = None
    heads: Optional[int] = None
    dropout: Optional[float] = None
    aggr: Optional[str] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    epochs: Optional[int] = None
    patience: Optional[int] = None
    batch_size: Optional[int] = None
    num_neighbors: Optional[List[int]] = None
    eval_every: Optional[int] = None

    # --- structural / bookkeeping options -----------------------------------
    residual: bool = False
    batchnorm: Optional[bool] = None
    #: Where the *graph* lives during mini-batch training.  ``"gpu"`` keeps the
    #: whole graph on the device (fastest sampling); ``"cpu"`` keeps it in host
    #: memory and only moves the sampled sub-graph to the device, which is what
    #: makes neighbour sampling scale to graphs that do not fit in GPU memory.
    data_device: str = "gpu"
    experiment: str = "main"
    seeds: Sequence[int] = (0, 1, 2)
    device: str = "auto"
    data_root: str = field(default_factory=default_root)
    results_dir: str = field(default_factory=lambda: os.path.join(TASK_DIR, "results"))
    tag: str = ""

    def resolved(self) -> "Config":
        """Fill every ``None`` knob from the dataset defaults, then the globals."""
        cfg = copy.deepcopy(self)
        merged = dict(GLOBAL_DEFAULTS)
        merged.update(DATASET_DEFAULTS.get(cfg.dataset, {}))
        # Mini-batch training needs its own (smaller) step size.
        if cfg.mode == "sample" and merged.get("sample_lr") is not None:
            merged["lr"] = merged["sample_lr"]
        for key in TUNABLE:
            if getattr(cfg, key) is None:
                value = merged[key]
                setattr(cfg, key, list(value) if isinstance(value, (list, tuple)) else value)
        return cfg

    @property
    def key(self) -> str:
        extra = f"_{self.tag}" if self.tag else ""
        return f"{self.dataset}_{self.model}_{self.mode}{extra}"


def resolve_fanout(num_neighbors: Sequence[int], num_layers: int) -> List[int]:
    """Return the per-hop fanout for the neighbour sampler.

    A stack of ``L`` message-passing layers needs *exactly* ``L`` sampled hops:
    layer 1 consumes the 1-hop neighbourhood, layer 2 consumes the 2-hop
    neighbourhood that the 1-hop nodes have already aggregated, and so on.
    Sampling fewer hops silently starves the deeper layers, so the fanout list
    is padded (repeating its last entry) or truncated to length ``L``.
    ``-1`` means "keep every neighbour" (see ``NeighborLoader``).
    """
    hops = max(num_layers, 1)
    fanout = list(num_neighbors)[:hops]
    while len(fanout) < hops:
        fanout.append(num_neighbors[-1] if num_neighbors else 10)
    return fanout


# --------------------------------------------------------------------------- #
# Single run (one seed)
# --------------------------------------------------------------------------- #
def _masks(data, device):
    return (data.train_mask.to(device), data.val_mask.to(device), data.test_mask.to(device))


@torch.no_grad()
def _evaluate(model, x, edge_index, y, mask, num_classes, device):
    model.eval()
    out = model(x, edge_index)
    logits = out[mask]
    return (accuracy(logits, y[mask]), macro_f1(logits, y[mask], num_classes))


def run_single(cfg: Config, data, num_features: int, num_classes: int, seed: int, device) -> dict:
    """Train one model with one seed; return test metrics and timings."""
    set_seed(seed)
    reset_peak_gpu_memory(device)

    model = build_model(
        cfg.model,
        in_channels=num_features,
        hidden_channels=cfg.hidden,
        num_classes=num_classes,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        heads=cfg.heads,
        aggr=cfg.aggr,
        batchnorm=cfg.batchnorm,
        residual=cfg.residual,
    ).to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask, val_mask, test_mask = _masks(data, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loader = None
    fanout: List[int] = []
    if cfg.mode == "sample":
        fanout = resolve_fanout(cfg.num_neighbors, cfg.num_layers)
        graph_device = device if cfg.data_device == "gpu" else torch.device("cpu")
        loader = NeighborLoader(
            data.to(graph_device),
            num_neighbors=fanout,
            batch_size=cfg.batch_size,
            input_nodes=train_mask.to(graph_device),
            shuffle=True,
            num_workers=0,
        )
    elif cfg.mode != "full":
        raise ValueError(f"mode must be 'full' or 'sample', got {cfg.mode!r}")

    best_val, best_state, best_epoch = -1.0, None, 0
    epochs_run = 0
    timer = Timer()
    timer.start()
    for epoch in range(1, cfg.epochs + 1):
        epochs_run = epoch
        model.train()
        if cfg.mode == "full":
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
        else:
            for batch in loader:
                if batch.x.device != device:
                    batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[: batch.batch_size]
                loss = F.cross_entropy(out, batch.y[: batch.batch_size])
                loss.backward()
                optimizer.step()

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val_acc, _ = _evaluate(model, x, edge_index, y, val_mask, num_classes, device)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch - best_epoch >= cfg.patience:
            break
    timer.stop()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    infer = Timer()
    with torch.no_grad():
        infer.start()
        out = model(x, edge_index)
        infer.stop()
    test_acc = accuracy(out[test_mask], y[test_mask])
    test_f1 = macro_f1(out[test_mask], y[test_mask], num_classes)
    val_acc = accuracy(out[val_mask], y[val_mask])

    return {
        "seed": seed,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "fit_time": timer.elapsed,
        "inference_time": infer.elapsed,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "params": count_parameters(model),
        "peak_mem_mb": peak_gpu_memory_mb(device),
        "fanout": fanout,
    }


# --------------------------------------------------------------------------- #
# Multi-seed driver
# --------------------------------------------------------------------------- #
def run(cfg: Config, verbose: bool = True, write: bool = True) -> dict:
    cfg = cfg.resolved()
    device = get_device(cfg.device)
    data, num_features, num_classes = load_dataset(cfg.dataset, cfg.data_root)
    stats = describe(data, num_features, num_classes)

    runs = [run_single(cfg, data, num_features, num_classes, s, device) for s in cfg.seeds]

    def col(name):
        return [r[name] for r in runs]

    result = {
        "config": {**asdict(cfg), "fanout": resolve_fanout(cfg.num_neighbors, cfg.num_layers)},
        "dataset_stats": stats,
        "params": runs[0]["params"],
        "runs": runs,
        "val_acc_mean": mean_std(col("val_acc"))[0],
        "test_acc_mean": mean_std(col("test_acc"))[0],
        "test_acc_std": mean_std(col("test_acc"))[1],
        "test_f1_mean": mean_std(col("test_f1"))[0],
        "test_f1_std": mean_std(col("test_f1"))[1],
        "fit_time_mean": mean_std(col("fit_time"))[0],
        "fit_time_std": mean_std(col("fit_time"))[1],
        "inference_time_mean": mean_std(col("inference_time"))[0],
        "epochs_mean": mean_std(col("best_epoch"))[0],
        "epochs_run_mean": mean_std(col("epochs_run"))[0],
        "peak_mem_mb": max(col("peak_mem_mb")),
    }

    if verbose:
        print(
            f"[{cfg.key}] params={result['params']:,}  "
            f"val={result['val_acc_mean']:.2f}  "
            f"test={fmt_mean_std(col('test_acc'))}  "
            f"F1={fmt_mean_std(col('test_f1'))}  "
            f"fit={result['fit_time_mean']:.2f}s  "
            f"infer={result['inference_time_mean']*1000:.1f}ms  "
            f"epochs={result['epochs_mean']:.0f}  "
            f"mem={result['peak_mem_mb']:.0f}MB",
            flush=True,
        )

    if write:
        detail_dir = ensure_dir(os.path.join(cfg.results_dir, "detail"))
        write_json(os.path.join(detail_dir, f"{cfg.key}.json"), result)
        append_csv(os.path.join(cfg.results_dir, "all_results.csv"), _csv_row(cfg, result, stats),
                   CSV_FIELDS)
    return result


def _csv_row(cfg: Config, result: dict, stats: dict) -> dict:
    return {
        "experiment": cfg.experiment,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "mode": cfg.mode,
        "hidden": cfg.hidden,
        "num_layers": cfg.num_layers,
        "heads": cfg.heads,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size if cfg.mode == "sample" else "-",
        "num_neighbors": ",".join(map(str, result["config"]["fanout"])) if cfg.mode == "sample" else "-",
        "aggr": cfg.aggr,
        "batchnorm": cfg.batchnorm if cfg.batchnorm is not None else (cfg.model == "gin"),
        "residual": cfg.residual,
        "params": result["params"],
        "val_acc_mean": round(result["val_acc_mean"], 3),
        "test_acc_mean": round(result["test_acc_mean"], 3),
        "test_acc_std": round(result["test_acc_std"], 3),
        "test_f1_mean": round(result["test_f1_mean"], 3),
        "test_f1_std": round(result["test_f1_std"], 3),
        "fit_time_mean": round(result["fit_time_mean"], 3),
        "fit_time_std": round(result["fit_time_std"], 3),
        "inference_time_mean": round(result["inference_time_mean"], 4),
        "epochs_mean": round(result["epochs_mean"], 1),
        "epochs_run_mean": round(result["epochs_run_mean"], 1),
        "peak_mem_mb": round(result["peak_mem_mb"], 1),
        "num_nodes": stats["num_nodes"],
        "num_edges": stats["num_edges"],
        "num_train": stats["num_train"],
        "num_test": stats["num_test"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ds = " (default: dataset specific -- see code/dataset.py)"
    p = argparse.ArgumentParser(
        description="Task 1: node classification (GCN/GAT/GraphSAGE/GIN, full vs. sampled)",
    )
    p.add_argument("--dataset", default="Cora", choices=["Cora", "Citeseer", "Flickr"])
    p.add_argument("--model", default="gcn", choices=["gcn", "gat", "sage", "gin"])
    p.add_argument("--mode", default="full", choices=["full", "sample"])
    p.add_argument("--hidden", type=int, default=None, help="hidden width" + ds)
    p.add_argument("--num-layers", type=int, default=None, help="number of message-passing layers" + ds)
    p.add_argument("--heads", type=int, default=None, help="GAT attention heads" + ds)
    p.add_argument("--dropout", type=float, default=None, help="dropout rate" + ds)
    p.add_argument("--aggr", default=None, help="GraphSAGE aggregation" + ds)
    p.add_argument("--lr", type=float, default=None, help="Adam learning rate" + ds)
    p.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay" + ds)
    p.add_argument("--epochs", type=int, default=None, help="max epochs" + ds)
    p.add_argument("--patience", type=int, default=None,
                   help="early stopping patience on validation accuracy" + ds)
    p.add_argument("--batch-size", type=int, default=None,
                   help="number of seed nodes per mini-batch (--mode sample)" + ds)
    p.add_argument("--num-neighbors", type=int, nargs="+", default=None,
                   help="fanout per hop, e.g. --num-neighbors 15 10 5" + ds)
    p.add_argument("--eval-every", type=int, default=None,
                   help="run full-graph validation every N epochs" + ds)
    p.add_argument("--residual", action="store_true",
                   help="add linear skip connections around hidden layers")
    p.add_argument("--batchnorm", dest="batchnorm", action="store_true", default=None,
                   help="force batch normalisation on (default: only for GIN)")
    p.add_argument("--no-batchnorm", dest="batchnorm", action="store_false")
    p.add_argument("--data-device", default=Config.data_device, choices=["gpu", "cpu"],
                   help="where the full graph is kept during mini-batch training")
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
    cfg = Config(dataset=args.dataset, model=args.model, mode=args.mode)
    for key in TUNABLE:
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            setattr(cfg, key, value)
    cfg.residual = args.residual
    cfg.batchnorm = args.batchnorm
    cfg.data_device = args.data_device
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
    run(cfg, write=not args.no_write)


if __name__ == "__main__":
    main()
