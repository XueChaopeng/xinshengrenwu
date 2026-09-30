"""Task 2 -- link prediction with GCN / GAT / GraphSAGE / GIN.

Same two training regimes as Task 1, so that they can be compared head to head:

* ``--mode full``   : the encoder performs message passing over the whole
  training graph each step; negatives are redrawn **once per epoch**.
* ``--mode sample`` : ``LinkNeighborLoader`` (the PyG sampler) draws a
  neighbour sub-graph around a batch of supervised edges; negatives are drawn
  inside each mini-batch.  Evaluation is always done on the full graph.

Metrics: ROC-AUC and Average Precision (AP), the standard pair for link
prediction, plus a split-wide Hits@K.

Examples
--------
>>> python train.py --dataset Cora --model gcn --mode full
>>> python train.py --dataset Flickr --model sage --mode sample --batch-size 2048
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
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DATASET_DEFAULTS, describe, default_root, load_dataset, split_edges  # noqa: E402
from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    Timer,
    append_csv,
    average_precision,
    count_parameters,
    ensure_dir,
    fmt_mean_std,
    get_device,
    hits_at_k,
    mean_std,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    roc_auc,
    set_seed,
    write_json,
)

TASK_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

TUNABLE = ("hidden", "num_layers", "heads", "dropout", "aggr", "lr",
           "weight_decay", "epochs", "patience", "batch_size",
           "num_neighbors", "eval_every")

GLOBAL_DEFAULTS = {
    "hidden": 128, "num_layers": 2, "heads": 8, "dropout": 0.0, "aggr": "mean",
    "lr": 0.01, "weight_decay": 0.0, "epochs": 200, "patience": 50,
    "batch_size": 512, "num_neighbors": [10, 10], "eval_every": 1,
}

CSV_FIELDS = [
    "experiment", "dataset", "model", "mode", "decoder", "hidden", "num_layers",
    "heads", "dropout", "aggr", "lr", "weight_decay", "batch_size",
    "num_neighbors", "params",
    "val_auc_mean", "test_auc_mean", "test_auc_std",
    "test_ap_mean", "test_ap_std", "test_hits50_mean",
    "fit_time_mean", "fit_time_std", "inference_time_mean",
    "epochs_mean", "peak_mem_mb",
    "num_nodes", "num_edges",
]


@dataclass
class Config:
    dataset: str = "Cora"
    model: str = "gcn"
    mode: str = "full"                     # "full" | "sample"
    decoder: str = "dot"                   # "dot" | "mlp"
    #: Negative-edge sampler for the full-graph regime: "uniform" (fast, on-device)
    #: or "pyg" (torch_geometric.utils.negative_sampling, filters real edges).
    neg_sampler: str = "uniform"

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
    disjoint_train_ratio: float = 0.0
    residual: bool = False
    batchnorm: Optional[bool] = None
    data_device: str = "gpu"
    experiment: str = "main"
    seeds: Sequence[int] = (0, 1, 2)
    device: str = "auto"
    data_root: str = field(default_factory=default_root)
    results_dir: str = field(default_factory=lambda: os.path.join(TASK_DIR, "results"))
    tag: str = ""

    def resolved(self) -> "Config":
        cfg = copy.deepcopy(self)
        merged = dict(GLOBAL_DEFAULTS)
        merged.update(DATASET_DEFAULTS.get(cfg.dataset, {}))
        # Mini-batch training has its own budget: a noisier gradient needs a
        # smaller step size, and on an edge-level task the number of steps per
        # epoch is (#supervised edges / batch size), which on Flickr is ~400 with
        # the full-graph batch size.  ``<name>_sample_*`` keys therefore override
        # lr / epochs / patience / batch_size / num_neighbors for --mode sample.
        if cfg.mode == "sample":
            for key in ("lr", "epochs", "patience", "batch_size", "num_neighbors"):
                override = merged.get(f"sample_{key}")
                if override is not None:
                    merged[key] = override
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
    """``L`` message-passing layers consume exactly ``L`` sampled hops."""
    hops = max(num_layers, 1)
    fanout = list(num_neighbors)[:hops]
    while len(fanout) < hops:
        fanout.append(num_neighbors[-1] if num_neighbors else 10)
    return fanout


# --------------------------------------------------------------------------- #
# Evaluation / negative sampling helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, x, edge_index, edge_label_index, edge_label):
    """Full-graph evaluation of a link-prediction model."""
    model.eval()
    z = model.encode(x, edge_index)
    logits = model.decode(z, edge_label_index)
    return {
        "auc": roc_auc(edge_label, logits),
        "ap": average_precision(edge_label, logits),
        "hits50": hits_at_k(edge_label, logits, 50),
    }


def _draw_negatives(pos_edge_index: torch.Tensor, num_nodes: int,
                    mode: str = "uniform") -> torch.Tensor:
    """Draw one negative edge per positive edge.

    ``"uniform"`` (default) samples both endpoints uniformly on the device.
    For these graphs the probability of accidentally drawing a *real* edge is
    negligible (Cora: 10,556 / 2,708^2 = 0.14 %, Flickr: 0.011 %), and unlike
    PyG's ``negative_sampling`` it never round-trips through host memory --
    measured on Cora, 0.6 ms per call versus 20.8 ms.

    ``"pyg"`` delegates to :func:`torch_geometric.utils.negative_sampling`,
    which additionally filters existing edges and self-loops at that cost.
    """
    n = pos_edge_index.size(1)
    if mode == "pyg":
        return negative_sampling(edge_index=pos_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=n)

    device = pos_edge_index.device
    neg = torch.randint(0, num_nodes, (2, n), device=device, dtype=pos_edge_index.dtype)
    self_loops = neg[0] == neg[1]
    if bool(self_loops.any()):
        k = int(self_loops.sum())
        neg[:, self_loops] = torch.randint(0, num_nodes, (2, k), device=device,
                                           dtype=pos_edge_index.dtype)
    return neg


def _warmup_metrics() -> None:
    """Pay the one-off lazy-initialisation cost of the metric backends.

    The first ``sklearn.metrics`` call in a fresh process takes over a second,
    which would otherwise be charged to whichever ``Timer`` wraps it (usually
    the training loop or the test evaluation).
    """
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7])
    roc_auc(labels, scores)
    average_precision(labels, scores)
    hits_at_k(labels, scores, 2)


# --------------------------------------------------------------------------- #
# Single run (one seed)
# --------------------------------------------------------------------------- #
def run_single(cfg: Config, data, num_features: int, seed: int, device) -> dict:
    set_seed(seed)
    reset_peak_gpu_memory(device)
    _warmup_metrics()

    train_data, val_data, test_data = split_edges(
        data, seed=seed, disjoint_train_ratio=cfg.disjoint_train_ratio,
    )

    model = build_model(
        cfg.model,
        in_channels=num_features,
        hidden_channels=cfg.hidden,
        decoder=cfg.decoder,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        heads=cfg.heads,
        aggr=cfg.aggr,
        batchnorm=cfg.batchnorm,
        residual=cfg.residual,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    num_nodes = int(data.num_nodes)
    x = train_data.x.to(device)
    train_edge_index = train_data.edge_index.to(device)
    train_pos = train_data.edge_label_index.to(device)

    val_x = val_data.x.to(device)
    val_edge_index = val_data.edge_index.to(device)
    val_label_index = val_data.edge_label_index.to(device)
    val_label = val_data.edge_label.to(device).float()

    loader = None
    fanout: List[int] = []
    if cfg.mode == "sample":
        fanout = resolve_fanout(cfg.num_neighbors, cfg.num_layers)
        graph_device = device if cfg.data_device == "gpu" else torch.device("cpu")
        td = train_data.to(graph_device)
        # ``RandomLinkSplit`` does not materialise negative training edges when
        # ``add_negative_train_samples=False``, so the loader needs a label
        # vector for the positive edges it will augment itself.
        train_edge_label = getattr(td, "edge_label", None)
        if train_edge_label is None:
            train_edge_label = torch.ones(td.edge_label_index.size(1), dtype=torch.float,
                                          device=graph_device)
        loader = LinkNeighborLoader(
            td,
            num_neighbors=fanout,
            batch_size=cfg.batch_size,
            edge_label_index=td.edge_label_index,
            edge_label=train_edge_label,
            neg_sampling_ratio=1.0,
            shuffle=True,
            num_workers=0,
        )
    elif cfg.mode != "full":
        raise ValueError(f"mode must be 'full' or 'sample', got {cfg.mode!r}")

    best_val_auc, best_state, best_epoch = -1.0, None, 0
    epochs_run = 0
    timer = Timer()
    timer.start()
    for epoch in range(1, cfg.epochs + 1):
        epochs_run = epoch
        model.train()
        if cfg.mode == "full":
            optimizer.zero_grad()
            neg = _draw_negatives(train_pos, num_nodes, cfg.neg_sampler)
            edge_label_index = torch.cat([train_pos, neg], dim=1)
            edge_label = torch.cat([
                torch.ones(train_pos.size(1), device=device),
                torch.zeros(neg.size(1), device=device),
            ])
            logits = model(x, train_edge_index, edge_label_index)
            loss = F.binary_cross_entropy_with_logits(logits, edge_label)
            loss.backward()
            optimizer.step()
        else:
            for batch in loader:
                if batch.x.device != device:
                    batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.edge_label_index)
                loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label.float())
                loss.backward()
                optimizer.step()

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            metrics = evaluate(model, val_x, val_edge_index, val_label_index, val_label)
            if metrics["auc"] > best_val_auc:
                best_val_auc = metrics["auc"]
                best_epoch = epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch - best_epoch >= cfg.patience:
            break
    timer.stop()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_x = test_data.x.to(device)
    test_edge_index = test_data.edge_index.to(device)
    test_label_index = test_data.edge_label_index.to(device)
    test_label = test_data.edge_label.to(device).float()

    infer = Timer()
    infer.start()
    test_metrics = evaluate(model, test_x, test_edge_index, test_label_index, test_label)
    infer.stop()

    return {
        "seed": seed,
        "val_auc": best_val_auc,
        "test_auc": test_metrics["auc"],
        "test_ap": test_metrics["ap"],
        "test_hits50": test_metrics["hits50"],
        "fit_time": timer.elapsed,
        "inference_time": infer.elapsed,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "params": count_parameters(model),
        "peak_mem_mb": peak_gpu_memory_mb(device),
        "fanout": fanout,
        "edge_split": {
            "train_edges": int(train_data.edge_label_index.size(1)),
            "val_edges": int(val_data.edge_label_index.size(1)),
            "test_edges": int(test_data.edge_label_index.size(1)),
            "msg_passing_edges": int(train_data.edge_index.size(1)),
        },
    }


# --------------------------------------------------------------------------- #
# Multi-seed driver
# --------------------------------------------------------------------------- #
def run(cfg: Config, verbose: bool = True, write: bool = True) -> dict:
    cfg = cfg.resolved()
    device = get_device(cfg.device)
    data, num_features, num_classes = load_dataset(cfg.dataset, cfg.data_root)
    stats = describe(data, num_features, num_classes)

    runs = [run_single(cfg, data, num_features, s, device) for s in cfg.seeds]

    def col(name):
        return [r[name] for r in runs]

    result = {
        "config": {**asdict(cfg), "fanout": resolve_fanout(cfg.num_neighbors, cfg.num_layers)},
        "dataset_stats": stats,
        "params": runs[0]["params"],
        "edge_split": runs[0]["edge_split"],
        "runs": runs,
        "val_auc_mean": mean_std(col("val_auc"))[0],
        "test_auc_mean": mean_std(col("test_auc"))[0],
        "test_auc_std": mean_std(col("test_auc"))[1],
        "test_ap_mean": mean_std(col("test_ap"))[0],
        "test_ap_std": mean_std(col("test_ap"))[1],
        "test_hits50_mean": mean_std(col("test_hits50"))[0],
        "fit_time_mean": mean_std(col("fit_time"))[0],
        "fit_time_std": mean_std(col("fit_time"))[1],
        "inference_time_mean": mean_std(col("inference_time"))[0],
        "epochs_mean": mean_std(col("best_epoch"))[0],
        "peak_mem_mb": max(col("peak_mem_mb")),
    }

    if verbose:
        print(
            f"[{cfg.key}] params={result['params']:,}  "
            f"val_AUC={result['val_auc_mean']:.2f}  "
            f"test_AUC={fmt_mean_std(col('test_auc'))}  "
            f"AP={fmt_mean_std(col('test_ap'))}  "
            f"fit={result['fit_time_mean']:.2f}s  "
            f"infer={result['inference_time_mean']*1000:.1f}ms  "
            f"epochs={result['epochs_mean']:.0f}  "
            f"mem={result['peak_mem_mb']:.0f}MB",
            flush=True,
        )

    if write:
        detail_dir = ensure_dir(os.path.join(cfg.results_dir, "detail"))
        write_json(os.path.join(detail_dir, f"{cfg.key}.json"), result)
        append_csv(os.path.join(cfg.results_dir, "all_results.csv"),
                   _csv_row(cfg, result, stats), CSV_FIELDS)
    return result


def _csv_row(cfg: Config, result: dict, stats: dict) -> dict:
    return {
        "experiment": cfg.experiment,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "mode": cfg.mode,
        "decoder": cfg.decoder,
        "hidden": cfg.hidden,
        "num_layers": cfg.num_layers,
        "heads": cfg.heads,
        "dropout": cfg.dropout,
        "aggr": cfg.aggr,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size if cfg.mode == "sample" else "-",
        "num_neighbors": ",".join(map(str, result["config"]["fanout"])) if cfg.mode == "sample" else "-",
        "params": result["params"],
        "val_auc_mean": round(result["val_auc_mean"], 2),
        "test_auc_mean": round(result["test_auc_mean"], 2),
        "test_auc_std": round(result["test_auc_std"], 2),
        "test_ap_mean": round(result["test_ap_mean"], 2),
        "test_ap_std": round(result["test_ap_std"], 2),
        "test_hits50_mean": round(result["test_hits50_mean"], 2),
        "fit_time_mean": round(result["fit_time_mean"], 2),
        "fit_time_std": round(result["fit_time_std"], 2),
        "inference_time_mean": round(result["inference_time_mean"], 4),
        "epochs_mean": round(result["epochs_mean"], 1),
        "peak_mem_mb": round(result["peak_mem_mb"], 1),
        "num_nodes": stats["num_nodes"],
        "num_edges": stats["num_edges"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ds = " (default: dataset specific -- see code/dataset.py)"
    p = argparse.ArgumentParser(
        description="Task 2: link prediction (GCN/GAT/GraphSAGE/GIN, full vs. sampled)")
    p.add_argument("--dataset", default="Cora", choices=["Cora", "Citeseer", "Flickr"])
    p.add_argument("--model", default="gcn", choices=["gcn", "gat", "sage", "gin"])
    p.add_argument("--mode", default="full", choices=["full", "sample"])
    p.add_argument("--decoder", default="dot", choices=["dot", "mlp"])
    p.add_argument("--neg-sampler", default=Config.neg_sampler, choices=["uniform", "pyg"],
                   help="negative-edge sampler used by the full-graph regime")
    p.add_argument("--hidden", type=int, default=None, help="hidden width" + ds)
    p.add_argument("--num-layers", type=int, default=None, help="message-passing layers" + ds)
    p.add_argument("--heads", type=int, default=None, help="GAT attention heads" + ds)
    p.add_argument("--dropout", type=float, default=None, help="dropout rate" + ds)
    p.add_argument("--aggr", default=None, help="GraphSAGE aggregation" + ds)
    p.add_argument("--lr", type=float, default=None, help="Adam learning rate" + ds)
    p.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay" + ds)
    p.add_argument("--epochs", type=int, default=None, help="max epochs" + ds)
    p.add_argument("--patience", type=int, default=None, help="early stopping patience" + ds)
    p.add_argument("--batch-size", type=int, default=None,
                   help="supervised edges per mini-batch (--mode sample)" + ds)
    p.add_argument("--num-neighbors", type=int, nargs="+", default=None,
                   help="fanout per hop, e.g. --num-neighbors 15 10" + ds)
    p.add_argument("--eval-every", type=int, default=None, help="validate every N epochs" + ds)
    p.add_argument("--disjoint-train-ratio", type=float, default=Config.disjoint_train_ratio,
                   help="fraction of training edges used for supervision only, so that "
                        "they are excluded from message passing")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--batchnorm", dest="batchnorm", action="store_true", default=None)
    p.add_argument("--no-batchnorm", dest="batchnorm", action="store_false")
    p.add_argument("--data-device", default=Config.data_device, choices=["gpu", "cpu"])
    p.add_argument("--experiment", default=Config.experiment)
    p.add_argument("--seeds", type=int, nargs="+", default=list(Config.seeds))
    p.add_argument("--device", default=Config.device)
    p.add_argument("--data-root", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--no-write", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(dataset=args.dataset, model=args.model, mode=args.mode, decoder=args.decoder,
                 neg_sampler=args.neg_sampler)
    for key in TUNABLE:
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            setattr(cfg, key, value)
    cfg.disjoint_train_ratio = args.disjoint_train_ratio
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
    run(config_from_args(args), write=not args.no_write)


if __name__ == "__main__":
    main()
