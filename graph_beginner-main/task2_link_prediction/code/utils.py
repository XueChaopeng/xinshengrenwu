"""Shared utilities for Task 1: node classification with PyTorch Geometric.

Contents
--------
* :func:`set_seed`          -- reproducible runs
* :func:`get_device`        -- "auto" device selection
* :class:`Timer`            -- CUDA-synchronised wall-clock timing
* :func:`accuracy` / :func:`macro_f1`
* :func:`append_csv` / :func:`write_json` -- result persistence helpers
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Iterable, Optional

import numpy as np
import torch

# Import the metric implementations once, at module load time.  Importing
# scikit-learn lazily inside the metric functions would charge several seconds
# of first-import time to whatever ``Timer`` happens to wrap the first call.
try:  # pragma: no cover - depends on the environment
    from sklearn.metrics import average_precision_score as _sk_average_precision
    from sklearn.metrics import roc_auc_score as _sk_roc_auc
except ImportError:  # pragma: no cover
    _sk_roc_auc = None
    _sk_average_precision = None


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """Resolve ``"auto"`` into cuda when available, otherwise cpu."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def peak_gpu_memory_mb(device: torch.device) -> float:
    """Peak allocated GPU memory in MiB (0.0 on CPU)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_peak_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class Timer:
    """Wall-clock timer that synchronises CUDA before/after measuring.

    Usable either explicitly (``t.start()`` / ``t.stop()``) or as a context
    manager (``with Timer() as t: ...``).  Elapsed segments accumulate.

    >>> with Timer() as t:
    ...     _ = model(x, edge_index)
    >>> t.elapsed            # doctest: +SKIP
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._t0: Optional[float] = None

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self) -> "Timer":
        if self._t0 is not None:  # a segment is already open: close it first
            self.stop()
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        if self._t0 is None:
            return self
        self._sync()
        self.elapsed += time.perf_counter() - self._t0
        self._t0 = None
        return self

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def reset(self) -> None:
        self.elapsed = 0.0
        self._t0 = None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy in percent."""
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item() * 100.0


def macro_f1(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 in percent (implemented with plain torch)."""
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) * 100.0


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Link-prediction metrics
# --------------------------------------------------------------------------- #
def roc_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Area under the ROC curve, in percent.

    Uses scikit-learn when available (it handles tied scores correctly) and
    falls back to an equivalent rank-based implementation.
    """
    if _sk_roc_auc is not None:
        y = labels.detach().cpu().numpy()
        if len(np.unique(y)) < 2:
            return float("nan")
        s = scores.detach().cpu().float().numpy()
        return float(_sk_roc_auc(y, s)) * 100.0
    return _roc_auc_rank(labels, scores) * 100.0


def _roc_auc_rank(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Mann-Whitney U formulation of the AUC (tie-aware); returns a fraction."""
    labels = labels.bool()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    cum = torch.cumsum(counts, 0).float()
    start = cum - counts.float()
    avg_rank = start + (counts.float() + 1.0) / 2.0
    ranks = avg_rank[inverse]
    return (ranks[labels].sum().item() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Average precision (area under the precision-recall curve), in percent."""
    if _sk_average_precision is not None:
        y = labels.detach().cpu().numpy()
        if y.sum() == 0:
            return float("nan")
        s = scores.detach().cpu().float().numpy()
        return float(_sk_average_precision(y, s)) * 100.0
    return _average_precision_torch(labels, scores) * 100.0


def _average_precision_torch(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.float()
    order = torch.argsort(scores, descending=True)
    labels = labels[order]
    cum_tp = torch.cumsum(labels, 0)
    precision = cum_tp / torch.arange(1, labels.numel() + 1, dtype=torch.float)
    total = labels.sum()
    if total == 0:
        return float("nan")
    return (precision * labels).sum().item() / total.item()


def hits_at_k(labels: torch.Tensor, scores: torch.Tensor, k: int) -> float:
    """Fraction of positives that land in the global top-``k`` of the scores.

    A cheap split-wide ranking metric; ``AUC`` / ``AP`` remain the primary
    numbers reported in the README.
    """
    labels = labels.bool()
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, int(labels.numel()))
    topk = torch.topk(scores, k).indices
    return float(labels[topk].sum()) / n_pos * 100.0


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def fmt_mean_std(values: Iterable[float], digits: int = 2) -> str:
    m, s = mean_std(values)
    if m != m:
        return "nan"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


# --------------------------------------------------------------------------- #
# Result persistence
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def append_csv(path: str, row: dict, fieldnames: Optional[list[str]] = None) -> None:
    """Append ``row`` to a CSV file, creating it with a header on first write."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if fieldnames is None:
        fieldnames = list(row.keys())
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.1f}s"
