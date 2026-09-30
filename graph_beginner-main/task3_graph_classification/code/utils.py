"""Shared utilities for Task 3: graph classification / graph regression (PyG).

Contents
--------
* :func:`set_seed`          -- reproducible runs
* :func:`get_device`        -- "auto" device selection
* :class:`Timer`            -- CUDA-synchronised wall-clock timing (explicit
  ``start()``/``stop()`` **and** context-manager usage; segments accumulate)
* :func:`accuracy` / :func:`macro_f1` / :func:`mae`
* :func:`count_parameters` / :func:`mean_std` / :func:`fmt_mean_std`
* :func:`ensure_dir` / :func:`write_json` / :func:`append_csv`
* :func:`peak_gpu_memory_mb` / :func:`reset_peak_gpu_memory`
* :func:`human_time`        -- pretty printing of a duration (Timer independent)
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
    manager (``with Timer() as t: ...``).  Elapsed segments accumulate, so a
    timer can be started/stopped once per epoch and still report a total.

    >>> with Timer() as t:
    ...     _ = model(batch.x, batch.edge_index, batch.batch)
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


def human_time(seconds: float) -> str:
    """``12.34s`` / ``3m07.5s`` / ``1h02m`` -- independent of :class:`Timer`."""
    if seconds != seconds:  # NaN
        return "nan"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{s:04.1f}s"
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{int(h)}h{int(m):02d}m"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy in percent (``logits`` may be ``[N]`` or ``[N, C]``)."""
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1) if logits.dim() > 1 else (logits > 0.5).long()
    return (pred.view(-1) == y.view(-1)).float().mean().item() * 100.0


def macro_f1(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 in percent (implemented with plain torch)."""
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1) if logits.dim() > 1 else (logits > 0.5).long()
    pred, y = pred.view(-1), y.view(-1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) * 100.0


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error (lower is better) on whatever scale ``target`` uses."""
    if pred.numel() == 0:
        return float("nan")
    return (pred.view(-1) - target.view(-1)).abs().mean().item()


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
