"""Turn ``results/all_results.csv`` into the markdown tables used in README.md.

The script is deliberately dependency-light (plain ``csv`` + a tiny markdown
writer) so it works in any environment.

Usage
-----
>>> python summarize.py                    # writes results/summary.md and prints it
>>> python summarize.py --experiment main  # only one group
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import TASK_DIR  # noqa: E402

DEFAULT_CSV = os.path.join(TASK_DIR, "results", "all_results.csv")


# --------------------------------------------------------------------------- #
# Small markdown helpers
# --------------------------------------------------------------------------- #
def md_table(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def _num(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _natural_key(value):
    """Sort ``bs512`` before ``bs2048`` instead of lexicographically."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


def pivot(rows: List[dict], index: str, columns: str, values: str,
          agg: str = "mean", digits: int = 2) -> str:
    """Pivot ``rows`` into a markdown table."""
    row_keys = sorted({r[index] for r in rows}, key=_natural_key)
    col_keys = sorted({r[columns] for r in rows}, key=_natural_key)
    buckets: Dict[tuple, List[float]] = {}
    for r in rows:
        try:
            v = float(r[values])
        except (TypeError, ValueError, KeyError):
            continue
        buckets.setdefault((r[index], r[columns]), []).append(v)

    def reduce(vals: List[float]):
        if not vals:
            return None
        if agg == "mean":
            return sum(vals) / len(vals)
        if agg == "max":
            return max(vals)
        if agg == "min":
            return min(vals)
        if agg == "sum":
            return sum(vals)
        raise ValueError(agg)

    table_rows = []
    for rk in row_keys:
        cells = []
        for ck in col_keys:
            v = reduce(buckets.get((rk, ck), []))
            cells.append("" if v is None else _num(v, digits))
        table_rows.append([rk] + cells)
    return md_table([index] + list(col_keys), table_rows)


def load_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def section_main(rows: List[dict]) -> str:
    rows = [r for r in rows if r.get("experiment") == "main"]
    if not rows:
        return "_no `main` rows yet_"
    parts = ["### 主实验：全图 vs 采样子图\n"]
    for mode, title in (("full", "**全图训练 (full-graph)**"),
                        ("sample", "**采样子图训练 (mini-batch sampling)**")):
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        parts.append(f"{title}\n")
        parts.append("测试准确率 Acc (%)\n")
        parts.append(pivot(sub, "dataset", "model", "test_acc_mean"))
        parts.append("\n训练时间 (s)\n")
        parts.append(pivot(sub, "dataset", "model", "fit_time_mean"))
        parts.append("")
    return "\n".join(parts)


def section_full_vs_sample(rows: List[dict]) -> str:
    rows = [r for r in rows if r.get("experiment") == "main"]
    if not rows:
        return ""
    by = {(r["dataset"], r["model"], r["mode"]): r for r in rows}
    out = [md_table(
        ["数据集", "模型", "全图 Acc (%)", "采样 Acc (%)", "Δ Acc",
         "全图时间 (s)", "采样时间 (s)", "加速比"],
        [[
            ds, model,
            _num(by[(ds, model, "full")]["test_acc_mean"]),
            _num(by[(ds, model, "sample")]["test_acc_mean"]),
            f'{float(by[(ds, model, "sample")]["test_acc_mean"]) - float(by[(ds, model, "full")]["test_acc_mean"]):+.2f}',
            _num(by[(ds, model, "full")]["fit_time_mean"], 1),
            _num(by[(ds, model, "sample")]["fit_time_mean"], 1),
            f'{float(by[(ds, model, "full")]["fit_time_mean"]) / max(float(by[(ds, model, "sample")]["fit_time_mean"]), 1e-9):.2f}x',
        ]
         for ds in sorted({r["dataset"] for r in rows})
         for model in sorted({r["model"] for r in rows})
         if (ds, model, "full") in by and (ds, model, "sample") in by]),
    ]
    return "### 全图 vs 采样：逐项对比\n\n" + "\n".join(out)


def section_pivot_group(rows: List[dict], experiment: str, title: str,
                        value: str, digits: int = 2) -> str:
    sub = [r for r in rows if r.get("experiment") == experiment]
    if not sub:
        return ""
    return f"### {title}\n\n{pivot(sub, 'tag', 'dataset/model', value, digits=digits)}\n"


def section_hparam(rows: List[dict], experiment: str, tag_col: str, title: str,
                   value: str = "test_acc_mean") -> str:
    """Pivot a hyper-parameter sweep: rows = swept value, columns = dataset/model."""
    sub = [r for r in rows if r.get("experiment") == experiment]
    if not sub:
        return ""
    for r in sub:
        r["dataset/model"] = f'{r["dataset"]}/{r["model"]}'
        r[tag_col] = r.get(tag_col, "")
    return f"### {title}\n\n{pivot(sub, tag_col, 'dataset/model', value)}\n"


def build_report(rows: List[dict]) -> str:
    parts = [
        "# Task 1 结果汇总（自动生成）\n",
        f"共 {len(rows)} 行配置结果。\n",
        section_main(rows),
        section_full_vs_sample(rows),
        section_hparam(rows, "hparam_lr", "lr", "学习率的影响（全图训练，测试准确率 %）"),
        section_hparam(rows, "hparam_depth", "num_layers", "网络层数的影响（全图训练，测试准确率 %）"),
        section_hparam(rows, "hparam_hidden", "hidden", "隐藏维度的影响（全图训练，测试准确率 %）"),
        section_pivot_group(rows, "sampling_batch", "采样 batch size 的影响（测试准确率 %）",
                            "test_acc_mean"),
        section_pivot_group(rows, "sampling_fanout", "采样 fanout 的影响（测试准确率 %）",
                            "test_acc_mean"),
        section_pivot_group(rows, "sampling_memory", "图存放位置：显存峰值 (MB)",
                            "peak_mem_mb", digits=0),
    ]
    return "\n".join(p for p in parts if p)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise Task-1 results as markdown")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--experiment", default=None, help="only include this group")
    p.add_argument("--out", default=os.path.join(TASK_DIR, "results", "summary.md"))
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"no results yet: {args.csv}")

    rows = load_rows(args.csv)
    if args.experiment:
        rows = [r for r in rows if r.get("experiment") == args.experiment]
    report = build_report(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
