"""Fill the ``<!-- *_RESULTS -->`` markers in the report/README files with tables
generated from each task's ``results/all_results.csv``.

Running this instead of hand-copying numbers guarantees that every figure in the
documentation matches the recorded measurements.

Usage
-----
>>> python _build_report_tables.py            # fill and report what changed
>>> python _build_report_tables.py --check    # only report, write nothing
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, Iterable, List, Sequence

ROOT = os.path.dirname(os.path.abspath(__file__))

TASK1 = os.path.join(ROOT, "task1_node_classification")
TASK2 = os.path.join(ROOT, "task2_link_prediction")
TASK3 = os.path.join(ROOT, "task3_graph_classification")
TASK4 = os.path.join(ROOT, "task4_knowledge_graph")


# --------------------------------------------------------------------------- #
# markdown helpers
# --------------------------------------------------------------------------- #
def natural(value):
    return [int(t) if t.isdigit() else str(t).lower() for t in re.split(r"(\d+)", str(value))]


def sort_key(value):
    """Numeric when possible, natural-string otherwise.

    ``natural`` alone maps both ``0.01`` and ``0.001`` to ``['0.', 1, '']``
    (because ``int('01') == int('001')``), which leaves learning-rate rows in an
    arbitrary order.  Parsing as a float first keeps sweeps properly sorted while
    still handling labels like ``f5-5``.
    """
    try:
        return (0, float(value), [])
    except (TypeError, ValueError):
        return (1, 0.0, natural(value))


def md_table(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def num(value, digits=2):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if f != f:                      # NaN
        return "—"
    return f"{f:.{digits}f}"


def mean_std(mean, std, digits=2):
    m = num(mean, digits)
    if m == "—":
        return "—"
    s = num(std, digits)
    return m if s == "—" else f"{m} ± {s}"


def load(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    # utf-8-sig also accepts plain utf-8; PowerShell's Export-Csv writes a BOM
    # that would otherwise become part of the first column name.
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pivot(rows: List[dict], index: str, columns: str, value_fn, digits=2) -> str:
    row_keys = sorted({r[index] for r in rows}, key=sort_key)
    col_keys = sorted({r[columns] for r in rows}, key=sort_key)
    index_map = {}
    for r in rows:
        index_map[(r[index], r[columns])] = r
    body = []
    for rk in row_keys:
        body.append([rk] + [value_fn(index_map.get((rk, ck))) for ck in col_keys])
    return md_table([index] + list(col_keys), body)


# --------------------------------------------------------------------------- #
# Task 1 -- node classification
# --------------------------------------------------------------------------- #
def task1_tables() -> Dict[str, str]:
    rows = load(os.path.join(TASK1, "results", "all_results.csv"))
    if not rows:
        return {}

    # The headline grid was first run while other CUDA jobs were active, which
    # inflated every wall-clock number (Flickr GCN full-graph: 213 s vs. 46 s on
    # an idle GPU).  A clean re-run lives in results/timing/ -- prefer it for the
    # speed comparison, and fall back to the original if it is absent.
    timing = os.path.join(TASK1, "results", "timing", "all_results.csv")
    headline = load(timing) if os.path.exists(timing) else rows

    main = [r for r in headline if r.get("experiment") == "main"]
    out = []

    def cell(r, col="test_acc_mean", std="test_acc_std"):
        return "—" if r is None else mean_std(r[col], r[std])

    if main:
        out.append("#### 2.2 不同神经网络的影响（测试准确率 %）\n")
        out.append(pivot(main, "dataset", "model", cell))
        out.append("\n注：Cora/Citeseer 为 3 个种子的均值 ± 标准差；Flickr 因单次训练"
                   "成本高（约 40–130 秒）用 1 个种子，故无标准差。\n")

        out.append("\n#### 2.3 全图训练 vs 采样子图训练\n")
        full = {(r["dataset"], r["model"]): r for r in main if r["mode"] == "full"}
        samp = {(r["dataset"], r["model"]): r for r in main if r["mode"] == "sample"}
        body = []
        for key in sorted(full, key=lambda k: (natural(k[0]), natural(k[1]))):
            f, s = full[key], samp.get(key)
            if s is None:
                continue
            try:
                delta = float(s["test_acc_mean"]) - float(f["test_acc_mean"])
                speed = float(f["fit_time_mean"]) / max(float(s["fit_time_mean"]), 1e-9)
                body.append([key[0], key[1].upper(),
                             num(f["test_acc_mean"]), num(s["test_acc_mean"]),
                             f"{delta:+.2f}",
                             num(f["fit_time_mean"], 1),
                             num(s["fit_time_mean"], 1),
                             f"{speed:.2f}×"])
            except (TypeError, ValueError):
                continue
        out.append("\n**性能与运行时间逐项对比**（准确率 %，时间秒，末列为全图/采样）\n")
        out.append(md_table(["数据集", "模型", "全图 Acc", "采样 Acc", "Δ Acc",
                             "全图时间", "采样时间", "全图/采样"], body))

        try:
            accs = [float(samp[k]["test_acc_mean"]) - float(full[k]["test_acc_mean"])
                    for k in full if k in samp]
            wins = sum(1 for a in accs if a >= 0)
            avg = sum(accs) / len(accs)
            out.append(f"\n> {len(accs)} 组对比中，采样训练在 **{wins}/{len(accs)}** 组上不劣于"
                       f"全图训练，平均提升 **{avg:+.2f}** 个百分点。\n")
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    main_tables = "\n".join(out)

    # The task README already carries a hand-written version of the headline
    # table, so only the sweeps are injected there; REPORT.md gets everything.
    sweeps = []
    for exp, title, col in (
            ("hparam_lr", "学习率的影响（全图训练，测试准确率 %）", "lr"),
            ("hparam_depth", "网络层数的影响（全图训练，测试准确率 %）", "num_layers"),
            ("hparam_hidden", "隐藏维度的影响（全图训练，测试准确率 %）", "hidden")):
        sub = [r for r in rows if r.get("experiment") == exp]
        if not sub:
            continue
        for r in sub:
            r["dataset/model"] = f'{r["dataset"]}/{r["model"].upper()}'
        sweeps.append(f"\n#### {title}\n")
        sweeps.append(pivot(sub, col, "dataset/model",
                            lambda r: "—" if r is None else num(r["test_acc_mean"])))

    for exp, title, value, digits in (
            ("sampling_batch", "采样 batch size 的影响（测试准确率 %）", "test_acc_mean", 2),
            ("sampling_fanout", "采样 fanout 的影响（测试准确率 %）", "test_acc_mean", 2),
            ("sampling_memory", "整图存放位置：峰值显存 (MB)", "peak_mem_mb", 0)):
        sub = [r for r in rows if r.get("experiment") == exp]
        if not sub:
            continue
        for r in sub:
            r["dataset/model"] = f'{r["dataset"]}/{r["model"].upper()}'
        sweeps.append(f"\n#### {title}\n")
        sweeps.append(pivot(sub, "dataset/model", "tag",
                            lambda r: "—" if r is None else num(r[value], digits)))

    result = {}
    if main_tables:
        result["<!-- TASK1_MAIN -->"] = main_tables
    if sweeps:
        result["<!-- TASK1_RESULTS -->"] = "\n".join(sweeps)
    return result


# --------------------------------------------------------------------------- #
# Task 2 -- link prediction
# --------------------------------------------------------------------------- #
def task2_tables() -> Dict[str, str]:
    rows = load(os.path.join(TASK2, "results", "all_results.csv"))
    main = [r for r in rows if r.get("experiment") == "main"]
    if not main:
        return {}
    out = []

    out.append("#### 3.2 不同神经网络的影响（测试集 AUC %）\n")
    out.append(pivot(main, "dataset", "model",
                     lambda r: "—" if r is None else mean_std(r["test_auc_mean"], r["test_auc_std"])))

    out.append("\n#### 3.3 全图训练 vs 采样子图训练\n")
    full = {(r["dataset"], r["model"]): r for r in main if r["mode"] == "full"}
    samp = {(r["dataset"], r["model"]): r for r in main if r["mode"] == "sample"}
    body = []
    for key in sorted(full, key=lambda k: (natural(k[0]), natural(k[1]))):
        f, s = full[key], samp.get(key)
        if s is None:
            continue
        try:
            delta = float(s["test_auc_mean"]) - float(f["test_auc_mean"])
            speed = float(f["fit_time_mean"]) / max(float(s["fit_time_mean"]), 1e-9)
            body.append([key[0], key[1].upper(),
                         num(f["test_auc_mean"]), num(s["test_auc_mean"]), f"{delta:+.2f}",
                         num(f["fit_time_mean"], 1),
                         num(s["fit_time_mean"], 1), f"{speed:.2f}×"])
        except (TypeError, ValueError):
            continue
    out.append(md_table(["数据集", "模型", "全图 AUC", "采样 AUC", "Δ AUC",
                         "全图时间", "采样时间", "全图/采样"], body))
    return {"<!-- TASK2_RESULTS -->": "\n".join(out)}


# --------------------------------------------------------------------------- #
# Task 3 -- graph classification
# --------------------------------------------------------------------------- #
def task3_tables() -> Dict[str, str]:
    rows = load(os.path.join(TASK3, "results", "all_results.csv"))
    if not rows:
        return {}
    out = []
    reg = {"ZINC"}

    def metric(r):
        if r is None:
            return "—"
        if r["dataset"] in reg:
            return f'MAE {num(r["test_mae_mean"], 4)}'
        return f'Acc {num(r["test_acc_mean"])}'

    main = [r for r in rows if r.get("experiment") == "main"]
    if main:
        out.append("#### 4.2 不同神经网络的影响（池化固定 mean）\n")
        for r in main:
            r["metric"] = metric(r)
        out.append(pivot(main, "dataset", "model", lambda r: "—" if r is None else r["metric"]))

    pooling = [r for r in rows if r.get("experiment") == "pooling"]
    if pooling:
        out.append("\n#### 4.3 池化方式的影响（作业重点）\n")
        out.append("**图分类（测试准确率 %）**\n")
        cls = [r for r in pooling if r["dataset"] not in reg]
        if cls:
            out.append(pivot(cls, "dataset", "pool",
                             lambda r: "—" if r is None else mean_std(r["test_acc_mean"], r["test_acc_std"])))
        zin = [r for r in pooling if r["dataset"] in reg]
        if zin:
            out.append("\n**ZINC 图回归（MAE，越低越好）**\n")
            out.append(pivot(zin, "pool", "model",
                             lambda r: "—" if r is None else mean_std(r["test_mae_mean"], r["test_mae_std"], 4)))
        out.append("\n**同一池化方式在不同模型下的表现（测试准确率 %）**\n")
        if cls:
            out.append(pivot(cls, "pool", "model",
                             lambda r: "—" if r is None else mean_std(r["test_acc_mean"], r["test_acc_std"])))

    for exp, title, col in (("hparam_depth", "网络层数的影响（% 或 MAE）", "num_layers"),
                            ("hparam_lr", "学习率的影响", "lr")):
        sub = [r for r in rows if r.get("experiment") == exp]
        if not sub:
            continue
        for r in sub:
            r["dataset/model"] = f'{r["dataset"]}/{r["model"].upper()}'
            r["metric"] = metric(r)
        out.append(f"\n#### {title}\n")
        out.append(pivot(sub, col, "dataset/model", lambda r: "—" if r is None else r["metric"]))

    return {"<!-- TASK3_RESULTS -->": "\n".join(out)}


# --------------------------------------------------------------------------- #
# Task 4 -- knowledge graph completion
# --------------------------------------------------------------------------- #
def task4_tables() -> Dict[str, str]:
    rows = load(os.path.join(TASK4, "results", "all_results.csv"))
    if not rows:
        return {}
    out = []
    main = [r for r in rows if r.get("experiment") == "main"]
    if main:
        out.append("#### 5.2 三个模型的对比（filtered ranking，% 除 MRR 外）\n")
        body = []
        for r in sorted(main, key=lambda r: (natural(r["dataset"]), natural(r["model"]))):
            body.append([r["dataset"], r["model"], r.get("dim", "—"),
                         num(r["test_mrr_mean"], 4),
                         num(r["test_hits1_mean"]), num(r["test_hits3_mean"]),
                         num(r["test_hits10_mean"]),
                         num(r["fit_time_mean"], 0),
                         r.get("epochs_run", "—")])
        out.append(md_table(["数据集", "模型", "维度", "MRR", "Hits@1", "Hits@3",
                             "Hits@10", "训练时间(s)", "实跑轮数"], body))
        out.append("\n> 训练时间包含每轮验证；`实跑轮数` 为早停实际执行的轮数，"
                   "用于核对是否因预算不足而欠拟合。\n")

    for exp, title, col, digits in (
            ("hparam_dim", "嵌入维度的影响", "dim", 4),
            ("hparam_lr", "学习率的影响", "lr", 4),
            ("hparam_neg", "负样本数的影响", "neg_samples", 4),
            ("hparam_gamma", "margin 的影响", "gamma", 4),
            ("ablation_inverse", "逆关系消融", "tag", 4)):
        sub = [r for r in rows if r.get("experiment") == exp]
        if not sub:
            continue
        for r in sub:
            r["dataset/model"] = f'{r["dataset"]}/{r["model"]}'
        out.append(f"\n#### {title}（测试 MRR）\n")
        out.append(pivot(sub, "dataset/model", col,
                         lambda r: "—" if r is None else num(r["test_mrr_mean"], digits)))

    return {"<!-- TASK4_RESULTS -->": "\n".join(out)}


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Fill report placeholders from result CSVs")
    p.add_argument("--check", action="store_true", help="report only, write nothing")
    p.add_argument("--only", nargs="+", default=None,
                   choices=["task1", "task2", "task3", "task4"],
                   help="only fill the markers of these tasks (useful while one task "
                        "is still being re-run and its CSV holds stale numbers)")
    args = p.parse_args()

    replacements: Dict[str, str] = {}
    for fn in (task1_tables, task2_tables, task3_tables, task4_tables):
        try:
            replacements.update(fn())
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {fn.__name__} failed: {type(exc).__name__}: {exc}")

    if args.only:
        wanted = tuple(f"TASK{n[-1]}" for n in args.only)
        replacements = {k: v for k, v in replacements.items()
                        if any(w in k for w in wanted)}

    targets = [os.path.join(ROOT, "REPORT.md"),
               os.path.join(TASK1, "README.md"),
               os.path.join(TASK2, "README.md"),
               os.path.join(TASK3, "README.md"),
               os.path.join(TASK4, "README.md")]

    for path in targets:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        original = text
        for marker, table in replacements.items():
            # NB: markers are the same in REPORT.md and the task READMEs, so a
            #     marker present in both files is filled in both.
            if marker in text:
                text = text.replace(marker, table)
        if text != original:
            if args.check:
                print(f"would update {os.path.relpath(path, ROOT)}")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"updated {os.path.relpath(path, ROOT)}")
        else:
            left = [m for m in replacements if m in text]
            state = f"({len(left)} markers left unfilled)" if left else ""
            print(f"unchanged {os.path.relpath(path, ROOT)} {state}")

    # Placeholders that never got data are surfaced explicitly so that a missing
    # experiment cannot silently ship as an empty section.
    for path in targets:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for m in re.findall(r"<!--\s*([A-Z0-9_]+_PLACEHOLDER|TASK\d_RESULTS)\s*-->", f.read()):
                print(f"  UNFILLED {os.path.relpath(path, ROOT)}: {m}")


if __name__ == "__main__":
    main()
