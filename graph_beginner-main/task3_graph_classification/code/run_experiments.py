"""Task 3 -- run every experiment and persist the results.

Experiment groups
-----------------
``main``          5 datasets x 4 models, pooling fixed to ``mean`` so that the
                  **architecture** is the only variable.  This is requirement 4
                  of the assignment (test GCN / GAT / GraphSAGE / GIN).
``pooling``       dataset x model x ``{mean, max, min, sum}``.  This is the
                  **pooling analysis** explicitly asked for in the README
                  (AvgPooling vs. MaxPooling vs. MinPooling).
``hparam_lr``     learning-rate sweep.
``hparam_depth``  1-4 message-passing layers.
``hparam_hidden`` hidden-width sweep.
``all``           every group above.

Everything is written to ``task3_graph_classification/results``:

* ``all_results.csv``            -- one row per (config, seed-average)
* ``detail/<config key>.json``   -- per-seed numbers, config, dataset stats

Parallel / sharded execution
----------------------------
The full grid takes hours, so the runner can be sharded and re-merged.  Each
shard gets its own results directory (this avoids several processes appending
to the same CSV at once), and ``--merge`` folds the shards back into the final
``all_results.csv`` + ``detail/``:

>>> python run_experiments.py --experiments all --seeds 0 1 2 \\
...     --datasets MUTAG ENZYMES --results-dir results/_shards/tu1
>>> python run_experiments.py --merge --results-dir results

Usage
-----
>>> python run_experiments.py --list
>>> python run_experiments.py --experiments main pooling
>>> python run_experiments.py --experiments all --seeds 0 1 2 --reset
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import time
from typing import Dict, Iterator, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train  # noqa: E402
from dataset import DATASETS, TU_DATASETS, ZINC_DATASET  # noqa: E402
from train import Config, TASK_DIR  # noqa: E402

MODELS = ("gcn", "gat", "sage", "gin")
POOLS = ("mean", "max", "min", "sum")
DEPTHS = (1, 2, 3, 4)
LR_SWEEP = (0.0005, 0.001, 0.005, 0.01)
HIDDEN_SWEEP = (16, 32, 64, 128)

#: ZINC costs roughly two orders of magnitude more per configuration than the
#: TUDatasets (10,000 training graphs, and a run does not early-stop before
#: ~100 epochs), so its depth and hidden sweeps are restricted to two
#: representative architectures.  ZINC *is* swept over all four models in
#: ``main`` and ``pooling`` -- the two groups the assignment asks for.
ZINC_SWEEP_MODELS = ("gcn", "gin")

#: Architectures used for the ZINC *pooling* study (see ``grid_pooling``).
ZINC_POOL_MODELS = ("gcn", "gin")

#: Columns that together identify one planned configuration.  Used by --resume
#: to skip work that a previous (possibly interrupted) run already finished.
RESUME_KEY = ("experiment", "dataset", "model", "pool", "hidden", "num_layers",
              "heads", "dropout", "lr", "weight_decay", "batch_size",
              "standardize", "batchnorm", "residual", "fold")


def resumable_key_from_cfg(cfg: Config) -> tuple:
    resolved = cfg.resolved()
    values = {f: getattr(resolved, f) for f in RESUME_KEY if f != "fold"}
    values["fold"] = resolved.folds          # Config.folds <-> CSV "fold"
    if values["batchnorm"] is None:          # same auto-rule train.py writes out
        values["batchnorm"] = resolved.model == "gin"
    return tuple(str(values[f]).lower() for f in RESUME_KEY)


def resumable_key_from_row(row: dict) -> tuple:
    return tuple(str(row.get(f, "")).lower() for f in RESUME_KEY)


def completed_keys(results_dir: str) -> set:
    """Keys of configurations already present in ``<results_dir>/all_results.csv``."""
    path = os.path.join(results_dir, "all_results.csv")
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {resumable_key_from_row(r) for r in csv.DictReader(f)}


def _grid(datasets: Sequence[str], models: Sequence[str],
          pools: Sequence[str] = ("mean",), **kwargs) -> Iterator[Config]:
    """Cartesian product of datasets x models x pools with extra fixed knobs."""
    experiment = kwargs.pop("experiment")
    for ds in datasets:
        for model in models:
            for pool in pools:
                yield Config(dataset=ds, model=model, pool=pool,
                             experiment=experiment, **kwargs)


# --------------------------------------------------------------------------- #
# Grid definitions -- each yields fully specified Config objects
# --------------------------------------------------------------------------- #
def grid_main() -> Iterator[Config]:
    """4 models x 5 datasets, pooling fixed to the mean readout."""
    yield from _grid(DATASETS, MODELS, experiment="main")


def grid_pooling() -> Iterator[Config]:
    """The pooling study asked for in the README: mean / max / min / sum.

    The four TUDatasets are cheap, so all four architectures are swept there.
    ZINC costs ~50 s per configuration, so its pooling study is restricted to
    the two architectures the GIN paper compares on ZINC (GIN itself, and GCN
    as the mean-aggregation baseline).
    """
    for ds in TU_DATASETS:
        yield from _grid((ds,), MODELS, POOLS, experiment="pooling")
    yield from _grid((ZINC_DATASET,), ZINC_POOL_MODELS, POOLS, experiment="pooling")


def grid_hparam_lr() -> Iterator[Config]:
    for ds in TU_DATASETS:
        for model in MODELS:
            for lr in LR_SWEEP:
                yield Config(dataset=ds, model=model, pool="mean", lr=lr,
                             tag=f"lr{lr:g}", experiment="hparam_lr")
    for model in ZINC_SWEEP_MODELS:
        for lr in (0.0001, 0.0005, 0.001, 0.005):
            yield Config(dataset=ZINC_DATASET, model=model, pool="mean", lr=lr,
                         tag=f"lr{lr:g}", experiment="hparam_lr")


def grid_hparam_depth() -> Iterator[Config]:
    for ds in TU_DATASETS:
        for model in MODELS:
            for nl in DEPTHS:
                yield Config(dataset=ds, model=model, pool="mean", num_layers=nl,
                             tag=f"L{nl}", experiment="hparam_depth")
    for model in ZINC_SWEEP_MODELS:
        for nl in DEPTHS:
            yield Config(dataset=ZINC_DATASET, model=model, pool="mean",
                         num_layers=nl, tag=f"L{nl}", experiment="hparam_depth")


def grid_hparam_hidden() -> Iterator[Config]:
    for ds in TU_DATASETS:
        for model in ("gcn", "sage"):
            for h in HIDDEN_SWEEP:
                yield Config(dataset=ds, model=model, pool="mean", hidden=h,
                             tag=f"h{h}", experiment="hparam_hidden")
    for model in ZINC_SWEEP_MODELS:
        for h in (32, 64, 128):
            yield Config(dataset=ZINC_DATASET, model=model, pool="mean", hidden=h,
                         tag=f"h{h}", experiment="hparam_hidden")


EXPERIMENTS: Dict[str, "callable"] = {
    "main": grid_main,
    "pooling": grid_pooling,
    "hparam_lr": grid_hparam_lr,
    "hparam_depth": grid_hparam_depth,
    "hparam_hidden": grid_hparam_hidden,
}


def build_plan(names: List[str], seeds, datasets=None, models=None,
               shard: Sequence[int] | None = None) -> List[Config]:
    if "all" in names:
        names = list(EXPERIMENTS)
    plan: List[Config] = []
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; choose from "
                             f"{sorted(EXPERIMENTS)} or 'all'")
        for cfg in EXPERIMENTS[name]():
            if datasets and cfg.dataset not in datasets:
                continue
            if models and cfg.model not in models:
                continue
            cfg.seeds = tuple(seeds)
            plan.append(cfg)
    if shard:
        index, total = shard
        # Interleaving (rather than slicing into blocks) spreads the expensive
        # datasets evenly, because every grid iterates dataset-major.
        plan = [cfg for i, cfg in enumerate(plan) if i % total == index]
    return plan


# --------------------------------------------------------------------------- #
# Shard merging
# --------------------------------------------------------------------------- #
def merge_results(root: str) -> int:
    """Fold every ``<root>/*/all_results.csv`` into ``<root>/all_results.csv``."""
    shards = sorted(glob.glob(os.path.join(root, "*", "all_results.csv")))
    if not shards:
        print(f"no shard CSVs found under {root}/*/all_results.csv")
        return 1

    fields: List[str] = []
    rows: List[dict] = []
    for path in shards:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for name in (r.fieldnames or []):
                if name not in fields:
                    fields.append(name)
            rows.extend(r)
        # detail/*.json for the same shard
        detail_src = os.path.join(os.path.dirname(path), "detail")
        detail_dst = os.path.join(root, "detail")
        os.makedirs(detail_dst, exist_ok=True)
        for jf in glob.glob(os.path.join(detail_src, "*.json")):
            shutil.copyfile(jf, os.path.join(detail_dst, os.path.basename(jf)))
        print(f"  merged {len(rows):5d} cumulative rows from {path}")

    # Deterministic ordering so the report tables are reproducible.
    def sort_key(row: dict):
        return (row.get("experiment", ""), row.get("dataset", ""),
                row.get("model", ""), row.get("pool", ""),
                float(row.get("num_layers") or 0), float(row.get("hidden") or 0),
                float(row.get("lr") or 0))

    rows.sort(key=sort_key)
    out = os.path.join(root, "all_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Run the Task-3 experiment grid",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=["main"],
                   help="experiment group names, or 'all'")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--results-dir", default=os.path.join(TASK_DIR, "results"))
    p.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS),
                   help="restrict the plan to these datasets (sharding helper)")
    p.add_argument("--models", nargs="+", default=None, choices=list(MODELS),
                   help="restrict the plan to these models (sharding helper)")
    p.add_argument("--shard", nargs=2, type=int, default=None,
                   metavar=("INDEX", "TOTAL"),
                   help="run only configuration INDEX, INDEX+TOTAL, ... of the "
                        "plan; several shards can run in parallel as long as "
                        "each one gets its own --results-dir (see --merge)")
    p.add_argument("--list", action="store_true", help="print the plan and exit")
    p.add_argument("--dry-run", action="store_true", help="same as --list")
    p.add_argument("--reset", action="store_true", help="delete previous results first")
    p.add_argument("--merge", action="store_true",
                   help="merge <results-dir>/*/all_results.csv into "
                        "<results-dir>/all_results.csv and exit")
    p.add_argument("--no-resume", action="store_true",
                   help="re-run configurations that are already in "
                        "<results-dir>/all_results.csv (default: skip them, so an "
                        "interrupted sweep simply continues where it stopped)")
    args = p.parse_args()

    if args.merge:
        sys.exit(merge_results(args.results_dir))

    # ``--reset`` must happen *before* the resume scan: otherwise the scan reads
    # the very CSV that is about to be deleted, skips everything it finds there,
    # and the run deletes those results while re-running nothing.  (That is how
    # the TUDataset ``main`` rows once vanished while only ZINC got re-run.)
    if args.reset and os.path.isdir(args.results_dir):
        shutil.rmtree(args.results_dir)
        print(f"removed {args.results_dir}")
        args.no_resume = True

    plan = build_plan(args.experiments, args.seeds, args.datasets, args.models,
                      args.shard)

    if not args.no_resume:
        done = completed_keys(args.results_dir)
        if done:
            before = len(plan)
            plan = [c for c in plan if resumable_key_from_cfg(c) not in done]
            print(f"resume: {before - len(plan)} configuration(s) already in "
                  f"{args.results_dir}/all_results.csv, {len(plan)} left", flush=True)

    if args.list or args.dry_run:
        print(f"{len(plan)} configurations, seeds={args.seeds}")
        for cfg in plan:
            print(f"  [{cfg.experiment:15s}] {cfg.key}")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    for cfg in plan:
        cfg.results_dir = args.results_dir

    print(f"\n=== running {len(plan)} configurations x {len(args.seeds)} seeds "
          f"===\n", flush=True)
    t0 = time.perf_counter()
    failures = []
    for i, cfg in enumerate(plan, 1):
        print(f"--- [{i}/{len(plan)}] {cfg.experiment}: {cfg.key}  "
              f"(elapsed {time.perf_counter() - t0:.0f}s)", flush=True)
        try:
            train.run(cfg, write=True)
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            failures.append((cfg.key, f"{type(exc).__name__}: {exc}"))
            print(f"!!! {cfg.key} FAILED: {type(exc).__name__}: {exc}", flush=True)

    print(f"\n=== done in {time.perf_counter() - t0:.0f}s "
          f"({len(plan) - len(failures)}/{len(plan)} succeeded) ===")
    for key, err in failures:
        print(f"  FAILED {key}: {err}")
    print(f"\nresults -> {os.path.join(args.results_dir, 'all_results.csv')}")


if __name__ == "__main__":
    main()
