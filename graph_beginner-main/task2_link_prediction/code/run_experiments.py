"""Task 2 -- run every link-prediction experiment and persist the results.

Experiment groups
-----------------
``main``           3 datasets x 4 models x {full-graph, mini-batch sampling}.
``decoder``        dot-product vs. MLP decoder on top of the same encoder.
``hparam_lr``      learning-rate sweep.
``hparam_depth``   network-depth sweep (1-4 message-passing layers).
``sampling_batch`` mini-batch size sweep for ``LinkNeighborLoader``.
``sampling_fanout``per-hop fanout sweep for ``LinkNeighborLoader``.
``all``            every group above.

Results land in ``task2_link_prediction/results``:
``all_results.csv`` plus ``detail/<config key>.json`` (per-seed numbers).

Usage
-----
>>> python run_experiments.py --list
>>> python run_experiments.py --experiments main
>>> python run_experiments.py --experiments all --reset
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Dict, Iterator, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train  # noqa: E402
from train import Config, TASK_DIR  # noqa: E402

DATASETS = ("Cora", "Citeseer", "Flickr")
MODELS = ("gcn", "gat", "sage", "gin")

LR_SWEEP = [0.001, 0.005, 0.01, 0.05]
DEPTHS = [1, 2, 3, 4]
BATCH_SWEEP = {
    "Cora": [128, 512, 2048, 4488],
    "Citeseer": [128, 512, 2048, 4096],
    "Flickr": [512, 2048, 8192],
}
FANOUT_SWEEP = {
    "Cora": [[5, 5], [10, 10], [15, 15]],
    "Citeseer": [[5, 5], [10, 10], [15, 15]],
    "Flickr": [[5, 5], [10, 10], [15, 10]],
}


def grid_main() -> Iterator[Config]:
    for ds in DATASETS:
        # Flickr has 900 k edges and its sampled runs cost minutes each, so the
        # headline grid uses one seed there and three on the small graphs.  The
        # seed count actually used is stored with every result row.
        ds_seeds = (0,) if ds == "Flickr" else (0, 1, 2)
        for model in MODELS:
            for mode in ("full", "sample"):
                yield Config(dataset=ds, model=model, mode=mode, seeds=ds_seeds,
                             experiment="main")


def grid_decoder() -> Iterator[Config]:
    for ds in DATASETS:
        for model in ("gcn", "sage"):
            for dec in ("dot", "mlp"):
                yield Config(dataset=ds, model=model, mode="full", decoder=dec,
                             tag=f"dec-{dec}", experiment="decoder")


def grid_hparam_lr() -> Iterator[Config]:
    for ds in DATASETS:
        for model in ("gcn", "sage"):
            for lr in LR_SWEEP:
                yield Config(dataset=ds, model=model, mode="full", lr=lr,
                             tag=f"lr{lr:g}", experiment="hparam_lr")


def grid_hparam_depth() -> Iterator[Config]:
    for ds in DATASETS:
        for model in MODELS:
            for nl in DEPTHS:
                yield Config(dataset=ds, model=model, mode="full", num_layers=nl,
                             tag=f"L{nl}", experiment="hparam_depth")


def grid_sampling_batch() -> Iterator[Config]:
    for ds in DATASETS:
        for model in ("gcn", "sage"):
            for bs in BATCH_SWEEP[ds]:
                yield Config(dataset=ds, model=model, mode="sample", batch_size=bs,
                             tag=f"bs{bs}", experiment="sampling_batch")


def grid_sampling_fanout() -> Iterator[Config]:
    for ds in DATASETS:
        for model in ("gcn", "sage"):
            for fan in FANOUT_SWEEP[ds]:
                yield Config(dataset=ds, model=model, mode="sample",
                             num_neighbors=list(fan), tag="f" + "-".join(map(str, fan)),
                             experiment="sampling_fanout")


EXPERIMENTS: Dict[str, "callable"] = {
    "main": grid_main,
    "decoder": grid_decoder,
    "hparam_lr": grid_hparam_lr,
    "hparam_depth": grid_hparam_depth,
    "sampling_batch": grid_sampling_batch,
    "sampling_fanout": grid_sampling_fanout,
}

GROUP_SEEDS = {
    "main": (0, 1, 2),
    "decoder": (0, 1),
    "hparam_lr": (0, 1),
    "hparam_depth": (0, 1),
    "sampling_batch": (0, 1),
    "sampling_fanout": (0, 1),
}


def build_plan(names: List[str], seeds) -> List[Config]:
    if "all" in names:
        names = list(EXPERIMENTS)
    plan: List[Config] = []
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)} or 'all'")
        group_seeds = tuple(seeds) if seeds else GROUP_SEEDS.get(name, (0,))
        for cfg in EXPERIMENTS[name]():
            # A grid may pin its own (smaller) seed set for expensive datasets;
            # an explicit --seeds on the command line always wins.
            if seeds:
                cfg.seeds = group_seeds
            elif cfg.seeds == Config.seeds:
                cfg.seeds = group_seeds
            plan.append(cfg)
    return plan


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Task-2 experiment grid",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=["main"])
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="override per-group seeds (main: 0 1 2, sweeps: 0 1)")
    p.add_argument("--results-dir", default=os.path.join(TASK_DIR, "results"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()

    plan = build_plan(args.experiments, args.seeds)

    if args.list or args.dry_run:
        print(f"{len(plan)} configurations")
        for cfg in plan:
            print(f"  [{cfg.experiment:15s}] {cfg.key:34s} seeds={list(cfg.seeds)}")
        return

    if args.reset and os.path.isdir(args.results_dir):
        shutil.rmtree(args.results_dir)
        print(f"removed {args.results_dir}")
    os.makedirs(args.results_dir, exist_ok=True)
    for cfg in plan:
        cfg.results_dir = args.results_dir

    print(f"\n=== running {len(plan)} configurations ===\n")
    t0 = time.perf_counter()
    failures = []
    for i, cfg in enumerate(plan, 1):
        print(f"--- [{i}/{len(plan)}] {cfg.experiment}: {cfg.key}", flush=True)
        try:
            train.run(cfg, write=True)
        except Exception as exc:  # noqa: BLE001
            failures.append((cfg.key, f"{type(exc).__name__}: {exc}"))
            print(f"!!! {cfg.key} FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(f"    elapsed {time.perf_counter() - t0:.0f}s\n", flush=True)

    print(f"=== done in {time.perf_counter() - t0:.0f}s "
          f"({len(plan) - len(failures)}/{len(plan)} succeeded) ===")
    for key, err in failures:
        print(f"  FAILED {key}: {err}")
    print(f"\nresults -> {os.path.join(args.results_dir, 'all_results.csv')}")


if __name__ == "__main__":
    main()
