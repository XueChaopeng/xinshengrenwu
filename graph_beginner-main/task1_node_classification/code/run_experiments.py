"""Task 1 -- run every experiment and persist the results.

Experiment groups
-----------------
``main``           3 datasets x 4 models x {full-graph, mini-batch sampling}.
                   This is the head-to-head comparison asked for in the README.
``hparam_lr``      learning-rate sweep (full-graph) on Cora / Citeseer / Flickr.
``hparam_depth``   network-depth sweep (full-graph), 1-4 message-passing layers.
``hparam_hidden``  hidden-width sweep (full-graph).
``sampling_batch`` mini-batch size sweep for the neighbour sampler.
``sampling_fanout``per-hop fanout sweep for the neighbour sampler.
``sampling_memory````--data-device gpu`` vs ``cpu``: the memory argument for
                   sampling (the graph stays in host RAM).
``all``            every group above.

Everything is written to ``task1_node_classification/results``:

* ``all_results.csv``            -- one row per (config, seed-average)
* ``detail/<config key>.json``   -- per-seed numbers, config, dataset stats

Usage
-----
>>> python run_experiments.py --list
>>> python run_experiments.py --experiments main
>>> python run_experiments.py --experiments all --seeds 0 1 2 --reset
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

#: Sweeps are chosen so that the seeding still gives every seed node a chance:
#: Cora/Citeseer only have 140 / 120 training nodes, Flickr has 44,625.
LR_SWEEP = [0.001, 0.005, 0.01, 0.05]
DEPTHS = [1, 2, 3, 4]
HIDDEN_SWEEP = [16, 32, 64, 128]
BATCH_SWEEP = {
    "Cora": [16, 32, 64, 128],
    "Citeseer": [16, 32, 64, 128],
    "Flickr": [256, 1024, 4096],
}
FANOUT_SWEEP = {
    "Cora": [[2, 2], [5, 5], [10, 10], [15, 15]],
    "Flickr": [[5, 5], [10, 10], [15, 10]],
}

#: Seeds used for the sweeps: a Cora run costs ~2 s while a Flickr run costs
#: ~40-130 s, so the large graph gets a single seed (the *trend* is what the
#: sweeps are for; the headline ``main`` table uses three seeds on the small
#: graphs).  Written up in the README and REPORT.
SWEEP_SEEDS = {"Cora": (0, 1), "Citeseer": (0, 1), "Flickr": (0,)}

#: Models swept per dataset.  Flickr costs ~40x a Cora run, so the sweeps there
#: use a single representative architecture (GCN) rather than all four; the
#: full 4-model comparison on Flickr is covered by ``main``.
SWEEP_MODELS = {
    "Cora": ("gcn", "gat", "sage", "gin"),
    "Citeseer": ("gcn", "gat", "sage", "gin"),
    "Flickr": ("gcn",),
}


# --------------------------------------------------------------------------- #
# Grid definitions -- each yields fully specified Config objects
# --------------------------------------------------------------------------- #
def grid_main() -> Iterator[Config]:
    for ds in DATASETS:
        # Flickr is ~40x larger than Cora/Citeseer: a single GAT/GIN sampled run
        # there costs several minutes, so the headline grid uses one seed for it
        # and three for the small graphs.  Documented in the README/report.
        ds_seeds = (0,) if ds == "Flickr" else (0, 1, 2)
        for model in MODELS:
            for mode in ("full", "sample"):
                yield Config(dataset=ds, model=model, mode=mode,
                             seeds=ds_seeds, experiment="main")


def grid_hparam_lr() -> Iterator[Config]:
    for ds in DATASETS:
        for model in SWEEP_MODELS[ds]:
            for lr in LR_SWEEP:
                yield Config(dataset=ds, model=model, mode="full", lr=lr,
                             seeds=SWEEP_SEEDS[ds], tag=f"lr{lr:g}",
                             experiment="hparam_lr")


def grid_hparam_depth() -> Iterator[Config]:
    for ds in DATASETS:
        for model in SWEEP_MODELS[ds]:
            for nl in DEPTHS:
                yield Config(dataset=ds, model=model, mode="full", num_layers=nl,
                             seeds=SWEEP_SEEDS[ds], tag=f"L{nl}",
                             experiment="hparam_depth")


def grid_hparam_hidden() -> Iterator[Config]:
    for ds in DATASETS:
        for model in ("gcn", "sage") if ds != "Flickr" else ("gcn",):
            for h in HIDDEN_SWEEP:
                yield Config(dataset=ds, model=model, mode="full", hidden=h,
                             seeds=SWEEP_SEEDS[ds], tag=f"h{h}",
                             experiment="hparam_hidden")


def grid_sampling_batch() -> Iterator[Config]:
    for ds in DATASETS:
        for model in ("gcn", "sage"):
            for bs in BATCH_SWEEP[ds]:
                yield Config(dataset=ds, model=model, mode="sample", batch_size=bs,
                             seeds=SWEEP_SEEDS[ds], tag=f"bs{bs}",
                             experiment="sampling_batch")


def grid_sampling_fanout() -> Iterator[Config]:
    for ds in DATASETS:
        for model in ("gcn", "sage"):
            for fan in FANOUT_SWEEP[ds]:
                yield Config(dataset=ds, model=model, mode="sample",
                             num_neighbors=list(fan), seeds=SWEEP_SEEDS[ds],
                             tag="f" + "-".join(map(str, fan)),
                             experiment="sampling_fanout")


def grid_sampling_memory() -> Iterator[Config]:
    """Same sampler, different home for the graph: GPU-VRAM vs. host RAM."""
    for dd in ("gpu", "cpu"):
        yield Config(dataset="Flickr", model="sage", mode="sample",
                     data_device=dd, tag=f"data-{dd}", experiment="sampling_memory")
    for dd in ("gpu", "cpu"):
        yield Config(dataset="Citeseer", model="sage", mode="sample",
                     data_device=dd, tag=f"data-{dd}", experiment="sampling_memory")


EXPERIMENTS: Dict[str, "callable"] = {
    "main": grid_main,
    "hparam_lr": grid_hparam_lr,
    "hparam_depth": grid_hparam_depth,
    "hparam_hidden": grid_hparam_hidden,
    "sampling_batch": grid_sampling_batch,
    "sampling_fanout": grid_sampling_fanout,
    "sampling_memory": grid_sampling_memory,
}

#: Seeds per group when ``--seeds`` is not given.  The headline comparison
#: (``main``) gets three seeds; the sweeps get two, which is enough for a trend
#: and keeps the whole grid inside a reasonable time budget.
GROUP_SEEDS = {
    "main": (0, 1, 2),
    "hparam_lr": (0, 1),
    "hparam_depth": (0, 1),
    "hparam_hidden": (0, 1),
    "sampling_batch": (0, 1),
    "sampling_fanout": (0, 1),
    "sampling_memory": (0,),
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


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Run the Task-1 experiment grid",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=["main"],
                   help="experiment group names, or 'all'")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="override the per-group default seeds (main: 0 1 2, sweeps: 0 1)")
    p.add_argument("--results-dir", default=os.path.join(TASK_DIR, "results"))
    p.add_argument("--list", action="store_true", help="print the plan and exit")
    p.add_argument("--dry-run", action="store_true", help="same as --list")
    p.add_argument("--reset", action="store_true", help="delete previous results first")
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
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
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
