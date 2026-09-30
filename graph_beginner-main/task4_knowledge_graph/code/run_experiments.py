"""Task 4 -- run every knowledge-graph-completion experiment and persist results.

Experiment groups
-----------------
``main``        {WN18RR, FB15k-237} x {TransE, RotatE, ConvE} -- the head-to-head
                model comparison asked for in the assignment.
``hparam_dim``  embedding-dimension sweep {50, 100, 200, 400}.
``hparam_lr``   learning-rate sweep {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}.
``hparam_neg``  number of negative samples {1, 4, 16, 64, 128} (TransE/RotatE).
``hparam_gamma``margin sweep {3, 6, 9, 12, 24} (TransE/RotatE).
``hparam_bern`` uniform vs. Bernoulli negative sampling.
``ablation_inverse``training with and without inverse relations (RotatE).
``all``         every group above.

Everything is written to ``task4_knowledge_graph/results``:

* ``all_results.csv``           -- one row per (configuration, seed-average)
* ``detail/<config key>.json``  -- per-seed numbers, per-epoch history, config

Usage
-----
>>> python run_experiments.py --list
>>> python run_experiments.py --experiments main
>>> python run_experiments.py --experiments hparam_dim --datasets WN18RR --seeds 0
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

from train import Config, TASK_DIR  # noqa: E402
import train  # noqa: E402

DATASETS = ("WN18RR", "FB15k-237")
MODELS = ("TransE", "RotatE", "ConvE")

DIM_SWEEP = [50, 100, 200, 400]
LR_SWEEP = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
NEG_SWEEP = [1, 4, 16, 64, 128]
GAMMA_SWEEP = [3.0, 6.0, 9.0, 12.0, 24.0]

#: Models used by the sweeps that only make sense for the margin-loss models.
MARGIN_MODELS = ("TransE", "RotatE")


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def grid_main(datasets, models) -> Iterator[Config]:
    for ds in datasets:
        for model in models:
            yield Config(dataset=ds, model=model, experiment="main")


def grid_hparam_dim(datasets, models) -> Iterator[Config]:
    for ds in datasets:
        for model in models:
            for dim in DIM_SWEEP:
                yield Config(dataset=ds, model=model, dim=dim,
                             tag=f"dim{dim}", experiment="hparam_dim")


def grid_hparam_lr(datasets, models) -> Iterator[Config]:
    for ds in datasets:
        for model in models:
            for lr in LR_SWEEP:
                yield Config(dataset=ds, model=model, lr=lr,
                             tag=f"lr{_fmt(lr)}", experiment="hparam_lr")


def grid_hparam_neg(datasets, models) -> Iterator[Config]:
    for ds in datasets:
        for model in [m for m in models if m in MARGIN_MODELS]:
            for k in NEG_SWEEP:
                yield Config(dataset=ds, model=model, neg_samples=k,
                             tag=f"neg{k}", experiment="hparam_neg")


def grid_hparam_gamma(datasets, models) -> Iterator[Config]:
    for ds in datasets:
        for model in [m for m in models if m in MARGIN_MODELS]:
            for g in GAMMA_SWEEP:
                yield Config(dataset=ds, model=model, gamma=g,
                             tag=f"g{_fmt(g)}", experiment="hparam_gamma")


def grid_hparam_bern(datasets, models) -> Iterator[Config]:
    for ds in datasets:
        for model in [m for m in models if m in MARGIN_MODELS]:
            for bern in (False, True):
                yield Config(dataset=ds, model=model, bern=bern,
                             tag="bern" if bern else "uniform", experiment="hparam_bern")


def grid_ablation_inverse(datasets, models) -> Iterator[Config]:
    """RotatE trained with forward+inverse relations vs. forward relations only.

    Both variants are *scored* in both directions; adding the inverse relations
    doubles the number of training triples per epoch and gives the head-side
    ranking its own relation embedding instead of reusing the forward one.
    """
    for ds in datasets:
        if "RotatE" not in models:
            continue
        yield Config(dataset=ds, model="RotatE", inverse_relations=True,
                     tag="fwd+inv", experiment="ablation_inverse")
        yield Config(dataset=ds, model="RotatE", inverse_relations=False,
                     tag="fwd-only", experiment="ablation_inverse")


EXPERIMENTS: Dict[str, "callable"] = {
    "main": grid_main,
    "hparam_dim": grid_hparam_dim,
    "hparam_lr": grid_hparam_lr,
    "hparam_neg": grid_hparam_neg,
    "hparam_gamma": grid_hparam_gamma,
    "hparam_bern": grid_hparam_bern,
    "ablation_inverse": grid_ablation_inverse,
}


def build_plan(names: List[str], seeds, datasets, models) -> List[Config]:
    if "all" in names:
        names = list(EXPERIMENTS)
    plan: List[Config] = []
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)} or 'all'")
        for cfg in EXPERIMENTS[name](datasets, models):
            cfg.seeds = tuple(seeds)
            plan.append(cfg)
    return plan


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Task-4 experiment grid",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=["main"],
                   help="experiment group names, or 'all'")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--results-dir", default=os.path.join(TASK_DIR, "results"))
    p.add_argument("--epochs", type=int, default=None,
                   help="override the epoch budget of every configuration")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override the batch size of every configuration")
    p.add_argument("--eval-every", type=int, default=None,
                   help="override the validation interval of every configuration")
    p.add_argument("--vocab-from", default="all", choices=["all", "train"],
                   help="entity/relation vocabulary source (see code/dataset.py)")
    p.add_argument("--list", action="store_true", help="print the plan and exit")
    p.add_argument("--dry-run", action="store_true", help="same as --list")
    p.add_argument("--reset", action="store_true", help="delete previous results first")
    args = p.parse_args()

    plan = build_plan(args.experiments, args.seeds, args.datasets, args.models)

    #: Recorded with every row so a reduced epoch budget cannot be mistaken for
    #: a full one.
    overrides = {"epochs": args.epochs, "batch_size": args.batch_size,
                 "eval_every": args.eval_every, "vocab_from": args.vocab_from}
    for cfg in plan:
        for key, value in overrides.items():
            if value is not None and not (key == "vocab_from" and value == "all"):
                setattr(cfg, key, value)

    if args.list or args.dry_run:
        print(f"{len(plan)} configurations, seeds={args.seeds}")
        for cfg in plan:
            print(f"  [{cfg.experiment:17s}] {cfg.key}")
        return

    if args.reset and os.path.isdir(args.results_dir):
        shutil.rmtree(args.results_dir)
        print(f"removed {args.results_dir}")

    os.makedirs(args.results_dir, exist_ok=True)
    for cfg in plan:
        cfg.results_dir = args.results_dir

    print(f"\n=== running {len(plan)} configurations x {len(args.seeds)} seeds ===")
    if args.epochs:
        print(f"    epoch budget overridden to {args.epochs}")
    if args.batch_size:
        print(f"    batch size overridden to {args.batch_size}")
    print()
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
