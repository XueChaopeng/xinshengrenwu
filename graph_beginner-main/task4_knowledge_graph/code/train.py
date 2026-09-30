"""Task 4 -- knowledge-graph completion with TransE / RotatE / ConvE.

Examples
--------
>>> # the two commands the assignment asks for
>>> python train.py --dataset WN18RR --model RotatE --epochs 200
>>> python train.py --dataset FB15k-237 --model TransE --dim 200
>>> # a quick smoke test
>>> python train.py --dataset WN18RR --model TransE --epochs 2 --eval-every 1 --seeds 0

Objectives
----------
* ``TransE`` / ``RotatE`` -- **negative sampling + margin ranking loss**
  (``--gamma``), ``--neg-samples`` corrupted triples per positive triple,
  corrupting the head or the tail (uniformly, or Bernoulli with ``--bern``).
* ``ConvE`` -- **1-vs-all (1-N) scoring**: every training triple is a query
  whose positive is its true tail and whose negatives are *all other entities*,
  trained with binary cross-entropy and label smoothing (``--label-smoothing``,
  default 0.1).  Use ``--conv-e-all-negatives false`` to fall back to sampled
  negatives when memory/time is tight (this is reported in the results table).

Protocol
--------
Both directions are scored for every test triple (the tail against all entities
with relation ``r``, the head against all entities with the trained *inverse*
relation ``r^-1``) and the standard **filtered** ranking removes the other known
true answers before ranking.  Reported metrics: MRR, Hits@1/3/10, MR, plus
training time and peak GPU memory.  Model selection uses **validation** MRR; the
test set is scored once with the best validation parameters.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (  # noqa: E402
    DATASET_DEFAULTS,
    MODEL_DEFAULTS,
    KGDataset,
    bernoulli_weights,
    build_filter_lookup,
    default_root,
    evaluate_filtered,
    load_dataset,
    sample_negatives,
)
from models import MODELS, build_model  # noqa: E402
from utils import (  # noqa: E402
    Timer,
    append_csv,
    count_parameters,
    ensure_dir,
    fmt_mean_std,
    get_device,
    human_time,
    mean_std,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    set_seed,
    write_json,
)

TASK_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

CSV_FIELDS = [
    "experiment", "dataset", "model", "dim", "lr", "batch_size", "epochs", "epochs_run",
    "patience", "neg_samples", "gamma", "label_smoothing", "objective", "filtered",
    "bern", "optimizer", "params",
    "val_mrr_mean", "val_mrr_std",
    "test_mrr_mean", "test_mrr_std",
    "test_hits1_mean", "test_hits1_std",
    "test_hits3_mean", "test_hits3_std",
    "test_hits10_mean", "test_hits10_std",
    "test_mr_mean", "test_mr_std",
    "best_epoch_mean", "fit_time_mean", "fit_time_std", "eval_time_mean",
    "peak_mem_mb", "num_entities", "num_relations",
    "num_train", "num_valid", "num_test",
]

#: Knobs left as ``None`` inherit from :data:`GLOBAL_DEFAULTS`, then the
#: per-dataset defaults, then (for ConvE) the per-model defaults.
TUNABLE = ("dim", "lr", "batch_size", "epochs", "patience", "neg_samples", "gamma",
           "eval_every", "label_smoothing")

GLOBAL_DEFAULTS = {
    "dim": 200, "lr": 0.001, "batch_size": 1024, "epochs": 200, "patience": 20,
    "neg_samples": 64, "gamma": 6.0, "eval_every": 5, "label_smoothing": 0.1,
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    dataset: str = "WN18RR"
    model: str = "RotatE"

    # --- tunable knobs: ``None`` means "inherit the defaults" ---------------
    dim: Optional[int] = None
    lr: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    patience: Optional[int] = None
    neg_samples: Optional[int] = None
    gamma: Optional[float] = None
    eval_every: Optional[int] = None
    label_smoothing: Optional[float] = None

    # --- objective / protocol ----------------------------------------------
    #: ConvE only: use every entity as a negative (the paper's 1-N objective).
    conv_e_all_negatives: bool = True
    #: Bernoulli negative sampling instead of uniform.
    bern: bool = False
    #: TransE/RotatE: train on forward *and* inverse relations (the standard
    #: trick that lets ``score_all_heads`` reuse ``score_all_tails``).  Turning
    #: it off trains only forward relations -- still scorable in both
    #: directions, but with a different (weaker) parameterisation.
    inverse_relations: bool = True
    #: filtered (standard) vs. raw ranking at evaluation time.
    filtered: bool = True
    #: entity regularisation for TransE/RotatE ("", "l2", "n3" or "n3+l2").
    regularise: str = ""
    #: L2 penalty coefficient for the regulariser above.
    reg_weight: float = 0.0
    optimizer: str = "adam"
    #: "all" (published-split vocabulary, default) or "train" (strict leak-free).
    vocab_from: str = "all"
    #: Global gradient-norm clipping; 0 disables it.  The reference KGE trainers
    #: clip at 1.0, which is what keeps RotatE's phase parameters stable.
    grad_clip: float = 1.0
    #: score-chunk size along the entity axis for 1-vs-all scoring.
    score_chunk: int = 8192
    #: evaluation batch size (triples per ranking step).
    eval_batch_size: int = 32

    experiment: str = "main"
    seeds: Sequence[int] = (0, 1, 2)
    device: str = "auto"
    data_root: str = field(default_factory=default_root)
    results_dir: str = field(default_factory=lambda: os.path.join(TASK_DIR, "results"))
    tag: str = ""

    def resolved(self) -> "Config":
        """Fill every ``None`` knob: globals -> dataset -> model overrides."""
        cfg = copy.deepcopy(self)
        merged = dict(GLOBAL_DEFAULTS)
        merged.update(DATASET_DEFAULTS.get(cfg.dataset, {}))
        merged.update(MODEL_DEFAULTS.get(cfg.model, {}))
        for key in TUNABLE:
            if getattr(cfg, key) is None:
                setattr(cfg, key, merged[key])
        return cfg

    @property
    def objective(self) -> str:
        if self.model == "ConvE":
            return "1vsall-bce" if self.conv_e_all_negatives else "bce-sampled-neg"
        return "margin-neg-sample"

    @property
    def key(self) -> str:
        extra = f"_{self.tag}" if self.tag else ""
        return f"{self.dataset}_{self.model}{extra}"


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def margin_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, gamma: float,
                label_smoothing: float = 0.0) -> torch.Tensor:
    """Max-margin ranking loss ``ReLU(gamma + s_neg - s_pos)``.

    With ``label_smoothing > 0`` the hinge is replaced by a softplus, which is a
    smooth surrogate (used by ``kge_framework`` when a soft margin is wanted);
    the default 0.0 keeps the classic TransE/RotatE objective.
    """
    if label_smoothing > 0:
        return F.softplus(gamma + neg_score - pos_score).mean()
    return F.relu(gamma + neg_score - pos_score).mean()


def bce_1vsall_loss(scores: torch.Tensor, targets: torch.LongTensor,
                    label_smoothing: float = 0.1) -> torch.Tensor:
    """Binary cross-entropy over all entities (the ConvE 1-N objective)."""
    n, num_e = scores.shape
    target = torch.zeros((n, num_e), device=scores.device)
    if label_smoothing > 0:
        target.fill_(label_smoothing / num_e)
        target.scatter_(1, targets.view(-1, 1), 1.0 - label_smoothing + label_smoothing / num_e)
    else:
        target.scatter_(1, targets.view(-1, 1), 1.0)
    return F.binary_cross_entropy_with_logits(scores, target)


def regularisation_penalty(model, head, relation, tail, kind: str) -> torch.Tensor:
    """Optional L2 / N3 regulariser for TransE and RotatE embeddings."""
    if not kind:
        return torch.zeros((), device=head.device)
    if model.__class__.__name__ == "TransE":
        h, r, t = model.entity[head], model.relation[relation], model.entity[tail]
        if kind == "n3":
            return ((F.normalize(h, 2, -1) ** 3).sum(-1)
                    + (F.normalize(r, 2, -1) ** 3).sum(-1)
                    + (F.normalize(t, 2, -1) ** 3).sum(-1)).mean()
        if kind == "l2":
            return (h.pow(2).sum(-1) + r.pow(2).sum(-1) + t.pow(2).sum(-1)).mean()
        if kind == "n3+l2":
            return (regularisation_penalty(model, head, relation, tail, "n3")
                    + regularisation_penalty(model, head, relation, tail, "l2"))
    if kind.startswith("l2"):  # RotatE: penalise the entity embedding magnitude
        h, t = model.entity[head], model.entity[tail]
        return (h.pow(2).sum(-1) + t.pow(2).sum(-1)).mean()
    raise ValueError(f"unsupported regulariser {kind!r} for {model.__class__.__name__}")


# --------------------------------------------------------------------------- #
# One epoch of training
# --------------------------------------------------------------------------- #
def train_epoch(model, dataset: KGDataset, train_triples: torch.LongTensor, optimizer,
                cfg: Config, device, bern_w: Optional[torch.Tensor],
                generator: Optional[torch.Generator]) -> float:
    model.train()
    if hasattr(model, "invalidate_cache"):
        model.invalidate_cache()
    n = train_triples.shape[0]
    perm = torch.randperm(n, device=device)
    losses: List[float] = []
    chunk = cfg.score_chunk

    for start in range(0, n, cfg.batch_size):
        idx = perm[start:start + cfg.batch_size]
        batch = train_triples[idx]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        optimizer.zero_grad(set_to_none=True)

        if cfg.model == "ConvE":
            if cfg.conv_e_all_negatives:
                scores = model.score_all_tails(h, r, chunk=chunk)       # [B, E]
                loss = bce_1vsall_loss(scores, t, cfg.label_smoothing)
            else:
                nh, nr, nt = sample_negatives(h, r, t, dataset.num_entities,
                                              cfg.neg_samples, generator, bern_w)
                all_h = torch.cat([h, nh])
                all_r = torch.cat([r, nr])
                all_t = torch.cat([t, nt])
                scores = model.score_all_tails(all_h, all_r, chunk=chunk)
                logits = scores.gather(1, all_t.view(-1, 1)).view(-1)
                labels = torch.cat([torch.ones(h.shape[0], device=device),
                                    torch.zeros(nh.shape[0], device=device)])
                if cfg.label_smoothing > 0:
                    labels = labels * (1 - cfg.label_smoothing) + cfg.label_smoothing * 0.5
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            # Pointwise scores only: materialising the full [B, E] tail matrix
            # (40,943 entities) *with its backward buffer* needs several GB and
            # does not fit a 4 GB consumer GPU at batch 1024 -- pointwise
            # scoring of B*(1+K) triples is the memory-lean equivalent used by
            # the reference frameworks for the margin objective.  The 1-vs-all
            # matrix is still used for chunked evaluation.
            pos = model.score(h, r, t)                                  # [B]
            nh, nr, nt = sample_negatives(h, r, t, dataset.num_entities,
                                          cfg.neg_samples, generator, bern_w)
            neg = model.score(nh, nr, nt).view(h.shape[0], cfg.neg_samples)  # [B, K]
            loss = margin_loss(pos.unsqueeze(1), neg, cfg.gamma, label_smoothing=0.0)
            if cfg.reg_weight > 0 and cfg.regularise:
                loss = loss + cfg.reg_weight * regularisation_penalty(
                    model, h, r, t, cfg.regularise)

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if hasattr(model, "invalidate_cache"):
            model.invalidate_cache()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else float("nan")


def build_filters(dataset: KGDataset, device) -> dict:
    """Lookup tables for the filtered protocol.

    Both tables are built from train + valid + test, i.e. all *known* true
    triples.  A query is the only row that could mask itself, and ``_mask``
    restores the query's own column afterwards, so including the evaluated split
    is safe and is exactly what "filtered" means in the KGE literature.
    """
    return {
        "tail": build_filter_lookup(dataset.all_true, dataset.num_entities, "tail").to(device),
        "head": build_filter_lookup(dataset.all_true, dataset.num_entities, "head").to(device),
    }


# --------------------------------------------------------------------------- #
# Single run (one seed)
# --------------------------------------------------------------------------- #
def run_single(cfg: Config, dataset: KGDataset, seed: int, device) -> dict:
    """Train with one seed; return validation/test metrics and timings."""
    set_seed(seed)
    reset_peak_gpu_memory(device)

    R = dataset.num_relations
    inverse = cfg.model in ("TransE", "RotatE") and cfg.inverse_relations
    num_rel_model = 2 * R if inverse else R

    model = build_model(cfg.model, dataset.num_entities, num_rel_model, dim=cfg.dim,
                        gamma=cfg.gamma, dataset=cfg.dataset).to(device)
    if inverse:  # score_all_heads(h, r) -> score_all_tails(h, r^-1 + R)
        model.inverse_index = torch.arange(R, 2 * R, device=device)

    train_triples = dataset.train.to(device)
    if inverse:
        inv = torch.stack([train_triples[:, 2], train_triples[:, 1] + R,
                           train_triples[:, 0]], dim=1)
        train_triples = torch.cat([train_triples, inv], dim=0)
    valid = dataset.valid.to(device)
    test = dataset.test.to(device)

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

    bern_w = bernoulli_weights(dataset, device) if cfg.bern else None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    filters = build_filters(dataset, device)

    best_val, best_state, best_epoch = -1.0, None, 0
    history: List[dict] = []
    epochs_run = 0
    timer = Timer()
    eval_time = 0.0
    timer.start()
    for epoch in range(1, cfg.epochs + 1):
        epochs_run = epoch
        loss = train_epoch(model, dataset, train_triples, optimizer, cfg, device,
                           bern_w, generator)
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            with Timer() as et:
                val = evaluate_filtered(model, valid, filters["tail"], dataset.num_entities,
                                        device, batch_size=cfg.eval_batch_size,
                                        filtered=cfg.filtered, chunk=cfg.score_chunk,
                                        head_lookup=filters["head"])
            eval_time += et.elapsed
            history.append({"epoch": epoch, "loss": loss, "val_mrr": val["mrr"],
                            "val_hits@1": val["hits@1"], "val_hits@10": val["hits@10"]})
            if val["mrr"] > best_val:
                best_val = val["mrr"]
                best_epoch = epoch
                best_state = copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()})
            if epoch - best_epoch >= cfg.patience:
                break
    timer.stop()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with Timer() as tt:
        test_metrics = evaluate_filtered(model, test, filters["tail"], dataset.num_entities,
                                         device, batch_size=cfg.eval_batch_size,
                                         filtered=cfg.filtered, chunk=cfg.score_chunk,
                                         head_lookup=filters["head"])

    return {
        "seed": seed,
        "val_mrr": best_val,
        "test": test_metrics,
        "test_mrr": test_metrics["mrr"],
        "test_hits@1": test_metrics["hits@1"],
        "test_hits@3": test_metrics["hits@3"],
        "test_hits@10": test_metrics["hits@10"],
        "test_mr": test_metrics["mr"],
        "fit_time": timer.elapsed,
        "test_time": tt.elapsed,
        "eval_time": eval_time,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "params": count_parameters(model),
        "peak_mem_mb": peak_gpu_memory_mb(device),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Multi-seed driver
# --------------------------------------------------------------------------- #
def run(cfg: Config, verbose: bool = True, write: bool = True) -> dict:
    cfg = cfg.resolved()
    device = get_device(cfg.device)
    dataset = load_dataset(cfg.dataset, cfg.data_root, verbose=verbose,
                           vocab_from=cfg.vocab_from)
    stats = dataset.describe()

    runs = []
    for s in cfg.seeds:
        r = run_single(cfg, dataset, s, device)
        runs.append(r)
        if verbose:
            print(f"    seed {s}: val_mrr={r['val_mrr']:.4f} "
                  f"test MRR={r['test_mrr']:.4f} H@1={r['test_hits@1']:.2f} "
                  f"H@3={r['test_hits@3']:.2f} H@10={r['test_hits@10']:.2f} "
                  f"epochs={r['epochs_run']} ({r['best_epoch']}) "
                  f"fit={human_time(r['fit_time'])}", flush=True)

    def col(name):
        return [r[name] for r in runs]

    def tcol(name):
        return [r["test"][name] for r in runs]

    result = {
        "config": {**asdict(cfg), "objective": cfg.objective,
                   "inverse_relations": cfg.model in ("TransE", "RotatE") and cfg.inverse_relations},
        "dataset_stats": stats,
        "params": runs[0]["params"],
        "runs": runs,
        "val_mrr_mean": mean_std(col("val_mrr"))[0],
        "val_mrr_std": mean_std(col("val_mrr"))[1],
        "test_mrr_mean": mean_std(tcol("mrr"))[0],
        "test_mrr_std": mean_std(tcol("mrr"))[1],
        "test_hits1_mean": mean_std(tcol("hits@1"))[0],
        "test_hits1_std": mean_std(tcol("hits@1"))[1],
        "test_hits3_mean": mean_std(tcol("hits@3"))[0],
        "test_hits3_std": mean_std(tcol("hits@3"))[1],
        "test_hits10_mean": mean_std(tcol("hits@10"))[0],
        "test_hits10_std": mean_std(tcol("hits@10"))[1],
        "test_mr_mean": mean_std(tcol("mr"))[0],
        "test_mr_std": mean_std(tcol("mr"))[1],
        "best_epoch_mean": mean_std(col("best_epoch"))[0],
        "fit_time_mean": mean_std(col("fit_time"))[0],
        "fit_time_std": mean_std(col("fit_time"))[1],
        "eval_time_mean": mean_std(col("eval_time"))[0],
        "peak_mem_mb": max(col("peak_mem_mb")),
    }

    if verbose:
        print(
            f"[{cfg.key}] params={result['params']:,}  obj={cfg.objective}  "
            f"MRR={fmt_mean_std(tcol('mrr'))}  "
            f"H@1={fmt_mean_std(tcol('hits@1'))}  "
            f"H@3={fmt_mean_std(tcol('hits@3'))}  "
            f"H@10={fmt_mean_std(tcol('hits@10'))}  "
            f"fit={human_time(result['fit_time_mean'])}  "
            f"epochs={result['best_epoch_mean']:.0f}/{mean_std(col('epochs_run'))[0]:.0f}  "
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
    r = result
    return {
        "experiment": cfg.experiment,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "dim": cfg.dim,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "epochs_run": round(mean_std([x["epochs_run"] for x in r["runs"]])[0], 1),
        "patience": cfg.patience,
        "neg_samples": cfg.neg_samples,
        "gamma": cfg.gamma,
        "label_smoothing": cfg.label_smoothing,
        "objective": cfg.objective,
        "filtered": cfg.filtered,
        "bern": cfg.bern,
        "optimizer": cfg.optimizer,
        "params": r["params"],
        "val_mrr_mean": round(r["val_mrr_mean"], 4),
        "val_mrr_std": round(r["val_mrr_std"], 4),
        "test_mrr_mean": round(r["test_mrr_mean"], 4),
        "test_mrr_std": round(r["test_mrr_std"], 4),
        "test_hits1_mean": round(r["test_hits1_mean"], 3),
        "test_hits1_std": round(r["test_hits1_std"], 3),
        "test_hits3_mean": round(r["test_hits3_mean"], 3),
        "test_hits3_std": round(r["test_hits3_std"], 3),
        "test_hits10_mean": round(r["test_hits10_mean"], 3),
        "test_hits10_std": round(r["test_hits10_std"], 3),
        "test_mr_mean": round(r["test_mr_mean"], 1),
        "test_mr_std": round(r["test_mr_std"], 1),
        "best_epoch_mean": round(r["best_epoch_mean"], 1),
        "fit_time_mean": round(r["fit_time_mean"], 2),
        "fit_time_std": round(r["fit_time_std"], 2),
        "eval_time_mean": round(r["eval_time_mean"], 2),
        "peak_mem_mb": round(r["peak_mem_mb"], 1),
        "num_entities": stats["num_entities"],
        "num_relations": stats["num_relations"],
        "num_train": stats["num_train"],
        "num_valid": stats["num_valid"],
        "num_test": stats["num_test"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    ds = " (default: dataset/model specific -- see code/dataset.py)"
    p = argparse.ArgumentParser(
        description="Task 4: knowledge-graph completion (TransE / RotatE / ConvE)")
    p.add_argument("--dataset", default="WN18RR", choices=["WN18RR", "FB15k-237"])
    p.add_argument("--model", default="RotatE", choices=list(MODELS))
    p.add_argument("--dim", type=int, default=None, help="embedding dimension" + ds)
    p.add_argument("--lr", type=float, default=None, help="learning rate" + ds)
    p.add_argument("--batch-size", type=int, default=None, help="triples per step" + ds)
    p.add_argument("--epochs", type=int, default=None, help="max epochs" + ds)
    p.add_argument("--patience", type=int, default=None,
                   help="early-stopping patience in epochs (on validation MRR)" + ds)
    p.add_argument("--neg-samples", type=int, default=None,
                   help="negatives per positive triple for TransE/RotatE" + ds)
    p.add_argument("--gamma", type=float, default=None, help="margin of the ranking loss" + ds)
    p.add_argument("--label-smoothing", type=float, default=None,
                   help="label smoothing for the ConvE BCE objective" + ds)
    p.add_argument("--eval-every", type=int, default=None,
                   help="run validation every N epochs" + ds)
    p.add_argument("--conv-e-all-negatives", type=_str2bool, default=True,
                   help="ConvE: score all entities per query (default true = the paper's 1-N)")
    p.add_argument("--bern", action="store_true",
                   help="Bernoulli negative sampling instead of uniform")
    p.add_argument("--no-inverse-relations", dest="inverse_relations",
                   action="store_false", default=True,
                   help="TransE/RotatE: train forward relations only (ablation)")
    p.add_argument("--filtered", dest="filtered", action="store_true", default=True,
                   help="filtered ranking (default)")
    p.add_argument("--no-filtered", dest="filtered", action="store_false",
                   help="raw ranking, reporting the standard-filtered numbers as well")
    p.add_argument("--regularise", default="", choices=["", "l2", "n3", "n3+l2"],
                   help="entity regulariser for TransE/RotatE")
    p.add_argument("--reg-weight", type=float, default=0.0)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adagrad"])
    p.add_argument("--vocab-from", default=Config.vocab_from, choices=["all", "train"],
                   help="build the entity/relation vocabulary from the whole dataset "
                        "(published protocol) or from train only (strict, drops OOV)")
    p.add_argument("--score-chunk", type=int, default=Config.score_chunk,
                   help="entity-axis chunk for 1-vs-all scoring (memory knob)")
    p.add_argument("--grad-clip", type=float, default=Config.grad_clip,
                   help="global gradient-norm clip; 0 disables it")
    p.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
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
    cfg = Config(dataset=args.dataset, model=args.model)
    for key in TUNABLE:
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    cfg.conv_e_all_negatives = args.conv_e_all_negatives
    cfg.bern = args.bern
    cfg.inverse_relations = args.inverse_relations
    cfg.filtered = args.filtered
    cfg.regularise = args.regularise
    cfg.reg_weight = args.reg_weight
    cfg.optimizer = args.optimizer
    cfg.vocab_from = args.vocab_from
    cfg.score_chunk = args.score_chunk
    cfg.grad_clip = args.grad_clip
    cfg.eval_batch_size = args.eval_batch_size
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
