"""Knowledge-graph dataset loading, batching and filtered-ranking evaluation.

Dataset format
--------------
Every dataset lives in ``data/<NAME>/{train,valid,test}.txt`` with one triple
per line, whitespace (typically tab) separated::

    head<TAB>relation<TAB>tail

Entities and relations are plain strings.  This is the format used by
``DeepGraphLearning/KnowledgeGraphEmbedding`` (wn18rr / FB15k-237) and by the
reference KGE framework ``Maxioo/kge_framework``.

Vocabulary (leak-free)
----------------------
Two vocabularies can be built, selected with ``vocab_from``:

``"train"``
    Entity and relation vocabularies come from the **training triples only**.
    Validation/test triples that mention an entity or relation unseen in
    training are **dropped** (they are out-of-vocabulary and cannot be scored)
    and the number of dropped triples is reported.  This is the strict
    leak-free setting; on WN18RR it drops 210 valid / 210 test triples, because
    the published split leaves 384 entities that appear only in valid/test.

``"all"`` (default)
    Vocabulary from train + valid + test.  This reproduces the **published**
    split statistics (WN18RR: 40,943 entities / 11 relations, FB15k-237:
    14,541 / 237) and therefore the standard evaluation protocol used by the
    papers.  Nothing is dropped, and no *label* information leaks -- the only
    thing the model learns is which symbols exist, not which triples are true.

Splits
------
The three files *are* the official split shipped with the dataset; this code
never re-splits them.  ``train`` is used to fit the embeddings, ``valid`` drives
early stopping / model selection, and ``test`` is touched exactly once, with the
best-on-validation parameters.  The "test" file additionally contains the
triples used for filtered ranking.

Negative sampling
-----------------
Training uses the standard *local closed world assumption*: for every positive
triple ``(h, r, t)`` a negative is produced by corrupting either the head or the
tail with a random entity drawn uniformly from the entity table (``--neg-samples``
negatives per positive).  A Bernoulli variant (``--bern``) picks the corrupted
side according to the ``bern`` weights of Wang et al. (2014) so that
one-to-many / many-to-one relations are not over-corrupted.

Filtered ranking (evaluation protocol)
--------------------------------------
For each test triple ``(h, r, t)`` the model scores **every** entity and the head
(resp. tail) is ranked among them.  Under the *filtered* setting any other
*known true* triple ``(h, r, t')`` / ``(h', r, t)`` is removed from the candidate
list before ranking, because those are also correct answers that simply happen
to live in train/valid/test.  Without filtering, a correct answer that appears
in the training set would be counted as a mistake.  Metrics: MRR, Hits@1/3/10
over the union of head- and tail-side rankings (the convention used by the
RotatE/ConvE papers).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

#: ``name -> sub-directory under ``data/`` `` (case/spelling as downloaded)
DATASETS = ("WN18RR", "FB15k-237")

#: Directory name used on disk for each dataset key.
DATA_DIRNAME = {"WN18RR": "WN18RR", "FB15k-237": "FB15k-237"}

#: Expected statistics, asserted by ``prepare_data.py --verify`` and printed at
#: the start of every run.  Source: the published WN18RR / FB15k-237 splits.
EXPECTED_STATS = {
    "WN18RR": dict(train=86835, valid=3034, test=3134, entities=40943, relations=11),
    "FB15k-237": dict(train=272115, valid=17535, test=20466, entities=14541, relations=237),
}

#: Per-dataset default hyper-parameters (mirrors ``task1``'s ``DATASET_DEFAULTS``).
#:
#: Epoch budgets are sized for a single consumer GPU (RTX 3050 Laptop, 4 GB):
#: one WN18RR epoch costs ~11 s for TransE and ~16 s for RotatE at dim 200 /
#: batch 1024, so the ``--epochs 200`` of the reference command would need
#: several hours per model.  The defaults below stop earlier and rely on early
#: stopping (``--patience``) on validation MRR; the experiment scripts record the
#: number of epochs actually run, and the README states the reduced budget.
#: ``ConvE`` additionally uses a smaller batch (its 1-vs-all loss over 40,943
#: entities is memory-hungry).
DATASET_DEFAULTS = {
    "WN18RR": dict(dim=200, lr=0.001, batch_size=1024, epochs=50, patience=10,
                   neg_samples=64, gamma=6.0, eval_every=5),
    "FB15k-237": dict(dim=200, lr=0.001, batch_size=1024, epochs=30, patience=5,
                      neg_samples=64, gamma=9.0, eval_every=5),
}

#: Extra per-model overrides applied on top of :data:`DATASET_DEFAULTS`.
#:
#: ``RotatE`` uses a larger step than ``TransE``.  An earlier version shrank it
#: to 1e-4 because the model appeared to collapse at 1e-3 -- but the real cause
#: was the entity initialisation (see :class:`models.RotatE`), not the step size.
#: With the reference-scale init, WN18RR validation MRR after 20 epochs is
#: 0.385 at 1e-3, versus 0.003 at 1e-4 on the old init.
#: ``ConvE`` uses the paper's smaller batch because its 1-vs-all loss covers all
#: 40,943 / 14,541 entities at once, and needs a longer budget than TransE/RotatE
#: because each epoch is a single 1-N step rather than many sampled ones.
MODEL_DEFAULTS = {
    "TransE": {},
    "RotatE": dict(lr=1e-3, epochs=50, patience=10, eval_every=5),
    "ConvE": dict(epochs=50, patience=10, lr=0.001, batch_size=256, eval_every=1),
}


def default_root() -> str:
    """``<task4>/data`` directory next to this file's parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, os.pardir, "data"))


# --------------------------------------------------------------------------- #
# Raw file reading
# --------------------------------------------------------------------------- #
def read_triples(path: str) -> List[Tuple[str, str, str]]:
    """Read a ``head<TAB>relation<TAB>tail`` file into a list of tuples."""
    triples: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: expected 3 fields, got {len(parts)}: {line!r}")
            triples.append((parts[0], parts[1], parts[2]))
    if not triples:
        raise ValueError(f"{path}: no triples found (truncated or bogus download?)")
    return triples


def data_dir(name: str, root: Optional[str] = None) -> str:
    if name not in DATA_DIRNAME:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    return os.path.join(root or default_root(), DATA_DIRNAME[name])


def raw_files_exist(name: str, root: Optional[str] = None) -> bool:
    d = data_dir(name, root)
    return all(os.path.exists(os.path.join(d, s + ".txt")) for s in ("train", "valid", "test"))


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    """``string -> contiguous id`` mapping, built from a fixed triple list."""

    def __init__(self, triples: Sequence[Tuple[str, str, str]]) -> None:
        self.ent2id: Dict[str, int] = {}
        self.rel2id: Dict[str, int] = {}
        for h, r, t in triples:
            for e in (h, t):
                if e not in self.ent2id:
                    self.ent2id[e] = len(self.ent2id)
            if r not in self.rel2id:
                self.rel2id[r] = len(self.rel2id)
        self.id2ent = [None] * len(self.ent2id)
        for e, i in self.ent2id.items():
            self.id2ent[i] = e

    @property
    def num_entities(self) -> int:
        return len(self.ent2id)

    @property
    def num_relations(self) -> int:
        return len(self.rel2id)


# --------------------------------------------------------------------------- #
# The dataset object
# --------------------------------------------------------------------------- #
class KGDataset:
    """A knowledge graph split into train / valid / test ``[N, 3]`` long tensors.

    Attributes
    ----------
    train, valid, test : torch.LongTensor
        ``[N, 3]`` tensors of ``(head_id, relation_id, tail_id)``.
    num_entities, num_relations : int
    vocab : Vocab
    dropped : dict
        How many valid/test triples were dropped because of an out-of-vocabulary
        entity or relation (``{"valid": int, "test": int}``).
    """

    def __init__(self, name: str, root: Optional[str] = None, verbose: bool = True,
                 vocab_from: str = "all") -> None:
        if vocab_from not in ("all", "train"):
            raise ValueError(f"vocab_from must be 'all' or 'train', got {vocab_from!r}")
        self.name = name
        self.root = root or default_root()
        self.dir = data_dir(name, self.root)
        self.vocab_from = vocab_from

        raw = {}
        for split in ("train", "valid", "test"):
            path = os.path.join(self.dir, f"{split}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found -- run `python prepare_data.py --datasets {name}` first")
            raw[split] = read_triples(path)
        self.raw_counts = {k: len(v) for k, v in raw.items()}

        # Leak-free vocabulary: training triples only, or the whole dataset
        # (the published-split convention).  See the module docstring.
        vocab_source = raw["train"] if vocab_from == "train" else (
            raw["train"] + raw["valid"] + raw["test"])
        self.vocab = Vocab(vocab_source)
        self.num_entities = self.vocab.num_entities
        self.num_relations = self.vocab.num_relations

        self.train = self._encode(raw["train"])
        self.valid, dropped_valid = self._encode_split(raw["valid"], "valid")
        self.test, dropped_test = self._encode_split(raw["test"], "test")
        self.dropped = {"valid": dropped_valid, "test": dropped_test}

        #: All known true triples, used by the filtered ranking protocol.
        self.all_true = torch.cat([self.train, self.valid, self.test], dim=0)

        if verbose:
            print(f"[{self.name}] train={self.train.shape[0]} valid={self.valid.shape[0]} "
                  f"test={self.test.shape[0]} entities={self.num_entities} "
                  f"relations={self.num_relations} vocab_from={self.vocab_from} "
                  f"dropped(valid/test)={self.dropped['valid']}/{self.dropped['test']}",
                  flush=True)

    # ---------------------------------------------------------------- encoding
    def _encode(self, triples: Sequence[Tuple[str, str, str]]) -> torch.LongTensor:
        e, r = self.vocab.ent2id, self.vocab.rel2id
        arr = np.asarray([(e[h], r[rel], e[t]) for h, rel, t in triples], dtype=np.int64)
        return torch.from_numpy(arr)

    def _encode_split(self, triples, split: str) -> Tuple[torch.LongTensor, int]:
        """Encode a split, dropping triples with out-of-vocabulary symbols."""
        e, r = self.vocab.ent2id, self.vocab.rel2id
        kept, dropped = [], 0
        for h, rel, t in triples:
            if h in e and t in e and rel in r:
                kept.append((e[h], r[rel], e[t]))
            else:
                dropped += 1
        if dropped:
            print(f"[{self.name}] WARNING: dropped {dropped} {split} triples with OOV "
                  f"entities/relations (vocabulary is built from train only)", flush=True)
        arr = np.asarray(kept, dtype=np.int64).reshape(-1, 3)
        return torch.from_numpy(arr), dropped

    # ------------------------------------------------------------------ helpers
    def describe(self) -> dict:
        """Statistics for the console log / JSON detail file."""
        return {
            "dataset": self.name,
            "vocab_from": self.vocab_from,
            "num_entities": int(self.num_entities),
            "num_relations": int(self.num_relations),
            "num_train": int(self.train.shape[0]),
            "num_valid": int(self.valid.shape[0]),
            "num_test": int(self.test.shape[0]),
            "num_raw_train": self.raw_counts["train"],
            "num_raw_valid": self.raw_counts["valid"],
            "num_raw_test": self.raw_counts["test"],
            "dropped_valid": int(self.dropped["valid"]),
            "dropped_test": int(self.dropped["test"]),
        }

    def expected_stats(self) -> dict:
        """Published split statistics used for verification.

        The published entity count corresponds to ``vocab_from="all"``;
        ``vocab_from="train"`` yields fewer entities by construction, so the
        entity check only applies to the default setting.
        """
        return EXPECTED_STATS.get(self.name, {})

    def verify(self) -> bool:
        """Compare the loaded statistics against :data:`EXPECTED_STATS`."""
        exp = self.expected_stats()
        if not exp:
            return True
        ok = (
            self.num_relations == exp["relations"]
            and self.raw_counts["train"] == exp["train"]
            and self.raw_counts["valid"] == exp["valid"]
            and self.raw_counts["test"] == exp["test"]
        )
        if self.vocab_from == "all":
            ok = ok and self.num_entities == exp["entities"]
        return ok


# --------------------------------------------------------------------------- #
# Negative sampling
# --------------------------------------------------------------------------- #
def bernoulli_weights(dataset: KGDataset, device: torch.device,
                      eps: float = 1e-12) -> torch.Tensor:
    """``[num_relations, 2]`` probability of corrupting the head vs. the tail.

    ``p_head = tph / (tph + hpt)`` with ``tph`` = average tails per head and
    ``hpt`` = average heads per tail for that relation (Wang et al., 2014).
    """
    num_r = dataset.num_relations
    tails_per_head = np.zeros(num_r, dtype=np.float64)
    heads_per_tail = np.zeros(num_r, dtype=np.float64)
    head_count: Dict[Tuple[int, int], int] = defaultdict(int)
    tail_count: Dict[Tuple[int, int], int] = defaultdict(int)
    distinct_heads = defaultdict(set)
    distinct_tails = defaultdict(set)

    tr = dataset.train.numpy()
    for h, r, t in tr:
        head_count[(h, int(r))] += 1
        tail_count[(int(r), t)] += 1
        distinct_heads[int(r)].add(int(h))
        distinct_tails[int(r)].add(int(t))
    for r in range(num_r):
        nh = max(len(distinct_heads[r]), 1)
        nt = max(len(distinct_tails[r]), 1)
        tails_per_head[r] = sum(tail_count[(r, t)] for t in distinct_tails[r]) / nt  # avg tails per head
        heads_per_tail[r] = sum(head_count[(h, r)] for h in distinct_heads[r]) / nh  # avg heads per tail
    p_head = tails_per_head / (tails_per_head + heads_per_tail + eps)
    w = np.stack([p_head, 1.0 - p_head], axis=1)
    return torch.from_numpy(w).to(device).float()


def sample_negatives(head: torch.LongTensor, rel: torch.LongTensor, tail: torch.LongTensor,
                     num_entities: int, num_neg: int, generator: Optional[torch.Generator] = None,
                     weights: Optional[torch.Tensor] = None) -> Tuple[torch.LongTensor,
                                                                     torch.LongTensor,
                                                                     torch.LongTensor]:
    """Corrupt head or tail of every positive triple ``num_neg`` times.

    Returns ``(neg_head, neg_rel, neg_tail)`` of shape ``[B*num_neg]``.  The
    relation is never corrupted, which is the standard KGE setting (the relation
    is given at test time as well).

    If ``weights`` (``[num_relations, 2]`` bernoulli probabilities) is given, the
    corrupted side is drawn per triple instead of uniformly.
    """
    device = head.device
    b = head.shape[0]
    rep_rel = rel.repeat_interleave(num_neg)
    rep_head = head.repeat_interleave(num_neg)
    rep_tail = tail.repeat_interleave(num_neg)
    rnd = torch.randint(0, num_entities, (b * num_neg,), device=device, generator=generator)
    if weights is None:
        corrupt_head = torch.rand(b * num_neg, device=device, generator=generator) < 0.5
    else:
        p_head = weights[rep_rel, 0]
        corrupt_head = torch.rand(b * num_neg, device=device, generator=generator) < p_head
    neg_head = torch.where(corrupt_head, rnd, rep_head)
    neg_tail = torch.where(corrupt_head, rep_tail, rnd)
    return neg_head, rep_rel, neg_tail


# --------------------------------------------------------------------------- #
# Filtered ranking evaluation
# --------------------------------------------------------------------------- #
def build_filter_lookup(triples: torch.LongTensor, num_entities: int,
                        mode: str) -> torch.LongTensor:
    """Pre-sort *other* true triples into a lookup table for the filtered protocol.

    Returns an ``[M, 3]`` long tensor of ``(x, y, flat_entity)`` rows, sorted by
    ``(x, y)``: for ``mode="tail"`` a row is ``(head, relation, tail)`` and the
    query ``(h, r, ?)`` must mask every tail with the same ``(h, r)``; for
    ``mode="head"`` a row is ``(relation, tail, head)`` and the query
    ``(?, r, t)`` masks every head with the same ``(r, t)``.

    The table must be built from the triples of the **other** splits plus the
    current split, so a query is never removed from its own candidate list.
    Duplicate rows are kept (they are harmless -- the same entity is just masked
    twice) because ``searchsorted`` on a sorted table stays cheap either way.
    """
    if triples.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.long)
    h, r, t = triples[:, 0].long(), triples[:, 1].long(), triples[:, 2].long()
    if mode == "tail":
        x, y, ent = h, r, t
    elif mode == "head":
        x, y, ent = r, t, h
    else:
        raise ValueError(f"mode must be 'tail' or 'head', got {mode!r}")
    keys = x * num_entities + y                                    # unique composite key
    order = torch.argsort(keys)
    return torch.stack([keys[order], ent[order]], dim=1)           # [M, 2]


def candidates_for(lookup: torch.LongTensor, x: torch.LongTensor, y: torch.LongTensor,
                   num_entities: int) -> torch.Tensor:
    """``[B, K]`` padded candidate tensor for each query ``(x_i, y_i)``.

    ``K = max_i count_i`` over the batch, and every row is **left-aligned**:
    slots ``0 .. count_i-1`` are filled and the remaining slots are ``-1``.  No
    row is ever truncated -- truncation would silently drop true answers (and
    with them the query's own target, which would then be masked away).
    """
    b = x.shape[0]
    if lookup.numel() == 0:
        return torch.full((b, 1), -1, dtype=torch.long)
    keys = lookup[:, 0]
    q = x * num_entities + y
    lo = torch.searchsorted(keys.contiguous(), q.contiguous(), right=False)
    hi = torch.searchsorted(keys.contiguous(), q.contiguous(), right=True)
    counts = hi - lo
    k = max(int(counts.max()), 1)
    out = torch.full((b, k), -1, dtype=torch.long)
    for i in range(b):
        c = int(counts[i])
        if c == 0:
            continue
        vals = lookup[lo[i]:hi[i], 1]                              # exactly c rows
        out[i, :vals.numel()] = vals
    return out


@torch.no_grad()
def evaluate_filtered(model, triples: torch.LongTensor, tail_lookup: torch.LongTensor,
                      num_entities: int, device: torch.device, batch_size: int = 128,
                      filtered: bool = True, chunk: int = 8192,
                      head_lookup: Optional[torch.LongTensor] = None) -> dict:
    """Filtered (or raw) ranking evaluation of ``triples``.

    Parameters
    ----------
    tail_lookup : torch.LongTensor
        Table from ``build_filter_lookup(all_true, E, "tail")`` -- masks the
        other tails that are true for the query's ``(head, relation)``.
    head_lookup : torch.LongTensor, optional
        Table from ``build_filter_lookup(all_true, E, "head")`` -- masks the
        other heads that are true for the query's ``(relation, tail)``.  The two
        sides need **different** tables; passing ``tail_lookup`` for both is a
        silent no-op on the head side.
    filtered : bool
        Apply the filtered protocol; otherwise every other entity competes.

    Returns ``{"mrr", "hits@1", "hits@3", "hits@10", "mr", "num_rankings"}``
    averaged over the union of the head- and tail-side rankings (2N rankings).

    Implementation notes
    --------------------
    ``model.score_all_tails`` / ``score_all_heads`` return the ``[B, E]`` block
    (chunked along the entity axis internally).  Filtering is done for the whole
    batch at once and the rank comes from two reductions, ``sum(s > s_q)`` and
    ``sum(s == s_q)``; the optimistic convention (an equally-scored entity counts
    as better with probability 1/2) is handled by the second reduction.  A naive
    per-query Python loop over the filter lists is ~100x slower.
    """
    model.eval()
    n = triples.shape[0]
    ranks = np.zeros(2 * n, dtype=np.int64)
    tl = tail_lookup if tail_lookup.numel() else None
    hl = head_lookup if (head_lookup is not None and head_lookup.numel()) else None

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        h = triples[start:stop, 0].to(device)
        r = triples[start:stop, 1].to(device)
        t = triples[start:stop, 2].to(device)

        # ---- tail side: rank the true tail among all entities --------------
        scores = model.score_all_tails(h, r, chunk=chunk)          # [B, E]
        if filtered and tl is not None:
            cands = candidates_for(tl.cpu(), h.cpu(), r.cpu(), num_entities).to(device)
            _mask(scores, cands, t)
        ranks[start:stop] = _rank_of(scores, t).cpu().numpy()

        # ---- head side: rank the true head via the inverse relation --------
        # NB: the query starts from the *tail* -- see Models.score_all_heads.
        scores = model.score_all_heads(r, t, chunk=chunk)
        if filtered and hl is not None:
            cands = candidates_for(hl.cpu(), r.cpu(), t.cpu(), num_entities).to(device)
            _mask(scores, cands, h)
        ranks[n + start:n + stop] = _rank_of(scores, h).cpu().numpy()

    ranks = ranks.astype(np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1) * 100.0),
        "hits@3": float(np.mean(ranks <= 3) * 100.0),
        "hits@10": float(np.mean(ranks <= 10) * 100.0),
        "mr": float(np.mean(ranks)),
        "num_rankings": int(ranks.size),
    }


def _mask(scores: torch.Tensor, candidates: torch.Tensor,
          keep: torch.LongTensor) -> None:
    """In-place: set every filtered candidate column to ``-inf``.

    Candidates are left-aligned and padded with ``-1`` (see
    :func:`candidates_for`); ``keep`` holds the query's own target, whose column
    is saved beforehand and restored afterwards so that the query is never
    filtered out of its own candidate list (which would give it rank 1 for free).
    """
    b, _ = scores.shape
    k = candidates.shape[1]
    if k == 0:
        return
    rows = torch.arange(b, device=scores.device)
    saved = scores.gather(1, keep.view(-1, 1))                      # [B, 1]
    idx = candidates.clamp_min(0)                                   # safe gather index
    valid = candidates >= 0
    rows2 = rows.unsqueeze(1).expand(b, k)
    scores[rows2[valid], idx[valid]] = float("-inf")
    scores[rows, keep] = saved.view(-1)


def _rank_of(scores: torch.Tensor, targets: torch.LongTensor) -> torch.LongTensor:
    """Optimistic 1-based rank of ``targets`` inside each row of ``scores``.

    ``equal`` counts the target itself, so an entity tied for the best score has
    ``greater == 0`` and must receive rank **1** under the optimistic
    convention.  An earlier version added an extra ``+1`` here, which shifted
    every rank up by one: Hits@1 was identically 0 for all three models and MRR
    was systematically understated (it reported ``1/(r+1)`` instead of ``1/r``).
    """
    target_score = scores.gather(1, targets.view(-1, 1))           # [B, 1]
    greater = (scores > target_score).sum(dim=1)
    equal = (scores == target_score).sum(dim=1)
    return (greater + (equal + 1) // 2).long()


def sparse_index(triples: torch.LongTensor, num_entities: int, mode: str) -> torch.LongTensor:
    """``[N, 3]`` ``(x, y, flat_entity)`` rows, sorted by the ``(x, y)`` key.

    Convenience wrapper around :func:`build_filter_lookup` that keeps the
    ``(x, y)`` columns explicit.
    """
    table = build_filter_lookup(triples, num_entities, mode)
    if table.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.long)
    keys, ent = table[:, 0], table[:, 1]
    return torch.stack([keys // num_entities, keys % num_entities, ent], dim=1)


def load_dataset(name: str, root: Optional[str] = None, verbose: bool = True,
                 vocab_from: str = "all") -> KGDataset:
    """Convenience wrapper used by ``train.py`` / ``run_experiments.py``."""
    return KGDataset(name, root=root, verbose=verbose, vocab_from=vocab_from)
