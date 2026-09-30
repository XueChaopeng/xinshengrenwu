"""Knowledge-graph embedding models: TransE, RotatE and ConvE.

Every model exposes the same interface so that a single training/evaluation
loop can drive all of them:

``forward(head, relation) -> [B, num_entities]``
    1-vs-all (a.k.a. 1-N) scoring: the score of every candidate entity against
    the query ``(head, relation)``.  **Higher is better** for all three models;
    the distance-based models return the negated distance.

``score_all_tails(head, relation, chunk)``
    Same as ``forward`` but chunked along the entity axis so that the ``[B, E]``
    score matrix never has to exist in full -- required for evaluation on
    WN18RR (40,943 entities) at a comfortable batch size.

``score_all_heads(relation, tail, chunk)``
    Scores all candidate **heads** for ``(?, relation, tail)``.  Implemented by
    training a matching *inverse relation* for every relation, which is the
    standard trick used by the KGE frameworks (``kge_framework``,
    ``KnowledgeGraphEmbedding``).  Note the argument order: the query is driven
    by the **tail**, because ``(t, r^-1, ?)`` is the mirror of ``(?, r, t)``.

``score(head, relation, tail) -> [B]``
    Pointwise score of the given (h, r, t) pairs -- used by the margin /
    full-softmax objectives.

Training objectives
-------------------
* **TransE / RotatE** -- *negative sampling with a margin*: for each positive
  triple, ``--neg-samples`` corrupted triples are produced and the loss is
  ``ReLU(gamma + s_pos - s_neg)`` (max-margin), optionally with ``--label-smoothing``
  cross-entropy instead (see :func:`task4_loss` in ``train.py``).  The corrupted
  triples are also passed through the same **1-vs-all** scorer, which is cheap
  because the tail-side score matrix is computed once.
* **ConvE** -- strictly **1-vs-all / 1-N binary cross-entropy**: every training
  triple is one query whose positive is the true tail and whose negatives are
  *all other entities*, with label smoothing (default 0.1).  This is the
  objective of the ConvE paper.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = ("TransE", "RotatE", "ConvE")


def _uniform_init(size, device=None):
    bound = 6.0 / math.sqrt(size[1])
    return torch.empty(size, device=device).uniform_(-bound, bound)


# --------------------------------------------------------------------------- #
# TransE
# --------------------------------------------------------------------------- #
class TransE(nn.Module):
    """Translating embeddings (Bordes et al., 2013).

    Principle
    ---------
    Relations are translations in the embedding space: a true triple satisfies
    ``h + r ~= t``.  The score is the *negative distance*

    .. math:: f(h, r, t) = -\\lVert h + r - t \\rVert_{p}

    so a triple is correct when ``t`` sits near ``h + r``.  Entities are
    L2-normalised before scoring (the standard trick that keeps the distance
    well behaved); no ``tanh`` is applied, because TransE relies on the
    unbounded translation geometry.

    ``norm`` selects the p of the norm (1 or 2); ``--gamma`` is the margin used
    by the max-margin training objective (see ``train.py``).
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int = 200,
                 norm: int = 2, init_entity_norm: bool = True) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.norm = norm
        self.init_entity_norm = init_entity_norm
        self.entity = _uniform_init((num_entities, dim))
        self.relation = _uniform_init((num_relations, dim))
        self.entity = nn.Parameter(self.entity)
        self.relation = nn.Parameter(self.relation)
        #: filled by the trainer with the (head-side) inverse relation ids
        self.register_buffer("inverse_index", torch.arange(num_relations), persistent=False)
        self._normalised_cache: Optional[torch.Tensor] = None
    # -- helpers ------------------------------------------------------------
    def _normalise(self, cache: bool = True) -> torch.Tensor:
        """L2-normalised entity table (cached outside training steps)."""
        if not self.init_entity_norm:
            return self.entity
        if cache and self._normalised_cache is not None:
            return self._normalised_cache
        out = F.normalize(self.entity, p=2, dim=-1)
        if cache:
            self._normalised_cache = out
        return out

    def invalidate_cache(self) -> None:
        self._normalised_cache = None

    def _dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.norm(a - b, p=self.norm, dim=-1)

    @staticmethod
    def _pairwise_l2(queries: torch.Tensor, table: torch.Tensor,
                     chunk: int = 8192) -> torch.Tensor:
        """``-||q - e||_2`` for every query/entity pair, computed memory-efficiently.

        Uses ``||q - e||^2 = ||q||^2 - 2 q.e + ||e||^2`` so that only the
        ``[B, chunk]`` score block and its ``[B, chunk, d]`` backward buffer
        exist at any time -- the naive broadcast difference materialises the
        whole ``[B, E, d]`` tensor and is what makes 40,943 entities at batch
        1024 exceed a 4 GB GPU.
        """
        q2 = queries.pow(2).sum(dim=-1, keepdim=True)              # [B, 1]
        e2 = table.pow(2).sum(dim=-1).unsqueeze(0)                 # [1, E]
        out = []
        for s in range(0, table.shape[0], chunk):
            blk = table[s:s + chunk]
            d2 = q2 - 2.0 * torch.mm(queries, blk.transpose(0, 1)) + e2[:, s:s + chunk]
            out.append(-d2.clamp_min_(0).sqrt_())
        return torch.cat(out, dim=1)

    @staticmethod
    def _pairwise_l1(queries: torch.Tensor, table: torch.Tensor,
                     chunk: int = 2048) -> torch.Tensor:
        """``-||q - e||_1``; the ``[B, chunk, d]`` difference is unavoidable."""
        out = []
        for s in range(0, table.shape[0], chunk):
            blk = table[s:s + chunk]                               # [c, d]
            out.append(-torch.abs(queries.unsqueeze(1) - blk.unsqueeze(0)).sum(dim=-1))
        return torch.cat(out, dim=1)

    # -- scoring ------------------------------------------------------------
    def score(self, head: torch.LongTensor, relation: torch.LongTensor,
              tail: torch.LongTensor) -> torch.Tensor:
        ent = self._normalise(cache=not self.training)
        h = ent[head]
        r = self.relation[relation]
        t = ent[tail]
        return -self._dist(h + r, t)

    def forward(self, head: torch.LongTensor, relation: torch.LongTensor) -> torch.Tensor:
        return self.score_all_tails(head, relation)

    def score_all_tails(self, head: torch.LongTensor, relation: torch.LongTensor,
                        chunk: int = 8192) -> torch.Tensor:
        ent = self._normalise(cache=not self.training)
        q = ent[head] + self.relation[relation]                    # [B, d]
        if self.norm == 2:
            return self._pairwise_l2(q, ent, chunk=max(chunk, 1024))
        return self._pairwise_l1(q, ent, chunk=min(chunk, 2048))

    def score_all_heads(self, relation: torch.LongTensor, tail: torch.LongTensor,
                        chunk: int = 8192) -> torch.Tensor:
        """Scores every candidate **head** for ``(?, relation, tail)``.

        The trained inverse relation mirrors ``(h, r, t)`` as ``(t, r^-1, h)``,
        so the query has to start from the **tail** and score every entity as a
        candidate head.  (Starting from the head -- which an earlier version of
        this file did -- silently scores the wrong quantity and roughly halves
        the filtered MRR, because ``evaluate`` averages the head-side and
        tail-side ranks.)
        """
        inv = self.inverse_index[relation]
        return self.score_all_tails(tail, inv, chunk=chunk)


# --------------------------------------------------------------------------- #
# RotatE
# --------------------------------------------------------------------------- #
class RotatE(nn.Module):
    """Rotation in complex space (Sun et al., 2019).

    Principle
    ---------
    Every entity is a point in ``C^d`` and every relation is an **element-wise
    rotation**: ``t ~= h o r`` with ``|r_i| = 1``.  The modulus constraint is
    enforced by parameterising the relation through its *phase*,

    .. math:: r_i = e^{i\\theta_i}, \\qquad \\theta_i = -\\pi + 2\\pi\\,u_i

    and the score is the negated L1 (or L2) distance in complex space

    .. math:: f(h,r,t) = -\\lVert h \\circ r - t\\rVert_p

    Implemented with the 2-component real formulation: the first half of the
    embedding axis is the real part, the second half the imaginary part.
    RotatE can model symmetry, antisymmetry, inversion and composition patterns.

    .. note::
       ``gamma`` is stored on the module for bookkeeping only.  It is **not**
       applied to the embeddings: a rotation preserves the modulus, so scaling
       ``h`` by ``gamma`` would break the geometry (it made RotatE collapse to
       MRR 0.0002 on WN18RR).  The margin is applied additively by the ranking
       loss in ``train.py``, exactly as in the reference implementation
       (``score = gamma - ||h o r - t||``).
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int = 200,
                 gamma: float = 12.0, norm: int = 1,
                 phase_init: str = "uniform") -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotatE needs an even dimension, got {dim}")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.half = dim // 2
        self.gamma = gamma
        self.norm = norm
        # The reference implementation initialises entities in
        # ``(-embedding_range, embedding_range)`` with
        # ``embedding_range = (gamma + epsilon) / dim``, epsilon = 2.
        # That keeps the L1 distance ``||h o r - t||`` on the same scale as the
        # margin: with dim=500, gamma=5 the range is +-0.014 and the initial
        # distance is ~2, i.e. comfortably inside the margin.
        # Initialising in (-1, 1) instead makes the initial distance ~134 while
        # the margin stays at 6, and RotatE then fails to move at all
        # (measured: WN18RR test MRR 0.002 with +-1 vs. healthy learning here).
        bound = (gamma + 2.0) / dim
        self.entity = nn.Parameter(torch.empty(num_entities, dim).uniform_(-bound, bound))
        if phase_init == "normal":  # the RotatE reference implementation's default
            phase = torch.empty(num_relations, self.half).normal_(0.0, 1.0)
            phase = (phase * math.pi).clamp(-math.pi + 1e-3, math.pi - 1e-3)
        else:
            phase = torch.empty(num_relations, self.half).uniform_(-math.pi + 1e-3,
                                                                   math.pi - 1e-3)
        #: phase angles in (-pi, pi); the actual rotation is exp(i * phase)
        self.rel_phase = nn.Parameter(phase)
        self.register_buffer("inverse_index", torch.arange(num_relations), persistent=False)

    def _rotation(self, relation: torch.LongTensor) -> torch.Tensor:
        """``[B, d]`` real/imag tensor of ``r = exp(i * theta)``."""
        theta = self.rel_phase[relation]                          # [B, half]
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

    @staticmethod
    def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise complex product for the half/half real-imag layout."""
        half = a.shape[-1] // 2
        ar, ai = a[..., :half], a[..., half:]
        br, bi = b[..., :half], b[..., half:]
        return torch.cat([ar * br - ai * bi, ar * bi + ai * br], dim=-1)

    def score(self, head: torch.LongTensor, relation: torch.LongTensor,
              tail: torch.LongTensor) -> torch.Tensor:
        rot = self._rotation(relation)
        h = self.entity[head]
        t = self.entity[tail]
        return -torch.norm(self._complex_mul(h, rot) - t, p=self.norm, dim=-1)

    def forward(self, head: torch.LongTensor, relation: torch.LongTensor) -> torch.Tensor:
        return self.score_all_tails(head, relation)

    def score_all_tails(self, head: torch.LongTensor, relation: torch.LongTensor,
                        chunk: int = 8192) -> torch.Tensor:
        rot = self._rotation(relation)                             # [B, d]
        q = self._complex_mul(self.entity[head], rot)              # [B, d]
        return self._complex_dist(q, self.entity, chunk=chunk)

    def _complex_dist(self, q: torch.Tensor, table: torch.Tensor,
                      chunk: int = 8192) -> torch.Tensor:
        """Negated L1/L2 distance in complex space, without the ``[B, E, d]`` blow-up.

        For the L2 norm the same expansion as in :meth:`TransE._pairwise_l2` is
        used, applied separately to the real and imaginary halves; for the L1
        norm (the RotatE paper's default) the difference has to be materialised,
        but only for ``chunk`` entities at a time.
        """
        half = table.shape[-1] // 2
        qr, qi = q[..., :half], q[..., half:]
        tr, ti = table[..., :half], table[..., half:]
        out = []
        step = chunk if self.norm == 1 else max(chunk, 1024)
        for s in range(0, table.shape[0], step):
            br, bi = tr[s:s + step], ti[s:s + step]                # [c, half]
            if self.norm == 2:
                q2 = qr.pow(2).sum(-1, keepdim=True) + qi.pow(2).sum(-1, keepdim=True)
                b2 = br.pow(2).sum(-1).unsqueeze(0) + bi.pow(2).sum(-1).unsqueeze(0)
                d2 = q2 - 2.0 * (torch.mm(qr, br.t()) + torch.mm(qi, bi.t())) + b2
                out.append(-d2.clamp_min_(0).sqrt_())
            else:
                dr = (qr.unsqueeze(1) - br.unsqueeze(0)).abs()     # [B, c, half]
                di = (qi.unsqueeze(1) - bi.unsqueeze(0)).abs()
                out.append(-(dr + di).sum(dim=-1))
        return torch.cat(out, dim=1)

    def score_all_heads(self, relation: torch.LongTensor, tail: torch.LongTensor,
                        chunk: int = 8192) -> torch.Tensor:
        """Scores every candidate **head** for ``(?, relation, tail)``.

        See :meth:`TransE.score_all_heads`: the query must start from the tail.
        """
        inv = self.inverse_index[relation]
        return self.score_all_tails(tail, inv, chunk=chunk)


# --------------------------------------------------------------------------- #
# ConvE
# --------------------------------------------------------------------------- #
class ConvE(nn.Module):
    """Convolutional 2D knowledge-graph embeddings (Dettmers et al., 2018).

    Principle
    ---------
    ``(head, relation)`` are concatenated, **reshaped into a 2-D image**
    (default ``10 x 20`` for dim 200), passed through a 2-D convolution, a
    batch-norm, a dropout, a fully connected projection back to ``dim``, and
    finally matched against the entity embedding matrix (the "1x1 convolution"
    of the paper, i.e. a matrix product).  A triple is scored by

    .. math:: f(h,r,t) = \\sigma\\big(\\mathrm{vec}(\\mathrm{conv}([\\bar h; \\bar r]))^\\top W\\big) \\cdot t

    Training uses **1-vs-all (1-N) binary cross-entropy**: the true tail is the
    positive and all other entities are negatives, with label smoothing (0.1).
    This is much cheaper per epoch than negative sampling for large graphs, but
    it needs the entity embedding matrix to be regularised (dropout + label
    smoothing + optional embedding dropout) to avoid overfitting.

    Standard hyper-parameters (paper defaults, used here): embedding dim 200,
    ``10 x 20`` reshape, 32 filters of ``3 x 3``, hidden dropout 0.2, input
    dropout 0.2, label smoothing 0.1, batch size 128-1024.
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int = 200,
                 embed_shape: tuple = (10, 20), num_filters: int = 32,
                 kernel_size: int = 3, input_dropout: float = 0.2,
                 hidden_dropout: float = 0.2, feature_dropout: float = 0.2,
                 embedding_dropout: float = 0.2, use_bias: bool = True) -> None:
        super().__init__()
        if embed_shape[0] * embed_shape[1] != dim:
            raise ValueError(
                f"embed_shape {tuple(embed_shape)} must multiply to dim = {dim} "
                f"(each of the head and the relation embedding is reshaped to that image)")
        if min(embed_shape) < kernel_size:
            raise ValueError(f"embed_shape {tuple(embed_shape)} is smaller than the "
                             f"{kernel_size}x{kernel_size} kernel")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.embed_shape = tuple(embed_shape)
        self.input_dropout = nn.Dropout(input_dropout)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.embedding_dropout = nn.Dropout2d(embedding_dropout)

        self.emb_e = nn.Parameter(torch.empty(num_entities, dim))
        self.emb_rel = nn.Parameter(torch.empty(num_relations, dim))
        bound = 6.0 / math.sqrt(dim)
        nn.init.uniform_(self.emb_e, -bound, bound)
        nn.init.uniform_(self.emb_rel, -bound, bound)

        self.inp_drop = self.input_dropout
        self.conv1 = nn.Conv2d(2, num_filters, (kernel_size, kernel_size), 1, 0, bias=use_bias)
        # bn0 normalises the 2-channel (head, relation) input image
        self.bn0 = nn.BatchNorm2d(2)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm1d(dim)
        #: size of the flattened convolution output
        conv_h = self.embed_shape[0] - kernel_size + 1
        conv_w = self.embed_shape[1] - kernel_size + 1
        self.fc = nn.Linear(num_filters * conv_h * conv_w, dim)
        self.register_buffer("inverse_index", torch.arange(num_relations), persistent=False)

    # -- scoring ------------------------------------------------------------
    def _embed(self, head: torch.LongTensor, relation: torch.LongTensor) -> torch.Tensor:
        """Concatenate + reshape ``(head, relation)`` into the conv input image."""
        e1 = self.emb_e[head].view(-1, 1, *self.embed_shape)
        e2 = self.emb_rel[relation].view(-1, 1, *self.embed_shape)
        stacked = torch.cat([e1, e2], dim=1)                       # [B, 2, 10, 20]
        return self.bn0(stacked)

    def _hidden(self, head: torch.LongTensor, relation: torch.LongTensor) -> torch.Tensor:
        x = self.inp_drop(self._embed(head, relation))
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

    def _dropout_entities(self) -> torch.Tensor:
        """Entity embedding matrix with 2-D dropout applied (training only).

        Dims 0 and 1 of the ``[num_entities, dim]`` table are treated as the
        spatial axes, which is what the reference ConvE implementation does with
        ``EmbeddingDropout2d``: whole embeddings are dropped, not individual
        coordinates.
        """
        if not self.training:
            return self.emb_e
        drop = self.embedding_dropout(self.emb_e.unsqueeze(-1).unsqueeze(-1))
        return drop.squeeze(-1).squeeze(-1)

    def score(self, head: torch.LongTensor, relation: torch.LongTensor,
              tail: torch.LongTensor) -> torch.Tensor:
        """Per-pair *logit* (pre-sigmoid) score."""
        x = self._hidden(head, relation)                           # [B, d]
        t = self.emb_e[tail]                                       # [B, d]
        return torch.sum(x * t, dim=-1)

    def forward(self, head: torch.LongTensor, relation: torch.LongTensor) -> torch.Tensor:
        return self.score_all_tails(head, relation)

    def score_all_tails(self, head: torch.LongTensor, relation: torch.LongTensor,
                        chunk: int = 8192) -> torch.Tensor:
        """``[B, num_entities]`` logits -- the ConvE 1-N scoring step."""
        x = self._hidden(head, relation)                           # [B, d]
        e = self._dropout_entities()                               # [E, d]
        return torch.mm(x, e.transpose(0, 1))

    def score_all_heads(self, relation: torch.LongTensor, tail: torch.LongTensor,
                        chunk: int = 8192) -> torch.Tensor:
        """Scores every candidate **head** for ``(?, relation, tail)``.

        See :meth:`TransE.score_all_heads`: the query must start from the tail.
        """
        inv = self.inverse_index[relation]
        return self.score_all_tails(tail, inv, chunk=chunk)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(name: str, num_entities: int, num_relations: int, dim: int = 200,
                gamma: float = 6.0, norm: Optional[int] = None, **kwargs) -> nn.Module:
    """Instantiate one of :data:`MODELS`.

    ``num_relations`` must already include the inverse relations if the trainer
    uses them (i.e. ``2 * R``), which is how ``train.py`` calls this factory.
    """
    if name == "TransE":
        return TransE(num_entities, num_relations, dim=dim,
                      norm=norm or (2 if "FB15k" in kwargs.get("dataset", "FB15k") else 1),
                      init_entity_norm=kwargs.get("init_entity_norm", True))
    if name == "RotatE":
        return RotatE(num_entities, num_relations, dim=dim, gamma=max(gamma, 1.0),
                      norm=norm or 1, phase_init=kwargs.get("phase_init", "normal"))
    if name == "ConvE":
        shape = kwargs.get("embed_shape") or embed_shape_for(dim)
        return ConvE(num_entities, num_relations, dim=dim, embed_shape=tuple(shape),
                     num_filters=kwargs.get("num_filters", 32),
                     kernel_size=kwargs.get("kernel_size", 3),
                     input_dropout=kwargs.get("input_dropout", 0.2),
                     hidden_dropout=kwargs.get("hidden_dropout", 0.2),
                     feature_dropout=kwargs.get("feature_dropout", 0.2),
                     embedding_dropout=kwargs.get("embedding_dropout", 0.2))
    raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")


def embed_shape_for(dim: int) -> tuple:
    """``(h, w)`` reshape for ConvE such that ``h * w == dim``.

    The head and the relation embedding are each reshaped to this image and
    stacked as the two channels of the convolution input, so the paper's
    ``10 x 20`` corresponds to ``dim = 200``.  Other dimensions use the closest
    near-square factorisation of ``dim`` that still leaves at least one output
    cell after a ``3 x 3`` convolution.
    """
    #: keys are ``dim``; values multiply to ``dim``
    table = {200: (10, 20), 100: (5, 20), 400: (20, 20), 50: (5, 10),
             20: (4, 5), 10: (2, 5)}
    if dim in table:
        return table[dim]
    for h in range(int(math.sqrt(dim)), 0, -1):
        if dim % h == 0:
            return (h, dim // h)
    return (1, dim)
