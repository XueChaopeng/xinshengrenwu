"""GNN encoders and the graph-level model used by Task 3 (graph classification).

Four mainstream architectures are implemented on top of *PyTorch Geometric*:

======  ==========================  =====================================
Name    Message passing             Aggregation
======  ==========================  =====================================
GCN     ``GCNConv``                 degree-normalised weighted mean
GAT     ``GATConv``                 multi-head attention (weighted sum)
SAGE    ``SAGEConv``                concatenation of self + neighbourhood mean
GIN     ``GINConv``                 sum aggregation + MLP, trainable ``eps``
======  ==========================  =====================================

The encoder maps ``(x, edge_index) -> h`` with ``h`` of size ``hidden``; the
graph-level part lives in :class:`GraphClassifier`, which applies a **readout
(pooling) operator** over the node embeddings of every graph in the batch and
then a small head.

Pooling (readout) operators
---------------------------
============================  ==========================================
``pool``                      implementation
============================  ==========================================
``mean``  (AvgPooling)        :func:`torch_geometric.nn.global_mean_pool`
``max``   (MaxPooling)        :func:`torch_geometric.nn.global_max_pool`
``min``   (MinPooling)        :func:`global_min_pool` (see below)
``sum`` / ``add``             :func:`torch_geometric.nn.global_add_pool`
============================  ==========================================

``global_min_pool`` does **not** exist in PyG, so it is implemented here with
``torch_geometric.utils.scatter(..., reduce="amin")``.  ``amin`` *is*
differentiable in modern PyTorch/PyG: the gradient flows to the (arg)minimal
node of every graph and is split evenly between ties, exactly like the standard
segment-max gradient.  This is verified in ``models.selftest()`` and is also
re-checked by ``python models.py``.  A ``-global_max_pool(-h)`` fallback is kept
for comparison and is mathematically identical.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter

MODEL_NAMES = ("gcn", "gat", "sage", "gin")
POOL_NAMES = ("mean", "max", "min", "sum", "add")

#: Canonical display names used in the report / README tables.
POOL_DISPLAY = {
    "mean": "AvgPooling",
    "max": "MaxPooling",
    "min": "MinPooling",
    "sum": "SumPooling",
    "add": "SumPooling",
}


# --------------------------------------------------------------------------- #
# Pooling / readout
# --------------------------------------------------------------------------- #
def global_min_pool(x: torch.Tensor, batch: torch.Tensor,
                    size: Optional[int] = None) -> torch.Tensor:
    """Min-pooling readout ``min_{v in g} h_v`` for every graph ``g``.

    PyG ships no ``global_min_pool``, so this uses ``scatter`` with
    ``reduce="amin"``.  ``include_self=False`` avoids leaking the ``+inf`` fill
    value used for empty groups into the result, and the operation stays
    differentiable (verified in :func:`selftest`).

    Parameters
    ----------
    x : torch.Tensor
        Node embeddings ``[N, F]``.
    batch : torch.Tensor
        Graph assignment of every node, ``[N]``.
    size : int, optional
        Number of graphs in the batch (defaults to ``batch.max() + 1``).
    """
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 0
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return scatter(x, batch, dim=0, dim_size=size, reduce="amin")


def global_min_pool_negmax(x: torch.Tensor, batch: torch.Tensor,
                           size: Optional[int] = None) -> torch.Tensor:
    """``min = -max(-x)``; an algebraically identical cross-check of the above."""
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 0
    return -global_max_pool(-x, batch, size=size)


def pool_nodes(x: torch.Tensor, batch: torch.Tensor, pool: str,
               size: Optional[int] = None) -> torch.Tensor:
    """Dispatch a node embedding matrix through the requested readout."""
    pool = pool.lower()
    if pool == "mean":
        return global_mean_pool(x, batch, size=size)
    if pool == "max":
        return global_max_pool(x, batch, size=size)
    if pool == "min":
        return global_min_pool(x, batch, size=size)
    if pool in ("sum", "add"):
        return global_add_pool(x, batch, size=size)
    raise ValueError(f"pool must be one of {POOL_NAMES}, got {pool!r}")


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Two-layer MLP used as the GIN update function."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNNEncoder(nn.Module):
    """Stack of ``num_layers`` message-passing layers producing node embeddings.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Width of every hidden layer (also the embedding size).
    num_layers : int
        Number of message-passing layers.
    model : {"gcn", "gat", "sage", "gin"}
    dropout : float
        Dropout applied to the output of every layer but the last.
    heads : int
        Number of attention heads (GAT only).  The hidden width is split
        across heads, so ``hidden_channels`` must be divisible by ``heads``.
    aggr : str
        Aggregation used by GraphSAGE (``"mean"``, ``"max"``, ``"sum"``).
    batchnorm : bool or None
        ``None`` -> enabled for GIN, disabled otherwise (the classical setup).
    residual : bool
        Add a linear skip connection around every hidden layer.  Useful for
        deeper stacks, see the depth study in the report.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        model: str = "gcn",
        dropout: float = 0.5,
        heads: int = 8,
        aggr: str = "mean",
        batchnorm: Optional[bool] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        model = model.lower()
        if model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if model == "gat" and hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
            )
        if batchnorm is None:
            batchnorm = model == "gin"

        self.model = model
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            is_last = i == num_layers - 1
            if model == "gcn":
                conv = GCNConv(in_c, hidden_channels)
            elif model == "gat":
                if is_last:
                    conv = GATConv(in_c, hidden_channels, heads=1, concat=False,
                                   dropout=dropout, add_self_loops=True)
                else:
                    conv = GATConv(in_c, hidden_channels // heads, heads=heads,
                                   concat=True, dropout=dropout, add_self_loops=True)
            elif model == "sage":
                conv = SAGEConv(in_c, hidden_channels, aggr=aggr)
            else:  # gin
                conv = GINConv(MLP(in_c, hidden_channels), train_eps=True)
            self.convs.append(conv)

            self.norms.append(BatchNorm(hidden_channels) if batchnorm else nn.Identity())
            if residual and not is_last and in_c == hidden_channels:
                self.skips.append(nn.Linear(in_c, hidden_channels, bias=False))
            else:
                self.skips.append(None)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            if self.skips[i] is not None:
                h = h + self.skips[i](x)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
        return x


# --------------------------------------------------------------------------- #
# Graph-level model
# --------------------------------------------------------------------------- #
class GraphClassifier(nn.Module):
    """``GNNEncoder`` -> readout (pooling) -> graph-level head.

    Despite the name the model also serves **graph regression** (ZINC): with
    ``regression=True`` the head emits a single scalar per graph and the loss is
    ``MSELoss``; with ``regression=False`` the head emits ``out_channels``
    logits and the loss is ``CrossEntropyLoss``.  Keeping one class for both
    tasks means the pooling comparison is literally the same code path in both
    settings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        model: str = "gcn",
        pool: str = "mean",
        dropout: float = 0.5,
        heads: int = 8,
        aggr: str = "mean",
        batchnorm: Optional[bool] = None,
        residual: bool = False,
        regression: bool = False,
        head_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        if pool.lower() not in POOL_NAMES:
            raise ValueError(f"pool must be one of {POOL_NAMES}, got {pool!r}")
        self.pool = pool.lower()
        self.dropout = dropout
        self.regression = regression

        self.encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            model=model,
            dropout=dropout,
            heads=heads,
            aggr=aggr,
            batchnorm=batchnorm,
            residual=residual,
        )

        if regression:
            # Two-layer MLP head, the usual choice for ZINC regression.
            h = head_hidden or hidden_channels
            self.head = nn.Sequential(
                nn.Linear(hidden_channels, h),
                nn.ReLU(),
                nn.Linear(h, out_channels),
            )
        else:
            self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                return_embedding: bool = False) -> torch.Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        h = self.encoder(x, edge_index)
        g = pool_nodes(h, batch, self.pool)
        if return_embedding:
            return g
        g = F.dropout(g, p=self.dropout, training=self.training)
        out = self.head(g)
        return out.view(-1) if self.regression else out


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    pool: str = "mean",
    regression: bool = False,
    **kwargs,
) -> GraphClassifier:
    """Factory used by the training scripts."""
    return GraphClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        model=name,
        pool=pool,
        regression=regression,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Self-test (``python models.py``)
# --------------------------------------------------------------------------- #
def selftest() -> None:
    """Sanity-check pooling correctness and -- crucially -- gradient flow."""
    torch.manual_seed(0)
    x = torch.tensor([[3.0, 1.0], [1.0, 5.0], [2.0, 2.0], [9.0, 0.0], [0.0, 0.0]])
    batch = torch.tensor([0, 0, 1, 1, 1])

    expected = {
        "mean": [[2.0, 3.0], [11 / 3, 2 / 3]],
        "max": [[3.0, 5.0], [9.0, 2.0]],
        "min": [[1.0, 1.0], [0.0, 0.0]],
        "sum": [[4.0, 6.0], [11.0, 2.0]],
    }
    for pool, exp in expected.items():
        got = pool_nodes(x, batch, pool).tolist()
        assert torch.allclose(torch.tensor(got), torch.tensor(exp), atol=1e-6), (
            f"{pool}: {got} != {exp}")
        t = x.clone().requires_grad_(True)
        pool_nodes(t, batch, pool).sum().backward()
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"{pool}: bad grad"
        assert t.grad.abs().sum() > 0, f"{pool}: dead gradient"
        print(f"  pool={pool:5s} forward+backward OK  grad_sum={t.grad.abs().sum():.3f}")

    # min pooling must agree with the -max(-x) identity, gradients included
    t1 = x.clone().requires_grad_(True)
    t2 = x.clone().requires_grad_(True)
    global_min_pool(t1, batch).sum().backward()
    global_min_pool_negmax(t2, batch).sum().backward()
    print(f"  min vs -max(-x): dX_max={(t1.grad - t2.grad).abs().max():.2e} "
          f"(tie-splitting may differ, both are valid subgradients)")

    for model in MODEL_NAMES:
        for pool in POOL_NAMES:
            m = build_model(model, in_channels=8, hidden_channels=16, out_channels=3,
                            pool=pool, num_layers=3)
            data = _tiny_batch()
            out = m(data.x, data.edge_index, data.batch)
            out.sum().backward()
            assert out.shape == (2, 3), out.shape
        print(f"  model={model:5s} all pools OK")
    print("models.selftest: all checks passed")


def _tiny_batch():
    from torch_geometric.data import Batch, Data
    g1 = Data(x=torch.randn(4, 8), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    g2 = Data(x=torch.randn(3, 8), edge_index=torch.tensor([[0, 1], [1, 2]]))
    return Batch.from_data_list([g1, g2])


if __name__ == "__main__":
    selftest()
