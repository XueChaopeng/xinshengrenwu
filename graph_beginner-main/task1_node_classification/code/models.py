"""GNN encoders used by Task 1 (node classification).

Four mainstream architectures are implemented on top of *PyTorch Geometric*:

======  ==========================  =====================================
Name    Message passing             Aggregation
======  ==========================  =====================================
GCN     ``GCNConv``                 degree-normalised weighted mean
GAT     ``GATConv``                 multi-head attention (weighted sum)
SAGE    ``SAGEConv``                concatenation of self + neighbourhood mean
GIN     ``GINConv``                 sum aggregation + MLP, trainable ``eps``
======  ==========================  =====================================

Every encoder maps ``(x, edge_index) -> h`` with ``h`` of size ``hidden``;
the task specific head (``NodeClassifier`` here) is kept separate so that the
*identical* encoder can be reused by Task 2 for link prediction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, BatchNorm

MODEL_NAMES = ("gcn", "gat", "sage", "gin")


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


class NodeClassifier(nn.Module):
    """``GNNEncoder`` followed by a linear classification head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        model: str = "gcn",
        dropout: float = 0.5,
        heads: int = 8,
        aggr: str = "mean",
        batchnorm: Optional[bool] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
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
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        if return_embedding:
            return h
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int,
    num_classes: int,
    **kwargs,
) -> NodeClassifier:
    """Factory used by the training scripts."""
    return NodeClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        model=name,
        **kwargs,
    )
