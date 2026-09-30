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


class DotProductDecoder(nn.Module):
    """Classic inner-product decoder: ``score(u, v) = <z_u, z_v>``."""

    def forward(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


class MLPDecoder(nn.Module):
    """Stronger decoder used by e.g. the PyG link-prediction tutorial.

    The endpoint embeddings are concatenated and pushed through an MLP; this can
    model non-symmetric relations that a plain dot product cannot.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout
        dims = [2 * in_channels] + [hidden_channels] * (num_layers - 1) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        return self.net(torch.cat([z[src], z[dst]], dim=-1)).view(-1)


class LinkPredictor(nn.Module):
    """``GNNEncoder`` + a decoder that scores candidate ``(u, v)`` pairs.

    The same object is used by both training regimes:

    * full-graph -- ``forward(x, edge_index, edge_label_index)`` where
      ``edge_index`` is the whole training graph;
    * mini-batch -- ``forward(batch.x, batch.edge_index, batch.edge_label_index)``
      where all three tensors describe the sampled sub-graph.  This works
      because ``LinkNeighborLoader`` re-indexes the edges of the sub-graph
      locally, exactly like ``NeighborLoader`` does for nodes.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        model: str = "gcn",
        dropout: float = 0.0,
        heads: int = 8,
        aggr: str = "mean",
        batchnorm: Optional[bool] = None,
        residual: bool = False,
        decoder: str = "dot",
    ) -> None:
        super().__init__()
        self.decoder_type = decoder
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
        if decoder == "dot":
            self.decoder = DotProductDecoder()
        elif decoder == "mlp":
            self.decoder = MLPDecoder(hidden_channels, hidden_channels, num_layers=2,
                                      dropout=dropout)
        else:
            raise ValueError(f"decoder must be 'dot' or 'mlp', got {decoder!r}")

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_label_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int,
    decoder: str = "dot",
    **kwargs,
) -> LinkPredictor:
    """Factory used by the training scripts."""
    return LinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        model=name,
        decoder=decoder,
        **kwargs,
    )
