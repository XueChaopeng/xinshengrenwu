"""Dataset loading and edge splitting for Task 2: link prediction.

Datasets
--------
``Cora``, ``Citeseer`` (Planetoid) and ``Flickr`` (GraphSAINT) -- exactly the
three graphs required by the assignment.  The node features are row-normalised
and the *graph structure* (not the labels) is what matters here.

Splits
------
Link prediction needs an **edge** split, not a node split.  We use
:class:`torch_geometric.transforms.RandomLinkSplit` with the split ratios used
throughout the link-prediction literature (e.g. the PyG "Introduction to link
prediction" tutorial):

* 85 % train / 5 % validation / 10 % test edges (``num_val=0.05``, ``num_test=0.1``)
* the graph is treated as undirected, so both ``(u, v)`` and ``(v, u)`` live in
  the same split -- this prevents the trivial "the reverse edge is the answer" leak
* ``disjoint_train_ratio=0.0``: the encoder performs message passing over **all**
  training edges and the supervision signal is those same edges.  This is the
  standard inductive-ish setup and, importantly, it keeps the full-graph and the
  mini-batch runs comparable (both see the same message-passing graph).
* negative edges are **not** baked into the training split
  (``add_negative_train_samples=False``); they are drawn on the fly: once per
  epoch for full-graph training, per mini-batch for the sampled regime.

The transformation returns three ``Data`` objects with an extra
``edge_label_index`` ``[2, E_split]`` and ``edge_label`` ``[E_split]`` describing
the supervised edges of that split.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.transforms import NormalizeFeatures

DATASETS = ("Cora", "Citeseer", "Flickr")

#: Default hyper-parameters per dataset.  ``sample_lr`` is used by the
#: mini-batch regime (noisier gradients need a smaller step size).
DATASET_DEFAULTS = {
    "Cora": dict(hidden=128, num_layers=2, lr=0.01, weight_decay=0.0, dropout=0.0,
                 epochs=200, patience=50, batch_size=512, num_neighbors=[10, 10],
                 sample_lr=0.01),
    "Citeseer": dict(hidden=128, num_layers=2, lr=0.01, weight_decay=0.0, dropout=0.0,
                     epochs=200, patience=50, batch_size=512, num_neighbors=[10, 10],
                     sample_lr=0.01),
    "Flickr": dict(hidden=256, num_layers=2, lr=0.005, weight_decay=0.0, dropout=0.0,
                   # Full-graph training costs ~0.6 s/epoch here, so it gets the
                   # same generous budget as the small graphs.  The *sampled*
                   # regime is the expensive one (a step per mini-batch of
                   # supervised edges, i.e. ~400 steps/epoch at batch 2048), so it
                   # overrides the budget through the ``sample_*`` keys below.
                   epochs=100, patience=30, batch_size=2048, num_neighbors=[10, 10],
                   sample_lr=0.002, sample_epochs=25, sample_patience=10,
                   sample_batch_size=16384),
}

#: Fraction of edges held out for validation / test.
NUM_VAL = 0.05
NUM_TEST = 0.1


def default_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, os.pardir, "data"))


def load_dataset(name: str, root: str | None = None, seed: int = 0) -> Tuple[Data, int, int]:
    """Load the raw graph (no split yet).

    Returns ``(data, num_features, num_classes)``.  ``num_classes`` is unused by
    the link-prediction loss but is kept so the signature matches Task 1.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")
    root = root or default_root()
    os.makedirs(root, exist_ok=True)

    transform = NormalizeFeatures()
    if name in ("Cora", "Citeseer"):
        dataset = Planetoid(root=root, name=name, split="public", transform=transform)
    else:
        dataset = Flickr(root=os.path.join(root, "Flickr"), transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def _normalise_split(d: Data) -> Data:
    """Unify the supervised-edge attribute names across PyG versions.

    PyG >= 2.6 stores the held-out edges as ``pos_edge_label_index`` /
    ``pos_edge_label`` (plus ``neg_edge_label_index`` / ``neg_edge_label`` for
    the automatically generated negatives), while older releases used a single
    ``edge_label_index`` / ``edge_label`` pair.  Both layouts are normalised to
    the latter so the trainer does not care which version is installed.
    """
    if getattr(d, "edge_label_index", None) is not None:
        return d

    pos_idx = d.pos_edge_label_index
    pos_lbl = d.pos_edge_label
    neg_idx = getattr(d, "neg_edge_label_index", None)
    if neg_idx is not None and neg_idx.numel() > 0:
        idx = torch.cat([pos_idx, neg_idx], dim=1)
        lbl = torch.cat([pos_lbl, getattr(d, "neg_edge_label")])
    else:
        idx, lbl = pos_idx, pos_lbl

    d.edge_label_index = idx
    d.edge_label = lbl
    return d


def split_edges(data: Data, num_val: float = NUM_VAL, num_test: float = NUM_TEST,
                seed: int = 0, disjoint_train_ratio: float = 0.0):
    """Split edges into ``(train_data, val_data, test_data)``.

    ``*_data.edge_index`` is the message-passing graph available to the encoder;
    ``*_data.edge_label_index`` / ``edge_label`` are the supervised edges of that
    split.  The training split holds **only positives** (negatives are drawn on
    the fly by the trainer), while the validation and test splits already carry
    an equal number of negative edges.
    """
    from torch_geometric.transforms import RandomLinkSplit

    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=False,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=1.0,
        split_labels=True,
    )
    torch.manual_seed(seed)
    train_data, val_data, test_data = transform(data)
    return tuple(_normalise_split(d) for d in (train_data, val_data, test_data))


def describe(data: Data, num_features: int, num_classes: int) -> dict:
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
        "avg_degree": round(2 * data.num_edges / max(data.num_nodes, 1), 3),
    }
