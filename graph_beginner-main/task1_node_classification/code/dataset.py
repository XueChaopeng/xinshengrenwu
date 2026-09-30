"""Dataset loading for Task 1: node classification.

Supported datasets
------------------
* ``Cora``     -- Planetoid, 2,708 nodes / 10,556 edges / 1,433 features / 7 classes
* ``Citeseer`` -- Planetoid, 3,327 nodes / 9,104 edges / 3,703 features / 6 classes
* ``Flickr``   -- 89,250 nodes / 899,756 edges / 500 features / 7 classes

All features are row-normalised (``NormalizeFeatures``, sum normalisation), which
is the standard preprocessing for bag-of-words node features.

Splits
------
Planetoid ships the public *Yang et al. (2016)* split (20 labelled nodes per
class for training, 500 validation and 1,000 test nodes).  Flickr ships the
split from the *gnn-benchmark* repository (used by GraphSAINT), with
train/val/test masks stored on the graph.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.transforms import NormalizeFeatures

#: ``dataset name -> dataset class``
DATASETS = {
    "Cora": Planetoid,
    "Citeseer": Planetoid,
    "Flickr": Flickr,
}

#: Sensible defaults per dataset.
#:
#: ``sample_lr`` is used instead of ``lr`` in ``--mode sample``: a mini-batch
#: gradient is much noisier than a full-graph one, so the sampler needs a
#: smaller step size (measured on Flickr: lr 0.01 -> 43.9%, lr 0.003 -> 51.2%
#: for GraphSAGE).  ``num_neighbors`` is the per-hop fanout; a stack of ``L``
#: layers consumes ``L`` hops.
DATASET_DEFAULTS = {
    "Cora": dict(hidden=64, num_layers=2, lr=0.01, weight_decay=5e-4, dropout=0.5,
                 epochs=200, patience=100, batch_size=64, num_neighbors=[10, 10],
                 sample_lr=0.01),
    "Citeseer": dict(hidden=64, num_layers=2, lr=0.01, weight_decay=5e-4, dropout=0.5,
                     epochs=200, patience=100, batch_size=64, num_neighbors=[10, 10],
                     sample_lr=0.01),
    "Flickr": dict(hidden=256, num_layers=2, lr=0.01, weight_decay=5e-4, dropout=0.5,
                   epochs=80, patience=25, batch_size=2048, num_neighbors=[5, 5],
                   sample_lr=0.003),
}


def default_root() -> str:
    """``<task>/data`` directory next to this file's parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, os.pardir, "data"))


def load_dataset(name: str, root: str | None = None) -> Tuple[Data, int, int]:
    """Load a node-classification dataset.

    Returns
    -------
    data : torch_geometric.data.Data
        Single graph with ``x``, ``edge_index``, ``y``, ``train_mask``,
        ``val_mask`` and ``test_mask``.
    num_features : int
    num_classes : int
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    root = root or default_root()
    os.makedirs(root, exist_ok=True)

    transform = NormalizeFeatures()  # row-sum normalisation of bag-of-words features
    if name in ("Cora", "Citeseer"):
        # Planetoid stores its files under <root>/<name>/{raw,processed}.
        dataset = Planetoid(root=root, name=name, split="public", transform=transform)
    else:
        # InMemoryDataset uses <root>/raw and <root>/processed, so give Flickr its
        # own root to keep the per-dataset layout identical to Planetoid's.
        dataset = Flickr(root=os.path.join(root, "Flickr"), transform=transform)

    data = dataset[0]
    # Some PyG versions do not attach the masks for Planetoid "public" splits here.
    if not hasattr(data, "train_mask") or data.train_mask is None:
        raise RuntimeError(f"{name}: dataset does not provide train/val/test masks")

    # Make mask access cheap and shape-stable.
    for split in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, split)
        if mask.dim() > 1:  # some datasets store [num_nodes, num_splits]
            setattr(data, split, mask[:, 0])
        setattr(data, split, getattr(data, split).bool())

    data = data.to(torch.device("cpu"))  # moved to device inside the trainer
    return data, dataset.num_features, dataset.num_classes


def describe(data: Data, num_features: int, num_classes: int) -> dict:
    """Dataset statistics used in the report and in the console log."""
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
        "num_train": int(data.train_mask.sum()),
        "num_val": int(data.val_mask.sum()),
        "num_test": int(data.test_mask.sum()),
        "avg_degree": round(2 * data.num_edges / max(data.num_nodes, 1), 3),
    }
