"""Dataset loading for Task 3: graph classification (TUDataset) and graph
regression (ZINC).

TUDataset
---------
Four small benchmark collections, all with **graph-level labels**:

==============  ======  =======  =========  ========  ==========================
Name            graphs  classes  features   avg |V|   notes
==============  ======  =======  =========  ========  ==========================
``MUTAG``       188     2        7          17.9      one-hot node labels (atoms)
``ENZYMES``     600     6        3          32.6      one-hot node labels
``PROTEINS``    1113    2        3          39.1      one-hot node labels
``IMDB-BINARY`` 1000    2        *none*     19.8      **degree features added**
==============  ======  =======  =========  ========  ==========================

``IMDB-BINARY`` ships *no* node features at all.  Following the GIN paper
(`Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019`) we use a
one-hot encoding of the node degree as the input feature
(:func:`add_degree_features`).  The encoding width is the maximum degree observed
in the dataset plus one (135 + 1 = 136 for IMDB-BINARY), which is exactly what
the reference implementation does.  Datasets that already carry features keep
theirs untouched.

Splits
------
TUDataset is tiny (188-1113 graphs).  The literature standard is 10-fold
cross-validation; the **default here is a stratified 80 / 10 / 10 split repeated
over 3 seeds (0, 1, 2)** so that every reported number carries a mean ± std.
The split is *stratified*: every class keeps its proportion in all three parts.
Use ``--folds 10`` (see ``train.py``) if you want the literature protocol
instead; the README states both and never mixes them.

ZINC
----
The *ZINC* subset (10,000 train / 1,000 val / 1,000 test) is a **graph-level
regression** task: predict the constrained logP of a molecule, scored with MAE
(lower is better).  PyG's own ``ZINC`` dataset downloads ``molecules.zip`` from
Dropbox, which is unreachable here, so the identical standard split is fetched
from the HuggingFace mirror (``graphs-datasets/ZINC``) as three JSONL files and
converted to :class:`torch_geometric.data.Data` objects by
``prepare_data.py``.  The on-disk representation matches PyG exactly: ``x`` is
int64 ``[N, 1]`` (the atomic number) and ``edge_attr`` is int64 ``[E, 1]`` (the
bond type).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset

# ``torch.load(weights_only=True)`` (the PyTorch 2.6+ default) does not know the
# PyG storage classes, which makes every dataset load emit a UserWarning and
# take the slow fallback path.  Allowlisting them is the fix PyG's own warning
# suggests and keeps the loading path identical to the fast one.
try:  # pragma: no cover - version dependent
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except Exception:  # noqa: BLE001 - older/newer PyG simply does not need this
    pass

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TU_DATASETS = ("MUTAG", "ENZYMES", "PROTEINS", "IMDB-BINARY")
ZINC_DATASET = "ZINC"
DATASETS = TU_DATASETS + (ZINC_DATASET,)

#: Datasets whose graphs carry no node features -> degree one-hot is added.
NEEDS_DEGREE_FEATURES = ("IMDB-BINARY",)

#: ``True`` for graph-level regression (MAE), ``False`` for classification.
REGRESSION = {name: False for name in TU_DATASETS}
REGRESSION[ZINC_DATASET] = True

#: Sensible defaults per dataset.  TUDatasets are small, so small batches with
#: enough optimisation steps; ZINC is 10x larger and is trained with the
#: standard 4-layer / lr 1e-3 recipe.
DATASET_DEFAULTS = {
    "MUTAG":       dict(hidden=64, num_layers=3, heads=8, dropout=0.5, lr=0.01,
                        weight_decay=5e-4, epochs=200, patience=50, batch_size=32),
    "ENZYMES":     dict(hidden=64, num_layers=3, heads=8, dropout=0.5, lr=0.01,
                        weight_decay=5e-4, epochs=200, patience=50, batch_size=32),
    "PROTEINS":    dict(hidden=64, num_layers=3, heads=8, dropout=0.5, lr=0.01,
                        weight_decay=5e-4, epochs=200, patience=50, batch_size=32),
    "IMDB-BINARY": dict(hidden=64, num_layers=3, heads=8, dropout=0.5, lr=0.01,
                        weight_decay=5e-4, epochs=200, patience=50, batch_size=32),
    "ZINC":        dict(hidden=64, num_layers=4, heads=8, dropout=0.0, lr=1e-3,
                        weight_decay=0.0, epochs=100, patience=25, batch_size=512),
}

#: Default pooling for every model.  ``gin`` uses the sum readout of the paper.
DEFAULT_POOL = {"gcn": "mean", "gat": "mean", "sage": "mean", "gin": "sum"}


def default_root() -> str:
    """``<task>/data`` directory next to this file's parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, os.pardir, "data"))


def is_regression(name: str) -> bool:
    return REGRESSION.get(name, False)


# --------------------------------------------------------------------------- #
# TUDataset
# --------------------------------------------------------------------------- #
def add_degree_features(graphs: List[Data], name: str = "") -> int:
    """Replace ``x`` by a one-hot encoding of the node degree (GIN paper).

    Only applied to datasets without node features (``IMDB-BINARY``).  The
    encoding width is ``max_degree + 1`` taken over the **whole dataset**, so
    every split shares the same feature space.

    ``graphs`` must be a plain list of :class:`~torch_geometric.data.Data`
    objects: ``InMemoryDataset.__getitem__`` hands out *copies*, so mutating
    ``dataset[i]`` in place would silently be discarded.  Returns the number of
    input features (unchanged width for datasets that already have features).
    """
    if graphs[0].x is not None:
        for g in graphs:  # defensive: keep every feature matrix float32
            if g.x.dtype != torch.float32:
                g.x = g.x.float()
        return int(graphs[0].x.size(-1))

    degrees = []
    for data in graphs:
        degrees.append(torch.bincount(data.edge_index[0], minlength=data.num_nodes))
    max_degree = int(max(int(d.max()) for d in degrees)) if degrees else 0

    for data, deg in zip(graphs, degrees):
        x = torch.zeros(data.num_nodes, max_degree + 1, dtype=torch.float)
        x[torch.arange(data.num_nodes), deg] = 1.0
        data.x = x

    print(f"  [{name or 'TUDataset'}] x was None -> one-hot degree features, "
          f"dim={max_degree + 1} (max degree {max_degree})", flush=True)
    return max_degree + 1


def load_tu_dataset(name: str, root: Optional[str] = None
                    ) -> Tuple[List[Data], int, int]:
    """Load one TUDataset collection.

    Returns ``(graphs, num_features, num_classes)`` where ``graphs`` is a plain
    list of ``Data`` objects with a **materialised** ``x`` -- required because
    ``IMDB-BINARY`` needs its degree features written in and PyG's in-memory
    dataset returns throw-away copies on every access.
    """
    if name not in TU_DATASETS:
        raise ValueError(f"unknown TUDataset {name!r}; choose from {TU_DATASETS}")
    root = root or default_root()
    dataset = TUDataset(root=root, name=name, use_node_attr=False)
    graphs = [d for d in dataset]
    num_features = add_degree_features(graphs, name)
    num_classes = int(dataset.num_classes)
    return graphs, num_features, num_classes


# --------------------------------------------------------------------------- #
# Stratified splits
# --------------------------------------------------------------------------- #
def stratified_split(
    y: Sequence[int],
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified train/val/test indices with a deterministic per-seed shuffle.

    Every class contributes ``fractions`` of its members to each split (at
    least one member to train, and to val/test whenever the class has >= 3
    members).  Classes are processed independently and the shuffle is driven by
    a ``numpy.random.default_rng(seed)``, so a given ``(y, seed)`` pair always
    yields the same split on any platform.
    """
    y = np.asarray([int(v) for v in y])
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_val = int(round(n * fractions[1]))
        n_test = int(round(n * fractions[2]))
        if fractions[1] > 0 and n >= 3:  # guarantee the class appears in the split
            n_val = max(n_val, 1)
        if fractions[2] > 0 and n >= 3:
            n_test = max(n_test, 1)
        while n - n_val - n_test < 1 and (n_val + n_test) > 0:
            if n_val >= n_test and n_val > (1 if fractions[1] > 0 else 0):
                n_val -= 1
            elif n_test > (1 if fractions[2] > 0 else 0):
                n_test -= 1
            else:
                break
        val.extend(idx[:n_val].tolist())
        test.extend(idx[n_val:n_val + n_test].tolist())
        train.extend(idx[n_val + n_test:].tolist())
    # Deterministic ordering keeps the DataLoader shuffling reproducible.
    return sorted(train), sorted(val), sorted(test)


def class_counts(y: Sequence[int], indices: Sequence[int]) -> Dict[str, int]:
    y = np.asarray([int(v) for v in y])
    vals, counts = np.unique(y[np.asarray(list(indices), dtype=int)], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


# --------------------------------------------------------------------------- #
# ZINC -- conversion + InMemoryDataset
# --------------------------------------------------------------------------- #
class ZincSplit(InMemoryDataset):
    """One split (``train`` / ``val`` / ``test``) of the ZINC subset.

    The heavy conversion from JSONL is done once by ``prepare_data.py``; this
    class only memory-loads ``<root>/processed/<split>.pt``.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        super().__init__(root, transform=transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, os.pardir, "ZINC_raw")

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self) -> str:
        return f"{self.split}.pt"

    def process(self) -> None:
        raise RuntimeError(
            f"{self.processed_paths[0]} is missing -- run "
            f"`python prepare_data.py --datasets ZINC` first."
        )


def zinc_paths(root: Optional[str] = None) -> Dict[str, str]:
    root = root or default_root()
    return {
        "raw_dir": os.path.join(root, "ZINC_raw"),
        "processed_dir": os.path.join(root, "ZINC_processed"),
        "stats": os.path.join(root, "ZINC_processed", "stats.json"),
    }


def convert_zinc_jsonl(path: str) -> List[Data]:
    """Turn one ZINC ``.jsonl`` file into ``torch_geometric.data.Data`` objects.

    Faithful to PyG's ``ZINC``: ``x`` is int64 ``[N, 1]`` (atomic number) and
    ``edge_attr`` is int64 ``[E, 1]`` (bond type 1..3).  ``y`` keeps the **raw**
    constrained logP; normalisation is applied by the trainer, not here.
    """
    graphs: List[Data] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            graphs.append(Data(
                x=torch.tensor(rec["node_feat"], dtype=torch.long),
                edge_index=torch.tensor(rec["edge_index"], dtype=torch.long),
                edge_attr=torch.tensor(rec["edge_attr"], dtype=torch.long),
                y=torch.tensor(rec["y"], dtype=torch.float),
            ))
    return graphs


def load_zinc(root: Optional[str] = None) -> Tuple[Dict[str, ZincSplit], Dict[str, float]]:
    """Load the three ZINC splits.

    Returns ``(splits, stats)`` where ``stats`` holds the **training-set** mean
    and std of the raw target (used to standardise for training; MAE is always
    reported back in raw logP units, see ``train.py``).
    """
    paths = zinc_paths(root)
    if not os.path.exists(paths["stats"]):
        raise FileNotFoundError(
            f"{paths['stats']} not found -- run `python prepare_data.py --datasets ZINC`"
        )
    with open(paths["stats"], "r", encoding="utf-8") as f:
        stats = json.load(f)

    def to_float(data: Data) -> Data:
        # Our encoders are generic float MLPs; PyG stores the atomic number as
        # an integer index of shape [N, 1], which we cast to float.
        data.x = data.x.float()
        return data

    splits = {
        s: ZincSplit(paths["processed_dir"], split=s, transform=to_float)
        for s in ("train", "val", "test")
    }
    return splits, stats


# --------------------------------------------------------------------------- #
# Unified entry point
# --------------------------------------------------------------------------- #
def describe_tu(graphs: List[Data], num_features: int, num_classes: int) -> dict:
    """Dataset statistics used in the report and in the console log."""
    num_nodes = sum(int(d.num_nodes) for d in graphs)
    num_edges = sum(int(d.num_edges) for d in graphs)
    ys = [int(d.y.item()) for d in graphs]
    counts = np.bincount(np.asarray(ys))
    return {
        "num_graphs": len(graphs),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": int(num_features),
        "num_classes": int(num_classes),
        "avg_nodes": round(num_nodes / max(len(graphs), 1), 2),
        "avg_edges": round(num_edges / max(len(graphs), 1), 2),
        "max_nodes": max(int(d.num_nodes) for d in graphs),
        "class_counts": ",".join(str(int(c)) for c in counts),
        "task": "classification",
    }


def describe_zinc(splits: Dict[str, ZincSplit], stats: Dict[str, float]) -> dict:
    train = splits["train"]
    ys = np.asarray([float(d.y.item()) for d in train])
    return {
        "num_graphs": sum(len(s) for s in splits.values()),
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "num_features": 1,
        "num_classes": 1,
        "num_edge_features": 1,
        "avg_nodes": round(float(np.mean([d.num_nodes for d in train])), 2),
        "avg_edges": round(float(np.mean([d.num_edges for d in train])), 2),
        "max_nodes": int(max(d.num_nodes for d in train)),
        "target_mean": round(float(stats["mean"]), 6),
        "target_std": round(float(stats["std"]), 6),
        "target_min": round(float(ys.min()), 4),
        "target_max": round(float(ys.max()), 4),
        "task": "regression",
    }
