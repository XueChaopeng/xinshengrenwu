"""Download and prepare all Task-3 datasets into ``task3_graph_classification/data``.

Datasets
--------
* ``MUTAG`` / ``ENZYMES`` / ``PROTEINS`` / ``IMDB-BINARY`` -- TUDataset, served
  by ``www.chrsmrrs.com/graphkerneldatasets`` (PyG's own mirror).  PyG downloads
  and processes them into ``data/<NAME>/{raw,processed}``.  A manual fallback
  downloads the very same ``<NAME>.zip`` when ``torch_geometric`` cannot.
* ``ZINC`` -- PyG's ``ZINC`` downloads ``molecules.zip`` from **Dropbox**, which
  is unreachable in this environment.  The same data is fetched from the
  HuggingFace **mirror** (``hf-mirror.com/datasets/graphs-datasets/ZINC``) as
  three JSONL files with resumable ``.part`` downloads and retries.

  .. note::
     That mirror serves the **full** ZINC (220,011 / 24,445 / 5,000 graphs,
     exactly PyG's ``ZINC Full`` = 249,456 graphs), *not* the 12,000-graph
     "ZINC Subset" used by the *Benchmarking GNNs* paper.  The standard subset
     therefore has to be reconstructed: PyG's own ``ZINC`` does exactly this
     with the ``{train,val,test}.index`` files from
     ``graphdeeplearning/benchmarking-gnns``, which index into the *per-split*
     pickles.  Those three index files are downloaded here through the GitHub
     API (``api.github.com``, ``Accept: application/vnd.github.raw``) and the
     per-split JSONL is subset accordingly -> 10,000 / 1,000 / 1,000 graphs,
     i.e. byte-for-byte PyG's ``ZINC(root, subset=True)``.

  The conversion produces ``torch_geometric.data.Data`` objects cached as
  ``data/ZINC_processed/{train,val,test}.pt`` (+ ``stats.json`` with the
  training-target mean/std and the provenance of the split).

The script is **idempotent**: everything already on disk is reused.

Usage
-----
>>> python prepare_data.py                       # everything
>>> python prepare_data.py --datasets ZINC
>>> python prepare_data.py --verify              # only print statistics
>>> python prepare_data.py --force               # ignore caches
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (  # noqa: E402
    DATASETS,
    TU_DATASETS,
    ZINC_DATASET,
    convert_zinc_jsonl,
    default_root,
    describe_tu,
    describe_zinc,
    load_tu_dataset,
    load_zinc,
    zinc_paths,
)

#: TUDataset mirror used by PyG itself.
TU_URL = "https://www.chrsmrrs.com/graphkerneldatasets"

#: HuggingFace mirror of the full PyG ZINC splits (JSONL, one graph per line).
ZINC_BASE_URL = "https://hf-mirror.com/datasets/graphs-datasets/ZINC/resolve/main"
ZINC_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")
#: Graph counts of the *full* ZINC files served by the mirror (== PyG "ZINC Full").
ZINC_FULL_COUNTS = {"train": 220011, "val": 24445, "test": 5000}
#: Graph counts of PyG's "ZINC Subset" (Benchmarking GNNs), selected by the
#: per-split index files below.
ZINC_EXPECTED = {"train": 10000, "val": 1000, "test": 1000}

#: Standard subset indices, one file per split, indexing into that split.
#: ``raw.githubusercontent.com`` is flaky here, so the GitHub API is primary.
ZINC_INDEX_URLS = (
    "https://api.github.com/repos/graphdeeplearning/benchmarking-gnns/contents"
    "/data/molecules/{split}.index",
    "https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns"
    "/master/data/molecules/{split}.index",
)
ZINC_INDEX_ACCEPT = {"api.github.com": "application/vnd.github.raw"}


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def download_resumable(url: str, dst: str, retries: int = 8, min_size: int = 1) -> str:
    """Download ``url`` to ``dst``, resuming a ``.part`` file and retrying.

    Returns ``dst``.  An existing, large-enough ``dst`` is reused untouched.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
        print(f"  cached  {os.path.basename(dst)} "
              f"({os.path.getsize(dst) / 1e6:.1f} MB)")
        return dst

    part = dst + ".part"
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        start = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if start and r.status != 206:  # server ignored Range -> restart
                    start = 0
                total = int(r.headers.get("Content-Length") or 0) + start
                print(f"  GET     {url}"
                      + (f"  (resume at {start / 1e6:.1f} MB)" if start else ""),
                      flush=True)
                done = start
                last_log = time.time()
                with open(part, "ab" if start else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last_log > 5:
                            last_log = time.time()
                            pct = f"{100 * done / total:.1f}%" if total else "?"
                            print(f"    {os.path.basename(dst)}: "
                                  f"{done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct})",
                                  flush=True)
            os.replace(part, dst)
            print(f"  saved   {os.path.basename(dst)} "
                  f"({os.path.getsize(dst) / 1e6:.1f} MB)")
            return dst
        except Exception as exc:  # noqa: BLE001 - retry the next attempt
            last_err = exc
            print(f"  ! attempt {attempt}/{retries}: {type(exc).__name__}: {exc}")
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"could not download {url}: {last_err}")


def prepare_tu_manual(name: str, root: str) -> str:
    """Fallback: fetch ``<name>.zip`` ourselves into ``data/<NAME>/raw``."""
    raw_dir = os.path.join(root, name, "raw")
    expected = os.path.join(raw_dir, f"{name}_graph_indicator.txt")
    if os.path.exists(expected):
        return "raw files already present"

    url = f"{TU_URL}/{name}.zip"
    zip_path = download_resumable(url, os.path.join(root, "_download", f"{name}.zip"))
    os.makedirs(raw_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            target = os.path.join(raw_dir, os.path.basename(member.filename))
            # A stale processed cache must be dropped when raw data changes.
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    processed = os.path.join(root, name, "processed")
    if os.path.isdir(processed):
        shutil.rmtree(processed)
    return f"unpacked from {url}"


# --------------------------------------------------------------------------- #
# TUDataset
# --------------------------------------------------------------------------- #
def prepare_tu(name: str, root: str, force: bool = False) -> None:
    print(f"[{name}]")
    processed = os.path.join(root, name, "processed", "data.pt")
    if force and os.path.isdir(os.path.join(root, name)):
        shutil.rmtree(os.path.join(root, name))
    if not os.path.exists(processed):
        try:
            load_tu_dataset(name, root)
        except Exception as exc:  # noqa: BLE001 - fall back to a manual download
            print(f"  ! PyG could not fetch {name} ({type(exc).__name__}: "
                  f"{str(exc)[:120]}); falling back to a manual download")
            print(f"  {prepare_tu_manual(name, root)}")
    graphs, nf, nc = load_tu_dataset(name, root)
    print(f"  ready -> {describe_tu(graphs, nf, nc)}")


# --------------------------------------------------------------------------- #
# ZINC
# --------------------------------------------------------------------------- #
def download_text(urls, retries: int = 5) -> str:
    """Try several URLs in order and return the downloaded text."""
    last_err: Exception | None = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                host = url.split("/")[2]
                headers = {"User-Agent": "Mozilla/5.0"}
                if host in ZINC_INDEX_ACCEPT:
                    headers["Accept"] = ZINC_INDEX_ACCEPT[host]
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read().decode("utf-8")
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last_err = exc
                time.sleep(min(2 ** attempt, 10))
        print(f"  ! {url.split('/')[2]} failed: {type(last_err).__name__}: "
              f"{str(last_err)[:100]}")
    raise RuntimeError(f"could not download any of {urls}: {last_err}")


def load_zinc_index(split: str, raw_dir: str, force: bool = False) -> list[int]:
    """Return the standard-subset indices for ``split`` (cached on disk)."""
    path = os.path.join(raw_dir, f"{split}.index")
    if os.path.exists(path) and not force:
        print(f"  cached  {split}.index")
    else:
        text = download_text([u.format(split=split) for u in ZINC_INDEX_URLS])
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  saved   {split}.index ({len(text)} bytes)")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # The files are a single comma-separated line with a trailing separator.
    return [int(x) for x in text.strip().rstrip(",").split(",") if x.strip()]


def prepare_zinc(root: str, force: bool = False, full: bool = False) -> None:
    print(f"[{ZINC_DATASET}]")
    paths = zinc_paths(root)
    os.makedirs(paths["raw_dir"], exist_ok=True)
    os.makedirs(paths["processed_dir"], exist_ok=True)

    if force:
        for n in list(ZINC_FILES) + [f"{s}.index" for s in ("train", "val", "test")]:
            for p in (os.path.join(paths["raw_dir"], n),
                      os.path.join(paths["raw_dir"], n + ".part")):
                if os.path.exists(p):
                    os.remove(p)

    # 1) download the three full jsonl splits (resumable, retried) -----------
    for name in ZINC_FILES:
        download_resumable(f"{ZINC_BASE_URL}/{name}",
                           os.path.join(paths["raw_dir"], name))

    # 2) select PyG's standard 12,000-graph subset --------------------------
    if full:
        print("  --zinc-full: keeping the ENTIRE ZINC (249,456 graphs); results "
              "are NOT comparable to the standard 12k ZINC benchmark")
        subsets = {s: None for s in ("train", "val", "test")}
    else:
        subsets = {}
        for split in ("train", "val", "test"):
            idx = load_zinc_index(split, paths["raw_dir"], force=force)
            if len(idx) != ZINC_EXPECTED[split]:
                raise RuntimeError(f"{split}.index has {len(idx)} entries, "
                                   f"expected {ZINC_EXPECTED[split]}")
            if max(idx) >= ZINC_FULL_COUNTS[split]:
                raise RuntimeError(
                    f"{split}.index references molecule {max(idx)} but the "
                    f"{split} file only holds {ZINC_FULL_COUNTS[split]} graphs")
            subsets[split] = idx

    # 3) convert to Data objects and cache as InMemoryDataset .pt -----------
    from torch_geometric.data import InMemoryDataset  # local import: keeps CLI fast

    all_y: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        out = os.path.join(paths["processed_dir"], f"{split}.pt")
        raw = os.path.join(paths["raw_dir"], f"{split}.jsonl")
        graphs = convert_zinc_jsonl(raw)
        if len(graphs) != ZINC_FULL_COUNTS[split]:
            print(f"  ! {split}: downloaded {len(graphs)} graphs, "
                  f"expected {ZINC_FULL_COUNTS[split]}")
        if subsets[split] is not None:
            graphs = [graphs[i] for i in subsets[split]]
        counts[split] = len(graphs)
        all_y[split] = torch.cat([g.y.view(-1) for g in graphs])
        if os.path.exists(out) and not force:
            print(f"  cached  {split}.pt ({len(graphs)} graphs, "
                  f"{os.path.getsize(out) / 1e6:.1f} MB)")
            continue
        data_list, slices = InMemoryDataset.collate(graphs)
        torch.save((data_list, slices), out)
        print(f"  saved   {split}.pt ({len(graphs)} graphs, "
              f"{os.path.getsize(out) / 1e6:.1f} MB)")

    # 4) training-target statistics (standardisation is the trainer's job) ---
    ytr = all_y["train"].double()
    yte = all_y["test"].double()
    stats = {
        "mean": float(ytr.mean()),
        "std": float(ytr.std(unbiased=False)),
        "train_mean": float(ytr.mean()),
        "train_std": float(ytr.std(unbiased=False)),
        "test_mean": float(yte.mean()),
        "test_std": float(yte.std(unbiased=False)),
        "train_min": float(ytr.min()),
        "train_max": float(ytr.max()),
        "test_min": float(yte.min()),
        "test_max": float(yte.max()),
        "count": int(ytr.numel()),
        "splits": counts,
        "subset": (not full),
        "source": f"{ZINC_BASE_URL}/{{train,val,test}}.jsonl",
        "split_source": ("graphdeeplearning/benchmarking-gnns "
                         "data/molecules/{train,val,test}.index "
                         "(same indices PyG's ZINC(subset=True) uses)"
                         if not full else "full ZINC, no subsetting"),
        "note": ("y is PyG's ZINC target field `logP_SA_cycle_normalized` "
                 "(penalized logP). It is NOT unit-variance: the standard 12k "
                 "subset has train mean/std as recorded above, and the "
                 "Benchmarking-GNNs literature reports MAE on this very scale. "
                 "train.py additionally standardises only if --standardize is "
                 "passed, and always converts MAE back to these units."),
    }
    with open(paths["stats"], "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  target stats: train mean={stats['mean']:.6f} std={stats['std']:.6f} "
          f"(n={stats['count']})")
    print(f"  target range: train [{stats['train_min']:.3f}, "
          f"{stats['train_max']:.3f}]  test [{stats['test_min']:.3f}, "
          f"{stats['test_max']:.3f}]")

    splits, stats = load_zinc(root)
    print(f"  ready -> {describe_zinc(splits, stats)}")


# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #
def verify(root: str) -> int:
    missing = 0
    for name in TU_DATASETS:
        try:
            graphs, nf, nc = load_tu_dataset(name, root)
            print(f"[{name}] OK  {describe_tu(graphs, nf, nc)}")
        except Exception as exc:  # noqa: BLE001
            missing += 1
            print(f"[{name}] MISSING: {type(exc).__name__}: {str(exc)[:200]}")
    try:
        splits, stats = load_zinc(root)
        print(f"[ZINC] OK  {describe_zinc(splits, stats)}")
    except Exception as exc:  # noqa: BLE001
        missing += 1
        print(f"[ZINC] MISSING: {type(exc).__name__}: {str(exc)[:200]}")
    print(f"\n{len(DATASETS) - missing}/{len(DATASETS)} datasets available under {root}")
    return missing


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Task-3 datasets (TUDataset + ZINC)")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--data-root", default=None)
    p.add_argument("--verify", action="store_true", help="only report availability")
    p.add_argument("--force", action="store_true", help="re-download / re-convert")
    p.add_argument("--zinc-full", action="store_true",
                   help="keep the full 249,456-graph ZINC instead of the standard "
                        "12,000-graph subset (NOT comparable to the literature)")
    args = p.parse_args()

    root = args.data_root or default_root()
    os.makedirs(root, exist_ok=True)
    print(f"data root: {root}\n")

    if args.verify:
        sys.exit(verify(root))

    for name in args.datasets:
        if name == ZINC_DATASET:
            prepare_zinc(root, force=args.force, full=args.zinc_full)
        else:
            prepare_tu(name, root, force=args.force)
    print()
    verify(root)


if __name__ == "__main__":
    main()
