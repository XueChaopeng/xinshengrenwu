"""Download and prepare all Task-2 datasets into ``task2_link_prediction/data``.

Datasets
--------
* ``Cora`` / ``Citeseer`` -- Planetoid, fetched from the PyG index
  (``github.com/kimiyoung/planetoid``).
* ``Flickr``             -- GraphSAINT's Flickr (89,250 nodes).  PyG serves this
  from Google Drive, which is unreachable in some networks, so this script
  falls back to DGL's public mirror ``https://data.dgl.ai/dataset/flickr.zip``
  which ships the *identical* four raw files
  (``adj_full.npz``, ``feats.npy``, ``class_map.json``, ``role.json``).
  The files are unpacked into ``data/Flickr/raw`` where PyG picks them up and
  performs its usual processing.

If you already ran Task 1, the raw files under
``task1_node_classification/data/<name>/raw`` can simply be copied here to skip
the download entirely.

Usage
-----
>>> python prepare_data.py                 # everything
>>> python prepare_data.py --datasets Cora Citeseer
>>> python prepare_data.py --verify        # only print statistics
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import default_root, describe, load_dataset  # noqa: E402

FLICKR_RAW_FILES = ["adj_full.npz", "feats.npy", "class_map.json", "role.json"]

#: Mirrors that ship the GraphSAINT Flickr raw files, tried in order.
FLICKR_ZIP_URLS = [
    "https://data.dgl.ai/dataset/flickr.zip",
    "https://data.dgl.ai/dataset/Flickr.zip",
]


def _download(url: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  cached  {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
        return dst
    print(f"  GET     {url}")
    tmp = dst + ".part"
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)
    print(f"  saved   {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    return dst


def prepare_flickr(root: str) -> str:
    """Make sure ``data/Flickr/raw`` holds the four GraphSAINT files."""
    raw_dir = os.path.join(root, "Flickr", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(raw_dir, f)) for f in FLICKR_RAW_FILES):
        return "raw files already present"

    last_err: Exception | None = None
    for url in FLICKR_ZIP_URLS:
        try:
            zip_path = _download(url, os.path.join(root, "_download", os.path.basename(url)))
            with zipfile.ZipFile(zip_path) as z:
                for member in z.infolist():
                    name = os.path.basename(member.filename)
                    if name not in FLICKR_RAW_FILES:
                        continue
                    target = os.path.join(raw_dir, name)
                    with z.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            missing = [f for f in FLICKR_RAW_FILES if not os.path.exists(os.path.join(raw_dir, f))]
            if missing:
                raise RuntimeError(f"zip is missing {missing}")
            return f"unpacked from {url}"
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_err = exc
            print(f"  ! {url} failed: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not obtain the Flickr raw files.  If you have access to Google Drive, "
        "simply run `python -c \"from torch_geometric.datasets import Flickr; "
        f"Flickr(root=r'{root}')\"` and PyG will download them itself.\n"
        f"Last error: {last_err}"
    )


def prepare(datasets, root: str) -> None:
    for name in datasets:
        print(f"[{name}]")
        if name == "Flickr":
            how = prepare_flickr(root)
            print(f"  {how}")
        data, nf, nc = load_dataset(name, root)
        stats = describe(data, nf, nc)
        print(f"  ready -> {stats}")


def verify(datasets, root: str) -> None:
    for name in datasets:
        try:
            data, nf, nc = load_dataset(name, root)
            print(f"[{name}] OK {describe(data, nf, nc)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] MISSING: {type(exc).__name__}: {str(exc)[:160]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Task-1 datasets")
    p.add_argument("--datasets", nargs="+", default=["Cora", "Citeseer", "Flickr"],
                   choices=["Cora", "Citeseer", "Flickr"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--verify", action="store_true", help="only report availability")
    args = p.parse_args()
    root = args.data_root or default_root()
    os.makedirs(root, exist_ok=True)
    print(f"data root: {root}\n")
    (verify if args.verify else prepare)(args.datasets, root)


if __name__ == "__main__":
    main()
