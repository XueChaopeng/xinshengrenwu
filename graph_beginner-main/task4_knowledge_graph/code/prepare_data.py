"""Download and validate the Task-4 knowledge-graph datasets.

Datasets
--------
* ``WN18RR``   -- WordNet-18 (RR), 40,943 entities / 11 relations /
                  86,835 train, 3,034 valid, 3,134 test triples.
* ``FB15k-237``-- Freebase-15k-237, 14,541 entities / 237 relations /
                  272,115 train, 17,535 valid, 20,466 test triples.

Both are fetched as raw files from the ``DeepGraphLearning/KnowledgeGraphEmbedding``
repository (the same reference data used by ``Maxioo/kge_framework``) into

    data/<NAME>/{train,valid,test}.txt

Downloads are written to a ``.part`` temporary file and only renamed on success,
so an interrupted transfer can never be mistaken for a complete dataset.  Every
mirror is retried with exponential back-off, and after downloading the file
statistics are compared against the published ones -- a truncated or bogus
download therefore fails loudly instead of silently producing a small graph.

Usage
-----
>>> python prepare_data.py                      # download + verify everything
>>> python prepare_data.py --datasets WN18RR
>>> python prepare_data.py --verify             # only print the statistics
>>> python prepare_data.py --verify --strict    # exit non-zero on a mismatch
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (  # noqa: E402
    DATA_DIRNAME,
    DATASETS,
    EXPECTED_STATS,
    KGDataset,
    default_root,
    read_triples,
)

SPLITS = ("train", "valid", "test")

#: Mirrors tried in order for every file.  ``raw.githubusercontent.com`` is the
#: primary (verified working) and ``github.com/.../raw/...`` is the fallback,
#: which redirects to a raw host.
URL_TEMPLATES = [
    "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/{dir}/{split}.txt",
    "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/{dir}/{split}.txt",
    "https://cdn.jsdelivr.net/gh/DeepGraphLearning/KnowledgeGraphEmbedding@master/data/{dir}/{split}.txt",
]

#: Directory name inside the upstream repository (case sensitive on raw hosts).
UPSTREAM_DIR = {"WN18RR": "wn18rr", "FB15k-237": "FB15k-237"}

RETRIES = 4
TIMEOUT = 180


def _report_hook(name: str):
    """Print progress at most every 2 seconds (works for chunked reads too)."""
    state = {"t": 0.0, "bytes": 0}

    def hook(nbytes: int, total: int) -> None:
        state["bytes"] += nbytes
        now = time.time()
        if now - state["t"] < 2.0:
            return
        state["t"] = now
        if total > 0:
            pct = 100.0 * state["bytes"] / total
            print(f"\r    {name}: {state['bytes'] / 1e6:8.1f}/{total / 1e6:.1f} MB "
                  f"({pct:5.1f}%)", end="", flush=True)
        else:
            print(f"\r    {name}: {state['bytes'] / 1e6:8.1f} MB", end="", flush=True)

    return hook


def download(url: str, dst: str, retries: int = RETRIES) -> None:
    """Download ``url`` to ``dst`` through a ``.part`` file, with retries."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"    GET {url} (attempt {attempt}/{retries})", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
                hook = _report_hook(os.path.basename(dst))
                total = int(r.headers.get("Content-Length") or 0)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    hook(len(chunk), total)
            print("", flush=True)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("downloaded file is empty")
            os.replace(tmp, dst)
            print(f"    saved {dst} ({os.path.getsize(dst) / 1e6:.2f} MB)", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - try the next mirror / attempt
            last_err = exc
            print(f"    ! attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"could not download {url}: {last_err}")


def ensure_split(name: str, split: str, root: str, force: bool = False) -> str:
    """Make sure ``data/<NAME>/<split>.txt`` exists; download it if not."""
    dst = os.path.join(root, DATA_DIRNAME[name], f"{split}.txt")
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and not force:
        print(f"  cached {dst} ({os.path.getsize(dst) / 1e6:.2f} MB)", flush=True)
        return dst
    last_err: Exception | None = None
    for template in URL_TEMPLATES:
        url = template.format(dir=UPSTREAM_DIR[name], split=split)
        try:
            download(url, dst)
            return dst
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_err = exc
            print(f"  ! mirror failed: {exc}", flush=True)
    raise RuntimeError(f"all mirrors failed for {name}/{split}: {last_err}")


def adopt_existing(name: str, root: str) -> None:
    """Reuse already-downloaded copies found in the repository's scratch dirs.

    An earlier partial attempt left data in ``data/raw`` and ``data/_probe``;
    a file is only adopted when its triple count matches the published split.
    """
    dst_dir = os.path.join(root, DATA_DIRNAME[name])
    exp = EXPECTED_STATS[name]
    candidates = [
        # (candidate path, which dataset/split it would be if valid)
        (os.path.join(root, "raw", "train.txt"), "train"),
        (os.path.join(root, "raw", "valid.txt"), "valid"),
        (os.path.join(root, "raw", "test.txt"), "test"),
        (os.path.join(root, "_probe", "wn18rr_train.txt"), "train"),
        (os.path.join(root, "_probe", "fb15k237_train.txt"), "train"),
    ]
    for src, split in candidates:
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, f"{split}.txt")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        try:
            n = len(read_triples(src))
        except Exception:  # noqa: BLE001
            continue
        if n != exp[split]:
            continue
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  adopted {src} -> {dst} ({n} triples)", flush=True)


def prepare(names, root: str, force: bool = False) -> None:
    for name in names:
        print(f"[{name}] downloading into {os.path.join(root, DATA_DIRNAME[name])}", flush=True)
        adopt_existing(name, root)
        for split in SPLITS:
            ensure_split(name, split, root, force=force)
        ds = KGDataset(name, root=root, verbose=True, vocab_from="all")
        ok = ds.verify()
        exp = EXPECTED_STATS[name]
        print(f"  expected: {exp}")
        print(f"  actual  : {ds.describe()}")
        print(f"  {'VERIFIED' if ok else 'MISMATCH -- data may be truncated or bogus'}\n",
              flush=True)
        if not ok:
            raise SystemExit(f"[{name}] statistics do not match the published split")


def verify(names, root: str, strict: bool = False) -> bool:
    all_ok = True
    for name in names:
        d = os.path.join(root, DATA_DIRNAME[name])
        missing = [s for s in SPLITS if not os.path.exists(os.path.join(d, s + ".txt"))]
        if missing:
            print(f"[{name}] MISSING {missing} in {d} -- run prepare_data.py first")
            all_ok = False
            continue
        try:
            ds = KGDataset(name, root=root, verbose=False, vocab_from="all")
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED to load: {type(exc).__name__}: {exc}")
            all_ok = False
            continue
        exp = EXPECTED_STATS[name]
        ok = ds.verify()
        all_ok = all_ok and ok
        print(f"[{name}] {'OK  ' if ok else 'FAIL'} "
              f"train={ds.raw_counts['train']} valid={ds.raw_counts['valid']} "
              f"test={ds.raw_counts['test']} entities={ds.num_entities} "
              f"relations={ds.num_relations}")
        print(f"        expected train={exp['train']} valid={exp['valid']} test={exp['test']} "
              f"entities={exp['entities']} relations={exp['relations']}"
              f"  -> {'match' if ok else 'MISMATCH'}")
        print(f"        rows after encoding: train={ds.train.shape[0]} valid={ds.valid.shape[0]} "
              f"test={ds.test.shape[0]}  OOV dropped(valid/test)="
              f"{ds.dropped['valid']}/{ds.dropped['test']}  (vocab_from={ds.vocab_from})")
        # Strict leak-free vocabulary: built from train only, OOV triples dropped.
        ds_train = KGDataset(name, root=root, verbose=False, vocab_from="train")
        print(f"        vocab_from=train: entities={ds_train.num_entities} "
              f"relations={ds_train.num_relations} "
              f"valid={ds_train.valid.shape[0]} test={ds_train.test.shape[0]} "
              f"dropped(valid/test)={ds_train.dropped['valid']}/{ds_train.dropped['test']} "
              f"(relations identical: {ds_train.num_relations == exp['relations']})")
    if strict and not all_ok:
        raise SystemExit(1)
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare the Task-4 knowledge-graph datasets")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--data-root", default=None)
    p.add_argument("--verify", action="store_true", help="only report the statistics")
    p.add_argument("--strict", action="store_true", help="with --verify: exit 1 on mismatch")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    args = p.parse_args()

    root = args.data_root or default_root()
    os.makedirs(root, exist_ok=True)
    print(f"data root: {root}\n")
    if args.verify:
        verify(args.datasets, root, strict=args.strict)
    else:
        prepare(args.datasets, root, force=args.force)


if __name__ == "__main__":
    main()
