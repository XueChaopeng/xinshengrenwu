"""检索器：BGE 编码 query + FAISS 内积（cosine）召回。"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .indexer import Embedder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = ROOT / "data" / "index"


class Retriever:
    def __init__(self, index_dir=None):
        index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        if not (index_dir / "index.faiss").exists():
            raise FileNotFoundError(f"索引不存在：{index_dir}/index.faiss；请先运行 src/indexer.py")
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        meta = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunks: list[str] = meta["chunks"]
        self.source = meta.get("source", "kb.pdf")
        self.embedder = Embedder()

    def retrieve(self, query: str, k: int = 10):
        q = self.embedder.encode(query, is_query=True)
        scores, idxs = self.index.search(q, k)
        results = []
        for j in range(min(k, self.index.ntotal)):
            i = int(idxs[0][j])
            if i < 0 or i >= len(self.chunks):
                continue
            results.append({
                "text": self.chunks[i],
                "score": float(scores[0][j]),
                "source": self.source,
            })
        return results


if __name__ == "__main__":
    r = Retriever()
    for q in ["什么是反向传播？", "注意力机制是什么？"]:
        print(f"\nQ: {q}")
        for d in r.retrieve(q, k=3):
            print(f"  {d['score']:.4f}  {d['text'][:60]!r}")
