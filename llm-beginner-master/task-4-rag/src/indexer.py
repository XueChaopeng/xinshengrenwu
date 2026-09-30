"""BGE embedding + FAISS 索引构建（从 data/kb.pdf 抽取文本，不索引 LaTeX 源）。

- Embedder：包装 BGE 小模型，query 侧加检索前缀，文档侧不加；统一 L2 归一化（这样内积等价 cosine）。
- build_index：PDF 文本 -> chunk -> embedding -> FAISS 内积索引，保存到 data/index/。
- 索引必须来自 data/kb.pdf，gold anchor 命中口径才真实。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "data" / "kb.pdf"
INDEX_DIR = ROOT / "data" / "index"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self, model_path=None):
        from sentence_transformers import SentenceTransformer
        model_path = model_path or str(ROOT / "models" / "bge-small-zh-v1.5")
        self.model = SentenceTransformer(model_path)

    def encode(self, texts, is_query: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def build_index(pdf_path=None, chunk_size: int = 256, overlap: int = 32, out_dir=None):
    import faiss
    pdf_path = Path(pdf_path or DEFAULT_PDF)
    out_dir = Path(out_dir or INDEX_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .chunker import extract_pdf_text, chunk_text
    print(f"[index] 抽取 {pdf_path.name} 文本 ...")
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"[index] chunks = {len(chunks)}")

    embedder = Embedder()
    vecs = embedder.encode(chunks)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # 已 L2 normalize，内积即 cosine
    index.add(vecs)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    meta = {"chunks": chunks, "chunk_size": chunk_size, "overlap": overlap,
            "source": str(pdf_path.name), "dim": dim, "count": len(chunks)}
    (out_dir / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[index] 已保存到 {out_dir}（{len(chunks)} chunks, dim={dim}）")
    return index, chunks


if __name__ == "__main__":
    build_index()
