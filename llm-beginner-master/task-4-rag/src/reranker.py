"""bge-reranker 精排：对 [query, doc] 文本对打分，重排召回结果。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "bge-reranker-base"


class Reranker:
    def __init__(self, model_path=None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(model_path or MODEL_PATH))

    def rerank(self, query: str, docs):
        """docs: list[dict]（含 text）。返回排好序的 docs，附上 rerank_score。"""
        if not docs:
            return []
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs)
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
        out = []
        for i in order:
            d = dict(docs[i])
            d["rerank_score"] = float(scores[i])
            out.append(d)
        return out


if __name__ == "__main__":
    from .retriever import Retriever
    rr = Reranker()
    ret = Retriever()
    docs = ret.retrieve("什么是反向传播？", k=20)
    top = rr.rerank("什么是反向传播？", docs)[:5]
    for d in top:
        print(f"  {d['rerank_score']:.4f}  {d['text'][:60]!r}")
