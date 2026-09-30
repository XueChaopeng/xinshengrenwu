"""端到端 RAG：answer(query) 串起 检索(+rerank) -> 生成，返回 {answer, sources}。"""
from __future__ import annotations

from .retriever import Retriever
from .generator import generate_answer


def answer(query: str, k: int = 10, use_rerank: bool = True, top_n: int = 4):
    """RAG 端到端。返回 dict(answer: str, sources: List[dict])。"""
    retriever = Retriever()
    docs = retriever.retrieve(query, k=k)

    if use_rerank:
        try:
            from .reranker import Reranker
            docs = Reranker().rerank(query, docs)[:top_n]
        except Exception:
            docs = docs[:top_n]
    else:
        docs = docs[:top_n]

    contexts = [d["text"] for d in docs]
    ans = generate_answer(query, contexts)
    return {"answer": ans, "sources": docs}


if __name__ == "__main__":
    r = answer("什么是反向传播？")
    print("answer:", r["answer"][:200])
    print("sources:", len(r["sources"]))
