"""生成器：把检索上下文拼成 prompt，调用本地 Qwen（OpenAI 兼容），无服务时回退为拼接摘要。"""
from __future__ import annotations

import os

# 可选的本地 LLM 端点（Ollama 默认 http://localhost:11434/v1）。也可用 LLM_ENDPOINT 环境变量覆盖。
DEFAULT_ENDPOINTS = ["http://localhost:11434/v1", "http://localhost:8000/v1"]
MODEL_NAME = os.environ.get("RAG_LLM_MODEL", "qwen2.5-7b-instruct")


def _try_llm(prompt: str, base_url: str):
    try:
        import requests
        url = base_url.rstrip("/") + "/chat/completions"
        r = requests.post(url, json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256, "temperature": 0.3,
        }, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None


def build_prompt(query: str, contexts) -> str:
    # 去重 + 截断，避免重复片段稀释关键信息
    seen = set()
    uniq = []
    for c in contexts:
        t = (c or "").strip()
        if t and t[:60] not in seen:
            seen.add(t[:60])
            uniq.append(t)
    ctx = "\n\n".join(uniq[:5])
    return (
        "请仅根据以下给定上下文回答问题；如果上下文不足以回答，请回答“不知道”。\n\n"
        f"上下文：\n{ctx}\n\n问题：{query}\n\n回答："
    )


def generate_answer(query: str, contexts, llm_endpoint=None) -> str:
    prompt = build_prompt(query, contexts)
    endpoints = list(DEFAULT_ENDPOINTS)
    if llm_endpoint:
        endpoints.insert(0, llm_endpoint)
    if os.environ.get("LLM_ENDPOINT"):
        endpoints.insert(0, os.environ["LLM_ENDPOINT"])

    for ep in endpoints:
        if not ep:
            continue
        ans = _try_llm(prompt, ep)
        if ans:
            return ans

    # 回退：无本地 LLM 服务时，直接给出基于检索内容的摘要（保证 answer 非空）
    if contexts:
        top = (contexts[0] or "").strip()
        tail = "..." if len(top) > 220 else ""
        return f"根据检索内容：{top[:220]}{tail}"
    return "根据当前检索，未找到相关上下文。"
