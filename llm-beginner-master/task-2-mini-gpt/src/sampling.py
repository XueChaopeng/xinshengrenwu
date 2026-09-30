"""采样策略：greedy / top-k / top-p / temperature。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next_token(logits, temperature: float = 1.0, top_k: int | None = None,
                      top_p: float | None = None):
    """从 logits（[vocab]）采样下一个 token 的 id（标量 int tensor）。

    - temperature <= 0（或 None）时退化为 greedy（argmax），避免除零。
    - top_k：只保留概率最大的 k 个，其余置 -inf。
    - top_p：按概率降序累积到阈值，其后的置 -inf，再重归一化。
    """
    # temperature == 0 -> greedy
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        if k > 0:
            v, _ = torch.topk(logits, k)
            logits = torch.where(
                logits < v[..., -1:],
                torch.full_like(logits, float("-inf")),
                logits,
            )

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # 去掉「累积概率已超过 top_p 且不是当前 token」的位置
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        # 恢复到原始顺序
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return torch.argmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)
