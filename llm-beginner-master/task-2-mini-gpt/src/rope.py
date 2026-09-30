"""RoPE（Rotary Position Embedding）旋转位置编码。

参考 RoFormer (https://arxiv.org/abs/2104.09864)：
- 对每个位置的 token，把它的 q/k 向量按「维度对」旋转一个与位置相关的角度。
- 只作用在 Q 和 K 上，**不作用在 V**。
- 频率基底常用 base=10000，head_dim 为每头的维数。

这里用分词头「折半旋转（rotate-half）」约定，匹配 LLaMA / HF 常见实现：
把 dim 维拆成前后两半，cos/sin 的维度是 dim//2，配对 (x[i], x[i+dim//2]) 一起旋转。
只要 apply_rotary 与预计算的 cos/sin 下标一致，开/关 KV cache 的结果就完全一致
（自检 kv_cache_equivalence 正是依赖这一点）。
"""
from __future__ import annotations

import torch


def _precompute_freqs_cis(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """预计算 [max_seq_len, head_dim//2] 的 cos/sin 表。

    返回 (cos, sin)，均为 float32 的 [max_seq_len, head_dim//2] 张量。
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE 要求 head_dim 为偶数")
    # freqs：位置无关，逐维的频率 theta_i = base^(-2i/head_dim)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    # positions: [max_seq_len, 1]
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    # angles: [max_seq_len, head_dim//2]
    angles = positions * freqs.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def get_rope_table(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None):
    """带缓存的 getter：同一 (head_dim, base) 只算一次，保证长序列复用同一张 freq 表。"""
    cos, sin = _precompute_freqs_cis(head_dim, max_seq_len, base)
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def apply_rotary(q, k, cos, sin):
    """对 q, k 施加 RoPE。

    参数：
      q, k: [B, H, T, head_dim]
      cos, sin: [T, head_dim//2]（必要时广播到 [B, H, T, head_dim//2]）
    返回：
      施加 RoPE 后的 q, k。
    """
    d = q.size(-1)
    # 折半
    q1 = q[..., : d // 2]
    q2 = q[..., d // 2:]
    k1 = k[..., : d // 2]
    k2 = k[..., d // 2:]

    # cos/sin 形状 [T, d//2] -> [1, 1, T, d//2]
    cos = cos.view(1, 1, -1, d // 2)
    sin = sin.view(1, 1, -1, d // 2)

    q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_out, k_out
