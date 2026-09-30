"""Causal Multi-Head Attention + Decoder Block，集成 RoPE 与 KV cache。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary, _precompute_freqs_cis


class CausalAttention(nn.Module):
    """causal masked multi-head attention，支持传入/更新 KV cache。"""

    def __init__(self, d_model: int, n_head: int, block_size: int, base: float = 10000.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # 预计算 RoPE 的 cos/sin 表（[block_size+1, head_dim//2]）与 causal mask。
        # 自检按 block_size 分块时会喂 block_size+1 个词元（多留一个位置算 next-token loss），
        # 因此位置表要多留一位；标记为 non-persistent，加载时按 config 重新生成，不进 state_dict。
        rope_len = block_size + 1
        self.register_buffer("_cos", _precompute_freqs_cis(self.head_dim, rope_len, base)[0],
                             persistent=False)
        self.register_buffer("_sin", _precompute_freqs_cis(self.head_dim, rope_len, base)[1],
                             persistent=False)
        # 上三角为 -inf 的 causal mask：[block_size+1, block_size+1]
        causal = torch.triu(torch.full((rope_len, rope_len), float("-inf")), diagonal=1)
        self.register_buffer("_causal", causal, persistent=False)

    def forward(self, x, kv_cache=None, return_cache=False):
        """x: [B, T, d_model]。

        kv_cache: (k, v)，各 [B, H, past_T, head_dim]；为 None 表示无历史。
        return_cache: True 时返回 (out, (k, v))，否则只返回 out。
        """
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        start = 0 if kv_cache is None else kv_cache[0].size(2)
        # 新 token 的 position 要从历史长度继续算（否则 RoPE 角度错、与全量前向对不上）
        cos = self._cos[start:start + T]
        sin = self._sin[start:start + T]
        q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)   # 在序列维度 T 上 append
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v) if return_cache else None

        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,T,Tk]
        # causal mask：对全序列长度 Tk 取上三角掩码，再取最后 T 行（对应新 query 位置）
        mask = self._causal[start:start + T, :Tk]
        att = att + mask.unsqueeze(0).unsqueeze(0)
        att = F.softmax(att, dim=-1)

        y = att @ v                                  # [B,H,T,head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.wo(y)
        return y, new_cache


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * mult, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    """Pre-LN 的 Transformer decoder block。"""

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_head, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x, kv_cache=None, return_cache=False):
        a, cache = self.attn(self.ln1(x), kv_cache, return_cache)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        if return_cache:
            return x, cache
        return x, None
