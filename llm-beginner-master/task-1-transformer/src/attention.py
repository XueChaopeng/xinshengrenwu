"""任务一：手写 scaled dot-product attention 与 Multi-Head Attention。

接口约定（见 task-1-transformer/README.md「实现约定」）：
    scaled_dot_product_attention(Q, K, V, mask=None)
        Q/K/V 形状 (B, H, T, D)；mask 形状可广播到 (B, H, T, T)，True = 被屏蔽。
    返回形状 (B, H, T, D)。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, need_weights=False):
    """Scaled dot-product attention（手写版）。

    公式：Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + mask) V
    其中 mask 在 logits 上生效：True 的位置填 -inf（而不是乘 0，否则 softmax
    后仍有概率泄漏），随后 softmax 概率为 0。

    need_weights=True 时返回 (out, attn_weights)，供可视化使用；默认只返回 out。
    """
    # Q, K, V: (B, H, T, D) -> logits: (B, H, T_q, T_k)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 乘 0 的坑：softmax(0)=exp(0)/Z 仍非零，必须填 -inf 才能让概率严格为 0
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V)
    if need_weights:
        return out, attn_weights
    return out


class MultiHeadAttention(nn.Module):
    """手写多头自注意力：Q/K/V 各自投影 -> 切分成 H 个头 -> 缩放点积 -> 拼接 -> 输出投影。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q/K/V 三套独立投影（不偷懒共用一套）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._last_weights = None  # 便于可视化时取出 (B, H, T, T) 注意力权重

    def forward(self, x, mask=None, need_weights=False):
        """x: (B, T, d_model)；mask: 形状可广播到 (B, H, T_q, T_k)，True=屏蔽。
        need_weights=True 时额外返回 (out, attn_weights (B,H,T,T))。"""
        B, T, _ = x.shape

        # 1. 投影
        Q = self.w_q(x)  # (B, T, d_model)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. 切分 head：(B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # 3. 缩放点积注意力（不调用 nn.MultiheadAttention / F.scaled_dot_product_attention）
        attn_out, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, need_weights=True)  # (B, H, T, d_k)

        # 4. 合并 head 并做输出投影；transpose 后必须 .contiguous() 再 view
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(attn_out)
        if need_weights:
            return out, attn_weights
        return out
