"""任务一：Transformer encoder block = multi-head attention + FFN + residual + LayerNorm。

采用 Pre-LN 结构（先 LayerNorm 再进子层，residual 加在子层输出上），
比 Post-LN 训练更稳定，是 GPT/BERT 之后的主流做法。
"""
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """单个 Transformer encoder block（自注意力 + 前馈 + 两个残差 + 两个 LayerNorm）。"""

    def __init__(self, d_model, n_heads, dropout=0.1, expansion=4):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, need_weights=False):
        """x: (B, T, d_model)；mask 同 MultiHeadAttention（True=屏蔽）。
        need_weights=True 时把本层注意力权重 (B,H,T,T) 存到 self.last_weights。"""
        # Pre-LN：x + Attn(LN(x))
        if need_weights:
            attn_out, w = self.attn(self.ln1(x), mask=mask, need_weights=True)
            self.last_weights = w
            x = x + self.dropout(attn_out)
        else:
            x = x + self.dropout(self.attn(self.ln1(x), mask=mask))
        # x + FFN(LN(x))
        x = x + self.ffn(self.ln2(x))
        return x
