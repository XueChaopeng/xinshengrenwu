"""MiniGPT：decoder-only 语言模型（RoPE + KV cache + 采样）。"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .attention import DecoderBlock
from .sampling import sample_next_token


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4,
                 n_head: int = 4, d_model: int = 128, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_seq_len = block_size      # 供自检按真实上下文长度切窗
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model

        self.wte = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache=None, return_cache: bool = False):
        """idx: [B, T] int64。返回 logits [B, T, vocab]；return_cache=True 时返回 (logits, cache)。
        cache: list，每层一个 (k, v)，各 [B, H, T, head_dim]。"""
        B, T = idx.shape
        x = self.wte(idx)
        new_cache = [] if return_cache else None
        for i, layer in enumerate(self.layers):
            lc = None if kv_cache is None else kv_cache[i]
            x, c = layer(x, lc, return_cache)
            if return_cache:
                new_cache.append(c)
        logits = self.lm_head(x)
        if return_cache:
            return logits, new_cache
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int = 50, top_k: int | None = None,
                 top_p: float | None = None, temperature: float = 1.0):
        """自回归生成，返回包含 prompt 的完整 id 序列（list[int]）。自动使用 KV cache。"""
        device = next(self.parameters()).device
        if isinstance(prompt_ids, torch.Tensor):
            idx = prompt_ids.long()
            if idx.dim() == 1:
                idx = idx.unsqueeze(0)
        else:
            idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        idx = idx.to(device)

        output = idx[0].tolist()
        prompt_len = idx.size(1)
        max_new_tokens = min(max_new_tokens, max(0, self.block_size - prompt_len))
        if idx.size(1) > self.block_size:
            idx = idx[:, :self.block_size]
            output = idx[0].tolist()

        # 先全量过一遍 prompt，拿到 logits 与各层 cache
        logits, cache = self(idx, return_cache=True)
        for _ in range(max_new_tokens):
            next_id = int(sample_next_token(logits[0, -1, :], temperature, top_k, top_p).item())
            output.append(next_id)
            next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, cache = self(next_tok, kv_cache=cache, return_cache=True)
        return output


def load_for_eval(ckpt_path):
    """从 ckpt 恢复 (model, tokenizer)。ckpt 含 config / state_dict。"""
    from .tokenizer import BPETokenizer
    ckpt_path = Path(ckpt_path)
    tokenizer = BPETokenizer.from_pretrained(str(ckpt_path.parent / "tokenizer.json"))
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    config = ckpt["config"]
    model = MiniGPT(**config)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, tokenizer
