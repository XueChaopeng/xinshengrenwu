"""任务一 M4：把同一个手写 attention 换成 causal mask，跑一个 toy 语言模型。

目的：验证「未来词元不泄漏」（self-check causal_mask 已保证），并预热任务二
（decoder-only + next-token prediction）。本脚本用仓库根目录的唐诗小语料
（poetryFromTang.txt）做字符级 next-token 训练，训练几步后采样几句看看。

用法（在 task-1-transformer/ 目录下）：
    python toy_lm.py [--steps 600] [--device cpu|cuda]
"""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import scaled_dot_product_attention
from src.block import TransformerBlock

ROOT = Path(__file__).resolve().parent
POETRY = ROOT.parent / "poetryFromTang.txt"

PAD, UNK = 0, 1


def load_text():
    if not POETRY.exists():
        raise SystemExit(f"找不到 {POETRY}，请先确认仓库根目录有唐诗语料。")
    return POETRY.read_text(encoding="utf-8")


def build_vocab(text, max_vocab=800):
    chars = {PAD: "[PAD]", UNK: "[UNK]"}
    cnt = {}
    for ch in text:
        cnt[ch] = cnt.get(ch, 0) + 1
    for ch, _ in sorted(cnt.items(), key=lambda kv: -kv[1]):
        if len(chars) >= max_vocab:
            break
        chars[len(chars)] = ch
    stoi = {ch: i for i, ch in chars.items()}
    return stoi


class ToyLM(nn.Module):
    """极简 decoder-only：embedding + N 个带 causal mask 的 TransformerBlock + 输出头。"""

    def __init__(self, vocab_size, d_model=96, n_heads=4, n_layers=2,
                 max_len=64, dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        """ids: (B, T)。用 causal mask + padding mask，预测每个位置的下一个词。"""
        B, T = ids.shape
        pad = (ids == self.pad_id).unsqueeze(1).unsqueeze(2)          # (B,1,1,T)
        causal = torch.triu(torch.ones(T, T, device=ids.device), diagonal=1).bool()
        mask = pad | causal.unsqueeze(0)                              # 广播到 (B,H,T,T)

        x = self.tok_emb(ids) + self.pos_emb(
            torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.head(self.norm(x))


def sample(model, stoi, itos, prompt, max_len=32, device="cpu"):
    model.eval()
    ids = [stoi.get(ch, UNK) for ch in prompt][-24:]
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / 0.8
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1).item())
            ids.append(nxt)
    return "".join(itos[i] for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    text = load_text()
    stoi = build_vocab(text)
    itos = {i: ch for ch, i in stoi.items()}
    data = [stoi.get(ch, UNK) for ch in text]
    print(f"语料 {len(text)} 字符，词表 {len(stoi)}，device={args.device}")

    model = ToyLM(vocab_size=len(stoi), max_len=args.seq_len,
                  pad_id=PAD).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_batches = max(1, (len(data) - args.seq_len - 1) // args.batch_size)
    print(f"每步 loss 基线 ≈ {math.log(len(stoi)):.2f}（均匀随机预测）")

    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        starts = [random.randrange(0, len(data) - args.seq_len - 1)
                  for _ in range(args.batch_size)]
        xs = torch.stack([torch.tensor(data[s:s + args.seq_len]) for s in starts])
        ys = torch.stack([torch.tensor(data[s + 1:s + args.seq_len + 1])
                          for s in starts])
        xs, ys = xs.to(args.device), ys.to(args.device)
        logits = model(xs)                          # (B,T,V)
        loss = F.cross_entropy(logits.reshape(-1, len(stoi)), ys.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0 or step == 1:
            print(f"step {step:4d}  loss={loss.item():.3f}  "
                  f"sample: {sample(model, stoi, itos, '床前明月光', device=args.device)}")

    out = ROOT / "toy_lm_sample.txt"
    gen = sample(model, stoi, itos, "床前明月光", max_len=24, device=args.device)
    out.write_text(gen, encoding="utf-8")
    print(f"\n最终采样（存入 {out.name}）：\n{gen}")


if __name__ == "__main__":
    main()
