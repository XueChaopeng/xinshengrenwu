"""任务二：在中文小语料（默认唐诗）上预训练一个 mini-GPT。

用法：
  python train.py                       # 默认参数，在 poetry 上训练
  python train.py --vocab-size 512 --block-size 128 --max-steps 3000
输出：
  ckpt/tokenizer.json     训练好的 BPE 词表
  ckpt/best.pt            dev 困惑度最低的模型 state_dict
  ckpt/config.json        模型超参
  ckpt/samples.txt        不同采样策略的生成样例
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.tokenizer import BPETokenizer
from src.model import MiniGPT

ROOT = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokens(tokenizer: BPETokenizer, path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    return np.array(tokenizer.encode(text), dtype=np.int64)


def get_batch(data: np.ndarray, block_size: int, batch_size: int):
    """采样一个 batch：从语料中随机取 batch_size 段长度为 block_size 的序列。"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size]) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_ppl(model: MiniGPT, dev_data: np.ndarray, block_size: int, max_tokens: int = 4096):
    """按模型上下文长度分块累加 NLL，再统一求困惑度（与 eval/run.py 一致）。"""
    model.eval()
    ids = dev_data[:max_tokens]
    nll, ntok = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block_size):
        window = ids[i:i + block_size + 1]
        if len(window) < 2:
            break
        x = torch.tensor([window[:-1]], dtype=torch.long, device=DEVICE)
        y = torch.tensor([window[1:]], dtype=torch.long, device=DEVICE)
        logits = model(x)
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="sum").item()
        ntok += y.size(1)
    return math.exp(nll / ntok) if ntok > 0 else float("inf")


def lr_schedule(step: int, max_steps: int, warmup: int, lr: float):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=str(ROOT / "data" / "train.txt"))
    ap.add_argument("--dev", type=str, default=str(ROOT / "data" / "dev.txt"))
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "ckpt"))
    args = ap.parse_args()

    seed_everything(args.seed)
    ROOT_ckpt = Path(args.ckpt_dir)
    ROOT_ckpt.mkdir(parents=True, exist_ok=True)

    # 1) 训练 tokenizer
    print(f"[tokenizer] vocab_size={args.vocab_size} ...")
    tokenizer = BPETokenizer()
    tokenizer.train(Path(args.train).read_text(encoding="utf-8"), vocab_size=args.vocab_size)
    tokenizer.save_pretrained(str(ROOT_ckpt / "tokenizer.json"))
    print(f"  tokenizer.vocab_size = {tokenizer.vocab_size}")

    # 2) 编码语料
    train_data = load_tokens(tokenizer, Path(args.train))
    dev_data = load_tokens(tokenizer, Path(args.dev))
    print(f"[data] train_tokens={len(train_data)} dev_tokens={len(dev_data)}")

    # 3) 建模型
    cfg = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "d_model": args.d_model,
        "dropout": args.dropout,
    }
    model = MiniGPT(**cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.2f}M, cfg={cfg}")

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    best_ppl = float("inf")
    best_step = 0
    start_t = time.time()

    for step in range(1, args.max_steps + 1):
        model.train()
        for g in optimizer.param_groups:
            g["lr"] = lr_schedule(step, args.max_steps, args.warmup, args.lr)
        x, y = get_batch(train_data, args.block_size, args.batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"  step {step}/{args.max_steps}  loss={loss.item():.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

        if step % args.eval_every == 0 or step == args.max_steps:
            ppl = estimate_ppl(model, dev_data, args.block_size)
            print(f"  === step {step}  dev_ppl={ppl:.2f}  best={best_ppl:.2f} ===")
            if ppl < best_ppl:
                best_ppl = ppl
                best_step = step
                torch.save({"config": cfg, "state_dict": model.state_dict(),
                            "step": step, "dev_ppl": ppl},
                           str(ROOT_ckpt / "best.pt"))
                print(f"      saved best.pt (step {step}, ppl {ppl:.2f})")

    # 4) 生成样例
    model.eval()
    prompts = tokenizer.encode("床前明月光")
    samples = {}
    samples["greedy"] = model.generate(prompts, max_new_tokens=40, temperature=0.0)
    samples["top_k_50"] = model.generate(prompts, max_new_tokens=40, top_k=50, temperature=0.8)
    samples["top_p_0.9"] = model.generate(prompts, max_new_tokens=40, top_p=0.9, temperature=0.8)
    samples["temperature_1.0"] = model.generate(prompts, max_new_tokens=40, temperature=1.0)
    with (ROOT_ckpt / "samples.txt").open("w", encoding="utf-8") as f:
        for name, ids in samples.items():
            f.write(f"--- {name} ---\n{tokenizer.decode(ids)}\n\n")
    (ROOT_ckpt / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    print(f"\n[done] best_ppl={best_ppl:.2f} @ step {best_step}, "
          f"elapsed={time.time() - start_t:.1f}s, ckpt -> {ROOT_ckpt}")


if __name__ == "__main__":
    main()
