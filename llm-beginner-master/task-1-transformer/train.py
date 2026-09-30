"""任务一：在 ChnSentiCorp 上训练 Transformer 情感分类器并保存 ckpt/best.pt。

用法（在 task-1-transformer/ 目录下）：
    python train.py [--epochs 5] [--batch-size 64] [--lr 3e-4]
                    [--d-model 128] [--n-heads 4] [--n-layers 4] [--max-len 128]
保存：ckpt/best.pt（state_dict）+ ckpt/config.json + ckpt/vocab.json + ckpt/train_log.json
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import (TransformerClassifier, build_char_vocab,
                       make_tokenize_fn)

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "ckpt"


class ChnSentiDataset(Dataset):
    """把 DataFrame 缓存为 (ids, label) 的简单 Dataset；批内 pad 到等长。"""

    def __init__(self, df, tokenize_fn, pad_id=0, max_len=128):
        self.seqs = []
        self.labels = []
        for _, row in df.iterrows():
            ids = tokenize_fn(row["text"])
            self.seqs.append(ids)
            self.labels.append(int(row["label"]))
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def collate_fn(batch, pad_id=0):
    ids_list, labels = zip(*batch)
    max_t = max(len(ids) for ids in ids_list)
    padded = torch.full((len(ids_list), max_t), pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
    return padded, torch.tensor(labels, dtype=torch.long)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(device)
            logits = model(ids)
            pred = logits.argmax(dim=-1)
            correct += (pred.cpu() == labels).sum().item()
            total += len(labels)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3, help="early stop 轮数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(ROOT / "data" / "train.parquet")
    dev_df = pd.read_parquet(ROOT / "data" / "validation.parquet")
    print(f"train={len(train_df)}  dev={len(dev_df)}")

    vocab = build_char_vocab(train_df["text"])
    pad_id = vocab["[PAD]"]
    print(f"vocab size = {len(vocab)} (含特殊符)")
    tokenize_fn = make_tokenize_fn(vocab, args.max_len)

    train_ds = ChnSentiDataset(train_df, tokenize_fn, pad_id=pad_id)
    dev_ds = ChnSentiDataset(dev_df, tokenize_fn, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    device = torch.device(args.device)
    model = TransformerClassifier(
        vocab_size=len(vocab), d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, num_classes=args.num_classes,
        max_len=args.max_len, dropout=args.dropout, pad_id=pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params = {n_params / 1e6:.2f}M  device = {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(train_loader)
    criterion = nn.CrossEntropyLoss()

    # cosine schedule + warmup（注意：LambdaLR 会给 lambda 再乘初始 lr，
    # 所以这里只返回 0~1 的比例因子，而不是绝对学习率）
    def lr_at(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_at)

    best_acc, best_epoch, bad_epochs = 0.0, -1, 0
    log = {"args": vars(args), "train_loss": [], "dev_acc": [], "best": {}}
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)
            n_seen += len(labels)
            step += 1

        avg_loss = total_loss / max(n_seen, 1)
        acc = evaluate(model, dev_loader, device)
        log["train_loss"].append(round(avg_loss, 4))
        log["dev_acc"].append(round(acc, 4))
        print(f"epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"dev_acc={acc:.4f}  ({time.time() - t0:.0f}s)")

        if acc > best_acc:
            best_acc, best_epoch, bad_epochs = acc, epoch, 0
            torch.save(model.state_dict(), CKPT_DIR / "best.pt")
            print(f"  -> 新 best，保存 ckpt/best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop @ epoch {epoch}（{args.patience} 轮无提升）")
                break

    config = {"vocab_size": len(vocab), "d_model": args.d_model,
              "n_heads": args.n_heads, "n_layers": args.n_layers,
              "num_classes": args.num_classes, "max_len": args.max_len,
              "dropout": args.dropout, "pad_id": pad_id}
    (CKPT_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (CKPT_DIR / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    log["best"] = {"dev_acc": best_acc, "epoch": best_epoch}
    (CKPT_DIR / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n训练完成。best dev_acc = {best_acc:.4f} @ epoch {best_epoch}，"
          f"ckpt 保存在 {CKPT_DIR}")


if __name__ == "__main__":
    main()
