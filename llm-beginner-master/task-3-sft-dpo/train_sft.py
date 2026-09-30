"""任务三 SFT：在 Qwen2.5-0.5B 上手写 LoRA 做指令微调。

数据格式（JSONL，每行一个 object）：
  {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
只对 assistant 内容算 loss。产出的 LoRA 权重保存到 ckpt/sft/。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.lora import inject_lora, save_lora_weights
from src.chat import get_tokenizer, format_messages, build_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/sft.jsonl")
    ap.add_argument("--base-model", type=str, default="models/Qwen2.5-0.5B")
    ap.add_argument("--out", type=str, default="ckpt/sft")
    ap.add_argument("--target", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = get_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取 SFT 数据
    examples = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    print(f"[sft] loaded {len(examples)} examples")

    # 加载基座 + 注入 LoRA
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    target = [t.strip() for t in args.target.split(",") if t.strip()]
    inject_lora(model, target, r=args.r, alpha=args.alpha)
    model.to(device)  # inject_lora 新建的参数在 cpu，移回训练设备
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] target={target} r={args.r} alpha={args.alpha} "
          f"trainable={trainable} total={total} ratio={trainable/total:.4f}")
    model.train()

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    best_loss = float("inf")

    global_step = 0
    while global_step < args.max_steps:
        # 用全量数据反复 shuffle 训练
        import random
        random.shuffle(examples)
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            input_ids, labels, attn = _make_batch(tokenizer, batch, args.max_length, device)
            logits = model(input_ids, attention_mask=attn).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = loss / args.grad_accum
            loss.backward()
            global_step += 1
            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                print(f"  step {global_step}/{args.max_steps}  loss={loss.item()*args.grad_accum:.4f}")
                if loss.item() < best_loss:
                    best_loss = loss.item()
            if global_step >= args.max_steps:
                break

    # 保存 LoRA 权重
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_lora_weights(model, out)
    # 保存一份合并后的权重便于直接推理/对比
    from src.lora import merge_lora
    merged = merge_lora(model)
    torch.save(merged.state_dict(), str(out / "merged_state_dict.pt"))
    (out / "done.json").write_text(json.dumps({"best_loss": best_loss}), encoding="utf-8")
    print(f"[done] sft ckpt -> {out}")


def _make_batch(tokenizer, batch, max_length, device):
    input_ids_list, labels_list, mask_list = [], [], []
    for ex in batch:
        messages = ex["messages"]
        text = format_messages(messages)
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        labels = build_labels(ids, messages)
        if ids.size(0) > max_length:
            ids = ids[:max_length]
            labels = labels[:max_length]
        input_ids_list.append(ids)
        labels_list.append(labels)
        mask_list.append(torch.ones_like(ids))
    max_len = max(x.size(0) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(mask_list), max_len), dtype=torch.long)
    for i, (ids, lab, m) in enumerate(zip(input_ids_list, labels_list, mask_list)):
        input_ids[i, :ids.size(0)] = ids
        labels[i, :lab.size(0)] = lab
        attn[i, :m.size(0)] = 1
    return input_ids.to(device), labels.to(device), attn.to(device)


if __name__ == "__main__":
    main()
