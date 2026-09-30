"""任务三 DPO：在 SFT 之上做偏好对齐。

数据格式（JSONL）：{"prompt": "...", "chosen": "...", "rejected": "..."}
policy 从 ckpt/sft 的 LoRA 权重初始化；reference 为冻结的 SFT 模型（只 forward）。
DPO loss = -log σ(β * ((log π_pol(chosen)-log π_ref(chosen)) - (log π_pol(rej)-log π_ref(rej))))
产出的 LoRA 权重保存到 ckpt/dpo/。
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

from src.lora import inject_lora, save_lora_weights, load_lora_weights
from src.chat import get_tokenizer, format_messages, build_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/dpo.jsonl")
    ap.add_argument("--base-model", type=str, default="models/Qwen2.5-0.5B")
    ap.add_argument("--sft-ckpt", type=str, default="ckpt/sft")
    ap.add_argument("--out", type=str, default="ckpt/dpo")
    ap.add_argument("--target", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = get_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = [t.strip() for t in args.target.split(",") if t.strip()]

    # 读取 DPO 数据
    examples = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    print(f"[dpo] loaded {len(examples)} preference pairs")

    # policy（trainable）与 reference（frozen）都从 base + SFT adapter 初始化
    policy = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
    inject_lora(policy, target, r=args.r, alpha=args.alpha)
    policy.to(device)
    load_lora_weights(policy, args.sft_ckpt)
    reference = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
    inject_lora(reference, target, r=args.r, alpha=args.alpha)
    reference.to(device)
    load_lora_weights(reference, args.sft_ckpt)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()
    policy.train()
    policy.config.use_cache = False

    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    global_step = 0
    while global_step < args.max_steps:
        import random
        random.shuffle(examples)
        # 打平为 (batch) 的 (chosen, rejected) 对
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            chosen_ids, chosen_labels = _encode_pair(tokenizer, batch, "chosen", args.max_length, device)
            rejected_ids, rejected_labels = _encode_pair(tokenizer, batch, "rejected", args.max_length, device)

            pi_chosen = _logprob(policy, chosen_ids, chosen_labels)
            pi_rejected = _logprob(policy, rejected_ids, rejected_labels)
            with torch.no_grad():
                ref_chosen = _logprob(reference, chosen_ids, chosen_labels)
                ref_rejected = _logprob(reference, rejected_ids, rejected_labels)
            ratio = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
            loss = -F.logsigmoid(args.beta * ratio).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            global_step += 1
            if global_step % 10 == 0:
                margin = (pi_chosen - pi_rejected).mean().item()
                print(f"  step {global_step}/{args.max_steps}  loss={loss.item():.4f}  "
                      f"reward_margin={margin:.4f}")
            if global_step >= args.max_steps:
                break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_lora_weights(policy, out)
    (out / "done.json").write_text(json.dumps({"beta": args.beta}), encoding="utf-8")
    print(f"[done] dpo ckpt -> {out}")


def _encode_pair(tokenizer, batch, key, max_length, device):
    ids_list, labels_list = [], []
    for ex in batch:
        prompt = ex["prompt"]
        resp = ex[key]
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]
        text = format_messages(messages)
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        labels = build_labels(ids, messages)
        if ids.size(0) > max_length:
            ids = ids[:max_length]
            labels = labels[:max_length]
        ids_list.append(ids)
        labels_list.append(labels)
    max_len = max(x.size(0) for x in ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
        input_ids[i, :ids.size(0)] = ids
        labels[i, :lab.size(0)] = lab
    return input_ids.to(device), labels.to(device)


def _logprob(model, input_ids, labels):
    """计算每个样本在目标 token（assistant 内容）上的对数概率之和。

    注意：不要加 @torch.no_grad()。policy 需要在图上做反向；reference 的调用由调用方包在 no_grad 里。
    """
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    shift_logp = log_probs[:, :-1].gather(-1, labels[:, 1:].clamp(min=0).unsqueeze(-1)).squeeze(-1)
    valid = (labels[:, 1:] != -100).float()
    total = (shift_logp * valid).sum(dim=1)
    return total


if __name__ == "__main__":
    main()
