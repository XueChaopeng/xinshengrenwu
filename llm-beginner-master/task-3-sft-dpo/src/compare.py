"""在同一批指令上对比 base / SFT / DPO 三个模型的输出，验证 SFT/DPO 的训练效果。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM

from src.lora import inject_lora, load_lora_weights
from src.chat import get_tokenizer, format_messages
BASE = ROOT / "models" / "Qwen2.5-0.5B"
SFT = ROOT / "ckpt" / "sft"
DPO = ROOT / "ckpt" / "dpo"
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
R = 16
ALPHA = 32.0

PROMPTS = [
    "请介绍一下深度学习",
    "用 Python 写一个冒泡排序",
    "床前明月光 的下一句是什么？",
    "什么是过拟合？",
    "什么是 RoPE 位置编码？",
]


def generate(model, tokenizer, prompt, max_new=64):
    messages = [{"role": "user", "content": prompt}]
    # prompt 模板（不含 assistant 回复）
    prompt_text = format_messages(messages) + "<|im_start|>assistant\n"
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    generated = out[0][ids.size(1):]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    tokenizer = get_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16).to(device)
    base.eval()

    sft_model = None
    dpo_model = None
    if SFT.exists():
        sft_model = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16).to(device)
        inject_lora(sft_model, TARGET, r=R, alpha=ALPHA)
        sft_model.to(device)
        load_lora_weights(sft_model, SFT)
        sft_model.eval()
        print("[load] SFT LoRA 已加载")
    if DPO.exists():
        dpo_model = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16).to(device)
        inject_lora(dpo_model, TARGET, r=R, alpha=ALPHA)
        dpo_model.to(device)
        load_lora_weights(dpo_model, DPO)
        dpo_model.eval()
        print("[load] DPO LoRA 已加载")

    for p in PROMPTS:
        print("\n" + "=" * 60)
        print(f"指令：{p}")
        print("-" * 60)
        print(f"[base] {generate(base, tokenizer, p)}")
        if sft_model is not None:
            print(f"[sft]  {generate(sft_model, tokenizer, p)}")
        if dpo_model is not None:
            print(f"[dpo]  {generate(dpo_model, tokenizer, p)}")


if __name__ == "__main__":
    main()
