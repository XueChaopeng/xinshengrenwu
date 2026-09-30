"""Qwen2.5 chat 模板与 loss masking。

模板（与 Qwen 官方 chat_template 一致）：
  <|im_start|>{role}\n{content}<|im_end|>\n
只对 assistant 回复内容计算 loss：user / system / 模板控制符 打 -100。
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Qwen2.5-0.5B 在本任务的路径；build_labels 用它恢复 tokenizer 以做逐 token 对齐。
_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"


@functools.lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(_MODEL_DIR))


def format_messages(messages: List[Dict[str, str]]) -> str:
    """把对话列表套成 Qwen chat 模板字符串。"""
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )


def _assistant_content_ranges(messages: List[Dict[str, str]]) -> List[Tuple[int, int]]:
    """返回所有 assistant 回复内容在模板字符串中的字符区间 [start, end)。"""
    ranges: List[Tuple[int, int]] = []
    cur = 0
    for m in messages:
        role = m["role"]
        content = m["content"]
        header = f"<|im_start|>{role}\n"
        footer = "<|im_end|>\n"
        cur += len(header)
        content_start = cur
        cur += len(content)
        content_end = cur
        cur += len(footer)
        if role == "assistant":
            ranges.append((content_start, content_end))
    return ranges


def build_labels(input_ids, messages: List[Dict[str, str]]):
    """根据对话结构给 input_ids 生成 labels（与 input_ids 同形状）。

    - 非 assistant 内容（system/user/模板控制符/role 名）置 -100。
    - assistant 内容位置的 label 设为该位置的 token id（训练时配合 shift 做 next-token 预测）。
    """
    tok = get_tokenizer()
    text = format_messages(messages)
    enc = tok(text, return_offsets_mapping=True)
    offsets = enc.offset_mapping

    if torch.is_tensor(input_ids):
        labels = input_ids.clone()
    else:
        labels = torch.tensor(list(input_ids), dtype=torch.long)
    labels = labels.clone()
    labels.fill_(-100)

    ranges = _assistant_content_ranges(messages)
    for i, (s, e) in enumerate(offsets):
        if i >= labels.size(0):
            break
        if s == 0 and e == 0:      # 特殊 token（im_start/im_end 等）无实义偏移，统一 -100
            continue
        for (cs, ce) in ranges:
            if s >= cs and e <= ce:
                labels[i] = int(input_ids[i])
                break
    return labels


def make_sft_batch(messages_list: List[List[Dict[str, str]]], max_length: int = 512):
    """把一批多轮对话编码成 (input_ids, labels, attention_mask)。

    - input_ids：chat 模板 token + 末尾追加一个 assistant 开头让模型续写。
    - labels：只保留 assistant 内容（经 shift 语义在 train_sft 里处理）。
    用于 train_sft.py。
    """
    tok = get_tokenizer()
    input_ids_list, labels_list, mask_list = [], [], []
    for messages in messages_list:
        text = format_messages(messages)
        ids = tok(text, return_tensors="pt").input_ids[0]
        labels = build_labels(ids, messages)
        if ids.size(0) > max_length:
            ids = ids[:max_length]
            labels = labels[:max_length]
        input_ids_list.append(ids)
        labels_list.append(labels)
        mask_list.append(torch.ones_like(ids))
    # pad
    max_len = max(x.size(0) for x in input_ids_list)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(mask_list), max_len), dtype=torch.long)
    for i, (ids, lab, m) in enumerate(zip(input_ids_list, labels_list, mask_list)):
        input_ids[i, :ids.size(0)] = ids
        labels[i, :lab.size(0)] = lab
        attn[i, :m.size(0)] = 1
    return input_ids, labels, attn
