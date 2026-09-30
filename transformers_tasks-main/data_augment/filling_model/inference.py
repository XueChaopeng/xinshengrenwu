# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

测试训练好的模型效果。

Author: pankeyu
Date: 2023/01/05
"""
from rich import print
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = 'cuda:0'
max_source_seq_len = 128
max_target_seq_len = 32                                        # [patched] bounded generation length
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/t5/model_best/')
model = T5ForConditionalGeneration.from_pretrained('./checkpoints/t5/model_best/')
# [patched] training ends each target with `tokenizer.eos_token` (= [SEP], id 102)
# while the config keeps the original T5 eos id; align them so generation stops
# at [SEP] instead of running to max_new_tokens.
model.config.eos_token_id = tokenizer.eos_token_id
model.to(device).eval()


def inference(masked_texts: list):
    """
    inference函数。

    Args:
        masked_texts (list): 掩码后的文字列表
    """
    inputs = tokenizer(
        text=masked_texts,
        truncation=True,
        max_length=max_source_seq_len,
        padding='max_length',
        return_tensors='pt'
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=max_target_seq_len,
    )
    outputs = [tokenizer.decode(output.cpu().numpy(), skip_special_tokens=True).replace(" ", "") \
                    for output in outputs]
    print(f'maksed text: {masked_texts}')
    print(f'output: {outputs}')


if __name__ == '__main__':
    masked_texts = [
        '"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：'
    ]
    inference(masked_texts)