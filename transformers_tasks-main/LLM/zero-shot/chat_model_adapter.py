# !/usr/bin/env python3
"""
[added] ChatGLM-compatible chat adapter.

The zero-shot demos in this folder call the ChatGLM-specific API

    response, history = model.chat(tokenizer, query, history=pre_history)

Running ChatGLM-6B itself is not possible on this machine (see the notes below),
so this module exposes the very same `.chat()` interface on top of any
instruction-tuned HuggingFace causal LM, letting the demo scripts stay unchanged.

Why not ChatGLM-6B here?
    * fp16 ChatGLM-6B needs ~12 GB of RAM/VRAM; this box has a 4 GB GPU and
      ~8 GB of free system RAM.
    * the int4 checkpoint (`THUDM/chatglm-6b-int4`) needs the cpm_kernels
      CPU/CUDA quantization kernels; they compile with gcc, which is not
      available on Windows, and a pure-torch dequantization of 6B parameters
      per generated token is far too slow (~minutes per reply).

Usage:
    LLM_ADAPTER=hf_chat CHATGLM_PATH=/path/to/instruct-model python llm_classification.py
"""
from typing import List, Optional, Tuple

import torch


class HFChatModel:
    """Wraps a causal LM so that it looks like a ChatGLM model to the demos."""

    def __init__(self, model, tokenizer, device: str = 'cpu', max_new_tokens: int = 96):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    # --- API used by the demo scripts -------------------------------------
    def to(self, device):
        self.device = device if isinstance(device, str) else str(device)
        self.model.to(self.device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def half(self):
        self.model.half()
        return self

    def float(self):
        self.model.float()
        return self

    def parameters(self):
        return self.model.parameters()

    def chat(
        self,
        tokenizer=None,
        query: str = '',
        history: Optional[List[Tuple[str, str]]] = None,
        **kwargs,
    ):
        messages = []
        for turn in (history or []):
            user, assistant = turn[0], turn[1]
            messages.append({'role': 'user', 'content': user})
            messages.append({'role': 'assistant', 'content': assistant})
        messages.append({'role': 'user', 'content': query})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        new_history = list(history or []) + [(query, response)]
        return response, new_history


def build_chat_model(model_path: str, device: str = 'cpu'):
    """Load `model_path` and return (tokenizer, HFChatModel)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    )
    if device.startswith('cuda'):
        model = model.half()
    else:
        model = model.float()
    model.to(device).eval()
    return tokenizer, HFChatModel(model, tokenizer, device=device)
