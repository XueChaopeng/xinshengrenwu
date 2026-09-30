"""任务一：TransformerClassifier —— 堆 N 层 encoder block 的中文情感分类器。

self-check 契约（README「实现约定」）：
    src/model.py 必须导出
      - class TransformerClassifier
      - load_for_eval(ckpt_path: str) -> (model, tokenize_fn)
    其中 tokenize_fn(text) -> LongTensor (T,)，model(ids (B,T)) -> logits (B, num_classes)。

分词方案：中文字符级 + [PAD]/[UNK]/[CLS]，无任何预训练 / 外部词表依赖。
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.block import TransformerBlock

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"


def build_char_vocab(texts, max_vocab=20000):
    """从文本列表构建字符级词表。texts 为空时返回最小词表（只含特殊符）。"""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, CLS_TOKEN: 2}
    counter = {}
    for t in texts:
        for ch in str(t):
            counter[ch] = counter.get(ch, 0) + 1
    # 按频次降序（保证稳定顺序），低频字符可丢弃
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    for ch, _ in ordered:
        if ch not in vocab:
            vocab[ch] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def make_tokenize_fn(vocab, max_len):
    """返回 tokenize_fn(text: str) -> LongTensor (T,)；文本超长截断、前补 [CLS]。"""
    pad_id, unk_id, cls_id = vocab[PAD_TOKEN], vocab[UNK_TOKEN], vocab[CLS_TOKEN]

    def tokenize_fn(text: str) -> torch.LongTensor:
        # 中文字符级 + 截断（给 [CLS] 留一个位置）
        chars = [cls_id] + [vocab.get(ch, unk_id) for ch in str(text)]
        chars = chars[:max_len]
        return torch.tensor(chars, dtype=torch.long)

    return tokenize_fn


class TransformerClassifier(nn.Module):
    """字符级 Transformer encoder 情感分类器（手写，无高层封装）。"""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 num_classes=2, max_len=128, dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.num_classes = num_classes

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)  # 可学习绝对位置编码

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids, need_weights=False):
        """ids: (B, T) -> logits: (B, num_classes)。自建 padding mask 屏蔽 PAD。
        need_weights=True 时额外返回 (logits, weights_list)，每层 (B,H,T,T)。"""
        B, T = ids.shape
        # padding mask：True = 屏蔽（PAD 位置不参与 attention）
        pad_mask = (ids == self.pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

        x = self.tok_emb(ids)  # (B,T,d_model)
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        weights_list = []
        for block in self.blocks:
            if need_weights:
                x = block(x, mask=pad_mask, need_weights=True)
                weights_list.append(block.last_weights)
            else:
                x = block(x, mask=pad_mask)

        # 取 [CLS]（恒为位置 0）的表示做分类
        cls_vec = x[:, 0]           # (B, d_model)
        logits = self.head(self.norm(cls_vec))
        if need_weights:
            return logits, weights_list
        return logits


def load_for_eval(ckpt_path: str):
    """从 ckpt/best.pt（及其同目录 config.json / vocab.json）重建模型与 tokenize_fn。"""
    ckpt_path = Path(ckpt_path)
    ckpt_dir = ckpt_path.parent

    config_path = ckpt_dir / "config.json"
    vocab_path = ckpt_dir / "vocab.json"
    if config_path.exists() and vocab_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        max_len = config["max_len"]
    else:
        # 兜底：单独一个 best.pt 文件（仅 state_dict）无法重建词表，
        # 此时要求 ckpt_path 是完整包（dict），兼容手动保存的 checkpoint。
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and "vocab" in state:
            model = TransformerClassifier(**state["model"])
            model.load_state_dict(state["model_state"])
            vocab = state["vocab"]
            max_len = state.get("max_len", model.max_len)
            model.eval()
            return model, make_tokenize_fn(vocab, max_len)
        raise FileNotFoundError(
            f"{ckpt_dir}/config.json 与 vocab.json 缺失，无法仅凭 state_dict 重建模型。"
            "请用 train.py 训练以生成完整的 ckpt/ 目录。"
        )

    keys = ("d_model", "n_heads", "n_layers", "num_classes", "max_len",
            "dropout", "pad_id")
    model = TransformerClassifier(vocab_size=len(vocab),
                                  **{k: config[k] for k in keys if k in config})
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, make_tokenize_fn(vocab, max_len)
