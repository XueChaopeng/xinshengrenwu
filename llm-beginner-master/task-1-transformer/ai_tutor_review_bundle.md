# AI Tutor Prompt · 任务一 熟悉 Transformer

把下面整段贴给 Claude / Qwen / DeepSeek 等大模型，连同你的代码一起，让它给针对性反馈。

---

## 角色设定

你是一位严格但耐心的深度学习课程助教，正在为学生 review《LLM-Beginner 大模型与智能体入门练习》任务一（熟悉 Transformer）的代码。请做一次系统的代码审查。

## 任务上下文

任务一要求学生从零手写：

1. Scaled dot-product attention（含 mask 处理）
2. Multi-head attention
3. 完整 Transformer encoder block（attention + FFN + residual + LayerNorm）
4. 用 padding mask 跑文本分类（ChnSentiCorp 中文情感分类）
5. 用 causal mask 跑 toy 语言模型预热
6. 注意力可视化

教学目的：让学生**真正理解** Transformer 内部机制，不允许调 `nn.MultiheadAttention` 之类的封装。

## 评审检查项

### 必检项（任一不过关都要指出并给出修复建议）

1. **scaled dot-product attention 的数学正确性**
   - softmax 是否对正确的维度（最后一维 K_seq）做？
   - 缩放因子是否是 `sqrt(d_k)` 而不是 `sqrt(d_model)`？
   - mask 处理：是否用 `-inf`（或 `-1e9`）填充被屏蔽位置，而不是简单乘 0？
2. **multi-head 的 reshape 顺序**
   - `(B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H)` 的 transpose 顺序是否对？
   - 输出 reshape 回去时是否调用了 `.contiguous()`？
3. **Padding mask vs causal mask**
   - padding mask 形状是否能广播到 `(B, 1, 1, T)` 或等价？
   - causal mask 是否上三角全 `-inf`？
4. **Residual + LayerNorm**
   - 是 Pre-LN 还是 Post-LN？两种都对，但要写得明确一致
   - residual 加在 LayerNorm 之前还是之后？
5. **注意力可视化（对应 M5，硬性交付）**
   - 是否产出 ≥ 3 张注意力热图（建议正面 / 负面 / 长句样本各一）？
   - 热图是否标注词元轴、选定了具体 layer / head？
   - 是否在正 / 负样本上对比，说明模型关注了哪些关键词？

### 加分项（指出改进空间即可，不算 fail）

1. 是否区分了 Q/K/V 三个投影矩阵（学生有时偷懒只用一个）
2. FFN 是否是两层 + 中间激活（通常 hidden = 4 * d_model）
3. 更深入的可视化分析（多头 / 多层对比、定量解读注意力分布）
4. 训练循环是否有 gradient clipping、warmup 等基础工程

## 输出格式

按以下结构返回反馈：

```
## 概览
（1-2 句总评：实现整体水平、关键问题数量）

## 必检项

### [项目名]
- 状态：通过 / 需要修复
- 现状：[引用学生代码片段]
- 问题：[具体说明]
- 修复建议：[代码片段]

（重复上面结构，覆盖所有必检项）

## 加分项观察
（按项简短指出，1-2 行/项）

## 优先级排序
（按修复重要性给出 3-5 条 actionable item）
```

---
## 我的代码

### src/attention.py

`python
"""任务一：手写 scaled dot-product attention 与 Multi-Head Attention。

接口约定（见 task-1-transformer/README.md「实现约定」）：
    scaled_dot_product_attention(Q, K, V, mask=None)
        Q/K/V 形状 (B, H, T, D)；mask 形状可广播到 (B, H, T, T)，True = 被屏蔽。
    返回形状 (B, H, T, D)。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, need_weights=False):
    """Scaled dot-product attention（手写版）。

    公式：Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + mask) V
    其中 mask 在 logits 上生效：True 的位置填 -inf（而不是乘 0，否则 softmax
    后仍有概率泄漏），随后 softmax 概率为 0。

    need_weights=True 时返回 (out, attn_weights)，供可视化使用；默认只返回 out。
    """
    # Q, K, V: (B, H, T, D) -> logits: (B, H, T_q, T_k)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 乘 0 的坑：softmax(0)=exp(0)/Z 仍非零，必须填 -inf 才能让概率严格为 0
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V)
    if need_weights:
        return out, attn_weights
    return out


class MultiHeadAttention(nn.Module):
    """手写多头自注意力：Q/K/V 各自投影 -> 切分成 H 个头 -> 缩放点积 -> 拼接 -> 输出投影。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q/K/V 三套独立投影（不偷懒共用一套）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._last_weights = None  # 便于可视化时取出 (B, H, T, T) 注意力权重

    def forward(self, x, mask=None, need_weights=False):
        """x: (B, T, d_model)；mask: 形状可广播到 (B, H, T_q, T_k)，True=屏蔽。
        need_weights=True 时额外返回 (out, attn_weights (B,H,T,T))。"""
        B, T, _ = x.shape

        # 1. 投影
        Q = self.w_q(x)  # (B, T, d_model)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. 切分 head：(B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # 3. 缩放点积注意力（不调用 nn.MultiheadAttention / F.scaled_dot_product_attention）
        attn_out, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, need_weights=True)  # (B, H, T, d_k)

        # 4. 合并 head 并做输出投影；transpose 后必须 .contiguous() 再 view
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(attn_out)
        if need_weights:
            return out, attn_weights
        return out
`

### src/block.py

`python
"""任务一：Transformer encoder block = multi-head attention + FFN + residual + LayerNorm。

采用 Pre-LN 结构（先 LayerNorm 再进子层，residual 加在子层输出上），
比 Post-LN 训练更稳定，是 GPT/BERT 之后的主流做法。
"""
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """单个 Transformer encoder block（自注意力 + 前馈 + 两个残差 + 两个 LayerNorm）。"""

    def __init__(self, d_model, n_heads, dropout=0.1, expansion=4):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, need_weights=False):
        """x: (B, T, d_model)；mask 同 MultiHeadAttention（True=屏蔽）。
        need_weights=True 时把本层注意力权重 (B,H,T,T) 存到 self.last_weights。"""
        # Pre-LN：x + Attn(LN(x))
        if need_weights:
            attn_out, w = self.attn(self.ln1(x), mask=mask, need_weights=True)
            self.last_weights = w
            x = x + self.dropout(attn_out)
        else:
            x = x + self.dropout(self.attn(self.ln1(x), mask=mask))
        # x + FFN(LN(x))
        x = x + self.ffn(self.ln2(x))
        return x
`

### src/model.py

`python
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
`