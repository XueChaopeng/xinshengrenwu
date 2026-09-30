"""任务一 M5：注意力热图可视化。

用法（训练出 ckpt/best.pt 后，在 task-1-transformer/ 目录下执行）：
    python visualize_attn.py

在验证集中挑 3 个样本（正面 / 负面 / 长句），画出指定层、指定头的
(T, T) 注意力权重热图，存到 figures/。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
CKPT = ROOT / "ckpt" / "best.pt"

# 中文字体：Windows 常见字体按顺序尝试
import matplotlib.font_manager as fm

_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/simsun.ttc",   # 宋体
]
for path in _CANDIDATES:
    if Path(path).exists():
        try:
            fm.fontManager.addfont(path)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def pick_samples(dev_df, n=1):
    """从 dev 里挑：正面、负面、最长句样本。返回 (text, label, tag)。"""
    pos = dev_df[dev_df["label"] == 1].sort_values("text", key=lambda s: s.str.len())
    neg = dev_df[dev_df["label"] == 0].sort_values("text", key=lambda s: s.str.len())
    longest = dev_df.loc[dev_df["text"].str.len().idxmax()]
    return [
        (pos.iloc[0]["text"], 1, "pos"),
        (neg.iloc[0]["text"], 0, "neg"),
        (longest["text"], int(longest["label"]), "long"),
    ]


def render(samples, layer=0, head=0):
    model, tokenize_fn = load_for_eval(str(CKPT))
    model.eval()

    for text, label, tag in samples:
        ids = tokenize_fn(text)
        with torch.no_grad():
            logits, weights = model(ids.unsqueeze(0), need_weights=True)
        pred = int(logits.argmax(-1).item())
        W = weights[layer][0, head]  # (T, T)
        T = W.size(0)

        # token 标签：ids[0]=[CLS]，其后逐字
        vocab = json.loads((CKPT.parent / "vocab.json").read_text(encoding="utf-8"))
        rev = {v: k for k, v in vocab.items()}
        labels = [rev.get(int(i), "?") if len(rev) == len(vocab)
                  else str(i) for i in ids.tolist()]

        fig, ax = plt.subplots(figsize=(max(6, T * 0.35), max(5, T * 0.35)))
        im = ax.imshow(W.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Key 位置")
        ax.set_ylabel("Query 位置")
        ax.set_title(
            f"Layer {layer + 1} / Head {head + 1}  标签={label}  预测={pred}\n"
            f"{text[:40]}{'…' if len(text) > 40 else ''}"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        FIG_DIR.mkdir(exist_ok=True)
        out = FIG_DIR / f"attn_{tag}_L{layer + 1}H{head + 1}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存 {out.name}  (T={T}, 标签={label}, 预测={pred})")


def main():
    dev_df = pd.read_parquet(ROOT / "data" / "validation.parquet")
    samples = pick_samples(dev_df)
    # 6 张图：正/负/长句 × 不同 layer/head，凑 ≥3 张
    render(samples, layer=0, head=0)
    render(samples, layer=0, head=1)
    render(samples, layer=3, head=2)
    print("完成。figure 列表：")
    for p in sorted(FIG_DIR.glob("attn_*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
