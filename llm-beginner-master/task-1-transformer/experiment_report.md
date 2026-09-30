# 任务一 · 实验观察报告

## 实验观察（正文）

本任务从零手写 scaled dot-product attention、多头注意力与 Pre-LN Transformer encoder block，全程未调用 `nn.MultiheadAttention`、`F.scaled_dot_product_attention` 或任何预训练权重。自检中手写 attention 与 torch 官方实现最大误差 7.2e-7，causal mask 下未来词元泄漏为 0，数值正确性有保证。默认配置（d=128, h=4, L=4, 1.35M 参数）在 ChnSentiCorp 上约 70 秒训满 8 轮，best dev acc = 0.8467，接近 0.85 基线；loss 从 0.70 平稳降至 0.14，第 5 轮后验证集进入平台期并 early stop，未见明显过拟合。

S1 消融（同参数量、同 8 轮训练）：h2/l4 = 0.8483、h8/l4 = 0.8508 与默认 h4/l4 = 0.8467 几乎无差异，砍到 2 层则降到 0.8433 且收敛更慢——该任务对 head 数不敏感、对深度略敏感。S2 结构性消融（2 层小模型、3000 条子集短训 3 轮）：去掉 residual 或 LayerNorm 仍能正常收敛（dev 0.700 / 0.723 / 0.748），浅层模型对二者依赖较弱，深层与长训练下的影响值得继续验证。

注意力可视化共出 9 张热图（figures/，正/负/长句 × L1H1、L1H2、L4H3）。配合词元级统计：浅层 head 的高权重集中在"的/不/了"等高频功能字，深层 head 更偏向内容词。自动挑中的一条负面样本被模型误判为正面（标签 0 → 预测 1），其深层 head 关注最多的词全落在外观/语气类词上，未形成负面词聚集——是一次典型的"注意力被表面措辞带偏"的可视化案例。

toy LM 复用同一套 attention 加 causal mask，在唐诗语料上做字符级 next-token 预测：600 步 loss 从 6.85 降到 1.96（随机基线 ≈ 6.68），采样产出"床前明月光，但见汉三[UNK]香泣"——已具备基本诗句韵律，为任务二预热 decoder-only 训练管线成功。

## 附录 A：S1 head/层数消融（d_model=128，epochs=8，early stop）

| 配置 | 参数量 | best dev acc | 达到轮次 |
|---|---|---|---|
| h=4, L=4（默认） | 1.35M | **0.8467** | epoch 5 |
| h=2, L=4 | 1.35M | 0.8483 | epoch 6 |
| h=8, L=4 | 1.35M | **0.8508** | epoch 7 |
| h=4, L=2 | 0.96M | 0.8433 | epoch 8 |

## 附录 B：S2 拆 residual / LayerNorm（d=64, h=4, L=2，3 轮，训练子集 3000 条）

| 变体 | dev acc（每轮） |
|---|---|
| 正常 | 0.524 → 0.648 → 0.700 |
| 去 residual | 0.595 → 0.689 → 0.723 |
| 去 LayerNorm | 0.530 → 0.723 → 0.748 |

结论：浅层短训练下两种结构都仍收敛（未发散），说明该设定下对 residual / LayerNorm 依赖不强；深层 / 长训练的消融留待后续。

## 附录 C：最终自检 eval/result.json

- `attention_correctness`：通过（max_abs_diff = 7.15e-07 < 1e-5）
- `causal_mask`：通过（leaked_diff = 0.0 < 1e-6）
- `classifier_accuracy`：通过（accuracy = 0.8467 ≥ 0.80；baseline 参考 0.85）

## 附录 D：产物清单

| 产物 | 位置 |
|---|---|
| 手写实现 | `src/attention.py`、`src/block.py`、`src/model.py` |
| 训练好的模型 | `ckpt/best.pt`（+ config/vocab/train_log） |
| 注意力热图 ×9 | `figures/attn_{pos,neg,long}_L{1,1,4}H{1,2,3}.png` |
| toy LM 采样 | `toy_lm_sample.txt` |
| 自检结果 | `eval/result.json` |
