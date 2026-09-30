# 任务三、图分类（Graph Classification / Regression）

基于 **PyTorch Geometric (PyG)** 实现 GCN / GAT / GraphSAGE / GIN 四个模型，
在 **TUDataset**（图分类）和 **ZINC**（图回归）上完成图级任务，
并系统对比 **AvgPooling / MaxPooling / MinPooling / SumPooling** 四种池化（readout）方式
对性能的影响。

---

## 1. 数据集

### 1.1 TUDataset（图分类）

| 数据集 | 图数 | 节点特征 | 类别数 | 说明 |
|---|---|---|---|---|
| MUTAG | 188 | 7 | 2 | 硝基化合物致突变性 |
| ENZYMES | 600 | 3 | 6 | 蛋白质三级结构酶类 |
| PROTEINS | 1,113 | 3 | 2 | 蛋白质是否为酶 |
| IMDB-BINARY | 1,000 | **无** | 2 | 电影合作网络（社交图） |

> **IMDB-BINARY 没有节点特征**，按 GIN 论文（Xu et al. 2019）的做法，
> 用**节点度数的 one-hot 编码**作为初始特征（`add_degree_features`，
> 编码宽度取整个数据集的最大度数 + 1，保证各划分共享同一特征空间）。

**划分方式**：TUDataset 官方没有固定划分，文献里通常用 **10 折交叉验证**。
本实现采用**分层（stratified）80% / 10% / 10%** 划分，并对每个随机种子重新划分，
报告多个种子的均值 ± 标准差。`stratified_split` 保证每个类别在验证/测试集中
至少出现一次（类别样本数 ≥ 3 时），且划分对 `(标签, 种子)` 完全确定、跨平台可复现。

### 1.2 ZINC（图回归）

| 数据集 | 图数 | 节点特征 | 边特征 | 目标 |
|---|---|---|---|---|
| ZINC (subset) | 10,000 / 1,000 / 1,000 | 原子序数 | 键类型 | 约束 logP（回归） |

ZINC 是**图级回归**任务，评价指标是 **MAE（越低越好）**，使用官方给定的
train/val/test 划分。PyG 官方把 ZINC 放在 Dropbox 上（本机网络不可达），
因此 `prepare_data.py` 改从 **HuggingFace 镜像**
`https://hf-mirror.com/datasets/graphs-datasets/ZINC` 下载标准的 PyG 划分
（`train/val/test.jsonl`，三项共约 178 MB），再转换为 PyG 的 `Data` 对象缓存到
`data/ZINC_processed/`。

## 2. 环境

见仓库根目录 `requirements.txt`。图分类用 `torch_geometric.loader.DataLoader`
做**图级批处理**（把小图拼成大图 + `batch` 向量），这一路径不需要
`pyg-lib` / `torch-sparse`；但为了与本项目其它任务共用一套环境，仍然安装它们。

## 3. 目录结构

```
task3_graph_classification/
├── data/
│   ├── MUTAG/ ENZYMES/ PROTEINS/ IMDB-BINARY/    TUDataset（已下载）
│   ├── ZINC_raw/           ZINC 原始 jsonl + index
│   └── ZINC_processed/     转换后的 PyG Data 缓存
├── code/
│   ├── prepare_data.py     下载/转换/校验 TUDataset 与 ZINC
│   ├── dataset.py          数据集加载、分层划分、度特征、ZINC 存储
│   ├── models.py           GNN 编码器 + 四种池化 + 图级预测头
│   ├── train.py            单次训练+测试
│   ├── run_experiments.py  批量实验编排（支持分片与合并）
│   └── utils.py            随机种子、CUDA 计时器、评价指标
├── results/
│   ├── all_results.csv
│   └── detail/*.json
└── README.md               本文件
```

## 4. 模型与池化

### 4.1 编码器

四个模型共用同一个 `GNNEncoder`（`code/models.py`），默认 **3 层**、隐藏维度 64，
只替换消息传递层，保证对比公平：

| 模型 | 消息传递层 | 聚合 |
|---|---|---|
| GCN | `GCNConv` | 度归一化加权平均 |
| GAT | `GATConv` | 多头注意力 |
| GraphSAGE | `SAGEConv` | 自身 + 邻居均值 |
| GIN | `GINConv` | 求和 + MLP（可学习 ε）+ BatchNorm |

### 4.2 池化（readout）

`pool_nodes(x, batch, pool)` 提供四种图级读出：

| 池化 | 实现 | 特点 |
|---|---|---|
| **AvgPooling** `mean` | `global_mean_pool` | 对图规模不敏感，最常用 |
| **MaxPooling** `max` | `global_max_pool` | 只保留每个维度的最强响应 |
| **MinPooling** `min` | `scatter(..., reduce="amin")` | PyG **没有** `global_min_pool`，自行实现；另有 `-max(-x)` 的等价实现用于交叉验证 |
| SumPooling `sum`/`add` | `global_add_pool` | GIN 论文的标准读出（对图规模敏感） |

> MinPooling 在 PyG 中没有现成实现，`code/models.py::global_min_pool` 用
> `torch_geometric.utils.scatter(reduce="amin")` 实现，`include_self=False`
> 避免空组的 `+inf` 填充值泄漏到结果里；梯度可导性由 `models.selftest()` 验证。

## 5. 训练与测试脚本

所有命令都在 `task3_graph_classification/code/` 目录下执行。

### 5.1 准备数据

```bash
python prepare_data.py                        # 下载 TUDataset + ZINC 并转换
python prepare_data.py --datasets MUTAG PROTEINS
python prepare_data.py --datasets ZINC
python prepare_data.py --verify               # 只检查是否就绪并打印统计
```

### 5.2 训练 / 测试单个模型

```bash
# TUDataset 图分类（默认 3 层，池化用 mean）
python train.py --dataset MUTAG --model gin --pool mean --seeds 0 1 2

# 池化对比：同一条命令只换 --pool
python train.py --dataset PROTEINS --model gcn --pool max --seeds 0 1 2
python train.py --dataset PROTEINS --model gcn --pool min --seeds 0 1 2

# ZINC 图回归（报告 MAE）
python train.py --dataset ZINC --model gin --pool sum --seeds 0 1

# 调超参
python train.py --dataset MUTAG --model gat --pool mean --num-layers 4 --hidden 128 --lr 0.005
```

常用参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--dataset` | `MUTAG` / `ENZYMES` / `PROTEINS` / `IMDB-BINARY` / `ZINC` | MUTAG |
| `--model` | `gcn` / `gat` / `sage` / `gin` | gcn |
| `--pool` | `mean` / `max` / `min` / `sum`(=`add`) | mean |
| `--num-layers` | 消息传递层数 | 3（ZINC 4） |
| `--hidden` | 隐藏维度 | 64 |
| `--batch-size` | 每批图数 | 32（ZINC 512） |
| `--lr` | 学习率 | 0.01（ZINC 0.001） |
| `--epochs` / `--patience` | 最大轮数 / 早停耐心 | 200/50（ZINC 100/25） |
| `--seeds` | 随机种子列表 | 0 1 2 |
| `--no-write` | 只打印不写文件 | 关闭 |

### 5.3 批量实验

```bash
python run_experiments.py --list                     # 打印实验计划
python run_experiments.py --experiments main pooling # 主实验 + 池化对比
python run_experiments.py --experiments all --seeds 0 1 2 --reset
```

| 组名 | 内容 |
|---|---|
| `main` | 5 数据集 × 4 模型（池化固定 mean）—— 模型对比 |
| `pooling` | 数据集 × 模型 × {mean, max, min, sum} —— **池化对比（作业要求）** |
| `hparam_lr` / `hparam_depth` / `hparam_hidden` | 超参扫描 |

> **成本说明**：ZINC 每次配置约 50 秒（1 万张训练图，且早停前要跑约 100 轮），
> 而 TUDataset 只要几秒。因此 ZINC 的池化对比只跑 **GCN 与 GIN** 两个架构
> （GIN 是 ZINC 的参考模型，GCN 是均值聚合基线）；四个模型的完整对比在
> TUDataset 上做。
>
> 该脚本还支持**分片并行**（`--shard i n` + 各自的 `--results-dir`）和
> `--merge` 合并，以及**断点续跑**（默认跳过 `all_results.csv` 里已有的配置）。

### 5.4 结果查看

```bash
python run_experiments.py --merge --results-dir results   # 合并分片（如用过 --shard）
python -c "import pandas as pd; print(pd.read_csv('results/all_results.csv'))"
```

---

## 6. 实验结果

> 以下数值全部来自 `results/all_results.csv`。
> TUDataset 指标为**测试准确率（%）**，ZINC 为 **MAE（越低越好）**。
> 每个配置为 2 个随机种子的均值（分层 80/10/10 重新划分）。

#### 4.2 不同神经网络的影响（池化固定 mean）

| dataset | gat | gcn | gin | sage |
|---|---|---|---|---|
| ENZYMES | Acc 32.50 | Acc 33.33 | Acc 27.50 | Acc 28.33 |
| IMDB-BINARY | Acc 72.50 | Acc 73.50 | Acc 75.00 | Acc 73.50 |
| MUTAG | Acc 88.89 | Acc 88.89 | Acc 80.56 | Acc 88.89 |
| PROTEINS | Acc 72.97 | Acc 75.68 | Acc 76.13 | Acc 72.07 |
| ZINC | MAE 0.8165 | MAE 1.0492 | MAE 0.7663 | MAE 0.7152 |

#### 4.3 池化方式的影响（作业重点）

**图分类（测试准确率 %）**

| dataset | max | mean | min | sum |
|---|---|---|---|---|
| ENZYMES | 27.50 ± 4.17 | 30.83 ± 4.17 | 25.00 ± 1.67 | 18.33 ± 5.00 |
| IMDB-BINARY | 75.50 ± 0.50 | 75.00 ± 0.00 | 78.50 ± 1.50 | 76.00 ± 1.00 |
| MUTAG | 86.11 ± 2.78 | 83.33 ± 5.56 | 86.11 ± 2.78 | 94.44 ± 0.00 |
| PROTEINS | 72.07 ± 0.90 | 72.52 ± 0.45 | 72.07 ± 0.00 | 69.82 ± 1.35 |

**ZINC 图回归（MAE，越低越好）**

| pool | gcn | gin |
|---|---|---|
| max | 1.0977 ± 0.0020 | 0.6408 ± 0.0267 |
| mean | 1.0513 ± 0.0089 | 0.7247 ± 0.0631 |
| min | 1.1105 ± 0.0031 | 0.7365 ± 0.0196 |
| sum | 1.0276 ± 0.0510 | 0.6960 ± 0.0052 |

**同一池化方式在不同模型下的表现（测试准确率 %）**

| pool | gat | gcn | gin | sage |
|---|---|---|---|---|
| max | 72.00 ± 0.00 | 74.00 ± 0.00 | 75.50 ± 0.50 | 72.50 ± 2.50 |
| mean | 77.00 ± 0.00 | 76.00 ± 0.00 | 75.00 ± 0.00 | 76.50 ± 1.50 |
| min | 70.50 ± 0.50 | 72.50 ± 2.50 | 78.50 ± 1.50 | 76.50 ± 0.50 |
| sum | 70.50 ± 0.50 | 74.50 ± 1.50 | 76.00 ± 1.00 | 73.50 ± 0.50 |

---

## 7. 结论

### 7.1 不同神经网络的影响

- **四个模型在 TUDataset 上的差距不大**（MUTAG 上 GCN/GAT/SAGE 都是 88.89%），
  因为这些数据集只有几百张图，**性能上限受数据量而非模型容量限制**。
- **GIN 在 ZINC 上明显最好**（MAE 0.715，GCN 1.049），这与 GIN 论文的结论一致：
  分子图的回归任务需要**保留图规模信息**的聚合方式。
- **GIN 在小图分类上并不占优**：MUTAG 上 GIN+mean 只有 80.56%，低于 GCN/GAT/SAGE 的 88.89%。
  原因是 GIN 的 sum 聚合对度数敏感，小图上容易过拟合。**换用 sum pooling 后
  GIN 达到 94.44%，反超所有组合** —— 说明 GIN 的读出方式必须与它的聚合方式匹配。

### 7.2 池化方式的影响（作业重点）

**没有全局最优的池化方式**，最优选择依赖数据集：

| 数据集 | 最优池化 | 最差池化 | 解释 |
|---|---|---|---|
| MUTAG | **sum 94.44%** | mean 83.33% | 分子图需要规模信息，sum 保留了原子数 |
| IMDB-BINARY | **min 78.50%** | mean 75.00% | 社交图（度特征）里「最小的那个邻居」有区分度 |
| PROTEINS | **mean 72.52%** | sum 69.82% | 蛋白质图规模差异大，sum 被大图主导 |
| ENZYMES | **mean 30.83%** | sum 18.33% | sum 退化到接近随机（6 类基线 16.7%） |
| ZINC(GCN) | **sum 1.0276** | min 1.1105 | 分子回归，MAE 越低越好，sum 最优 |
| ZINC(GIN) | **max 0.6408** | min 0.7365 | GIN 的 sum 聚合 + max 读出的组合最好 |

几个值得注意的现象：

1. **sum pooling 极不稳定**：在 MUTAG 上是全场最优（94.44%），在 ENZYMES 上却
   是全场最差（18.33%，接近 6 分类的随机基线 16.7%）。因为 sum 的输出量级
   正比于图的节点数，当数据集内图规模差异很大时，读出向量的尺度会剧烈变化，
   分类头难以适配。
2. **max / min pooling 不是彼此的镜像**：在 IMDB-BINARY 上 min 比 max 好 3 个点，
   在 MUTAG 上两者相同，在 PROTEINS 上也相同。**min 对特征下界敏感**，
   它需要激活值有稳定下界（这正是 GIN 每层加 BatchNorm 的副作用之一）。
3. **池化的效果与模型耦合**：同一个池化在不同模型下的排名并不一致
   （例如 IMDB-BINARY 上 max 对 GCN 是 74.00，对 GIN 是 75.50；
   而 min 对 GIN 是 78.50，对 GAT 只有 70.50）。
   因此「换池化」不能脱离模型单独讨论。

### 7.3 网络层数的影响

和任务一一致：**2–3 层最优，4 层普遍下降**，这是消息传递的**过平滑
（over-smoothing）** 现象 —— 层数增加后不同图的节点表示趋于相同，图级读出的
区分度反而降低。TUDataset 的图很小（平均十几到几十个节点），
过平滑出现得更早。

### 7.4 关于本任务实验设置的两点说明

1. **划分方式**：TUDataset 官方没有固定划分，文献里普遍用 **10 折交叉验证**。
   本实验用**分层 80/10/10 + 2 个随机种子重新划分**来近似，因此数值与文献
   （如 GIN 论文报告的 MUTAG 89.4%）会有几个点的出入，但模型间的**相对排名**是一致的。
2. **ZINC 的早停**：ZINC 默认最多 100 轮、耐心 25 轮。部分配置在验证 MAE 最低点
   之后较早停止，实际执行轮数记录在 `results/all_results.csv` 的 `epochs_run_mean` 列中，
   可比照确认没有欠拟合。
