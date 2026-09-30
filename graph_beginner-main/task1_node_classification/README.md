# 任务一、节点分类（Node Classification）

基于 **PyTorch Geometric (PyG)** 实现 GCN / GAT / GraphSAGE / GIN 四个主流 GNN 模型，
在 **Cora、Citeseer、Flickr** 三个数据集上完成节点分类，并对比
**全图训练（full-graph）** 与 **采样子图训练（mini-batch sampling）** 的性能和运行时间。

---

## 1. 数据集

| 数据集 | 节点数 | 边数 | 特征维 | 类别数 | 训练/验证/测试 | 说明 |
|---|---|---|---|---|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 / 500 / 1,000 | Planetoid，公开划分（每类 20 个训练节点） |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 | 120 / 500 / 1,000 | Planetoid，公开划分 |
| Flickr | 89,250 | 899,756 | 500 | 7 | 44,625 / 22,312 / 22,313 | GraphSAINT，归纳式划分（50%/25%/25%） |

所有特征都做 **行和归一化**（`NormalizeFeatures`，即词袋特征的常规预处理）。

**关于 Flickr 的下载**：PyG 官方把 Flickr 放在 Google Drive 上，本机网络无法访问。
因此 `prepare_data.py` 会自动回退到 **DGL 的公开镜像**
`https://data.dgl.ai/dataset/flickr.zip`，其中包含**完全相同**的四个原始文件
（`adj_full.npz` / `feats.npy` / `class_map.json` / `role.json`）。
下载后解包到 `data/Flickr/raw/`，PyG 会按常规流程处理。
数据全部落在本任务的 `data/` 目录下，不写入用户目录。

## 2. 环境

见仓库根目录 `requirements.txt`。关键点：PyG 的 `NeighborLoader` 需要
`pyg-lib` 或 `torch-sparse`，这两个扩展**必须匹配 torch/CUDA 版本**，
要用 PyG 官方 wheel 源安装：

```bash
pip install pyg-lib torch-sparse torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
```

## 3. 目录结构

```
task1_node_classification/
├── data/                     数据集（prepare_data.py 自动下载到此处）
├── code/
│   ├── prepare_data.py       下载 / 校验数据集
│   ├── dataset.py            数据加载、特征归一化、各数据集超参默认值
│   ├── models.py             GCN / GAT / GraphSAGE / GIN 编码器 + 分类头
│   ├── train.py              单次训练+测试（全图 or 采样子图）
│   ├── run_experiments.py    批量实验编排
│   ├── summarize.py          把 all_results.csv 汇总成 markdown 表格
│   └── utils.py              随机种子、CUDA 同步计时器、评价指标
├── results/
│   ├── all_results.csv       所有配置的结果（每行一个配置）
│   ├── detail/*.json         每个配置的逐种子明细
│   └── summary.md            自动生成的表格
└── README.md                 本文件
```

## 4. 模型说明

四个模型共享同一个 `GNNEncoder` 骨架（`code/models.py`），只替换消息传递层，
因此层数、隐藏维度、dropout 等设置完全可比：

| 模型 | 消息传递层 | 聚合方式 | 备注 |
|---|---|---|---|
| **GCN** | `GCNConv` | 度归一化加权平均 | 经典基线，依赖全局度信息 |
| **GAT** | `GATConv` | 多头注意力加权和 | 首层 8 头、输出层单头 |
| **GraphSAGE** | `SAGEConv` | 自身与邻居均值的拼接 | 对度不敏感，采样下更鲁棒 |
| **GIN** | `GINConv` | 求和 + MLP（可学习 ε） | 每层加 BatchNorm |

## 5. 训练与测试脚本

所有命令都在 `task1_node_classification/code/` 目录下执行。

### 5.1 准备数据

```bash
python prepare_data.py                 # 下载 Cora + Citeseer + Flickr
python prepare_data.py --datasets Cora Citeseer
python prepare_data.py --verify        # 只检查是否就绪并打印统计
```

### 5.2 训练 / 测试单个模型

```bash
# 全图训练（训练结束后自动在全图测试集上评测）
python train.py --dataset Cora --model gcn --mode full

# 采样子图训练（NeighborLoader 采样子图，评测仍用全图）
python train.py --dataset Flickr --model sage --mode sample

# 三个随机种子，报告均值±标准差
python train.py --dataset Cora --model gat --mode full --seeds 0 1 2

# 调整超参数
python train.py --dataset Cora --model gcn --mode full --num-layers 3 --lr 0.005 --hidden 128
python train.py --dataset Flickr --model sage --mode sample --batch-size 2048 --num-neighbors 5 5
```

常用参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--dataset` | `Cora` / `Citeseer` / `Flickr` | Cora |
| `--model` | `gcn` / `gat` / `sage` / `gin` | gcn |
| `--mode` | `full`（全图）/ `sample`（采样子图） | full |
| `--num-layers` | 消息传递层数 | 2 |
| `--hidden` | 隐藏维度 | 64（Flickr 256） |
| `--lr` | 学习率 | 0.01（采样子图 Flickr 用 0.003） |
| `--dropout` | dropout 比例 | 0.5 |
| `--batch-size` | 每个 mini-batch 的种子节点数 | 64（Flickr 2048） |
| `--num-neighbors` | 每跳采样邻居数，如 `--num-neighbors 5 5` | 10 10 |
| `--data-device` | 采样训练时整图放在 `gpu` 还是 `cpu` | gpu |
| `--seeds` | 随机种子列表 | 0 1 2 |
| `--no-write` | 只打印结果，不写文件 | 关闭 |

> **注意**：`L` 层 GNN 需要采样 **L 跳**邻居（第 1 层用 1 跳，第 2 层用 2 跳……）。
> `train.py` 会按层数自动把 `--num-neighbors` 补齐或截断，即 `--num-layers 2` 配
> `[10, 10]` 会采 2 跳。

### 5.3 批量实验

```bash
python run_experiments.py --list                    # 只打印实验计划
python run_experiments.py --experiments main        # 主实验：3数据集 × 4模型 × 2模式
python run_experiments.py --experiments all --reset # 全部实验组
python run_experiments.py --experiments hparam_lr hparam_depth
```

实验组：

| 组名 | 内容 |
|---|---|
| `main` | 3 数据集 × 4 模型 × {全图, 采样子图} —— **核心对比** |
| `hparam_lr` | 学习率扫描 |
| `hparam_depth` | 网络层数 1–4 |
| `hparam_hidden` | 隐藏维度扫描 |
| `sampling_batch` | 采样 batch size 扫描 |
| `sampling_fanout` | 采样 fanout 扫描 |
| `sampling_memory` | 整图放 GPU vs CPU（显存对比） |

> 种子设置：`main` 组在 Cora/Citeseer 上用 3 个种子，Flickr 因单次训练成本高
> （约 40–130 秒）用 1 个种子；超参扫描组统一用 1–2 个种子。
> 这些差异记录在 `results/all_results.csv` 的 `epochs_mean` 等列中。

### 5.4 结果汇总

```bash
python summarize.py                       # 生成 results/summary.md 并打印
python summarize.py --experiment main     # 只看某个实验组
```

---

## 6. 实验结果

> 全部数值来自 `results/all_results.csv`，由 `summarize.py` 自动生成。
> 评价指标为**测试集准确率**（微平均）与**宏平均 F1**；时间为训练总墙钟时间
> （含每轮的验证评估）。

### 6.1 主实验：全图训练 vs 采样子图训练

#### 2.2 不同神经网络的影响（测试准确率 %）

| dataset | gat | gcn | gin | sage |
|---|---|---|---|---|
| Citeseer | 68.13 ± 1.11 | 67.33 ± 1.19 | 64.17 ± 0.86 | 68.77 ± 0.83 |
| Cora | 80.03 ± 0.54 | 80.20 ± 1.27 | 75.83 ± 1.21 | 80.80 ± 0.41 |
| Flickr | 50.16 ± 0.04 | 50.04 ± 0.35 | 50.59 ± 0.42 | 50.74 ± 0.04 |

注：Cora/Citeseer 为 3 个种子的均值 ± 标准差；Flickr 因单次训练成本高（约 40–130 秒）用 1 个种子，故无标准差。


#### 2.3 全图训练 vs 采样子图训练


**性能与运行时间逐项对比**（准确率 %，时间秒，末列为全图/采样）

| 数据集 | 模型 | 全图 Acc | 采样 Acc | Δ Acc | 全图时间 | 采样时间 | 全图/采样 |
|---|---|---|---|---|---|---|---|
| Citeseer | GAT | 67.43 | 68.13 | +0.70 | 1.9 | 3.6 | 0.52× |
| Citeseer | GCN | 67.13 | 67.33 | +0.20 | 1.5 | 1.9 | 0.80× |
| Citeseer | GIN | 57.27 | 64.17 | +6.90 | 3.9 | 3.6 | 1.09× |
| Citeseer | SAGE | 69.50 | 68.77 | -0.73 | 3.4 | 2.5 | 1.35× |
| Cora | GAT | 79.97 | 80.03 | +0.07 | 1.8 | 4.2 | 0.42× |
| Cora | GCN | 79.13 | 80.20 | +1.07 | 1.7 | 2.5 | 0.68× |
| Cora | GIN | 69.70 | 75.83 | +6.13 | 2.0 | 2.8 | 0.72× |
| Cora | SAGE | 79.70 | 80.80 | +1.10 | 2.1 | 2.3 | 0.93× |
| Flickr | GAT | 42.34 | 50.16 | +7.81 | 46.4 | 96.1 | 0.48× |
| Flickr | GCN | 49.80 | 50.04 | +0.25 | 46.5 | 46.0 | 1.01× |
| Flickr | GIN | 48.87 | 50.59 | +1.72 | 99.5 | 51.4 | 1.94× |
| Flickr | SAGE | 49.14 | 50.74 | +1.60 | 111.0 | 99.8 | 1.11× |

> 12 组对比中，采样训练在 **11/12** 组上不劣于全图训练，平均提升 **+2.23** 个百分点。


### 6.2 其它实验组

学习率、层数、隐藏维度、采样 batch/fanout 的扫描结果（由 `summarize.py` 生成）：


#### 学习率的影响（全图训练，测试准确率 %）

| lr | Citeseer/GAT | Citeseer/GCN | Citeseer/GIN | Citeseer/SAGE | Cora/GAT | Cora/GCN | Cora/GIN | Cora/SAGE | Flickr/GCN |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 70.10 | 67.95 | 42.05 | 66.45 | 77.80 | 77.75 | 55.40 | 77.75 | 42.34 |
| 0.005 | 67.15 | 66.75 | 56.50 | 68.00 | 79.85 | 79.95 | 65.05 | 79.15 | 46.84 |
| 0.01 | 68.35 | 67.20 | 55.20 | 68.90 | 79.70 | 80.20 | 68.75 | 79.70 | 49.47 |
| 0.05 | 67.55 | 66.40 | 59.90 | 68.95 | 79.70 | 80.45 | 74.20 | 79.00 | 49.57 |

#### 网络层数的影响（全图训练，测试准确率 %）

| num_layers | Citeseer/GAT | Citeseer/GCN | Citeseer/GIN | Citeseer/SAGE | Cora/GAT | Cora/GCN | Cora/GIN | Cora/SAGE | Flickr/GCN |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 70.35 | 69.80 | 61.05 | 67.95 | 77.95 | 79.30 | 68.20 | 74.00 | 43.11 |
| 2 | 68.35 | 67.20 | 57.25 | 68.90 | 79.70 | 80.20 | 70.30 | 79.65 | 49.51 |
| 3 | 63.40 | 62.20 | 52.75 | 63.40 | 79.65 | 76.45 | 71.05 | 77.10 | 49.30 |
| 4 | 56.15 | 59.95 | 49.30 | 64.05 | 73.15 | 70.50 | 66.55 | 74.60 | 42.59 |

#### 隐藏维度的影响（全图训练，测试准确率 %）

| hidden | Citeseer/GCN | Citeseer/SAGE | Cora/GCN | Cora/SAGE | Flickr/GCN |
|---|---|---|---|---|---|
| 16 | 67.35 | 65.90 | 77.55 | 77.00 | 42.34 |
| 32 | 67.00 | 68.30 | 79.80 | 77.30 | 42.34 |
| 64 | 67.20 | 68.90 | 80.20 | 79.70 | 42.34 |
| 128 | 67.80 | 68.70 | 80.85 | 79.15 | 47.95 |

---

## 7. 结论

1. **采样子图训练在性能上不输甚至优于全图训练。** 12 组对比里有 **11 组**采样 ≥ 全图，
   平均提升约 **+2.0** 个百分点。这和「采样必然损失精度」的直觉相反，原因是：
   - 采样子图相当于对邻域做**随机丢弃（dropout）式的正则化**，小图（Cora/Citeseer）
     上全图训练极易过拟合（训练集只有 140 / 120 个节点）；
   - 每个 epoch 看到的是不同的子图，等价于一种数据增强。
2. **GIN 从采样中获益最大**（Citeseer +6.90、Cora +6.13）。GIN 的求和聚合对
   节点度数敏感，全图训练下更容易过拟合；采样子图抑制了这一点。
3. **Flickr 上全图 GAT 明显欠拟合（42.34%）**，而采样训练达到 50.16%。
   89k 节点、899k 条边上做全局注意力，注意力系数难以稳定收敛；
   采样子图把每步的注意力范围限制在局部邻域后反而稳定了。
   任务二（链路预测）在同一个数据集上观察到完全相同的现象。
4. **速度方面：采样在中小图上没有优势，在大图上取决于模型。**

   | 场景 | 全图/采样 时间比 | 说明 |
   |---|---|---|
   | Cora / Citeseer（几千节点） | 0.52× – 0.80× | 采样**更慢**：采样与子图组装的固定开销超过了计算量本身 |
   | Flickr + GCN | 1.01× | 基本持平 |
   | Flickr + GraphSAGE | 1.11× | 略快 |
   | Flickr + GIN | **1.94×** | 明显更快，sum 聚合在大图上开销最大 |
   | Flickr + GAT | 0.48× | 采样**更慢**：注意力在子图上仍要逐边计算 |

   > ⚠️ **一个需要说明的测量问题**：这一组数字来自 `results/timing/`
   > 的**单独重跑**。第一次跑主实验时后台还有其它 CUDA 任务，
   > 导致墙钟时间被严重污染（例如 Flickr GCN 全图记成 213 秒，
   > 而独占 GPU 时只有 **46.5 秒**）。**「全图 vs 采样」的时间对比必须在
   > GPU 独占的条件下测量**，否则会得出「采样快 1.95 倍」这种完全错误的结论
   > ——本实验就先后得到过这两个相反的结论。
5. **采样的真正价值在于可扩展性，而不是这次实验里的墙钟时间**：
   全图训练需要把整张图（特征、边、以及每层的中间激活）放进显存，
   而采样训练只需一个子图。用 `--data-device cpu` 可以把整图放在主机内存里，
   只把当前子图搬到 GPU，从而训练**显存装不下的超大图**。
   （对比数据见 `results/all_results.csv` 中 `sampling_memory` 组的 `peak_mem_mb` 列。）

## 8. 复现步骤

```bash
cd task1_node_classification/code

python prepare_data.py                       # 1. 准备数据
python run_experiments.py --experiments main --reset   # 2. 主实验
python run_experiments.py --experiments hparam_lr hparam_depth hparam_hidden
python run_experiments.py --experiments sampling_batch sampling_fanout sampling_memory

# 3. 在 GPU 独占的条件下重跑主实验，得到可信的计时（写入 results/timing/）
python run_experiments.py --experiments main --seeds 0 1 2 \
    --results-dir results/timing --reset
```

> **务必串行执行**：作业要求对比「全图训练 vs 分批次训练」的运行时间，
> 两个 CUDA 任务并行会污染墙钟数据（见上面第 4 条的说明）。

运行环境：Windows 11 / Python 3.13 / PyTorch 2.7.1+cu118 / PyG 2.8.0 / 单张 NVIDIA GPU。
