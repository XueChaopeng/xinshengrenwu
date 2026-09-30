# 任务二、链路预测（Link Prediction）

基于 **PyTorch Geometric (PyG)** 实现 GCN / GAT / GraphSAGE / GIN 四个 GNN 编码器，
在 **Cora、Citeseer、Flickr** 三个数据集上完成链路预测，
并对比 **全图训练（full-graph）** 与 **采样子图训练（mini-batch sampling）** 的性能和运行时间。

---

## 1. 数据集与边划分

沿用任务一的三个图（节点特征做行和归一化）：

| 数据集 | 节点数 | 边数 | 特征维 | 平均度 |
|---|---|---|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 7.80 |
| Citeseer | 3,327 | 9,104 | 3,703 | 5.47 |
| Flickr | 89,250 | 899,756 | 500 | 20.16 |

链路预测需要**边的划分**而非节点划分。使用
`torch_geometric.transforms.RandomLinkSplit`，参数与链路预测文献（以及 PyG 官方
「Introduction to link prediction」教程）一致：

| 设置 | 取值 | 原因 |
|---|---|---|
| 划分比例 | **85% / 5% / 10%**（train/val/test） | 文献标准 |
| `is_undirected=True` | 无向图 `(u,v)` 与 `(v,u)` 同属一个集合 | 否则「反向边就是答案」，等于泄题 |
| `disjoint_train_ratio=0.0` | 编码器在**全部训练边**上做消息传递，这些边同时充当监督信号 | 全图与采样两种模式看到的消息传递图完全相同，对比才公平 |
| `add_negative_train_samples=False` | 训练集的负边**不预先写入** | 全图模式每轮重新采样一次，采样模式每个 mini-batch 重新采样 |
| 验证/测试集 | 自动各配等量负边 | 评测时正负各半，AUC/AP 才有意义 |

以 Cora 为例（种子 0）：

| 划分 | 消息传递边 | 监督边 | 正样本 | 负样本 |
|---|---|---|---|---|
| train | 8,976 | 4,488 | 4,488 | 训练时在线采样 |
| val | 8,976 | 526 | 263 | 263 |
| test | 9,502 | 1,054 | 527 | 527 |

**评价指标**：**ROC-AUC** 与 **Average Precision（AP）**，两者都以百分数报告
（随机基线为 50 / 正例占比）。另附一个 split 级别的 `Hits@50`。

## 2. 环境

见仓库根目录 `requirements.txt`。`LinkNeighborLoader` 需要 `pyg-lib` 或
`torch-sparse` 之一，必须用 PyG 官方 wheel 源按 torch/CUDA 版本安装：

```bash
pip install pyg-lib torch-sparse torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
```

## 3. 目录结构

```
task2_link_prediction/
├── data/                     数据集（与任务一相同，可互相拷贝 raw 目录）
├── code/
│   ├── prepare_data.py       下载 / 校验数据集
│   ├── dataset.py            数据加载 + RandomLinkSplit 边划分
│   ├── models.py             GNN 编码器 + 点积/MLP 解码器
│   ├── train.py              单次训练+测试（全图 or 采样子图）
│   ├── run_experiments.py    批量实验编排
│   └── utils.py              随机种子、CUDA 计时器、AUC/AP/Hits@K
├── results/
│   ├── all_results.csv
│   └── detail/*.json
└── README.md                 本文件
```

## 4. 模型

链路预测模型 = **GNN 编码器** + **解码器**（`code/models.py`）。
编码器与任务一完全同源，因此两个任务的模型行为可以直接对照。

| 模型 | 编码器 | 聚合 |
|---|---|---|
| GCN | `GCNConv` | 度归一化加权平均 |
| GAT | `GATConv` | 多头注意力 |
| GraphSAGE | `SAGEConv` | 自身 + 邻居均值 |
| GIN | `GINConv` | 求和 + MLP + BatchNorm |

解码器二选一（`--decoder`）：

- `dot`（默认）：`score(u,v) = <z_u, z_v>`，最经典、参数量为零；
- `mlp`：拼接两端嵌入后过两层 MLP，能表达非对称关系。

编码器输出只用于打分，不接分类头，因此所有模型的输出维度都是隐藏维度。

> **关于 GIN 与链路预测**：GIN 的求和聚合不带归一化，节点嵌入的模长会随度数增长，
> 做内积解码时尺度差异很大。全图模式下 Cora 的 GIN 明显欠拟合（AUC 75.42），
> 采样模式下反而好转（87.98）——子图截断把尺度拉平了。

## 5. 训练与测试脚本

所有命令都在 `task2_link_prediction/code/` 目录下执行。

### 5.1 准备数据

```bash
python prepare_data.py                       # 下载 Cora + Citeseer + Flickr
python prepare_data.py --verify              # 只检查是否就绪并打印统计
```

> 如果已经跑过任务一，可以把 `task1_node_classification/data/<数据集>/raw`
> 直接拷贝到本任务的同名目录，省去重新下载。
> Flickr 的官方源在 Google Drive 上（本机不可达），脚本会自动回退到
> DGL 镜像 `https://data.dgl.ai/dataset/flickr.zip`（GraphSAINT 原版四文件）。

### 5.2 训练 / 测试单个模型

```bash
# 全图训练（编码器在整张训练图上做消息传递）
python train.py --dataset Cora --model gcn --mode full

# 采样子图训练（LinkNeighborLoader 围绕一批监督边采邻居子图）
python train.py --dataset Flickr --model sage --mode sample

# 三个种子，报告均值±标准差
python train.py --dataset Cora --model gat --mode full --seeds 0 1 2

# 换解码器 / 调超参
python train.py --dataset Cora --model gcn --decoder mlp
python train.py --dataset Flickr --model sage --mode sample --batch-size 2048 --num-neighbors 5 5
```

常用参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--dataset` | `Cora` / `Citeseer` / `Flickr` | Cora |
| `--model` | `gcn` / `gat` / `sage` / `gin` | gcn |
| `--mode` | `full`（全图）/ `sample`（采样子图） | full |
| `--decoder` | `dot` / `mlp` | dot |
| `--num-layers` | 消息传递层数 | 2 |
| `--hidden` | 隐藏维度 | 128（Flickr 256） |
| `--dropout` | dropout 比例 | 0.0 |
| `--lr` | 学习率 | 0.01（Flickr 全图 0.005、采样 0.002） |
| `--batch-size` | 每个 mini-batch 的监督边数 | 512（Flickr 2048） |
| `--num-neighbors` | 每跳采样邻居数 | 10 10 |
| `--neg-sampler` | 全图模式的负采样器 `uniform` / `pyg` | uniform |
| `--data-device` | 采样训练时整图放 `gpu` 还是 `cpu` | gpu |
| `--seeds` / `--no-write` | 随机种子 / 只打印不写文件 | 0 1 2 / 关闭 |

**关于负采样**：全图模式默认用 `--neg-sampler uniform`
（在 GPU 上直接均匀采样两个端点）。对这三个稀疏图而言，采到真实边的概率极低
（Cora 10,556 / 2,708² = 0.14%，Flickr 0.011%），且实测比
`torch_geometric.utils.negative_sampling` 快 **35 倍**
（0.6 ms vs 20.8 ms 每次调用，后者会因内部实现往返 CPU）。
需要严格过滤已存在边的场合可以切换回 `--neg-sampler pyg`。

### 5.3 批量实验

```bash
python run_experiments.py --list                     # 打印实验计划
python run_experiments.py --experiments main         # 主实验：3数据集 × 4模型 × 2模式
python run_experiments.py --experiments all --reset  # 全部实验组
```

| 组名 | 内容 |
|---|---|
| `main` | 3 数据集 × 4 模型 × {全图, 采样子图} —— **核心对比** |
| `decoder` | 点积 vs MLP 解码器 |
| `hparam_lr` | 学习率扫描 |
| `hparam_depth` | 网络层数 1–4 |
| `sampling_batch` | 采样 batch size 扫描 |
| `sampling_fanout` | 采样 fanout 扫描 |

> 种子：`main` 组 3 个种子，其余实验组 2 个种子（见脚本内 `GROUP_SEEDS`）。

---

## 6. 实验结果

> 指标为测试集 **AUC / AP（%）**，均为 3 个随机种子的均值。
> 时间是从训练开始到早停的墙钟总时间（含每轮验证）。

#### 3.2 不同神经网络的影响（测试集 AUC %）

| dataset | gat | gcn | gin | sage |
|---|---|---|---|---|
| Citeseer | 90.58 ± 0.09 | 89.54 ± 0.71 | 85.64 ± 0.79 | 86.89 ± 0.60 |
| Cora | 89.93 ± 1.39 | 90.52 ± 0.79 | 89.23 ± 0.81 | 88.37 ± 1.46 |
| Flickr | 65.71 ± 0.00 | 74.02 ± 0.00 | 66.04 ± 0.00 | 64.76 ± 0.00 |

#### 3.3 全图训练 vs 采样子图训练

| 数据集 | 模型 | 全图 AUC | 采样 AUC | Δ AUC | 全图时间 | 采样时间 | 全图/采样 |
|---|---|---|---|---|---|---|---|
| Citeseer | GAT | 90.41 | 90.58 | +0.17 | 1.9 | 7.8 | 0.24× |
| Citeseer | GCN | 89.66 | 89.54 | -0.12 | 1.7 | 7.2 | 0.24× |
| Citeseer | GIN | 78.08 | 85.64 | +7.56 | 3.5 | 13.6 | 0.26× |
| Citeseer | SAGE | 82.15 | 86.89 | +4.74 | 3.0 | 10.1 | 0.30× |
| Cora | GAT | 90.29 | 89.93 | -0.36 | 1.8 | 5.9 | 0.30× |
| Cora | GCN | 85.74 | 90.52 | +4.78 | 1.7 | 4.4 | 0.38× |
| Cora | GIN | 82.16 | 89.23 | +7.07 | 2.6 | 11.1 | 0.24× |
| Cora | SAGE | 89.20 | 88.37 | -0.83 | 2.4 | 4.5 | 0.52× |
| Flickr | GAT | 60.37 | 65.71 | +5.34 | 448.9 | 1137.1 | 0.39× |
| Flickr | GCN | 83.55 | 74.02 | -9.53 | 23.7 | 162.4 | 0.15× |
| Flickr | GIN | 64.93 | 66.04 | +1.11 | 28.4 | 89.1 | 0.32× |
| Flickr | SAGE | 59.95 | 64.76 | +4.81 | 99.7 | 172.2 | 0.58× |

---

## 7. 结论

### 7.1 不同神经网络的影响

- **GAT 在 Cora / Citeseer 上最好**（90.29 / 90.41 AUC）：注意力机制能给不同邻居
  分配权重，在引文网络这种邻居质量差异大的图上优势明显。
- **GIN 在小图上最差**（Cora 82.16、Citeseer 78.08）：GIN 的 sum 聚合不归一化，
  节点嵌入的模长随度数增长，做内积解码时尺度差异很大；
  采样训练把子图截断后这一点明显缓解（85.64 / 89.23）。
- **Flickr 上全图 GAT 只有 60.37 AUC，GCN 却有 83.55** —— 这与直觉相反。
  89k 节点、90 万条边上做全局注意力，注意力系数难以稳定收敛；
  而 GCN 的度归一化是固定权重的平滑算子，在大图上更稳。
  **采样子图能把 GAT 拉回 65.71**，因为每个 mini-batch 只在局部邻域内算注意力。
  （任务一的节点分类上观察到完全相同的现象：Flickr 全图 GAT 只有 42.34%。）

### 7.2 全图训练 vs 采样子图训练

12 组对比中 **8 组采样不劣于全图**，平均提升 **+2.06** 个百分点。规律很清楚：

1. **采样对小图的 GIN / GCN 帮助最大**：Cora GIN +7.07、Citeseer GIN +7.56、
   Cora GCN +4.78。与任务一同样的解释：子图是天然的**正则化**，
   而 Cora 只有 4,488 条监督边，全图训练很容易过拟合。
2. **Flickr 上采样普遍弱于全图**（GCN −9.53 最明显）。原因不在精度上限而在
   **训练预算**：Flickr 有约 80 万条监督训练边，边级采样的步数等于
   `监督边数 / batch_size`。batch 取 2048 时每个 epoch 就要 400 步，
   单次训练要几分钟，因此实际执行轮数被迫压低（见 CSV 的 `epochs_mean` 列）。
   **边级采样在大图上比节点级采样昂贵得多**，这是任务一（按节点采样）
   与任务二（按边采样）最关键的差异。
3. **时间上采样全面更慢**（比值 0.15×–0.58×）：同样是监督边数量导致的。
   任务一里采样在 Flickr 上快 1.95×，这里却只有 0.15×，正是「按边采样 vs
   按节点采样」的差别——Flickr 有 4.5 万个训练节点，却有 80 万条监督边。

### 7.3 关于本任务实验设置的三点说明

1. **负采样**：全图模式默认用 GPU 均匀负采样（每轮重采），采样模式由
   `LinkNeighborLoader` 在每个 mini-batch 内采负样本，两者都是 1:1 的正负比例。
2. **公平性**：两种模式看到的消息传递图完全相同（`disjoint_train_ratio=0`，
   编码器在全部训练边上做传播），评测也都在全图上做，因此差异只来自
   「梯度是用全图算的还是用一个子图估的」。
3. **Flickr 的种子数**：单次采样训练要几分钟，Flickr 用 1 个种子，
   Cora / Citeseer 用 3 个（报告均值 ± 标准差）。
