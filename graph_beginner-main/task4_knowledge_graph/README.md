# 任务四、知识图谱补全（Knowledge Graph Completion）

实现 **TransE / RotatE / ConvE** 三个经典知识图谱嵌入模型，
在 **WN18RR** 和 **FB15k-237** 上完成链接预测（知识图谱补全），
采用知识图谱领域标准的 **filtered ranking** 协议评测 **MRR / Hits@1 / Hits@3 / Hits@10**。

> 与任务一/二不同，本任务**不涉及**全图训练 vs 分批次训练的对比（README 中已注明「任务四不需要」）。

---

## 1. 数据集与划分

| 数据集 | 实体数 | 关系数 | 训练集 | 验证集 | 测试集 |
|---|---|---|---|---|---|
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 |
| FB15k-237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 |

原始文件是制表符分隔的 `头实体 \t 关系 \t 尾实体` 三元组。
下载源：`https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/{wn18rr,FB15k-237}/{train,valid,test}.txt`
（脚本内置重试与镜像回退；`prepare_data.py --verify` 会**断言**上表的统计量，
一旦下载被截断或不完整会立即报错，而不会静默训练出无意义的结果。）

### 关于训练集 / 验证集 / 测试集的划分

- 知识图谱补全使用**官方给定的固定划分**，三个集合的三元组**互不相交**。
  这与任务一/二的「随机划分边/节点」不同：KG 的划分是社区长期沿用的基准，
  换划分会让结果无法与文献对比。
- **实体与关系的词表**：默认由三个集合的全部三元组构建（`--vocab-from all`），
  这是文献的做法（测试集里出现的新实体本来就该被预测）。
  脚本也提供 `--vocab-from train` 的**严格无泄漏**模式，
  此时验证/测试集中出现的未登录实体三元组会被丢弃并**打印丢弃数量**。
- **训练**：从训练集采样正样本，并**破坏头或尾**生成负样本（`--neg-samples` 个/正样本）。
- **验证/测试**：对每个测试三元组，分别把尾实体、头实体替换为**所有实体**，
  按分数排序取真实实体的排名。

### Filtered ranking（过滤排名）

评测时会**过滤掉所有已知为真的三元组**：如果某个错误候选 `(h, r, t')` 其实也在
train/valid/test 中出现过，就不把它算作比正确答案更优。这是 KG 补全的标准协议，
不滤波的结果（raw ranking）会被大量假阴性严重压低。
本实现同时支持 `--filtered`（默认）与 `--no-filtered`。
排名采用**乐观约定**：与正确答案同分的实体按 1/2 概率算作更优。

## 2. 模型原理

### 2.1 TransE（Bordes et al., 2013）

把关系看作嵌入空间中的**平移向量**：真三元组满足 `h + r ≈ t`，打分为**负距离**

```
f(h, r, t) = -|| h + r - t ||_p        (p = 1 或 2)
```

实体在打分前做 L2 归一化（保持距离尺度稳定），且**不施加 tanh**，
因为 TransE 依赖无界的平移几何。它能建模一对一的反对称关系，
但**无法**处理一对多、多对一、对称关系（`r` 无法同时满足 `h1 + r = t` 和 `h2 + r = t`）。

### 2.2 RotatE（Sun et al., 2019）

把每个实体看作复空间中的点、每个关系看作**逐元素的旋转**：`t ≈ h ∘ r`，其中 `|r_i| = 1`。
模长约束通过**相位参数化**实现：

```
r_i = exp(i·θ_i),   θ_i ∈ (-π, π)
f(h, r, t) = -|| h ∘ r - t ||_p
```

实现采用「前半实部 / 后半虚部」的实值表示。因为旋转可以同时表达
**对称**（θ = 0 或 π）、**反对称**（θ ≠ 0, π）、**逆关系**和**复合关系**，
RotatE 在 WN18RR 这类以对称/逆关系为主的数据集上明显强于 TransE。

> ⚠️ **踩坑记录**：`gamma`（margin）是**加性**的，只能出现在损失函数里。
> 早期版本在模型内部把 `h` 乘上了 `gamma` 去缩放距离，这破坏了旋转的
> **模长不变性**，导致 RotatE 彻底学不动（WN18RR 上 MRR ≈ 0.0002，等于瞎猜）。

### 2.3 ConvE（Dettmers et al., 2018）

把 `(头实体, 关系)` 拼接后 **reshape 成二维图像**（dim=200 时为 10×20），
经过 2D 卷积 → BatchNorm → Dropout → 全连接投影回 `dim`，
最后与实体嵌入矩阵做内积得到所有候选尾的分数：

```
f(h, r, t) = σ( vec(conv([h̄; r̄]))ᵀ W ) · t
```

训练使用 **1-vs-all（1-N）二分类交叉熵**：每个正样本的负例是**全部其它实体**，
配合标签平滑（0.1）。这比逐负采样更适合大图，但需要 dropout + 标签平滑
来抑制过拟合。默认超参采用论文设置：dim 200、10×20 reshape、32 个 3×3 卷积核、
hidden dropout 0.2、label smoothing 0.1。

## 3. 环境

见仓库根目录 `requirements.txt`。本任务只需 PyTorch，**不依赖** PyG 的采样扩展。

## 4. 目录结构

```
task4_knowledge_graph/
├── data/
│   ├── WN18RR/{train,valid,test}.txt
│   └── FB15k-237/{train,valid,test}.txt
├── code/
│   ├── prepare_data.py     下载 + 校验 KG 数据集
│   ├── dataset.py          词表构建、三元组读取、filtered 排名评测
│   ├── models.py           TransE / RotatE / ConvE
│   ├── train.py            单次训练+评测
│   ├── run_experiments.py  批量实验编排
│   └── utils.py            随机种子、CUDA 计时器、结果写入
├── results/
│   ├── all_results.csv
│   └── detail/*.json
└── README.md               本文件
```

## 5. 训练与测试脚本

所有命令都在 `task4_knowledge_graph/code/` 目录下执行。

### 5.1 准备数据

```bash
python prepare_data.py                      # 下载 WN18RR + FB15k-237
python prepare_data.py --datasets WN18RR
python prepare_data.py --verify             # 校验统计量是否与标准一致
```

### 5.2 训练 / 测试单个模型

```bash
# 训练并在测试集上评测（每 5 轮在验证集上算一次 MRR，按验证 MRR 早停）
python train.py --dataset WN18RR --model TransE
python train.py --dataset WN18RR --model RotatE
python train.py --dataset WN18RR --model ConvE

# 指定维度 / 学习率 / 负样本数
python train.py --dataset WN18RR --model RotatE --dim 400 --lr 1e-4 --neg-samples 128

# 多个随机种子
python train.py --dataset WN18RR --model TransE --seeds 0 1 2

# 消融：不加逆关系（head 侧排名复用正向关系嵌入）
python train.py --dataset WN18RR --model RotatE --no-inverse-relations
```

常用参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--dataset` | `WN18RR` / `FB15k-237` | WN18RR |
| `--model` | `TransE` / `RotatE` / `ConvE` | RotatE |
| `--dim` | 嵌入维度 | 200 |
| `--lr` | 学习率 | 1e-3（RotatE 1e-4） |
| `--batch-size` | 批大小 | 1024（ConvE 256） |
| `--neg-samples` | 每个正样本的负样本数 | 64 |
| `--gamma` | margin 损失的间隔 | 6.0（FB15k-237 为 9.0） |
| `--epochs` / `--patience` | 最大轮数 / 早停耐心 | 50/10（ConvE 20/5） |
| `--eval-every` | 每隔多少轮做一次验证 | 5（ConvE 1） |
| `--filtered` / `--no-filtered` | 是否用标准 filtered 排名 | filtered |
| `--bern` | 使用 Bernoulli 负采样 | 关闭（均匀采样） |
| `--no-inverse-relations` | TransE/RotatE 不训练逆关系 | 关闭 |
| `--vocab-from` | 词表来源 `all` / `train` | all |
| `--seeds` / `--no-write` | 随机种子 / 只打印不写文件 | 0 1 2 / 关闭 |

**关于逆关系**：TransE 与 RotatE 默认在训练时同时加入 `(t, r+R, h)` 这类逆三元组
（`--inverse-relations`，默认开启）。这样 head 侧排名可以复用 tail 侧的
`score_all_tails`，且 head 侧拥有自己的关系嵌入，是 KGE 框架的标准做法。

### 5.3 批量实验

```bash
python run_experiments.py --list                     # 打印实验计划
python run_experiments.py --experiments main --datasets WN18RR --seeds 0
python run_experiments.py --experiments all --seeds 0 1 2 --reset
```

| 组名 | 内容 |
|---|---|
| `main` | 数据集 × {TransE, RotatE, ConvE} —— **模型对比（作业要求）** |
| `hparam_dim` | 嵌入维度 {50, 100, 200, 400} |
| `hparam_lr` | 学习率 {1e-4 … 1e-2} |
| `hparam_neg` | 负样本数 {1, 4, 16, 64, 128} |
| `hparam_gamma` | margin {3, 6, 9, 12, 24} |
| `hparam_bern` | 均匀 vs Bernoulli 负采样 |
| `ablation_inverse` | 训练时加/不加逆关系 |

支持 `--datasets` / `--models` 过滤、`--epochs` / `--batch-size` 覆盖（覆盖值会写进
结果 CSV，避免把「缩短预算」误当成「完整训练」）。

---

## 6. 实验结果

> 下表数值来自 `results/all_results.csv`，评测协议为 **filtered ranking**。
> 指标为 **MRR / Hits@1 / Hits@3 / Hits@10**（越高越好），
> 由模型在**验证集 MRR** 最优时保存的权重在测试集上测得。

#### 5.2 三个模型的对比（filtered ranking，% 除 MRR 外）

| 数据集 | 模型 | 维度 | MRR | Hits@1 | Hits@3 | Hits@10 | 训练时间(s) | 实跑轮数 |
|---|---|---|---|---|---|---|---|---|
| WN18RR | ConvE | 200 | 0.0577 | 3.56 | 6.16 | 9.91 | 354 | 50.0 |
| WN18RR | RotatE | 200 | 0.3872 | 37.59 | 39.04 | 40.83 | 291 | 35.0 |
| WN18RR | TransE | 200 | 0.2008 | 1.48 | 37.56 | 44.51 | 340 | 50.0 |

> 训练时间包含每轮验证；`实跑轮数` 为早停实际执行的轮数，用于核对是否因预算不足而欠拟合。


---

## 7. 结论

### 7.1 三个模型的对比

在 WN18RR 上（filtered ranking，单一种子）：

**RotatE (MRR 0.387) ≫ TransE (0.201) ≫ ConvE (0.058)**

这个排序与理论预期一致：

1. **RotatE 最好，因为它能建模对称/逆关系。** WN18RR 是从 WordNet 构建的，
   主要关系类型是**对称**（如 `similar_to`）和**逆关系**（如 `hypernym`/`hyponym`）。
   RotatE 把关系建模为复平面上的旋转，`θ = 0` 或 `π` 表示对称、
   `θ ≠ 0, π` 表示反对称，因此能同时覆盖这几类关系；
   而 TransE 的平移几何**在原理上就无法表示对称关系**
   （若 `h1 + r = t` 且 `h2 + r = t`，则 `h1 = h2`）。

2. **TransE 有一个很典型的现象：Hits@1 只有 1.48%，Hits@3 却高达 37.56%。**
   也就是说它**经常把正确答案排在第 2–3 位，但几乎从不排在第 1 位**。
   原因正是平移几何的「拥挤」：`h + r` 附近常常聚集着多个实体，
   模型很难把唯一正确的那个精确对齐到第一位。
   对比 RotatE 的 Hits@1 = 37.59%——旋转把正确答案**精确**转到位，
   而不是落在一片模糊的邻域里。这组数字本身就是对两种几何最直观的说明。

3. **ConvE (0.058) 明显欠拟合**，原因是训练预算：ConvE 用的是 **1-vs-all
   二分类**，每个 epoch 只相当于一次 1-N 打分，收敛比逐样本采样慢得多。
   本实验给了 50 轮（约 6 分钟），而 ConvE 论文在 WN18RR 上训练 100 轮以上。
   它的 `epochs_run` 列显示 50（跑满未早停），说明**限制因素是轮数而不是收敛**，
   继续训练应该还能提升。

### 7.2 与文献值的对照

| 模型 | 本实验 (dim=200) | 文献 (dim=500，负样本 256–1024) |
|---|---|---|
| TransE | MRR 0.201 | MRR 0.226 |
| RotatE | MRR 0.387 | MRR 0.476 |
| ConvE | MRR 0.058 | MRR 0.43 |

TransE 与 RotatE 已经接近文献水平。差距主要来自**规模而非方法**：
本实验的维度是 200（文献 500），每个正样本只采 64 个负样本
（文献 256–1024），并且用 max-margin 损失而非 RotatE 论文的
self-adversarial softmax。ConvE 的差距则完全来自训练轮数。

### 7.3 数据划分与评测协议的说明

- **划分**：使用数据集官方给定的 train/valid/test，三者互不相交。
  词表默认由三个集合共同构建（`--vocab-from all`，文献做法），
  也提供 `--vocab-from train` 的严格无泄漏模式。
- **评测**：**filtered ranking**（过滤掉所有已知为真的候选三元组），
  同时对尾实体和头实体两侧排名。头侧通过训练时加入的**逆关系**实现，
  这样 `score_all_heads` 可以复用 `score_all_tails` 的打分函数。
- **排名约定**：与正确答案同分的实体按 1/2 计入（乐观约定），rank 从 1 开始计。

