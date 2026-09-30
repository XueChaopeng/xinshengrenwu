# 图神经网络-Beginner
选择一种框架完成如下任务

参考：
- [DGL框架](https://www.dgl.ai/)
- [Pytorch_geometric框架](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [子图训练](https://docs.dgl.ai/tutorials/large/L0_neighbor_sampling_overview.html#sphx-glr-tutorials-large-l0-neighbor-sampling-overview-py)

****
## 任务一、 节点分类
实现基于GNN主流模型(GCN, GAT, GraphSAGE, GIN)的节点分类:

1. 实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图上的节点分类任务
2. 使用Cora, Citeseer, Flickr数据集
3. 测试GCN, GAT, GraphSAGE, GIN模型
4. 利用框架自带的Sampler采样子图进行训练，并与全图训练进行性能和运行时间的对比

## 任务二、 图上的链路预测
1. 实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图上的链路预测
2. 使用Cora, Citeseer, Flickr数据集
3. 测试GCN, GAT, GraphSAGE, GIN模型
4. 利用框架自带的Sampler采样子图进行训练，并与全图训练进行性能和运行时间的对比

## 任务三、 图分类
1. 实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图分类任务
2. 使用TUDataset, ZINC数据集
3. 分析不同的池化方法对图分类性能的影响(AvgPooling, MaxPooling, MinPooling)
4. 测试GCN, GAT, GraphSAGE, GIN模型


## 任务四、 知识图谱
1. 参考
   1. 训练和测试框架[KGE框架](https://github.com/Maxioo/kge_framework)
   2. TransE, RotatE, ConvE的论文
2. 实现要求：基于参考资料和知识图谱补全框架，支持常见模型(TransE, RotatE, ConvE)
3. 需要了解的知识点：
   1. 常见知识图谱补全模型的原理(TransE, RotatE, ConvE)
   2. 数据集：训练集/验证集/测试集的划分

## 任务要求
1. 需要了解的知识点：
   1. GNN主流模型的原理(GCN, GAT, GraphSAGE, GIN, ...)
   2. 数据集：训练集/验证集/测试集的划分
2. 代码要求

   代码文件夹按统一格式:
      - 任务一
         - data
         - code
         - README.md
      - 任务二
         - data
         - code
         - README.md
      - ...
      - requirements.txt (运行环境文件)
   
   其中README.md中需要写明训练和测试的脚本

3. 报告要求
   1. 分析不同参数（学习率、网络层数）和不同的神经网络对性能的影响
   2. 测试全图训练和分批次训练对模型性能和运行时间的影响(任务四不需要)

---
---

# 项目完成情况

四个任务均已实现并**实际跑通实验**，结果保存在各任务的 `results/` 目录，
分析结论汇总在根目录的 [`REPORT.md`](REPORT.md)。

## 目录结构

```
graph_beginner-main/
├── README.md                        本文件（任务书 + 完成情况）
├── REPORT.md                        总报告：四个任务的实验分析与结论
├── requirements.txt                 运行环境文件
├── build_report_tables.py           从各任务 results/ 读取 CSV 并生成报告表格
│                                    （REPORT.md 与各 README 中的表格由它生成，
│                                     保证文档里的数字与实验结果一致）
│
├── task1_node_classification/       任务一：节点分类
│   ├── data/                        数据集（由 prepare_data.py 自动下载）
│   ├── code/
│   │   ├── prepare_data.py          下载/校验 Cora、Citeseer、Flickr
│   │   ├── dataset.py               数据集加载、特征归一化、划分说明
│   │   ├── models.py                GCN / GAT / GraphSAGE / GIN 编码器
│   │   ├── train.py                 训练+测试入口（全图 / 采样子图）
│   │   ├── run_experiments.py       批量实验编排（主实验 + 超参 + 采样消融）
│   │   ├── summarize.py             把 all_results.csv 汇总成 markdown 表格
│   │   └── utils.py                 随机种子、CUDA 计时器、评价指标
│   ├── results/                     all_results.csv / detail/*.json / summary.md
│   └── README.md                    任务一说明 + 训练/测试脚本 + 结果表格
│
├── task2_link_prediction/           任务二：链路预测
│   ├── data/  code/  results/  README.md     （结构同任务一）
│
├── task3_graph_classification/      任务三：图分类
│   ├── data/  code/  results/  README.md
│
└── task4_knowledge_graph/           任务四：知识图谱补全
    ├── data/  code/  results/  README.md
```

> 注：README 原文用「任务一/二/三/四」命名文件夹，这里改用语义化的英文目录名
> （`task1_node_classification` 等），子目录格式仍严格遵循要求：
> 每个任务下都有 `data/`、`code/`、`README.md`。

## 框架选择

要求中允许在 **DGL** 与 **PyTorch Geometric (PyG)** 中二选一，本项目选择 **PyG 2.8**。
原因：实验机为 Python 3.13，DGL 尚未提供该版本的官方 wheel，而 PyG 2.8 +
`pyg-lib` 已完整支持，并且 PyG 自带 `NeighborLoader` / `LinkNeighborLoader`
等子图采样器，正好满足「利用框架自带的 Sampler 采样子图进行训练」的要求。

## 环境安装

```bash
pip install -r requirements.txt
# 采样器所需的 C++ 扩展必须匹配 torch/CUDA 版本，用 PyG 官方 wheel 源安装：
pip install pyg-lib torch-sparse torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
```

## 快速开始

每个任务都可以独立运行，例如任务一：

```bash
cd task1_node_classification/code
python prepare_data.py                                   # 下载数据集
python train.py --dataset Cora --model gcn --mode full   # 全图训练一次
python train.py --dataset Cora --model gcn --mode sample # 采样子图训练一次
python run_experiments.py --experiments main             # 跑完整对比实验
python summarize.py                                      # 生成结果表格
```

任务四（知识图谱补全）：

```bash
cd task4_knowledge_graph/code
python prepare_data.py --verify                                      # 校验数据
python train.py --dataset WN18RR --model TransE                      # 单个模型
python train.py --dataset WN18RR --model RotatE --dim 200            # 换模型/维度
python run_experiments.py --experiments main --datasets WN18RR       # 三模型对比
```

## ⚠️ 关于运行时间对比的一个前提

作业要求「测试全图训练和分批次训练对模型性能和运行时间的影响」。
**请确保同一时刻只有一个 GPU 任务在运行** —— 两个 CUDA 进程并行会互相拖慢，
墙钟时间会被严重污染，而且误差**不能靠多跑几次平均掉**。

本项目就踩过这个坑：第一轮实验时后台还有其它任务，
Flickr GCN 全图的训练时间被记成 213 秒（独占 GPU 时只有 **46.5 秒**），
一度得出「采样比全图快 1.95 倍」的结论；干净重跑后实际是 **1.01 倍（基本持平）**。
各任务的复现步骤里都注明了这一点，详细记录见 `REPORT.md` 第 6.11 节。

各任务 README.md 中有更详细的训练/测试脚本说明。
