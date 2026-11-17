# ADRN

**Overlap-Aware Deep Probabilistic Reconstruction Network**  
面向不平衡金融数据分类的重叠感知深度概率重构网络：根据用户提供的论文独立实现与实验。

> 本项目是独立复现，不是论文作者官方代码。本文实验基于实际训练和独立测试；未使用论文报告值填充运行结果。论文部分超参数与预处理细节未披露，本实现的设置见下文，实验结果不保证与原文相同。

## 方法

编码器采用 `Linear → GELU → Linear → LayerNorm`，利用两类可学习各向同性高斯混合基元构建观测概率；随后依次应用：

- **RPDB**：基于 Bhattacharyya 系数计算跨类冲突与类内冗余，形成可靠度加权结构概率。
- **MPSDR**：提取结构概率高于观测概率的非负差值，作为少数类支持缺口。
- **OARG**：利用两类基元责任度和跨类重叠构造局部门控，调节补偿幅度。

训练执行编码器监督预训练、按真实类别 KMeans 初始化基元、端到端联合优化；验证集 PR-AUC（Average Precision）选择检查点，并在验证集最大化 F2 确定阈值，再在独立测试集上评估。预处理仅在训练集拟合，`duration`、`ID` 和标签均不作为输入。

## 实际实验结果

下表是本仓库运行得到的独立测试集结果，固定一次 8:1:1 分层划分，种子 0–4 独立初始化，报告均值 ± 样本标准差；PR-AUC 按 Average Precision 计算。

| 数据集 | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| GMSC | 0.2708 ± 0.0220 | 0.6273 ± 0.0262 | 0.3773 ± 0.0159 | 0.4952 ± 0.0021 | 0.8337 ± 0.0012 | 0.3665 ± 0.0053 |
| Bank | 0.3247 ± 0.0579 | 0.6355 ± 0.0572 | 0.4244 ± 0.0387 | 0.5268 ± 0.0127 | 0.7925 ± 0.0037 | 0.4481 ± 0.0115 |
| Default | 0.3079 ± 0.0092 | 0.8015 ± 0.0302 | 0.4446 ± 0.0087 | 0.6065 ± 0.0121 | 0.7505 ± 0.0078 | 0.5243 ± 0.0062 |

![三组数据集测试结果](figures/test_metrics.png)

以下每张图展示种子 0 的测试集 PR/ROC 曲线、验证集训练曲线和按验证集阈值计算的测试集混淆矩阵：

![GMSC 实验图](figures/gmsc_evaluation.png)
![Bank 实验图](figures/bank_evaluation.png)
![Default 实验图](figures/default_evaluation.png)

## TPE 搜索与横向基线

每个数据集进行 5 次候选搜索（前 2 次为 TPE 启动采样），搜索空间为隐层 `{32,64}`、潜在维数 `{8,16}`、每类基元数 `{4,8}`、学习率 `[0.0005,0.002]`（对数采样）和采样指数 β `[0.25,0.75]`。只按验证集 AP 选配置，再对固定配置执行五次独立初始化并在测试集评估；搜索过程及各次结果存于 `results/*_tpe.json`。这是便于复现的小规模搜索，不能视为原文未公开搜索范围的等价实现。

| 数据集 | TPE 后 F1 | TPE 后 F2 | TPE 后 PR-AUC |
|---|---:|---:|---:|
| GMSC | 0.3673 ± 0.0128 | 0.4941 ± 0.0035 | 0.3690 ± 0.0030 |
| Bank | 0.4193 ± 0.0310 | 0.5332 ± 0.0148 | 0.4572 ± 0.0111 |
| Default | 0.4497 ± 0.0105 | 0.6074 ± 0.0064 | 0.5220 ± 0.0164 |

另以相同划分、验证集 F2 阈值和五次运行比较下列基线；这是自行指定配置的独立实现，无法保证与论文基线配置完全相同。MLP 和 Focal MLP 的隐藏层为 64/16、均使用 β=0.5 抽样，Focal 设置 α=0.25、γ=2；XGBoost 为 150 棵树、最大深度 4；SMOTE 仅用于训练集，但对独热编码列也做插值，可能产生非整数的类别表示。

| 数据集 | 基线 | F1 | F2 | PR-AUC |
|---|---|---:|---:|---:|
| GMSC | MLP | 0.3794 ± 0.0061 | 0.4967 ± 0.0023 | 0.3685 ± 0.0017 |
| GMSC | Focal MLP | 0.3791 ± 0.0123 | 0.4926 ± 0.0030 | 0.3664 ± 0.0020 |
| GMSC | XGBoost | 0.4152 ± 0.0072 | 0.5241 ± 0.0044 | 0.4066 ± 0.0019 |
| GMSC | SMOTE + XGBoost | 0.3632 ± 0.0058 | 0.5067 ± 0.0035 | 0.3738 ± 0.0023 |
| BANK | MLP | 0.4344 ± 0.0260 | 0.5254 ± 0.0055 | 0.4510 ± 0.0070 |
| BANK | Focal MLP | 0.4004 ± 0.0128 | 0.5192 ± 0.0089 | 0.4453 ± 0.0045 |
| BANK | XGBoost | 0.4530 ± 0.0114 | 0.5561 ± 0.0060 | 0.4867 ± 0.0019 |
| BANK | SMOTE + XGBoost | 0.3759 ± 0.0065 | 0.5174 ± 0.0031 | 0.4442 ± 0.0026 |
| DEFAULT | MLP | 0.4371 ± 0.0120 | 0.6172 ± 0.0043 | 0.5330 ± 0.0054 |
| DEFAULT | Focal MLP | 0.4415 ± 0.0158 | 0.6116 ± 0.0046 | 0.5290 ± 0.0036 |
| DEFAULT | XGBoost | 0.4571 ± 0.0072 | 0.6226 ± 0.0045 | 0.5483 ± 0.0021 |
| DEFAULT | SMOTE + XGBoost | 0.4473 ± 0.0042 | 0.6183 ± 0.0035 | 0.5455 ± 0.0029 |

本次 XGBoost 在三组数据的 F1、F2、PR-AUC 上均高于本次固定配置的 ADRN；因此本次结果不支持 ADRN 在这些条件下全面优于该基线。图中同时保留固定配置和 TPE 配置，TPE 配置并未根据测试集结果选择。

![GMSC 对比](figures/gmsc_comparison.png)
![Bank 对比](figures/bank_comparison.png)
![Default 对比](figures/default_comparison.png)

## Bank 基元数量及不平衡率实验

以下为种子 0 的单次测试。K 敏感性使用 4、6、8、10、12，预训练 4 轮、联合训练 12 轮，其余配置与原固定配置相同；不平衡率实验仅对训练集少数类无放回抽样，保留训练集多数类，目标 IR 为 10、50、100，预处理在抽样后的训练集重新拟合，验证集和测试集划分保持不变，K=8。由于只运行一次且训练轮数与主实验不同，仅用于观察趋势。

![Bank 基元数量](figures/bank_k_sensitivity.png)
![Bank 不平衡率](figures/bank_ir_sensitivity.png)

完整数值及实际类别比例见 `results/sensitivity.json`。

## 重构机制分析

以下为固定配置、种子 0 模型在独立测试集的内部量，不用于选择模型。散点图显示观测概率与结构概率，直方图显示最终补偿，右图显示局部重叠与门控的对应关系；散点为固定随机种子抽取的至多 2500 个样本。

| 数据集 | 测试正类数 | 正类中存在正缺口的样本 | 平均门控值 | 平均概率补偿 |
|---|---:|---:|---:|---:|
| GMSC | 1003 | 607 | 0.1887 | 0.0162 |
| Bank | 529 | 422 | 0.5891 | 0.1275 |
| Default | 663 | 381 | 0.4133 | 0.0938 |

![GMSC 机制分析](figures/gmsc_mechanism.png)
![Bank 机制分析](figures/bank_mechanism.png)
![Default 机制分析](figures/default_mechanism.png)

运行 `python -m experiments.mechanism` 需要先运行主实验生成本地种子 0 检查点。

## 五次消融实验

依据论文设置四种结构变体，沿用固定配置、相同划分及种子 0–4；下表为测试集均值 ± 样本标准差。`w/o Deficit` 使用有符号残差，`w/o Latent` 跳过编码器预训练。各方案在同一固定测试集上重复训练，本次结果中部分消融均值高于完整模型，不能据此声称各模块都带来提升。

| 数据集 | 方案 | F1 | F2 | PR-AUC |
|---|---|---:|---:|---:|
| GMSC | 完整模型 | 0.3773 ± 0.0159 | 0.4952 ± 0.0021 | 0.3665 ± 0.0053 |
| GMSC | w/o Reliability | 0.3754 ± 0.0101 | 0.4934 ± 0.0040 | 0.3641 ± 0.0019 |
| GMSC | w/o Deficit | 0.3681 ± 0.0124 | 0.4915 ± 0.0018 | 0.3623 ± 0.0079 |
| GMSC | w/o Gate | 0.3691 ± 0.0112 | 0.4937 ± 0.0019 | 0.3638 ± 0.0016 |
| GMSC | w/o Latent | 0.3666 ± 0.0105 | 0.4778 ± 0.0052 | 0.3489 ± 0.0051 |
| BANK | 完整模型 | 0.4244 ± 0.0387 | 0.5268 ± 0.0127 | 0.4481 ± 0.0115 |
| BANK | w/o Reliability | 0.4300 ± 0.0344 | 0.5274 ± 0.0100 | 0.4426 ± 0.0103 |
| BANK | w/o Deficit | 0.4347 ± 0.0258 | 0.5341 ± 0.0088 | 0.4492 ± 0.0080 |
| BANK | w/o Gate | 0.4285 ± 0.0310 | 0.5330 ± 0.0062 | 0.4481 ± 0.0088 |
| BANK | w/o Latent | 0.3889 ± 0.0323 | 0.4916 ± 0.0172 | 0.3473 ± 0.0241 |
| DEFAULT | 完整模型 | 0.4446 ± 0.0087 | 0.6065 ± 0.0121 | 0.5243 ± 0.0062 |
| DEFAULT | w/o Reliability | 0.4468 ± 0.0132 | 0.6148 ± 0.0052 | 0.5280 ± 0.0045 |
| DEFAULT | w/o Deficit | 0.4480 ± 0.0145 | 0.6090 ± 0.0115 | 0.5275 ± 0.0038 |
| DEFAULT | w/o Gate | 0.4479 ± 0.0096 | 0.6083 ± 0.0061 | 0.5241 ± 0.0055 |
| DEFAULT | w/o Latent | 0.4261 ± 0.0054 | 0.6082 ± 0.0056 | 0.4789 ± 0.0009 |

![GMSC 五次消融](figures/gmsc_ablation_means.png)
![Bank 五次消融](figures/bank_ablation_means.png)
![Default 五次消融](figures/default_ablation_means.png)

## Welch 检验与多重比较

固定配置 ADRN 与四类基线、四种消融的五次测试 F1、F2、PR-AUC 分别采用双侧 Welch t 检验；横向对比与消融各自形成 **36 项**比较，各自进行 Benjamini–Hochberg 调整，全部原始 p 值和调整值见 `results/significance.json`。横向 36 项中 8 项、消融 36 项中 6 项调整后 p < 0.05；显著性本身不表明 ADRN 占优，方向需结合实际均值判断。

**统计限制：**五次结果共享同一训练、验证和测试划分，且每组仅 5 次，Welch 检验不能反映跨划分波动，其独立性与正态性假设也较难充分验证；这些 p 值只作为固定划分下的探索结果，不应外推为稳健的统计结论。论文横向比较含 7 个基线、63 项，本项目仅有 4 个基线，故校正范围与论文不同。

## 实验约定与差异

- 三组数据分别为 [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)、[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) 和 [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)。Bank 剔除 `duration`，Default 剔除 `ID`。本实现分别得到 10、50、33 个特征；Default 与论文表格的 29 维不同，原因包括编码方案未完全明确。
- 划分随机种子 42；数值缺失用训练集中位数填补及标准化，类别缺失用训练集众数填补并独热编码。Default 的 SEX、EDUCATION、MARRIAGE 按类别编码。GMSC 实验使用 Kaggle 数据的公开副本；数据文件未提交至仓库。
- 本次使用 CPU、Adam、学习率 0.001、批大小 512、隐层 64、潜在维数 16、每类 4 个基元、采样指数 β=0.5、预训练 3 轮、联合训练 6 轮、梯度裁剪 5。论文未完整给出搜索空间和每项训练配置，本实验使用明确固定的近似配置，并补充了 5 次候选的 TPE 搜索；这一小规模搜索不等同于论文未公开的完整搜索配置。
- 五次运行共享同一划分，标准差反映随机初始化和采样波动，不代表跨数据划分的泛化波动。另行实现四类横向基线与 Bank 单次 K/不平衡率实验；SCL-TPE、NDU+XGBoost 和 CISNN 尚未实现。
- 四种消融已各完成五次，检验使用本项目的固定配置与四个已实现基线；论文其余三类基线仍待实现。

## 运行

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.download_uci
# 从 Kaggle 下载 GMSC 的 cs-training.csv 放入 data/raw/
python -m unittest discover -s tests -v
python -m experiments.run --seeds 0 1 2 3 4
python -m experiments.tune --trials 5 --seeds 0 1 2 3 4
python -m experiments.baselines --seeds 0 1 2 3 4
python -m experiments.sensitivity
python -m experiments.ablation --seeds 0 1 2 3 4
python -m experiments.plot_summary
python -m experiments.plot_extra
python -m experiments.plot_ablation_means
python -m experiments.mechanism
python -m experiments.significance
```

运行产物写入 `results/`、`figures/` 和本地 `checkpoints/`。检查点目前只包含模型权重和参数，不包含预处理器，不应直接作为完整推理包使用。原始数据与检查点默认不上传。
