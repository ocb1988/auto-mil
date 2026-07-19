# 计算病理学多实例学习方法综述

## 研究范围与证据边界

本综述基于提供的 61 份逐篇方法摘要，系统梳理了计算病理学（Computational Pathology, CPath）中多实例学习（Multiple Instance Learning, MIL）的技术演进。研究范围涵盖从早期的注意力池化、RNN聚合到近期的Transformer、图神经网络（GNN）、状态空间模型（SSM/Mamba）及基础模型微调等主流架构。

**证据边界说明：**
1.  **数据异常处理**：`08_REMIX...` 为DFT材料科学论文，虽在索引中保留链接，但不纳入MIL方法比较；`09_S4_MIL...` 与 `27_MAMBA_MIL...` 均指向 MambaMIL，合并论述；`14_IIB_MIL...` 为受限版，以完整版 `IIB_MIL.md` 为准并双链引用；`48_INTER_MIL...` 为 MedIA 期刊扩展版，不冒充 MICCAI 原始章节。
2.  **公平性声明**：不同方法在编码器预训练权重、数据集划分（如 TCGA vs. 自建集）及评价指标上存在差异，本文仅依据摘要事实进行定性归纳，不进行跨论文的定量排名。

## 技术演进与分类框架

计算病理学 MIL 方法可大致划分为以下四个技术阶段/流派：
1.  **经典注意力与聚类驱动**：以 AB-MIL 和 CLAM 为代表，利用注意力机制实现白盒定位，或通过伪标签聚类提升判别力。
2.  **上下文感知与图结构建模**：引入空间拓扑信息，通过 GCN、动态图或知识图谱捕捉实例间的局部或全局依赖。
3.  **长序列高效建模（Transformer & SSM）**：针对 WSI 海量 Patch 带来的 $O(N^2)$ 复杂度瓶颈，发展出 Nystrom、FlashAttention、ALiBi 以及线性复杂度的 S4/Mamba 架构。
4.  **鲁棒性增强与基础模型融合**：结合对比学习、反事实推理、域适应及大模型重嵌入，解决噪声标签、分布偏移及小样本问题。

## 逐方法创新点

### 1. 经典注意力与聚类驱动范式

*   **AB-MIL (Attention-based Deep Multiple Instance Learning)**
    提出基于可训练注意力机制的集合聚合操作，证明了其符合对称函数分解定理以保证置换不变性，实现了从黑盒分类到关键实例白盒定位的跨越 [详细解读](method_summaries/01_AB_MIL_Attention-based%20Deep%20Multiple%20Instance%20Learning.md)。

*   **CLAM (Data Efficient and Weakly Supervised Computational Pathology on WSI)**
    并行分支设计支持多分类，并通过选取高/低注意力分数的补丁生成伪标签，结合 Smooth SVM 损失进行实例级聚类约束，显著提升了弱监督下的数据效率和特征判别力 [详细解读](method_summaries/06_CLAM_MIL_Data%20Efficient%20and%20Weakly%20Supervised%20Computational%20Pathology%20on%20WSI.md)。

*   **DTFD-MIL (Double-Tier Feature Distillation MIL)**
    首次推导并证明在 AB-MIL 范式下获取 Instance 级概率的方法，指出该概率比 Attention Score 更能准确定位阳性区域，并结合双层 MIL 架构进行特征蒸馏以缓解过拟合 [详细解读](method_summaries/11_DTFD_MIL_Double-Tier%20Feature%20Distillation%20MIL%20for%20Histopathology%20WSI%20Classification.md)。

*   **ADD-MIL (Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology)**
    将 MIL 预测头重构为实例特征的线性加和，实现了无需后处理的、与 Shapley 值等价的内在空间解释性，兼顾高精度与逐类贡献可视化 [详细解读](method_summaries/12_ADD_MIL_Additive%20MIL_%20Intrinsically%20Interpretable%20MIL%20for%20Pathology.md)。

*   **IIB-MIL (Integrated Instance-Level and Bag-Level MIL with Label Disambiguation)**
    协同整合实例级与袋级监督，利用动量更新的原型和置信度银行动态生成软标签以消除实例级伪标签噪声，实现更精准的病理分析 [详细解读](method_summaries/14_IIB_MIL_Integrated%20instance-level%20and%20bag-level%20MIL%20with%20label%20disambiguation.md) [详细解读](method_summaries/IIB_MIL.md)。

*   **ILRA-MIL (Exploring Low-Rank Property in Multiple Instance Learning)**
    提出无需标签的低秩约束损失（LRC），模拟低秩子空间结构以替代无标签场景下的对比学习；并将全局注意力转化为迭代低秩交叉注意力，以线性复杂度建模大规模跨实例相关性 [详细解读](method_summaries/ILRA_MIL.md)。

*   **FR-MIL / FR-MIL++ (Distribution Re-calibration based MIL)**
    通过基于关键实例的特征重校准和单次自注意力池化解决分布偏移；FR-MIL++ 进一步引入 VQ-VAE 对关键实例进行离散潜变量建模，替代显式幅度损失以提升稳定性 [详细解读](method_summaries/FR_MIL.md)。

*   **CAMIL (Context-Aware Multiple Instance Learning for Cancer Detection)**
    结合 Nystromformer 捕获全局上下文，并利用基于特征相似度的动态邻接掩码约束注意力计算，从而同时捕捉全局语义与局部生物拓扑结构 [详细解读](method_summaries/25_CA_MIL_Context-Aware%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md)。

*   **ACMIL (Attention-Challenging Multiple Instance Learning)**
    引入多分支注意力机制配合多样性正则化以捕捉多样化判别模式，并结合随机 Top-K 实例掩码（STKIM）抑制注意力过度集中，有效缓解过拟合 [详细解读](method_summaries/26_AC_MIL_Attention-Challenging%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md)。

*   **AEM (Attention Entropy Maximization)**
    通过引入带有余弦退火调度的注意力负熵正则化，以极简方式缓解注意力过度集中和过拟合问题，显著提升泛化性能 [详细解读](method_summaries/38_AEM_MIL_Attention%20Entropy%20Maximization%20for%20MIL%20based%20WSI%20Classification.md)。

*   **TDA-MIL (Top-Down Attention-based MIL)**
    引入可学习的任务相关 Token 和特征选择模块，在两阶段自注意力过程中实现“自上而下”的任务聚焦，提升对细微病灶的识别精度 [详细解读](method_summaries/40_TDA_MIL_Top-Down%20Attention-based%20Multiple%20Instance%20Learning%20for%20Whole%20Slide%20Image%20Analysis.md)。

*   **PSA-MIL (Probabilistic Spatial Attention-Based MIL)**
    构建基于可学习距离衰减先验的概率自注意力机制，放松标准 Q/K L2 归一化假设，结合动态空间剪枝实现高效高精度的空间上下文建模 [详细解读](method_summaries/41_PSA_MIL_Probabilistic%20Spatial%20Attention-Based%20MIL%20for%20Whole%20Slide%20Image%20Classification.md)。

*   **MiCo (Multiple Instance Learning with Context-Aware Clustering)**
    引入可学习的语义锚点，利用 Cluster Route 聚合跨区域相似实例以增强上下文感知，并通过 Cluster Reducer 精简锚点消除语义冗余，解决空间异质性 [详细解读](method_summaries/39_MICO_MIL_Multiple%20Instance%20Learning%20with%20Context-Aware%20Clustering.md)。

### 2. 上下文感知与图/超图结构建模

*   **Patch-GCN (Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks)**
    将 WSI 建模为基于物理邻近性的 2D 点云图，利用带残差和密集连接的图卷积网络聚合局部空间上下文，提升生存预测性能 [详细解读](method_summaries/07_PGCN_MIL_Context-Aware%20Survival%20Prediction%20using%20Patch-based%20Graph%20Convolutional%20Networks.md)。

*   **WiKG (Dynamic Graph Representation with Knowledge-aware Attention)**
    通过可学习的 Head/Tail 嵌入动态构建有向知识图，利用三元组注意力机制聚合邻居信息，克服传统方法中空间拓扑受限和实例交互缺失的问题 [详细解读](method_summaries/18_WIKG_MIL_Dynamic%20Graph%20Representation%20with%20Knowledge-aware%20Attention%20for%20WSI%20Analysis.md)。

*   **AMD-MIL (Agent Aggregator with Mask Denoise Mechanism)**
    引入可训练 Agent 矩阵替代传统池化 Agent，并结合基于 Value 投影的动态掩码去噪机制，实现高效线性复杂度的特征聚合 [详细解读](method_summaries/19_AMD_MIL_Agent%20Aggregator%20with%20Mask%20Denoise%20Mechanism%20for%20Histopathology%20WSI%20Analysis.md)。

*   **MicroMIL (Graph-Based MIL for Context-Aware Diagnosis)**
    将在线深度聚类与可微分的 Hard Gumbel-Softmax 结合，提取代表性图像并构建稀疏图进行消息传递，解决显微镜图像缺乏空间坐标和高冗余的问题 [详细解读](method_summaries/33_MICRO_MIL_Graph-Based%20MIL%20for%20Context-Aware%20Diagnosis%20with%20Microscopic%20Images.md)。

*   **DyHG (Dynamic Hypergraph Representation for Bone Metastasis)**
    通过低秩策略和 Gumbel-Softmax 采样动态构建超图，捕捉 Patch 间的高阶生物关联，并在骨转移癌症分析中取得 SOTA 性能 [详细解读](method_summaries/34_DYHG_MIL_Dynamic%20Hypergraph%20Representation%20for%20Bone%20Metastasis%20Cancer%20Analysis.md)。

*   **DAG (Deformable Attention Graph Representation Learning)**
    引入基于绝对坐标和可学习偏移的动态邻居采样机制，构建能自适应聚焦形态学相关区域的加权图，提升复杂空间结构的建模能力 [详细解读](method_summaries/44_DAG_MIL_Deformable%20attention%20graph%20representation%20learning%20for%20histopathology%20WSI%20analysis.md)。

*   **GDF-MIL (Rethinking Multi-Instance Learning through Graph-Driven Fusion)**
    通过 Gumbel-Softmax 软聚类压缩包表示，在紧凑表示上构建稀疏动态图，并利用双路径门控机制自适应融合局部上下文与残差特征 [详细解读](method_summaries/43_GDF_MIL_Rethinking%20Multi-Instance%20Learning%20through%20Graph-Driven%20Fusion.md)。

### 3. 长序列高效建模：Transformer 变体与状态空间模型 (SSM)

*   **TransMIL (Transformer based Correlated Multiple Instance Learning)**
    构建相关 MIL 框架，利用 Nystrom 近似 Transformer 和金字塔位置编码（PPEG），解决长序列计算复杂度高的问题，实现高精度与强可解释性 [详细解读](method_summaries/04_TRANS_MIL_Transformer%20based%20Correlated%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md)。

*   **Long-MIL (Scaling Long Contextual MIL)**
    引入基于 2D 欧氏距离的线性偏置（2d-ALiBi）和 FlashAttention，实现具备长度外推能力的长上下文 WSI 分析，提升形状变化大的图像分类性能 [详细解读](method_summaries/22_LONG_MIL_Scaling%20Long%20Contextual%20MIL%20for%20Histopathology%20WSI%20Analysis.md)。

*   **RetMIL (Retentive Multiple Instance Learning)**
    引入线性保留机制和分层聚合架构，解决 Transformer 面临的长序列高计算成本问题，在保持 SOTA 精度的同时实现高效推理 [详细解读](method_summaries/28_RET_MIL_Retentive%20Multiple%20Instance%20Learning%20for%20Histopathological%20WSI%20Classification.md)。

*   **S4-MIL (Structured State Space Models for MIL in Digital Pathology)**
    将对角化状态空间模型（S4D）直接嵌入 MIL 作为序列编码器，通过线性卷积高效聚合病理图像块序列，显著降低计算复杂度 [详细解读](method_summaries/58_STATE_SPACE_MIL_Structured%20state%20space%20models%20for%20MIL%20in%20digital%20pathology.md)。

*   **MambaMIL (Enhancing Long Sequence Modeling with Sequence Reordering)**
    合并 `09_S4_MIL` 与 `27_MAMBA_MIL` 内容。设计 SR-Mamba 模块，包含感知实例分布的序列重排序、双分支 SSM 建模及门控融合，在保持线性复杂度的同时解决长序列全局依赖建模和过拟合问题 [详细解读](method_summaries/09_S4_MIL_Efficiently%20Modeling%20Long%20Sequences%20with%20Structured%20State%20Spaces.md) [详细解读](method_summaries/27_MAMBA_MIL_Enhancing%20Long%20Sequence%20Modeling%20with%20Sequence%20Reordering%20in%20CPath.md)。

*   **2DMamba (Efficient State Space Model for Image Representation)**
    创新 2D 选择性扫描算法和硬件感知 CUDA 算子，在保持线性复杂度的同时保留图像的 2D 空间连续性，提升 WSI 分析效率 [详细解读](method_summaries/36_MAMBA2D_MIL_2DMamba_%20Efficient%20State%20Space%20Model%20for%20Image%20Representation.md)。

*   **MoMIL (Multi-order enhanced multiple instance learning)**
    构建原始、翻转和转置三种序列顺序，利用 SSD 模型高效捕捉 WSI 的多方向空间上下文，并通过轻量级融合模块实现高性能分析 [详细解读](method_summaries/MO_MIL.md)。

*   **MSMMIL (Multi-scan Mamba-based Multiple Instance Learning)**
    引入 Grid 和 Layer 两种互补的多向扫描策略以及轻量级 GCA 块，在保持 Mamba 线性复杂度的前提下提升判别性特征提取能力 [详细解读](method_summaries/MSM_MIL.md)。

*   **StableMIL (Entropy-Stabilized Attention-based MIL)**
    提出熵稳定注意力机制，从信息论角度证明能将最大熵波动控制在常数级别；结合随机投影 2D 旋转位置编码（RPRoPE）消除坐标各向异性，解决注意力坍塌和泛化差问题 [详细解读](method_summaries/Stable_MIL.md)。

*   **FourierMIL (Fourier Filtering-based Multiple Instance Learning)**
    引入自适应令牌填充和全通频域滤波（APFF）模块，在频率域实现高效且保留高频细节的令牌混合，解决 Transformer 成本高和细粒度特征丢失问题 [详细解读](method_summaries/37_FOURIER_MIL_Fourier%20filtering-based%20multiple%20instance%20learning%20for%20whole%20slide%20image%20analysis.md)。

### 4. 鲁棒性增强、数据增强与基础模型融合

*   **DSMIL (Dual-stream MIL Network for WSI Classification with SSL Contrastive Learning)**
    首次将 SimCLR 应用于 WSI 分析，通过双流架构（Max-pooling 识别关键实例 + 距离加权注意力）和自监督对比学习，缓解正负样本不平衡及大 Bag 训练困难 [详细解读](method_summaries/05_DS_MIL_Dual-stream%20MIL%20Network%20for%20WSI%20Classification%20with%20SSL%20Contrastive%20Learning.md)。

*   **DGMIL (Distribution Guided Multiple Instance Learning)**
    通过自监督 MAE 初始化特征，利用负样本聚类和马氏距离生成伪标签，迭代训练线性投影头以精炼特征空间分布，实现高性能弱监督分类 [详细解读](method_summaries/10_DG_MIL_Distribution%20Guided%20Multiple%20Instance%20Learning%20for%20Whole%20Slide%20Image%20Classification.md)。

*   **IBMIL (Interventional Bag Multi-Instance Learning)**
    构建混淆因子字典并利用后门调整公式进行干预性训练，消除由数据集偏差引起的虚假相关性，提升模型鲁棒性 [详细解读](method_summaries/15_IB_MIL_Interventional%20Bag%20Multi-Instance%20Learning%20On%20Whole-Slide%20Pathological%20Images.md)。

*   **RankMix (Data Augmentation for Classifying WSIs with Diverse Sizes)**
    通过伪标签排序选取关键 Patch 并保持相对顺序，实现不同尺寸 WSI 特征的有效混合，结合两阶段自训练提升分类性能 [详细解读](method_summaries/16_RANKMIX_MIL_Data%20Augmentation%20for%20Classifying%20WSIs%20with%20Diverse%20Sizes.md)。

*   **MHIM-MIL (Masked Hard Instance Mining)**
    构建 EMA 驱动的 Teacher-Student 框架，利用 Teacher 注意力分数对高显著性实例进行掩码，隐式挖掘硬实例并施加一致性约束 [详细解读](method_summaries/17_MHIM_MIL_MIL%20Framework%20with%20Masked%20Hard%20Instance%20Mining%20for%20WSI%20Classification.md)。

*   **PseMix (Pseudo-Bag Mixup Augmentation)**
    采用“Bag 原型聚类 + 表型微调”策略划分伪袋，通过 Mask-based 混合和 r-mix 机制将 Mixup 扩展到不规则 WSI Bag，提升泛化能力 [详细解读](method_summaries/21_PSEBMIX_MIL_Pseudo-Bag%20Mixup%20Augmentation%20for%20MIL%20Based%20Whole%20Slide%20Image%20Classification.md)。

*   **DGR-MIL (Exploring Diverse Global Representation in MIL)**
    引入可学习的全局向量和基于 DPP 的正交性多样性损失，结合正例对齐机制，以线性复杂度有效建模 WSI 中实例的多样性 [详细解读](method_summaries/23_DGR_MIL_Exploring%20Diverse%20Global%20Representation%20in%20MIL%20for%20WSI%20Classification.md)。

*   **cDP-MIL (Robust Multiple Instance Learning via Cascaded Dirichlet Process)**
    利用 Stick-Breaking 过程参数化 DP，动态学习每个簇的均值和全协方差矩阵以刻画高阶特征分布；在幻灯片级利用贝叶斯非参数分类器实现鲁棒预测与不确定性量化 [详细解读](method_summaries/24_CDP_MIL_cDP-MIL_%20Robust%20Multiple%20Instance%20Learning%20via%20Cascaded%20Dirichlet%20Process.md)。

*   **SCMIL (Sparse Context-aware MIL for Predicting Cancer Survival)**
    通过可学习的软过滤机制去除噪声，利用融合形态与空间信息的稀疏自注意力捕获局部 patch 交互，结合队列统计先验预测个体化生存概率分布 [详细解读](method_summaries/29_SC_MIL_Sparse%20Context-aware%20MIL%20for%20Predicting%20Cancer%20Survival%20Probability%20Distribution%20in%20WSI.md)。

*   **NcIEMIL (Rethinking Decoupled MIL Framework)**
    将 Slide 级监督信号传递到 Instance 级进行去噪筛选，并行处理通道和空间注意力并通过交叉注意力融合，解决信息冗余和噪声干扰 [详细解读](method_summaries/30_NCIE_MIL_Rethinking%20Decoupled%20MIL%20Framework%20for%20Histopathological%20Slide%20Classification.md)。

*   **R²T (Feature Re-Embedding: Towards Foundation Model-Level Performance)**
    提出 Re-embedded Regional Transformer 模块，通过区域局部自注意力和跨区域全局交互，实现实例特征在线重嵌入，使传统 MIL 性能达到基础模型水平 [详细解读](method_summaries/31_RRT_MIL_Towards%20Foundation%20Model-Level%20Performance%20in%20Computational%20Pathology.md)。

*   **PAMIL (Dynamic Policy-Driven Adaptive Multi-Instance Learning)**
    引入基于强化学习的动态策略实例选择（DPIS）和融合历史信息的特征表示（SFFR），利用子袋预测和自引导奖励/惩罚机制协同优化，解决采样盲目性 [详细解读](method_summaries/32_PA_MIL_Dynamic%20Policy-Driven%20Adaptive%20Multi-Instance%20Learning%20for%20WSI%20Classification.md)。

*   **Inter-MIL (Self-interactive learning: Fusion and evolution of multi-scale histomorphology features)**
    迭代地利用 Slide 级注意力引导 Tile 级编码器微调，辅以对抗去噪和对比预训练，实现小样本病理图像中细粒度与全局特征的深度融合 [详细解读](method_summaries/48_INTER_MIL_Predicting%20molecular%20traits%20through%20self-interactive%20multi-instance%20learning.md)。

*   **LNPL-MIL (Learning from Noisy Pseudo Labels)**
    SP-LNPL 在特征空间聚类构建 Super Patch 并过滤假阳性；TOD-MIL 利用实例顺序和分布信息，通过 Bag 级语义引导注意力缓解标签歧义 [详细解读](method_summaries/49_LNPL_MIL_Learning%20from%20noisy%20pseudo%20labels%20for%20promoting%20MIL%20in%20WSI.md)。

*   **SMMILe (Accurate spatial quantification in computational pathology with MIL)**
    理论分析揭示 IAMIL 优势，利用 NIC 卷积、无参实例 Dropout、超块采样及 MRF 细化网络，解决召回率低问题，实现分类与高精度空间量化双重突破 [详细解读](method_summaries/50_SMMILE_MIL_Accurate%20spatial%20quantification%20in%20computational%20pathology%20with%20MIL.md)。

*   **WHOLE SLIDE IMAGES BASED CANCER SURVIVAL PREDICTION USING ATTENTION GUIDED DEEP MIL**
    结合表型聚类和注意力驱动的多实例学习，实现高效、可解释且高精度的全切片图像癌症生存预测 [详细解读](method_summaries/51_DEEPATTNMISL_MIL_WSI%20based%20cancer%20survival%20prediction%20using%20attention%20guided%20deep%20MIL.md)。

*   **MS-DA-MIL (Multi-scale Domain-adversarial Multiple-instance CNN)**
    结合 MIL、域对抗训练和多尺度分析，在两阶段训练中分别优化域不变特征提取器和多尺度 Bag 分类器，实现染色差异鲁棒的癌症亚型分类 [详细解读](method_summaries/52_MS_DA_MIL_Multi-scale%20domain-adversarial%20MIL%20CNN%20for%20cancer%20subtype%20classification.md)。

*   **Mixed Supervision Transformer (Gleason Grading)**
    结合随机掩码策略，同时利用 Slide 级和经 SLIC 处理的 Instance 级标签，克服标签噪声，实现高精度 Gleason 分级 [详细解读](method_summaries/53_MIXED_SUPERVISION_MIL_Multiple%20instance%20learning%20with%20mixed%20supervision%20in%20Gleason%20grading.md)。

*   **MDMIL (Targeting tumor heterogeneity multiplex-detection-based MIL)**
    IQGM 动态筛选高质量实例生成内部查询；MDM 融合内外查询实现鲁棒检测；基于记忆的对比损失强制类内紧凑和类间分离，应对肿瘤异质性 [详细解读](method_summaries/54_MDMIL_Targeting%20tumor%20heterogeneity%20multiplex-detection-based%20MIL.md)。

*   **RAM-MIL (Retrieval-Augmented Multiple Instance Learning)**
    利用注意力权重构建实例分布，使用最优传输距离检索最相似外部 Bag 进行特征融合，降低表征空间内在维度，提升域外泛化能力 [详细解读](method_summaries/55_RAM_MIL_Retrieval-augmented%20multiple%20instance%20learning.md)。

*   **VINO (Transformer-Based Video-Structure Multi-Instance Learning)**
    将 WSI 划分为空间相邻视频片段，利用参数共享 Transformer 和类特定 Token，实现兼具上下文感知能力和端到端训练效率的分类与定位 [详细解读](method_summaries/56_VINO_MIL_Transformer-based%20video-structure%20MIL%20for%20WSI%20classification.md)。

*   **Prov-GigaPath (A whole-slide foundation model for digital pathology)**
    结合 DINOv2 局部特征提取器和 LongNet 全局序列编码器，在大规模真实世界数据上实现 Gigapixel WSI 高效建模，取得多项 SOTA 性能 [详细解读](method_summaries/57_PROV_GIGAPATH_MIL_A%20whole-slide%20foundation%20model%20for%20digital%20pathology%20from%20real-world%20data.md)。

*   **CIMIL (Boosting MIL models based on counterfactual inference)**
    基于反事实推理的分层子包评估生成高精度伪标签，利用 Instance Classifier Embedding 作为 Prompt 细化特征，同步提升 Bag 和 Instance 预测性能 [详细解读](method_summaries/59_CIMIL_Boosting%20MIL%20models%20based%20on%20counterfactual%20inference.md)。

### 5. 早期奠基与特定任务方法

*   **Campanella et al. (Clinical-grade computational pathology using weakly supervised deep learning on WSI)**
    引入超参数 K 放松严格 MIL 假设，利用 RNN 整合 Top-K 图块特征捕捉协同效应，在超大规模无标注数据上实现临床级癌症检测 [详细解读](method_summaries/45_CAMPANELLA_MIL_Clinical-grade%20computational%20pathology%20using%20weakly%20supervised%20deep%20learning%20on%20WSI.md)。

*   **RMDL (Recalibrated multi-instance deep learning for whole slide gastric image classification)**
    两阶段框架：全卷积定位网络筛选高判别力 Patch，再通过融合局部全局特征的注意力机制重新校准实例权重，实现高精度胃病理分类 [详细解读](method_summaries/46_RMDL_Recalibrated%20multi-instance%20deep%20learning%20for%20whole%20slide%20gastric%20image%20classification.md)。

*   **CAMEL (Weakly supervised learning framework for histopathology image segmentation)**
    组合多实例学习（cMIL）自动从图像级标签生成高质量实例级标签，结合级联增强和约束机制，实现仅需图像级监督的高精度分割 [详细解读](method_summaries/47_CMIL_CAMEL_Weakly%20supervised%20learning%20framework%20for%20histopathology%20image%20segmentation.md)。

*   **DT-MIL (Deformable Transformer for Multi-instance Learning on Histopathological Image)**
    引入可变形 Transformer 编码器和解码器，在保留实例二维位置上下文的同时高效聚合特征，实现高性能弱监督分类 [详细解读](method_summaries/DT_MIL.md)。

*   **Mixup (Beyond Empirical Risk Minimization)**
    通过线性插值随机样本对及其标签构建虚拟训练数据，简单有效地正则化神经网络，提升泛化能力和鲁棒性 [详细解读](method_summaries/02_MIXUP_MIL_mixup_%20Beyond%20Empirical%20Risk%20Minimization.md)。

## 横向比较与发展趋势

1.  **从全局平均到精细定位**：早期方法如 AB-MIL 确立了注意力池化的地位，后续 CLAM、DTFD-MIL 等通过伪标签和概率推导，将定位精度提升至 Instance 级别。
2.  **从局部池化到全局/图上下文**：DSMIL、TransMIL 引入了全局或局部上下文约束，而 Patch-GCN、WiKG、DAG 等方法则显式建模实例间的拓扑关系，解决了无序集合忽略空间结构的问题。
3.  **计算效率的革命**：面对 WSI 的长序列特性，Transformer 的 $O(N^2)$ 瓶颈催生了 Nystrom、FlashAttention、ALiBi 等优化，进而推动了 S4/Mamba 等线性复杂度状态空间模型的兴起（S4-MIL, MambaMIL, 2DMamba）。
4.  **鲁棒性与泛化性**：随着基础模型（GigaPath）的出现，MIL 逐渐从纯监督转向半监督/自监督预训练后的微调。同时，反事实推理（CIMIL）、域适应（MS-DA-MIL）和数据增强（RankMix, PseMix）成为解决分布偏移和噪声的关键手段。

## 方法选择建议

*   **若关注可解释性与基线性能**：推荐参考 **AB-MIL** 和 **CLAM**，它们是许多后续工作的基准。
*   **若 WSI 尺寸极大且显存受限**：优先考虑 **MambaMIL**、**2DMamba** 或 **Long-MIL**，它们提供了线性或近线性的计算复杂度。
*   **若需捕捉复杂的组织空间结构**：**Patch-GCN**、**WiKG** 或 **DAG** 等图/超图方法是更好的选择。
*   **若数据存在严重噪声或分布偏移**：**IBMIL**（干预学习）、**CIMIL**（反事实）或 **MS-DA-MIL**（域适应）提供了针对性的解决方案。
*   **若追求极致精度且算力充足**：可尝试结合基础模型微调的 **Prov-GigaPath** 或 **R²T**。

## 文档覆盖索引

以下索引确保输入中的每个 link 至少出现一次，方便从总综述跳转到所有逐篇摘要：

*   [method_summaries/01_AB_MIL_Attention-based%20Deep%20Multiple%20Instance%20Learning.md](method_summaries/01_AB_MIL_Attention-based%20Deep%20Multiple%20Instance%20Learning.md)
*   [method_summaries/02_MIXUP_MIL_mixup_%20Beyond%20Empirical%20Risk%20Minimization.md](method_summaries/02_MIXUP_MIL_mixup_%20Beyond%20Empirical%20Risk%20Minimization.md)
*   [method_summaries/04_TRANS_MIL_Transformer%20based%20Correlated%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md](method_summaries/04_TRANS_MIL_Transformer%20based%20Correlated%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md)
*   [method_summaries/05_DS_MIL_Dual-stream%20MIL%20Network%20for%20WSI%20Classification%20with%20SSL%20Contrastive%20Learning.md](method_summaries/05_DS_MIL_Dual-stream%20MIL%20Network%20for%20WSI%20Classification%20with%20SSL%20Contrastive%20Learning.md)
*   [method_summaries/06_CLAM_MIL_Data%20Efficient%20and%20Weakly%20Supervised%20Computational%20Pathology%20on%20WSI.md](method_summaries/06_CLAM_MIL_Data%20Efficient%20and%20Weakly%20Supervised%20Computational%20Pathology%20on%20WSI.md)
*   [method_summaries/07_PGCN_MIL_Context-Aware%20Survival%20Prediction%20using%20Patch-based%20Graph%20Convolutional%20Networks.md](method_summaries/07_PGCN_MIL_Context-Aware%20Survival%20Prediction%20using%20Patch-based%20Graph%20Convolutional%20Networks.md)
*   [method_summaries/08_REMIX_MIL_A%20General%20and%20Efficient%20Framework%20for%20MIL%20based%20WSI%20Classification.md](method_summaries/08_REMIX_MIL_A%20General%20and%20Efficient%20Framework%20for%20MIL%20based%20WSI%20Classification.md) *(注：此为DFT材料科学论文，不纳入MIL比较)*
*   [method_summaries/09_S4_MIL_Efficiently%20Modeling%20Long%20Sequences%20with%20Structured%20State%20Spaces.md](method_summaries/09_S4_MIL_Efficiently%20Modeling%20Long%20Sequences%20with%20Structured%20State%20Spaces.md) *(注：与27合并为MambaMIL)*
*   [method_summaries/10_DG_MIL_Distribution%20Guided%20Multiple%20Instance%20Learning%20for%20Whole%20Slide%20Image%20Classification.md](method_summaries/10_DG_MIL_Distribution%20Guided%20Multiple%20Instance%20Learning%20for%20Whole%20Slide%20Image%20Classification.md)
*   [method_summaries/11_DTFD_MIL_Double-Tier%20Feature%20Distillation%20MIL%20for%20Histopathology%20WSI%20Classification.md](method_summaries/11_DTFD_MIL_Double-Tier%20Feature%20Distillation%20MIL%20for%20Histopathology%20WSI%20Classification.md)
*   [method_summaries/12_ADD_MIL_Additive%20MIL_%20Intrinsically%20Interpretable%20MIL%20for%20Pathology.md](method_summaries/12_ADD_MIL_Additive%20MIL_%20Intrinsically%20Interpretable%20MIL%20for%20Pathology.md)
*   [method_summaries/14_IIB_MIL_Integrated%20instance-level%20and%20bag-level%20MIL%20with%20label%20disambiguation.md](method_summaries/14_IIB_MIL_Integrated%20instance-level%20and%20bag-level%20MIL%20with%20label%20disambiguation.md) *(注：受限版，以IIB_MIL.md为准)*
*   [method_summaries/15_IB_MIL_Interventional%20Bag%20Multi-Instance%20Learning%20On%20Whole-Slide%20Pathological%20Images.md](method_summaries/15_IB_MIL_Interventional%20Bag%20Multi-Instance%20Learning%20On%20Whole-Slide%20Pathological%20Images.md)
*   [method_summaries/16_RANKMIX_MIL_Data%20Augmentation%20for%20Classifying%20WSIs%20with%20Diverse%20Sizes.md](method_summaries/16_RANKMIX_MIL_Data%20Augmentation%20for%20Classifying%20WSIs%20with%20Diverse%20Sizes.md)
*   [method_summaries/17_MHIM_MIL_MIL%20Framework%20with%20Masked%20Hard%20Instance%20Mining%20for%20WSI%20Classification.md](method_summaries/17_MHIM_MIL_MIL%20Framework%20with%20Masked%20Hard%20Instance%20Mining%20for%20WSI%20Classification.md)
*   [method_summaries/18_WIKG_MIL_Dynamic%20Graph%20Representation%20with%20Knowledge-aware%20Attention%20for%20WSI%20Analysis.md](method_summaries/18_WIKG_MIL_Dynamic%20Graph%20Representation%20with%20Knowledge-aware%20Attention%20for%20WSI%20Analysis.md)
*   [method_summaries/19_AMD_MIL_Agent%20Aggregator%20with%20Mask%20Denoise%20Mechanism%20for%20Histopathology%20WSI%20Analysis.md](method_summaries/19_AMD_MIL_Agent%20Aggregator%20with%20Mask%20Denoise%20Mechanism%20for%20Histopathology%20WSI%20Analysis.md)
*   [method_summaries/21_PSEBMIX_MIL_Pseudo-Bag%20Mixup%20Augmentation%20for%20MIL%20Based%20Whole%20Slide%20Image%20Classification.md](method_summaries/21_PSEBMIX_MIL_Pseudo-Bag%20Mixup%20Augmentation%20for%20MIL%20Based%20Whole%20Slide%20Image%20Classification.md)
*   [method_summaries/22_LONG_MIL_Scaling%20Long%20Contextual%20MIL%20for%20Histopathology%20WSI%20Analysis.md](method_summaries/22_LONG_MIL_Scaling%20Long%20Contextual%20MIL%20for%20Histopathology%20WSI%20Analysis.md)
*   [method_summaries/23_DGR_MIL_Exploring%20Diverse%20Global%20Representation%20in%20MIL%20for%20WSI%20Classification.md](method_summaries/23_DGR_MIL_Exploring%20Diverse%20Global%20Representation%20in%20MIL%20for%20WSI%20Classification.md)
*   [method_summaries/24_CDP_MIL_cDP-MIL_%20Robust%20Multiple%20Instance%20Learning%20via%20Cascaded%20Dirichlet%20Process.md](method_summaries/24_CDP_MIL_cDP-MIL_%20Robust%20Multiple%20Instance%20Learning%20via%20Cascaded%20Dirichlet%20Process.md)
*   [method_summaries/25_CA_MIL_Context-Aware%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md](method_summaries/25_CA_MIL_Context-Aware%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md)
*   [method_summaries/26_AC_MIL_Attention-Challenging%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md](method_summaries/26_AC_MIL_Attention-Challenging%20Multiple%20Instance%20Learning%20for%20WSI%20Classification.md)
*   [method_summaries/27_MAMBA_MIL_Enhancing%20Long%20Sequence%20Modeling%20with%20Sequence%20Reordering%20in%20CPath.md](method_summaries/27_MAMBA_MIL_Enhancing%20Long%20Sequence%20Modeling%20with%20Sequence%20Reordering%20in%20CPath.md) *(注：与09合并为MambaMIL)*
*   [method_summaries/28_RET_MIL_Retentive%20Multiple%20Instance%20Learning%20for%20Histopathological%20WSI%20Classification.md](method_summaries/28_RET_MIL_Retentive%20Multiple%20Instance%20Learning%20for%20Histopathological%20WSI%20Classification.md)
*   [method_summaries/29_SC_MIL_Sparse%20Context-aware%20MIL%20for%20Predicting%20Cancer%20Survival%20Probability%20Distribution%20in%20WSI.md](method_summaries/29_SC_MIL_Sparse%20Context-aware%20MIL%20for%20Predicting%20Cancer%20Survival%20Probability%20Distribution%20in%20WSI.md)
*   [method_summaries/30_NCIE_MIL_Rethinking%20Decoupled%20MIL%20Framework%20for%20Histopathological%20Slide%20Classification.md](method_summaries/30_NCIE_MIL_Rethinking%20Decoupled%20MIL%20Framework%20for%20Histopathological%20Slide%20Classification.md)
*   [method_summaries/31_RRT_MIL_Towards%20Foundation%20Model-Level%20Performance%20in%20Computational%20Pathology.md](method_summaries/31_RRT_MIL_Towards%20Foundation%20Model-Level%20Performance%20in%20Computational%20Pathology.md)
*   [method_summaries/32_PA_MIL_Dynamic%20Policy-Driven%20Adaptive%20Multi-Instance%20Learning%20for%20WSI%20Classification.md](method_summaries/32_PA_MIL_Dynamic%20Policy-Driven%20Adaptive%20Multi-Instance%20Learning%20for%20WSI%20Classification.md)
*   [method_summaries/33_MICRO_MIL_Graph-Based%20MIL%20for%20Context-Aware%20Diagnosis%20with%20Microscopic%20Images.md](method_summaries/33_MICRO_MIL_Graph-Based%20MIL%20for%20Context-Aware%20Diagnosis%20with%20Microscopic%20Images.md)
*   [method_summaries/34_DYHG_MIL_Dynamic%20Hypergraph%20Representation%20for%20Bone%20Metastasis%20Cancer%20Analysis.md](method_summaries/34_DYHG_MIL_Dynamic%20Hypergraph%20Representation%20for%20Bone%20Metastasis%20Cancer%20Analysis.md)
*   [method_summaries/36_MAMBA2D_MIL_2DMamba_%20Efficient%20State%20Space%20Model%20for%20Image%20Representation.md](method_summaries/36_MAMBA2D_MIL_2DMamba_%20Efficient%20State%20Space%20Model%20for%20Image%20Representation.md)
*   [method_summaries/37_FOURIER_MIL_Fourier%20filtering-based%20multiple%20instance%20learning%20for%20whole%20slide%20image%20analysis.md](method_summaries/37_FOURIER_MIL_Fourier%20filtering-based%20multiple%20instance%20learning%20for%20whole%20slide%20image%20analysis.md)
*   [method_summaries/38_AEM_MIL_Attention%20Entropy%20Maximization%20for%20MIL%20based%20WSI%20Classification.md](method_summaries/38_AEM_MIL_Attention%20Entropy%20Maximization%20for%20MIL%20based%20WSI%20Classification.md)
*   [method_summaries/39_MICO_MIL_Multiple%20Instance%20Learning%20with%20Context-Aware%20Clustering.md](method_summaries/39_MICO_MIL_Multiple%20Instance%20Learning%20with%20Context-Aware%20Clustering.md)
*   [method_summaries/40_TDA_MIL_Top-Down%20Attention-based%20Multiple%20Instance%20Learning%20for%20Whole%20Slide%20Image%20Analysis.md](method_summaries/40_TDA_MIL_Top-Down%20Attention-based%20Multiple%20Instance%20Learning%20for%20Whole%20Slide%20Image%20Analysis.md)
*   [method_summaries/41_PSA_MIL_Probabilistic%20Spatial%20Attention-Based%20MIL%20for%20Whole%20Slide%20Image%20Classification.md](method_summaries/41_PSA_MIL_Probabilistic%20Spatial%20Attention-Based%20MIL%20for%20Whole%20Slide%20Image%20Classification.md)
*   [method_summaries/43_GDF_MIL_Rethinking%20Multi-Instance%20Learning%20through%20Graph-Driven%20Fusion.md](method_summaries/43_GDF_MIL_Rethinking%20Multi-Instance%20Learning%20through%20Graph-Driven%20Fusion.md)
*   [method_summaries/44_DAG_MIL_Deformable%20attention%20graph%20representation%20learning%20for%20histopathology%20WSI%20analysis.md](method_summaries/44_DAG_MIL_Deformable%20attention%20graph%20representation%20learning%20for%20histopathology%20WSI%20analysis.md)
*   [method_summaries/45_CAMPANELLA_MIL_Clinical-grade%20computational%20pathology%20using%20weakly%20supervised%20deep%20learning%20on%20WSI.md](method_summaries/45_CAMPANELLA_MIL_Clinical-grade%20computational%20pathology%20using%20weakly%20supervised%20deep%20learning%20on%20WSI.md)
*   [method_summaries/46_RMDL_Recalibrated%20multi-instance%20deep%20learning%20for%20whole%20slide%20gastric%20image%20classification.md](method_summaries/46_RMDL_Recalibrated%20multi-instance%20deep%20learning%20for%20whole%20slide%20gastric%20image%20classification.md)
*   [method_summaries/47_CMIL_CAMEL_Weakly%20supervised%20learning%20framework%20for%20histopathology%20image%20segmentation.md](method_summaries/47_CMIL_CAMEL_Weakly%20supervised%20learning%20framework%20for%20histopathology%20image%20segmentation.md)
*   [method_summaries/48_INTER_MIL_Predicting%20molecular%20traits%20through%20self-interactive%20multi-instance%20learning.md](method_summaries/48_INTER_MIL_Predicting%20molecular%20traits%20through%20self-interactive%20multi-instance%20learning.md)
*   [method_summaries/49_LNPL_MIL_Learning%20from%20noisy%20pseudo%20labels%20for%20promoting%20MIL%20in%20WSI.md](method_summaries/49_LNPL_MIL_Learning%20from%20noisy%20pseudo%20labels%20for%20promoting%20MIL%20in%20WSI.md)
*   [method_summaries/50_SMMILE_MIL_Accurate%20spatial%20quantification%20in%20computational%20pathology%20with%20MIL.md](method_summaries/50_SMMILE_MIL_Accurate%20spatial%20quantification%20in%20computational%20pathology%20with%20MIL.md)
*   [method_summaries/51_DEEPATTNMISL_MIL_WSI%20based%20cancer%20survival%20prediction%20using%20attention%20guided%20deep%20MIL.md](method_summaries/51_DEEPATTNMISL_MIL_WSI%20based%20cancer%20survival%20prediction%20using%20attention%20guided%20deep%20MIL.md)
*   [method_summaries/52_MS_DA_MIL_Multi-scale%20domain-adversarial%20MIL%20CNN%20for%20cancer%20subtype%20classification.md](method_summaries/52_MS_DA_MIL_Multi-scale%20domain-adversarial%20MIL%20CNN%20for%20cancer%20subtype%20classification.md)
*   [method_summaries/53_MIXED_SUPERVISION_MIL_Multiple%20instance%20learning%20with%20mixed%20supervision%20in%20Gleason%20grading.md](method_summaries/53_MIXED_SUPERVISION_MIL_Multiple%20instance%20learning%20with%20mixed%20supervision%20in%20Gleason%20grading.md)
*   [method_summaries/54_MDMIL_Targeting%20tumor%20heterogeneity%20multiplex-detection-based%20MIL.md](method_summaries/54_MDMIL_Targeting%20tumor%20heterogeneity%20multiplex-detection-based%20MIL.md)
*   [method_summaries/55_RAM_MIL_Retrieval-augmented%20multiple%20instance%20learning.md](method_summaries/55_RAM_MIL_Retrieval-augmented%20multiple%20instance%20learning.md)
*   [method_summaries/56_VINO_MIL_Transformer-based%20video-structure%20MIL%20for%20WSI%20classification.md](method_summaries/56_VINO_MIL_Transformer-based%20video-structure%20MIL%20for%20WSI%20classification.md)
*   [method_summaries/57_PROV_GIGAPATH_MIL_A%20whole-slide%20foundation%20model%20for%20digital%20pathology%20from%20real-world%20data.md](method_summaries/57_PROV_GIGAPATH_MIL_A%20whole-slide%20foundation%20model%20for%20digital%20pathology%20from%20real-world%20data.md)
*   [method_summaries/58_STATE_SPACE_MIL_Structured%20state%20space%20models%20for%20MIL%20in%20digital%20pathology.md](method_summaries/58_STATE_SPACE_MIL_Structured%20state%20space%20models%20for%20MIL%20in%20digital%20pathology.md)
*   [method_summaries/59_CIMIL_Boosting%20MIL%20models%20based%20on%20counterfactual%20inference.md](method_summaries/59_CIMIL_Boosting%20MIL%20models%20based%20on%20counterfactual%20inference.md)
*   [method_summaries/DT_MIL.md](method_summaries/DT_MIL.md)
*   [method_summaries/FR_MIL.md](method_summaries/FR_MIL.md)
*   [method_summaries/IIB_MIL.md](method_summaries/IIB_MIL.md)
*   [method_summaries/ILRA_MIL.md](method_summaries/ILRA_MIL.md)
*   [method_summaries/MO_MIL.md](method_summaries/MO_MIL.md)
*   [method_summaries/MSM_MIL.md](method_summaries/MSM_MIL.md)
*   [method_summaries/Stable_MIL.md](method_summaries/Stable_MIL.md)
