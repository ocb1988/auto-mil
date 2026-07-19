# 51_DEEPATTNMISL_MIL_WSI based cancer survival prediction using attention guided deep MIL 方法总结

> 证据说明：输入为完整论文预印本（arXiv:2009.11169v1），包含摘要、引言、方法论、实验及结论。公式提取基本完整，关键符号定义清晰。无明显的页面或公式缺失导致无法理解的情况。

## 一、论文基本信息

- **论文标题**：WHOLE SLIDE IMAGES BASED CANCER SURVIVAL PREDICTION USING ATTENTION GUIDED DEEP MULTIPLE INSTANCE LEARNING NETWORKS
- **作者**：Jiawen Yao, Xinliang Zhu, Jitendra Jonagaddala, Nicholas Hawkins, Junzhou Huang
- **发表年份**：2020 (Preprint), 发表于 Medical Image Analysis 65, 2020
- **会议/期刊**：Medical Image Analysis (MedIA)
- **论文链接/DOI/arXiv ID**：arXiv:2009.11169v1 / DOI: 10.1016/j.media.2020.101789
- **代码仓库**：https://github.com/uta-smile/DeepAttnMISL_MEDIA
- **研究任务**：基于全切片图像（WSI）的癌症患者生存预测（Survival Prediction）
- **数据模态**：数字病理学全切片图像（WSIs），弱监督标签（患者级别的生存时间 $t_i$ 和删失状态 $\delta_i$）

## 二、论文整体概述

### 1. 核心问题
传统基于图像的生存预测模型依赖判别性补丁（patch）标注，难以扩展到大规模数据集；现有的WSI生存模型通常局限于从WSI中提取的关键补丁或聚类，且聚合方式（如平均池化）不够灵活，无法有效捕捉肿瘤异质性。此外，WSI尺寸巨大（GB/TB级），直接处理计算成本极高，且缺乏像素级标注。

### 2. 整体方法
提出 **DeepAttnMISL** (Deep Attention Multiple Instance Survival Learning) 框架。该方法采用多实例学习（MIL）范式，将患者视为一个“包”（Bag），将患者的表型聚类（Phenotype Clusters）视为“实例”（Instances）。
主要流程包括：
1.  **预处理与聚类**：从WSI中提取组织补丁，使用预训练网络（如VGG）提取特征，并在患者级别进行K-means聚类得到表型聚类。
2.  **Siamese MI-FCN**：使用共享权重的孪生全卷积网络（MI-FCN）分别处理每个表型聚类中的补丁集合，生成每个表型的局部表示。
3.  **Attention-based MIL Pooling**：利用注意力机制对各个表型的局部表示进行加权聚合，生成患者级别的表示。
4.  **生存风险预测**：通过全连接层输出风险评分，并使用负对数偏似然损失函数进行端到端训练。

### 3. 主要贡献
- 引入注意力机制到深度多实例生存学习中，实现了可训练的加权聚合，比固定池化更灵活。
- 使用Siamese MI-FCN网络从不同表型聚类中学习形态特异性特征。
- 提供了良好的可解释性，能够定位与预后相关的重要区域和模式。
- 在两个大型WSI数据集（NLST肺癌，MCO结直肠癌）上验证了方法的有效性，优于现有基线。

## 三、方法总结

### 方法 1：DeepAttnMISL 整体架构

#### 1. 核心思想与解决的问题
- **目标问题**：如何在没有像素级标注的情况下，从巨大的WSI中高效地学习患者级别的生存风险预测，并解决肿瘤异质性问题。
- **现有方法的局限**：ROI-based方法依赖人工选择感兴趣区域，代表性不足；WSISA等方法需要全局聚类且分阶段训练，不可扩展且聚合方式固定；传统MIL方法未充分利用表型结构的语义信息。
- **核心思想**：将WSI中的补丁聚类为具有生物学意义的“表型”，利用MIL框架将这些表型作为实例，通过注意力机制自适应地聚合这些表型特征以预测生存风险。
- **创新点**：
    1.  **表型级MIL**：不直接使用原始补丁，而是使用聚类后的表型作为MIL的实例，减少了实例数量并增强了语义鲁棒性。
    2.  **Siamese MI-FCN**：针对变长输入的表型补丁集合，使用全卷积网络提取局部表示，权重共享保证一致性。
    3.  **可微注意力聚合**：将注意力机制嵌入MIL池化层，实现端到端训练，同时提供注意力热力图作为可解释性依据。

#### 2. 详细结构与数据流
- **输入**：
    - 患者 $i$ 的 $C$ 个表型聚类。
    - 第 $j$ 个表型包含 $m_j$ 个补丁的特征向量。
    - 标签 $(t_i, \delta_i)$。
- **处理流程**：
    1.  **Patch Extraction & Feature Extraction**: 从WSI提取 $500 \times 500$ 补丁，使用ImageNet预训练VGG-16提取高维特征（文中未明确指定最终特征维度 $d$，但后续提到MI-FCN输入通道为 $d$，通常VGG fc层前特征为4096或类似，但在MI-FCN中作为输入通道）。
    2.  **Clustering**: 对患者所有补丁的特征进行K-means聚类，确定 $C$ 个表型（实验中测试 $C=6,8,10,12$）。
    3.  **Siamese MI-FCN**: 对于每个表型 $j$，将其包含的 $m_j$ 个补丁特征堆叠成张量，输入到MI-FCN。MI-FCN由Conv-ReLU层对组成，最后接Global Pooling，输出该表型的局部表示 $r_j$。由于是Siamese结构，所有表型共享同一套MI-FCN参数。
    4.  **Attention MIL Pooling**: 收集所有表型的局部表示 $R = \{r_1, ..., r_C\}$。计算每个表型的注意力权重 $a_k$，并对 $r_k$ 进行加权求和得到患者级表示 $z$。
    5.  **Risk Prediction**: 将 $z$ 输入全连接层（FC: 64->32->1），输出风险评分 $o_i$。
- **输出**：患者风险评分 $o_i$。
- **模块在整体网络中的位置**：位于特征提取之后，损失计算之前。
- **与其他模块的连接方式**：接收来自聚类模块的表型补丁特征组，输出患者级向量给分类头。

#### 3. 数学公式

**注意力权重计算：**
$$ a_k = \frac{\exp\{w^\top \tanh(V r_k^\top)\}}{\sum_{j=1}^{C} \exp\{w^\top \tanh(V r_j^\top)\}} \quad (2) $$
其中：
- $r_k \in \mathbb{R}^{M}$ 是第 $k$ 个表型的局部表示（$M$为表示维度，文中Table 1显示为64）。
- $V \in \mathbb{R}^{L \times M}$ 是可训练参数矩阵（文中Table 1暗示中间维度 $L$ 可能为64或根据具体实现，公式中 $w$ 维度需匹配）。*注：原文公式(2)中 $w \in \mathbb{R}^{L \times 1}$，故 $w^\top \tanh(...)$ 结果为标量。*
- $w \in \mathbb{R}^{L \times 1}$ 是可训练参数向量。
- $\tanh(\cdot)$ 为逐元素非线性激活。

**患者级表示聚合：**
$$ z = \sum_{k=1}^{C} a_k r_k \quad (1) $$

**Loss Function (Negative Log Partial Likelihood):**
$$ \mathcal{L}(o_i) = \sum_{i} \delta_i (-o_i + \log \sum_{j: t_j \ge t_i} \exp(o_j)) \quad (6) $$
其中：
- $\delta_i$ 是事件指示变量（1为死亡，0为删失）。
- $o_i$ 是模型预测的风险评分。
- 求和项 $\sum_{j: t_j \ge t_i}$ 是在时间 $t_i$ 时仍处于风险集中的患者集合 $R(t_i)$。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| Patch Input | Patch Features | $m_j \times d$ | $m_j$为第j个表型的补丁数，$d$为VGG提取的特征维度 |
| MI-FCN Output | Local Representation ($r_j$) | $1 \times M$ (或 $M$) | 文中Table 1显示Output size为64，即 $M=64$ |
| Aggregation Input | All Representations ($R$) | $C \times M$ | $C$为表型数量，$M=64$ |
| Attention Weights | $a_k$ | Scalar per cluster | 归一化的注意力分数 |
| Patient Rep | $z$ | $1 \times M$ | 患者级特征向量 |
| FC Layer 1 | Hidden | $1 \times 32$ | Table 1显示 Fully-Con. 64->32 |
| FC Layer 2 | Risk Score ($o_i$) | $1 \times 1$ | 最终风险评分 |

*注意：Table 1中MI-FCN的Input写为 $1 \times m_i \times d$，Output为 64。这意味着MI-FCN内部进行了空间维度的压缩和通道变换。*

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MI_FCN(nn.Module):
    """
    Siamese MI-FCN for a single phenotype cluster.
    Input: Patches features of one phenotype [Batch_Size, Num_Patches, Feature_Dim]
    Output: Local representation [Batch_Size, Hidden_Dim]
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(MI_FCN, self).__init__()
        # 根据论文描述：Conv-ReLU pair(s). 
        # 实验部分提到 "one convolutional layer, one ReLU layer" 效果最好 (Table 4)
        # 假设 input_dim 是 VGG 特征维度 (例如 4096 或 2048)，或者如果是图像块则是卷积输入
        # 论文中提到 "1x1 conv layer", 暗示是对特征向量操作而非空间卷积，因为输入已是特征
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1) # 1D Conv over feature dim? 
        # 或者如果输入是 [Num_Patches, Channels, Height, Width] 则用2D Conv
        # 根据公式 r_j 是向量，且输入是 patch features，这里假设输入已经是展平的特征向量序列
        # 修正：论文Figure 3显示 "Instance (Phenotype) -> Convolution -> ReLU ... -> Pool"
        # 如果输入是 patch features (vector)，则 Conv1d 合适。
        # 若输入是图像patch，则需先展平或保持空间。鉴于VGG提取的是fc层前的特征，通常是向量。
        # 但为了通用性，假设输入是 [N_patches, D_features]
        
        # 重新审视：论文说 "input is a set of features from mi patches... organized as 1 x mi x d"
        # 这看起来像是一个序列。Conv1d 是合理的。
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        
        # 如果只有一层 Conv-ReLU:
        # self.fc_final = nn.Linear(hidden_dim, hidden_dim) # 可选，取决于具体实现细节
        # 论文Table 1 Output size 64. 假设 hidden_dim = 64.

    def forward(self, x):
        # x shape: [Batch, N_patches, D_features]
        # Transpose to [Batch, D_features, N_patches] for Conv1d
        x = x.transpose(1, 2) 
        
        x = self.conv1(x)
        x = self.relu(x)
        # Global Average Pooling over the patch dimension (dim 2)
        x = self.pool(x).squeeze(-1) # Shape: [Batch, Hidden_Dim]
        return x

class AttentionMILPooling(nn.Module):
    """
    Attention-based MIL Pooling
    Input: Local representations R [Batch, C, M]
    Output: Patient representation z [Batch, M]
    """
    def __init__(self, input_dim=64, attn_dim=64):
        super(AttentionMILPooling, self).__init__()
        self.V = nn.Linear(input_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, R):
        # R: [Batch, C, M]
        # Calculate attention scores
        # A = tanh(R @ V^T) -> [Batch, C, L]
        A = torch.tanh(self.V(R)) 
        # Scores = A @ w^T -> [Batch, C, 1]
        Scores = self.w(A).squeeze(-1) # [Batch, C]
        
        # Softmax
        Alpha = F.softmax(Scores, dim=1) # [Batch, C]
        
        # Weighted Sum
        # z = sum(alpha_k * r_k)
        # Alpha: [Batch, C], R: [Batch, C, M]
        # Expand Alpha for broadcasting
        Alpha_expanded = Alpha.unsqueeze(-1) # [Batch, C, 1]
        z = torch.sum(Alpha_expanded * R, dim=1) # [Batch, M]
        
        return z, Alpha

class DeepAttnMISL(nn.Module):
    def __init__(self, patch_feature_dim, num_phenotypes=10, hidden_dim=64):
        super(DeepAttnMISL, self).__init__()
        self.num_phenotypes = num_phenotypes
        # Siamese MI-FCN: Shared weights across all phenotypes
        self.mi_fcn = MI_FCN(patch_feature_dim, hidden_dim)
        
        # Attention Pooling
        self.attention_pool = AttentionMILPooling(input_dim=hidden_dim, attn_dim=hidden_dim)
        
        # Survival Head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, phenotype_patches_list, labels=None):
        """
        phenotype_patches_list: List of tensors, each tensor is [Batch, N_patches_in_cluster, Patch_Feat_Dim]
        Note: In practice, batches might need padding or dynamic handling if N_patches varies significantly.
        The paper implies processing per patient/batch where each patient has C clusters.
        """
        local_representations = []
        
        # Process each phenotype cluster through the shared MI-FCN
        for i in range(self.num_phenotypes):
            patches = phenotype_patches_list[i] # [Batch, Mi, D]
            # Handle empty clusters if any (though clustering usually ensures non-empty or masked)
            rep = self.mi_fcn(patches)
            local_representations.append(rep)
            
        # Stack to [Batch, C, M]
        R = torch.stack(local_representations, dim=1)
        
        # Attention Pooling
        z, alpha = self.attention_pool(R)
        
        # Survival Prediction
        h = self.relu(self.fc1(z))
        o = self.fc2(h) # Risk score
        
        return o, alpha

    def loss(self, outputs, labels):
        """
        Negative Log Partial Likelihood Loss
        outputs: [Batch, 1] risk scores
        labels: tuple (time, event_indicator)
        """
        times, events = labels
        # Sort by time to handle risk sets correctly
        # This requires careful implementation of the risk set summation
        # Simplified logic for pseudo-code:
        batch_size = outputs.size(0)
        loss = 0.0
        for i in range(batch_size):
            if events[i] == 1: # Only uncensored contribute to likelihood
                # Risk set R(ti): all j such that tj >= ti
                # In mini-batch training, this is often approximated or computed over the whole batch
                # Assuming we compute within the current batch for simplicity, 
                # though full dataset ranking is ideal for Cox PH.
                # Paper Eq 6 sums over j in R(ti).
                
                # Get indices in risk set
                risk_set_indices = torch.where(times >= times[i])[0]
                
                # Numerator: exp(oi)
                num = torch.exp(outputs[i])
                
                # Denominator: sum(exp(oj)) for j in R(ti)
                denom = torch.sum(torch.exp(outputs[risk_set_indices]))
                
                loss += -torch.log(num / denom)
                
        return loss / batch_size
```

#### 6. 实现提示
- **关键网络组件**：`MI_FCN` 使用 `Conv1d` 处理特征序列，配合 `AdaptiveAvgPool1d`。`AttentionMILPooling` 使用两层线性变换加 Tanh 和 Softmax。
- **重要超参数**：
    - 聚类数量 $C$：实验中测试 6, 8, 10, 12，最佳通常为 6-10。
    - MI-FCN 层数：实验表明 1 层 Conv-ReLU 效果最佳（Table 4）。
    - 优化器：Adam, LR=$10^{-4}$, Weight Decay=$5 \times 10^{-4}$。
    - Early Stopping：基于验证集 Loss。
- **归一化/激活方式**：MI-FCN 中使用 ReLU；注意力机制中使用 Tanh 和 Softmax。
- **维度对齐方式**：MI-FCN 输出维度需与 Attention 输入维度一致（文中均为 64）。
- **实现注意事项**：
    - **Risk Set 计算**：Cox Loss 的计算依赖于全局排序或当前 Batch 内的相对排序。如果在 Mini-batch 中计算，需确保正确处理 $t_j \ge t_i$ 的条件。
    - **空聚类处理**：如果某个患者在某些聚类中没有补丁，需将该聚类的注意力权重强制设为 0 或跳过该聚类，避免除以零或无效梯度。
    - **特征提取**：外部使用 VGG-16 提取补丁特征，无需在 PyTorch 模型中包含 VGG 主干，除非进行微调（文中主要使用预训练特征）。

#### 7. 计算与资源开销
- **理论计算复杂度**：远低于直接处理 Gigapixel WSI。复杂度主要取决于补丁数量 $N$ 和聚类数 $C$。MI-FCN 是轻量级的（1层 Conv）。Attention 计算复杂度为 $O(C \cdot M^2)$，非常低。
- **参数量**：较少。主要是 MI-FCN 的卷积核和 Attention 层的线性层参数。
- **FLOPs/MACs**：未提供具体数值，但由于避免了深层 CNN 在 WSI 上的滑动窗口推理，FLOPs 显著降低。
- **显存开销**：较低，因为只加载补丁特征和小型网络。
- **推理速度**：快，适合大规模数据集。
- **论文是否提供效率对比**：文中提到相比 WSISA 等需要大量补丁采样的方法，本方法更高效，但未给出精确的 FPS 或 GFLOPs 对比表格。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：基于 H&E 染色 WSIs 的癌症生存分析。
- **可迁移到的任务/数据集**：其他类型的病理图像分类/回归（如癌症亚型分类）、任何基于 MIL 的弱监督医学图像分析任务。
- **迁移所需调整**：调整特征提取器（如换用 ResNet）、调整聚类算法（如 Spectral Clustering）、调整生存损失函数（如改为 Binary Cross Entropy 用于分类）。
- **适用条件**：拥有患者级别的标签，缺乏实例级标签；数据量大，存在异质性。
- **潜在限制**：依赖 K-means 聚类的质量；对补丁特征提取器的依赖性较强。

#### 9. 实验与消融证据
- **主要性能结果**：
    - **MCO 数据集**：C-index 最高达 0.606 (1M patches, c=6)。
    - **NLST 数据集**：C-index 最高达 0.6963。
- **相对基线的提升**：优于 WSISA, DeepMISL, Lasso-Cox, MTLSA 等基线。在 NLST 上显著优于所有基线。
- **相关消融实验**：
    - **聚类数量**：c=6 或 8 表现较好，过多聚类（12）性能下降。
    - **MI-FCN 深度**：1 层优于 2-3 层。
    - **Siamese 有效性**：移除 Siamese 结构（直接对特征做 Attention）性能大幅下降（C-index 从 ~0.60 降至 ~0.54）。
    - **注意力 vs Max/Mean**：Attention 优于 Max 和 Mean pooling。
    - **Gated Attention**：Plain Attention 略优于 Gated Attention。
- **作者结论**：提出的 DeepAttnMISL 在性能和可解释性上均优于现有方法。
- **证据是否充分**：在两个独立的大型数据集上进行了广泛比较和消融，证据充分。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 中 | 注意力MIL已有前人工作（Ilse et al.），本文的创新在于将其应用于生存分析并结合表型聚类，属于应用层面的改进和系统化整合。 |
| 技术可行性 | 高 | 模块标准，易于复现，依赖成熟的预训练模型和聚类算法。 |
| 实现难度 | 低 | 代码逻辑清晰，开源代码可用。 |
| 架构相关性 | 高 | 专门针对WSI的大尺度和异质性设计。 |
| 可迁移性 | 高 | MIL和注意力机制是通用组件。 |
| 计算成本 | 低 | 相比端到端CNN处理WSI，计算成本低得多。 |

#### 11. 一句话总结
DeepAttnMISL 通过结合表型聚类和注意力驱动的多实例学习，实现了高效、可解释且高精度的全切片图像癌症生存预测。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **表型级 MIL 建模**：将原始的 Patch 聚类为 Phenotype 作为 MIL 的 Instance，既降低了计算复杂度，又引入了语义层面的鲁棒性，解决了 Patch 级噪声和冗余问题。
- **可解释性可视化**：通过 Attention 权重生成 Heatmap，直观展示模型关注的肿瘤区域，符合临床需求。

### 2. 方法之间的关系
- **与 WSISA 的关系**：WSISA 也是基于聚类的生存预测，但它是两阶段的（先聚类选簇，再训练生存模型），且聚合方式固定。DeepAttnMISL 是端到端的，且使用 Attention 动态聚合。
- **与 DeepMISL 的关系**：DeepMISL 是本文的前期工作，使用了 MIL 但没有 Attention，且对所有 Cluster 平等对待。DeepAttnMISL 引入了 Attention 来区分不同 Cluster 的重要性。

### 3. 复现可行性
- **代码是否公开**：是，GitHub 上有官方代码。
- **方法描述是否完整**：是，提供了详细的网络结构、损失函数和超参数。
- **关键配置是否明确**：是，包括 Patch 大小、预训练模型、优化器等。
- **预计复现难点**：
    - **Cox Loss 的实现**：正确构建 Risk Set 并进行批量计算可能需要仔细处理索引和排序。
    - **数据预处理**：WSI 的读取、背景去除、Patch 提取和特征提取流水线较为繁琐，需确保与论文设置一致。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：Attention-based MIL Pooling 已成为 WSI 分析的标准组件之一，可直接采用。
- **需要改造的设计**：如果应用于非生存任务（如分类），需替换 Loss 函数；如果希望进一步提升精度，可考虑引入 Transformer 替代简单的 FCN 进行局部特征提取。
- **可能形成的新研究思路**：结合图神经网络（GCN）来建模表型之间的空间关系（论文提到 Li et al. 的 GCN 方法需要图结构知识，而本文方法无需此先验，是一个优势，但也意味着忽略了空间拓扑，这是一个潜在的改进方向）。

### 5. 阅读备注
- 论文强调“Patient-wise Clustering”而非“Database-wise Clustering”，这是其可扩展性的关键。
- 实验中发现 Ensemble 模型并未带来显著提升，说明单模型已具备较强的泛化能力。
- Kaplan-Meier 曲线和 Log-rank 检验证明了模型在风险分层上的临床价值。
