# 39_MICO_MIL_Multiple Instance Learning with Context-Aware Clustering 方法总结

> 证据说明：输入为完整论文文本（11页），包含摘要、引言、方法论、实验及附录。公式提取基本完整，关键数学符号清晰。代码仓库链接已提供。无明显的页面缺失或公式乱码导致无法理解的情况。

## 一、论文基本信息

- **论文标题**：MiCo: Multiple Instance Learning with Context-Aware Clustering for Whole Slide Image Analysis
- **作者**：Junjian Li, Jin Liu, Hulin Kuang, Hailin Yue, Mengshen He, Jianxin Wang
- **发表年份**：2026 (arXiv:2506.18028v3, 12 Jan 2026)
- **会议/期刊**：arXiv预印本 (cs.CV)
- **论文链接/DOI/arXiv ID**：https://arxiv.org/abs/2506.18028
- **代码仓库**：https://github.com/junjianli106/MiCo
- **研究任务**：全切片图像（WSI）分析，具体包括癌症亚型分类（Cancer Subtyping）和生存预测（Survival Prediction）。
- **数据模态**：数字病理学全切片图像（H&E染色），通过多模态基础模型TITAN提取的Patch特征。

## 二、论文整体概述

### 1. 核心问题
全切片图像（WSI）具有极高的空间异质性。病理上重要的同类组织结构往往在解剖学上分散分布（如淋巴血管肿瘤栓子从瘤周延伸至远处血管系统）。现有的多实例学习（MIL）方法难以有效建模这些分散组织之间的长距离依赖关系和跨区域的语义交互，导致语义碎片化或无法捕捉跨区域的 intra-tissue 相关性。

### 2. 整体方法
提出了一种名为 **MiCo** 的新框架，即“具有上下文感知聚类（Context-Aware Clustering）的多实例学习”。
- **核心机制**：使用可学习的语义锚点（Semantic Anchors）作为聚类中心。
- **Cluster Route (CluRoute)**：动态连接远距离但属于同一组织类型的实例，通过特征相似度将分散的实例聚合到语义锚点，增强区域内（intra-tissue）的相关性。
- **Cluster Reducer (CluReducer)**：合并冗余的语义锚点，促进不同语义组之间的信息交换，消除语义碎片化，增强区域间（inter-tissue）的语义关联。
- **架构**：由多层上下文感知聚类模块堆叠而成，每层包含上述两个模块。

### 3. 主要贡献
1. 提出了MiCo框架，专门解决WSI中的空间异质性和语义碎片化问题。
2. 设计了Cluster Route模块，通过直推估计器（Straight-Through Estimator）实现可微分的实例到锚点的分配，聚合跨区域的相似语义。
3. 设计了Cluster Reducer模块，通过非线性变换减少锚点数量，整合冗余信息并增强不同语义类别间的交互。
4. 在9个大型公共癌症数据集上的广泛实验表明，MiCo在生存预测和癌症亚型分类任务上优于最先进的方法。

## 三、方法总结

### 方法 1：上下文感知聚类模块 (Context-Aware Clustering Module)

#### 1. 核心思想与解决的问题
- **目标问题**：WSI中形态相似的病理结构在空间上分散，传统MIL方法难以捕捉这种非局部的语义一致性；同时，固定的聚类或注意力机制可能导致语义冗余或忽略不同组织类型间的复杂交互。
- **现有方法的局限**：基于Attention的方法（如AMIL）关注局部关键实例，缺乏跨区域语义联系；基于Transformer的方法（如TransMIL）将所有交互视为同质，忽略了肿瘤微环境的动态异质性；基于图的方法依赖固定邻域，无法适应形态连续性。
- **核心思想**：引入一组可学习的“语义锚点”（Semantic Anchors）作为抽象的语义代表。通过“路由”机制将分散的实例动态分配给最相关的锚点，从而建立跨区域的语义连接；随后通过“缩减”机制优化锚点表示，减少冗余并促进不同语义组间的交流。
- **创新点**：
    1. 使用可学习锚点替代硬编码或静态聚类中心。
    2. 结合STRAIGHT-THROUGH ESTIMATOR实现离散分配的可微训练。
    3. 显式地分离了“增强同类跨区域联系”（Route）和“优化/融合语义”（Reducer）两个过程。

#### 2. 详细结构与数据流
- **输入**：
    - 实例特征矩阵 $H \in \mathbb{R}^{M \times d}$，其中 $M$ 是Patch数量，$d$ 是特征维度。
    - 语义锚点矩阵 $S \in \mathbb{R}^{K \times d}$，其中 $K$ 是锚点数量（默认64）。
- **处理流程**：
    1. **Cluster Route**:
        - 计算实例 $h_m$ 与所有锚点 $s_k$ 的余弦相似度，得到对齐矩阵 $A_l \in \mathbb{R}^{M \times K}$。
        - 使用 `argmax` 确定每个实例所属的最相关锚点，并通过 Straight-Through Estimator 生成软/硬分配矩阵 $\hat{A}_l$。
        - 根据分配矩阵 $\hat{A}_l$ 聚合被分配到同一锚点的实例特征，生成上下文感知的锚点表示 $\tilde{s}_k$。
        - 利用更新后的锚点语义 $\tilde{s}_k$ 对原始实例特征 $h_m$ 进行残差更新，得到增强后的实例特征 $h'_m$。
    2. **Cluster Reducer**:
        - 接收聚合后的锚点表示 $\tilde{S} \in \mathbb{R}^{K \times d}$。
        - 转置后通过MLP进行非线性变换，输出维度减半的特征 $\tilde{S}' \in \mathbb{R}^{d \times K/2}$。
        - 转置回 $\mathbb{R}^{K/2 \times d}$，得到精简且富含交互信息的锚点集合。
    3. **迭代**：更新后的实例特征和锚点可以送入下一层Context-Aware Clustering模块。
- **输出**：
    - 更新后的实例特征 $H' \in \mathbb{R}^{M \times d}$（用于后续池化）。
    - 精简后的语义锚点 $\tilde{S}_{new} \in \mathbb{R}^{K/2 \times d}$。
- **模块在整体网络中的位置**：位于特征提取器（TITAN Encoder）之后，Bag-level Pooling（Attention Pooling）之前。通常堆叠多层。
- **与其他模块的连接方式**：输入来自Encoder的Patch特征和上一级的Anchor；输出传递给下一层Module或直接进入Classifier前的Pooling层。

#### 3. 数学公式

**1. 相似度计算：**
$$ A_l(m,k) = \frac{h_m^\top s_k}{\|h_m\| \cdot \|s_k\|} $$
其中 $A_l \in \mathbb{R}^{M \times K}$ 量化实例与锚点的语义对齐程度。

**2. 实例-锚点分配（Straight-Through Estimator）：**
$$ \hat{A}_l = \text{one-hot}(\arg \max(A_l)) + A_l - \text{sg}(A_l) $$
其中 $\text{sg}(\cdot)$ 是停止梯度算子。前向传播时 $\hat{A}_l$ 是二进制矩阵（每行一个hot），反向传播时梯度通过 $A_l$ 流动。

**3. 锚点聚合（Context-Aware Representation）：**
$$ \tilde{s}_k = \frac{1}{N_k} \sum_{m=1}^{M} \hat{A}_l(m,k) \cdot h_m, \quad N_k = \sum_{m=1}^{M} \hat{A}_l(m,k) $$
$\tilde{s}_k \in \mathbb{R}^d$ 捕获了分配给第 $k$ 个锚点的所有实例共享的形态语义。

**4. 实例特征更新：**
$$ h'_m = h_m + \text{MLP}(h_m + \hat{A}_l(m,k) \cdot \tilde{s}_k) $$
注意：这里 $\hat{A}_l(m,k)$ 指示实例 $m$ 被分配到的锚点 $k$。该步骤将跨区域的上下文语义注入到单个实例特征中。

**5. 锚点缩减（Cluster Reducer）：**
$$ \tilde{S}' = \text{MLP}(\tilde{S}^\top) $$
其中 $\tilde{S} \in \mathbb{R}^{K \times d}$ 是所有聚合后的锚点矩阵。$\tilde{S}' \in \mathbb{R}^{d \times K/2}$ 是经过MLP处理后的中间表示，最终转置得到新的锚点集。这实现了锚点数量的减半和语义的融合。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | $H$ | $M \times d$ | WSI中所有Patch的特征矩阵 |
| 输入 | $S$ | $K \times d$ | 初始或上一级传递的语义锚点矩阵 |
| 中间 | $A_l$ | $M \times K$ | 实例与锚点的余弦相似度矩阵 |
| 中间 | $\hat{A}_l$ | $M \times K$ | 离散化的分配矩阵（One-Hot） |
| 中间 | $\tilde{S}$ | $K \times d$ | 聚合后的上下文感知锚点 |
| 中间 | $H'$ | $M \times d$ | 更新后的实例特征 |
| 输出 | $\tilde{S}_{new}$ | $K/2 \times d$ | 缩减后的新锚点矩阵 |

*注：$M$ 为Patch数，$d$ 为特征维数（TITAN提取通常为768或类似高维），$K$ 为锚点数（默认64）。*

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterRoute(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d = d
        self.k = k
        # MLP for feature update
        self.mlp_update = nn.Sequential(
            nn.Linear(d * 2, d), # Input: concat(h_m, context)
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, H, S):
        """
        H: [M, d] Instance features
        S: [K, d] Semantic anchors
        """
        M, d = H.shape
        
        # 1. Compute Cosine Similarity
        # Normalize
        H_norm = F.normalize(H, p=2, dim=1)
        S_norm = F.normalize(S, p=2, dim=1)
        
        # A_l: [M, K]
        A_l = torch.matmul(H_norm, S_norm.T)
        
        # 2. Assignment via Straight-Through Estimator
        # argmax along K dimension -> [M] indices
        assign_indices = torch.argmax(A_l, dim=1)
        
        # One-hot encoding [M, K]
        one_hot_assign = F.one_hot(assign_indices, num_classes=self.k).float()
        
        # STE: Forward uses one_hot, Backward uses raw A_l gradients
        # In PyTorch, we can implement this manually or use a custom autograd function
        # Here simplified logic:
        # grad_flow = A_l - stop_gradient(A_l) + one_hot
        # However, standard implementation often just passes gradient through A_l 
        # while using one_hot for aggregation.
        
        # 3. Aggregate Anchors
        # Sum features assigned to each anchor
        # scatter_add requires careful indexing
        # S_agg: [K, d]
        S_agg = torch.zeros(self.k, d, device=H.device)
        counts = torch.zeros(self.k, device=H.device)
        
        # Efficient aggregation using scatter_add
        # expand H to match indices shape? No, use index_select/scatter
        # Let's use a loop or advanced indexing for clarity in pseudo-code
        # S_agg[assign_indices] += H  <-- This is not directly supported for sum without scatter
        
        # Correct scatter_add usage:
        # We need to sum H[m] into S_agg[k] where k = assign_indices[m]
        # Reshape H for scatter: [M, 1, d]
        # Indices: [M, 1]
        
        S_agg.scatter_add_(0, assign_indices.unsqueeze(1).expand(-1, d), H)
        counts.scatter_add_(0, assign_indices, torch.ones(M, device=H.device))
        
        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)
        S_context = S_agg / counts.unsqueeze(1) # [K, d]
        
        # 4. Update Instance Features
        # For each instance m, get its assigned anchor context S_context[assign_indices[m]]
        # Context vector for each instance: [M, d]
        instance_context = S_context[assign_indices] 
        
        # Concatenate original feature and context
        combined = torch.cat([H, instance_context], dim=1) # [M, 2d]
        
        # Apply MLP
        updated_H = H + self.mlp_update(combined)
        
        return updated_H, S_context

class ClusterReducer(nn.Module):
    def __init__(self, d, k_in, k_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(k_in, k_out), # Note: Paper says MLP on Transpose, so input dim is K
            nn.ReLU(),
            # The paper implies the output is K/2 anchors of dim d.
            # Formula: S' = MLP(S^T). S^T is [d x K]. 
            # If MLP is linear layer applied per row (feature dim d)? 
            # Or is it a transformation of the anchor space?
            # Re-reading: "transpose S to S^T ... apply MLP ... S' in R^{d x K/2}"
            # This suggests the MLP operates on the 'd' dimension? 
            # Wait, if S^T is [d, K], and output is [d, K/2], 
            # usually MLPs operate on the last dimension. 
            # If it's a Linear layer, it would be Linear(K, K/2) applied to each of the d rows?
            # Yes, effectively reducing the number of anchors K while keeping dim d.
            nn.Linear(k_in, k_out) # Applied across the K dimension for each feature channel?
            # Actually, standard Linear(in_features, out_features) applies to the last dim.
            # So if input is [d, K], Linear(K, K/2) works perfectly.
        )
        self.d = d
        self.k_in = k_in
        self.k_out = k_out

    def forward(self, S):
        """
        S: [K, d] Aggregated anchors
        """
        # Transpose to [d, K]
        S_T = S.T 
        
        # Apply MLP. Since S_T is [d, K], and we want [d, K/2]
        # We treat each of the d channels independently? 
        # Or is the MLP shared across d?
        # "apply a MLP to model nonlinear interactions"
        # If it's a single Linear layer Linear(K, K/2), it reduces K.
        # But does it mix features? 
        # If we use Linear(K, K/2), it mixes the anchor indices but keeps feature dims separate.
        # To mix features, we might need a different structure, but based on formula (5):
        # S' = MLP(S^T). Shape matches [d, K/2].
        # Implementation: View as [d*K, 1]? No.
        # Most likely: The MLP is a Linear layer that reduces the K dimension.
        # However, to capture "semantic dependencies", mixing d is important.
        # Let's assume the MLP is defined on the feature space first?
        # No, formula explicitly transposes.
        # Let's stick to the shape constraint: Output [d, K/2].
        # A common way to reduce K while mixing D is:
        # S_T: [d, K] -> reshape [d, K] -> Linear?
        # If we use nn.Linear(K, K/2), it acts on the last dim.
        
        # Alternative interpretation: Maybe the MLP is on the feature dim d?
        # But output shape is fixed by K/2.
        
        # Let's implement as: Reduce K dimension using a learned projection that might involve feature mixing?
        # Simplest valid interpretation of Eq 5 given shapes:
        # S_T is [d, K]. We want [d, K/2].
        # We can use a Linear layer on the K dimension.
        
        # To allow feature interaction, we might transpose back, process, then transpose?
        # But let's follow the text strictly: MLP(S^T).
        
        # Assuming MLP is a sequence of layers operating on the 'K' axis for each 'd'.
        # In PyTorch, Linear(K, K/2) works on the last dimension.
        
        reduced_S_T = self.mlp(S_T) # [d, K/2]
        
        # Transpose back to [K/2, d]
        S_new = reduced_S_T.T
        
        return S_new

class MiCoModule(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.route = ClusterRoute(d, k)
        self.reducer = ClusterReducer(d, k, k // 2)
        
    def forward(self, H, S):
        # Route: Updates H, produces intermediate S_context
        H_updated, S_context = self.route(H, S)
        
        # Reducer: Takes S_context, outputs new compact S
        S_new = self.reducer(S_context)
        
        return H_updated, S_new
```

#### 6. 实现提示
- **关键网络组件**：
    - `ClusterRoute`: 需要实现自定义的 Straight-Through Estimator 逻辑，或者在PyTorch中利用 `torch.argmax` 的前向行为和手动控制梯度的反向行为（例如 `grad_output = input_grad + (one_hot - input).detach()` 这种技巧，或者直接使用 `stop_gradient` 操作符如果框架支持，否则需写Custom Autograd Function）。
    - `ClusterReducer`: 简单的线性投影或MLP，关键在于维度变换 `Transpose -> Linear -> Transpose`。
- **重要超参数**：
    - 锚点数量 $K$：论文测试了 32, 64, 128，最佳为 64。
    - 特征维度 $d$：取决于编码器（TITAN通常为768或更高）。
    - MLP隐藏层维度：未明确说明，通常设为 $d$ 或 $d/2$。
- **归一化/激活方式**：
    - 相似度计算前对 $H$ 和 $S$ 进行 L2 归一化。
    - MLP中使用 ReLU 激活函数（推测，常见配置）。
- **维度对齐方式**：
    - `ClusterRoute` 输出更新的 $H$ ($M \times d$) 和新的 $S$ ($K \times d$)。
    - `ClusterReducer` 将 $S$ 的维度从 $K$ 减半到 $K/2$。如果有多层模块，需注意 $K$ 的变化。论文图示暗示每层可能保持 $K$ 或逐步减少？公式(5)明确说 $K \to K/2$。如果堆叠多层，第二层的输入锚点数将是第一层的一半。需确认是否每层重置 $K$ 或持续减半。通常为了保持表达能力，可能会在层间重新初始化或保持恒定，但根据公式推导，若直接堆叠，$K$ 会指数级下降。*假设*：论文可能在每层开始时重新初始化锚点，或者“Multi-layered”指的是对同一组锚点进行多次Route/Reducer迭代而不改变K？看图1，箭头指向下一层，且标注了 $K \times d \to (K/2) \times d \to K \times d$ (在Reducer后似乎又变回了K? 不，图1右侧Reducer输出是 $(K/2) \times d$，然后有一个箭头指回左侧的 $K \times d$ 吗？不，图1显示的是单层的内部结构。文字描述 "multi-layered... each organized by..."。如果Reducer减半，下一层输入就是 $K/2$。这可能是一个设计细节，复现时需确认是否每层独立或串联减半。鉴于消融实验中只提到了移除模块，未提及层数对K的影响，**建议假设**：如果是深层网络，可能需要每层结束后通过插值或复制将锚点数量恢复至 $K$，或者论文仅使用单层/两层且 $K$ 较小。但在伪代码中严格遵循公式 $K \to K/2$。
- **实现注意事项**：
    - `scatter_add` 用于高效聚合。
    - ST estimator 的实现是关键，确保梯度能回传到 $A_l$ 进而影响 $S$。

#### 7. 计算与资源开销
- **理论计算复杂度**：
    - Route: 相似度计算 $O(M \cdot K \cdot d)$。聚合 $O(M \cdot d)$。更新 $O(M \cdot d)$。
    - Reducer: MLP $O(d \cdot K \cdot K_{out})$ 或 $O(d \cdot K)$ 如果是线性。
    - 总体复杂度主要由 $M$（Patch数，通常数千）主导，线性于 $M$。
- **参数量**：
    - 主要参数来自 MLP 和 Anchor 初始化。Anchor 本身是可学习参数 $K \times d$。MLP 参数量相对较小。
- **FLOPs/MACs**：
    - 相比 Transformer 的 $O(M^2)$ 自注意力，MiCo 的复杂度更低，因为 $K \ll M$。
- **显存开销**：
    - 存储 $M \times K$ 的相似度矩阵 $A_l$ 可能较大，若 $M$ 很大需分块或优化。
- **推理速度**：
    - 由于避免了全局自注意力，推理速度应快于 TransMIL 等基于Transformer的方法。
- **论文是否提供效率对比**：
    - 文中主要对比准确率（C-Index, ACC, F1, AUC），未提供详细的 FLOPs 或 FPS 对比表格，但声称其有效性。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：计算病理学 WSIs 分析（生存预测、癌症亚型）。
- **可迁移到的任务/数据集**：任何基于 MIL 的任务，特别是当数据存在显著的类内空间分散性时（如遥感图像分割、视频动作识别中的片段关联）。
- **迁移所需调整**：
    - 编码器需适配新数据模态。
    - 锚点数量 $K$ 需根据数据复杂度调整。
    - 若数据没有明显的“空间”概念，Cluster Route 的语义聚合依然有效，但物理意义减弱。
- **适用条件**：实例数量 $M$ 较大，且存在潜在的语义聚类结构。
- **潜在限制**：对锚点初始化敏感；若 $K$ 设置不当（过大或过小）性能下降明显（如图3所示）。

#### 9. 实验与消融证据
- **主要性能结果**：
    - **生存预测**：平均 C-Index 0.680，优于 AMIL (0.642), TransMIL (0.649) 等。在 BLCA, BRCA, GBMLGG, HNSC, KIRC, KIRP, LUAD 七个数据集上均取得最好或接近最好的成绩。
    - **癌症亚型**：TCGA-BRCA 和 TCGA-NSCLC 上，ACC 0.927, F1 0.903, AUC 0.967，优于 PatchGCN (ACC 0.918) 等基线。
- **相对基线的提升**：在多个数据集上显著超越 Attention-based 和 Graph-based 方法。
- **相关消融实验**：
    - w/o Semantic Anchors Init: C-Index 降至 0.668。
    - w/o CluReducer: C-Index 降至 0.659。
    - w/o CluRoute: C-Index 降至 0.656。
    - 证明所有组件均不可或缺。
- **作者结论**：MiCo 能有效解决空间异质性问题，上下文感知聚类增强了语义连贯性。
- **证据是否充分**：在9个数据集上的广泛实验和详细的消融研究提供了充分证据。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 将聚类思想引入MIL，提出动态路由和锚点缩减机制，区别于传统的Attention或Graph池化。 |
| 技术可行性 | 高 | 基于标准的PyTorch操作，ST estimator有成熟实现方案，计算复杂度可控。 |
| 实现难度 | 中 | 需仔细处理ST estimator的梯度流和scatter_add的索引对齐，以及多层堆叠时的维度管理。 |
| 架构相关性 | 高 | 专为WSI的空间异质性设计，对具有强空间结构的MIL任务高度相关。 |
| 可迁移性 | 中 | 依赖于“语义锚点”的概念，可迁移到其他MIL任务，但需重新校准锚点数量和初始化策略。 |
| 计算成本 | 低 | 相比 $O(M^2)$ 的Transformer，复杂度为 $O(M \cdot K)$，效率较高。 |

#### 11. 一句话总结
MiCo 通过引入可学习的语义锚点，利用 Cluster Route 模块聚合跨区域的相似实例以增强上下文感知，并通过 Cluster Reducer 模块精简锚点以消除语义冗余，从而有效解决了全切片图像分析中的空间异质性问题。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **Context-Aware Clustering 范式**：将实例聚类中心作为“语义锚点”，并允许其在训练过程中动态更新，而非静态K-means。
- **Straight-Through Estimator 在MIL中的应用**：优雅地解决了离散分配（Assignments）不可导的问题，使得基于聚类的MIL方法可以进行端到端训练。
- **解耦的 Route 和 Reducer**：明确区分了“信息收集/路由”和“信息压缩/融合”两个步骤，这种模块化设计有助于理解和管理复杂的语义交互。

### 2. 方法之间的关系
- **Cluster Route** 类似于一种动态的、基于内容的注意力机制，但它强制将实例绑定到特定的语义原型（Anchor），具有更强的结构化约束。
- **Cluster Reducer** 类似于降维或瓶颈层，但它作用于语义空间（Anchor Space）而非特征空间，旨在优化语义表示的紧凑性和多样性。
- 两者共同构成了一个迭代优化的闭环：Route 利用当前 Anchor 细化实例，Reducer 根据细化的实例反馈优化 Anchor。

### 3. 复现可行性
- **代码是否公开**：是，GitHub 链接已提供。
- **方法描述是否完整**：是，公式、维度、算法步骤清晰。
- **关键配置是否明确**：是，$K=64$，Optimizer=Ranger，Epoch=200等均有说明。
- **预计复现难点**：
    1. **ST Estimator 的具体实现**：虽然原理清楚，但不同框架下的具体代码实现（如何精确阻断前向梯度而保留反向梯度）需要小心处理。
    2. **多层堆叠的维度处理**：Reducer 将 $K$ 减半，后续层如何处理 $K$ 的变化（是继续减半还是重置？）在正文中略显模糊，需参考代码或尝试两种策略。
    3. **TITAN 编码器的集成**：需要使用特定的预训练模型提取特征。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：ST Estimator 用于离散分配的梯度回传；基于锚点的语义聚合机制。
- **需要改造的设计**：针对非病理学的其他模态（如自然图像），可能需要调整相似度度量或锚点初始化方式。
- **可能形成的新研究思路**：
    - 将 Cluster Reducer 推广到更通用的语义压缩模块。
    - 探索动态调整 $K$ 值的策略，而不是固定值。
    - 结合对比学习，使 Anchor 更具判别性。

### 5. 阅读备注
- 论文强调“Context-Aware”，其核心在于通过 Anchor 建立的跨实例语义连接。
- 实验部分展示了良好的可解释性（Figure 2），Heatmap 与 Ground Truth 肿瘤区域对齐良好，证明了模型确实关注了有意义的病理结构。
- 注意区分 $A_l$ (相似度) 和 $\hat{A}_l$ (分配矩阵) 的区别，前者用于梯度，后者用于聚合。
