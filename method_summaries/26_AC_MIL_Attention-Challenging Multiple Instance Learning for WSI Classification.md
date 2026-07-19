# 26_AC_MIL_Attention-Challenging Multiple Instance Learning for WSI Classification 方法总结

> 证据说明：输入为完整论文文本（含正文及附录）。公式提取基本完整，关键超参数和实验设置均有明确说明。代码仓库链接已提供。

## 一、论文基本信息

- **论文标题**：Attention-Challenging Multiple Instance Learning for Whole Slide Image Classification
- **作者**：Yunlong Zhang, Honglin Li, Yunxuan Sun, Sunyi Zheng, Chenglu Zhu, Lin Yang
- **发表年份**：2024 (arXiv:2311.07125v4)
- **会议/期刊**：arXiv预印本 (未注明最终录用会议，但基于CVPR/MICCAI等常见投稿习惯推测可能为相关顶会或期刊，此处以arXiv ID为准)
- **论文链接/DOI/arXiv ID**：https://arxiv.org/abs/2311.07125
- **代码仓库**：https://github.com/dazhangyu123/ACMIL
- **研究任务**：全切片图像（WSI）分类
- **数据模态**：数字病理学图像（H&E染色），提取为Patch级别的特征向量

## 二、论文整体概述

### 1. 核心问题
在基于多实例学习（MIL）的WSI分类中，现有的注意力机制往往过度集中在少数具有判别性的实例上（即“注意力值集中”现象）。这种集中导致模型容易过拟合，泛化能力下降。论文通过熵与损失函数的负相关性分析，证实了注意力值的过度集中与过拟合紧密相关。

### 2. 整体方法
提出 **ACMIL (Attention-Challenging MIL)** 框架，基于基线 ABMIL 改进，包含两个核心模块：
1.  **Multiple Branch Attention (MBA)**：引入多个注意力分支，每个分支捕捉不同的判别性模式（patterns），并通过语义正则化和多样性正则化确保分支间的差异性，从而捕获更多样的判别性实例。
2.  **Stochastic Top-K Instance Masking (STKIM)**：在训练阶段，随机屏蔽Top-K高注意力值的实例，并将它们的注意力权重重新分配给剩余实例，以抑制对极少数实例的过度依赖。推理时不使用该掩码。

### 3. 主要贡献
- 揭示了WSI分类中注意力值集中与过拟合之间的强负相关关系。
- 提出了MBA模块，通过多分支结构和多样性损失解决单一分支无法捕捉所有判别模式的问题。
- 提出了STKIM策略，通过随机屏蔽高注意力实例来分散注意力，且无需教师模型或额外预训练。
- 在三个数据集（CAMELYON16, BRACS, LBC）和两种骨干网络（ResNet-18, ViT-S/16）上取得了SOTA性能。

## 三、方法总结

### 方法 1：Multiple Branch Attention (MBA)

#### 1. 核心思想与解决的问题
- **目标问题**：单个注意力分支倾向于捕捉简单的或特定的判别模式，忽略其他复杂的模式（如肿瘤的不同形态、纹理差异），导致信息遗漏。
- **现有方法的局限**：标准ABMIL使用单一注意力头，难以覆盖所有判别性子结构；多头注意力（MHA）虽然能捕捉不同概念，但缺乏显式的多样性约束，可能导致头部学习到相似的概念。
- **核心思想**：并行使用 $M$ 个独立的注意力分支，每个分支专注于捕捉一种特定的判别模式。
- **创新点**：引入了**多样性正则化（Diversity Regularization）**，强制不同分支生成的热力图（attention maps）之间保持低相似度，从而确保每个分支学习独特的判别特征。

#### 2. 详细结构与数据流
- **输入**：实例特征集合 $H = \{h_n\}_{n=1}^N$，其中 $h_n \in \mathbb{R}^d$。
- **处理流程**：
    1.  对于第 $i$ 个分支 ($i=1...M$)，计算其注意力权重 $a_{in}$ 和对应的模式嵌入 $z_i$。
    2.  每个分支末端连接一个MLP分类器，输出预测 $\hat{Y}_i$，并计算语义损失 $L_p$。
    3.  计算所有分支间热力图的余弦相似度，计算多样性损失 $L_d$。
    4.  对所有分支的注意力权重取平均得到Bag级注意力 $a$。
    5.  利用平均注意力 $a$ 聚合实例特征得到Bag嵌入 $z$，或直接对各个分支的模式嵌入 $z_i$ 取平均。
- **输出**：Bag级预测 $\hat{Y}$，以及各分支的中间表示用于损失计算。
- **模块在整体网络中的位置**：替换ABMIL中的单注意力聚合层，位于特征提取器之后，Bag分类器之前。
- **与其他模块的连接方式**：MBA的输出作为STKIM的输入（或在STKIM之后进行聚合，具体见下文STKIM部分，通常STKIM作用于注意力生成后、聚合前）。根据图4，STKIM作用于MBA生成的注意力值上。

#### 3. 数学公式

**Gated Attention Mechanism (用于每个分支):**
$$ a_{in} = \frac{\exp\{w^T(\tanh(V_1 h_n) \odot \text{sigm}(V_2 h_n))\}}{\sum_{j=1}^N \exp\{w^T(\tanh(V_1 h_j) \odot \text{sigm}(V_2 h_j))\}} $$
其中 $V_1, V_2 \in \mathbb{R}^{L \times M}, w \in \mathbb{R}^{L \times 1}$ 为可学习参数。

**Semantic Regularization Loss (语义正则化):**
$$ L_p = -\frac{1}{M} \sum_{i=1}^M \left[ Y \log \hat{Y}_i + (1-Y) \log (1-\hat{Y}_i) \right] $$
其中 $\hat{Y}_i = g_i(z_i)$ 是第 $i$ 个分支的预测。

**Diversity Regularization Loss (多样性正则化):**
$$ L_d = \frac{2}{M(M-1)} \sum_{i=1}^M \sum_{j=i+1}^M \cos(a_i, a_j) $$
其中 $a_i = \{a_{i1}, \dots, a_{iN}\}$ 是第 $i$ 个分支的所有注意力值组成的向量（即热力图），$\cos(\cdot)$ 为余弦相似度函数。

**Bag Aggregation:**
$$ a = \frac{1}{M} \sum_{i=1}^M a_i $$
$$ z = \sum_{n=1}^N a_n h_n = \frac{1}{M} \sum_{i=1}^M \left( \sum_{n=1}^N a_{in} h_n \right) = \frac{1}{M} \sum_{i=1}^M z_i $$

**Total Loss:**
$$ L = L_b + L_p + L_d $$
其中 $L_b$ 为Bag级的交叉熵损失。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | Instance Features $H$ | $(N, d)$ | $N$为实例数，$d$为特征维度(ResNet:512->256, ViT:384->128) |
| MBA内部 | Attention Weights $A_i$ | $(N)$ | 第 $i$ 个分支的注意力分布 |
| MBA内部 | Pattern Embedding $z_i$ | $(d')$ | 第 $i$ 个分支聚合后的Bag特征 |
| MBA内部 | Branch Prediction $\hat{Y}_i$ | $(1)$ | 第 $i$ 个分支的二分类概率 |
| 多样性损失 | Heatmap Similarity | Scalar | $M \times M$ 矩阵的平均余弦相似度 |
| 输出 | Bag Prediction $\hat{Y}$ | $(1)$ | 最终Bag分类概率 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.V1 = nn.Linear(input_dim, hidden_dim)
        self.V2 = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (N, input_dim)
        act1 = torch.tanh(self.V1(x))
        act2 = torch.sigmoid(self.V2(x))
        energy = self.w(act1 * act2) # (N, 1)
        attention = F.softmax(energy, dim=0) # (N, 1)
        return attention.squeeze(-1) # (N,)

class MBA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_branches=5, bag_hidden_dim=128):
        super().__init__()
        self.num_branches = num_branches
        self.attention_heads = nn.ModuleList([
            GatedAttention(input_dim, hidden_dim) for _ in range(num_branches)
        ])
        # MLP for each branch prediction
        self.branch_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, bag_hidden_dim),
                nn.ReLU(),
                nn.Linear(bag_hidden_dim, 1)
            ) for _ in range(num_branches)
        ])
        
    def forward(self, H):
        # H: (N, input_dim)
        N = H.size(0)
        all_attention_maps = []
        pattern_embeddings = []
        branch_predictions = []
        
        for i in range(self.num_branches):
            # 1. Get attention weights
            attn_weights = self.attention_heads[i](H) # (N,)
            all_attention_maps.append(attn_weights)
            
            # 2. Aggregate to get pattern embedding z_i
            # z_i = sum(a_in * h_n)
            z_i = torch.sum(attn_weights.unsqueeze(1) * H, dim=0) # (input_dim,)
            pattern_embeddings.append(z_i)
            
            # 3. Predict from pattern embedding
            pred_i = self.branch_predictors[i](z_i).squeeze(-1) # (1,)
            branch_predictions.append(pred_i)
            
        # Average attention maps for final aggregation
        avg_attn = torch.stack(all_attention_maps).mean(dim=0) # (N,)
        
        # Final Bag embedding using average attention
        bag_embedding = torch.sum(avg_attn.unsqueeze(1) * H, dim=0) # (input_dim,)
        
        return bag_embedding, avg_attn, all_attention_maps, branch_predictions

    def compute_loss(self, bag_embedding, all_attention_maps, branch_predictions, Y):
        # L_p: Semantic loss for each branch
        probs = torch.sigmoid(torch.stack(branch_predictions)) # (M,)
        # Assuming binary classification, Y is scalar 0 or 1
        # Expand Y to match batch size if necessary, here assuming single sample per step for simplicity of formula mapping
        # In code, usually handle batch dimension. Here Y is target label.
        ce_loss_per_branch = F.binary_cross_entropy_with_logits(
            torch.stack(branch_predictions), 
            torch.full_like(torch.stack(branch_predictions), float(Y))
        )
        L_p = ce_loss_per_branch.mean()
        
        # L_d: Diversity loss
        # Calculate cosine similarity between all pairs of attention maps
        maps_tensor = torch.stack(all_attention_maps) # (M, N)
        # Normalize maps
        maps_norm = F.normalize(maps_tensor, p=2, dim=1)
        # Cosine similarity matrix (M, M)
        sim_matrix = torch.mm(maps_norm, maps_norm.T)
        # Remove diagonal (self-similarity)
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_off_diag = sim_matrix[~mask].view(sim_matrix.size(0), -1)
        L_d = sim_off_diag.mean()
        
        # L_b: Bag level loss
        # Note: The paper defines L_b separately, but typically uses the final bag_embedding
        # For simplicity, we assume a separate classifier head on bag_embedding exists outside this module
        # Or we can use the mean of branch predictions? Paper says "bag prediction is generated based on aggregated bag embedding"
        # Let's assume L_b is computed externally or via a shared head. 
        # Here we just return components needed for L_b calculation if needed.
        
        return L_p, L_d
```

#### 6. 实现提示
- **关键网络组件**：`GatedAttention` 类需严格遵循公式(3)。`MBA` 类包含 $M$ 个并行的注意力头和MLP。
- **重要超参数**：
    - $M$ (Branches数量): CAMELYON16/ResNet设为2，其他情况设为5。消融实验显示 $M=5$ 效果最佳。
    - Hidden Dim: ResNet backbone后接FC层降至256维，ViT降至128维。
- **归一化/激活方式**：注意力权重使用Softmax；Gated Attention中使用Tanh和Sigmoid；MLP中使用ReLU。
- **维度对齐方式**：多样性损失计算前需对注意力向量进行L2归一化 (`F.normalize`)。
- **实现注意事项**：多样性损失 $L_d$ 的计算涉及所有分支对的余弦相似度，当 $M$ 较大时注意内存开销，但通常 $M$ 较小（如5）。

#### 7. 计算与资源开销
- **理论计算复杂度**：MBA将计算量增加约 $M$ 倍（因为 $M$ 个并行分支），但由于分支间无交互，易于并行加速。
- **参数量**：增加 $M$ 个注意力头的参数和 $M$ 个小型MLP分类器的参数。相对于整个WSI分类系统，增量很小。
- **FLOPs/MACs**：Table 7显示，加入MBA后FLOPs仅从201M微增至202M (ResNet) 或 84M至85M (ViT)，几乎不变。
- **显存开销**：Table 7显示峰值显存略有增加（例如ResNet下从0.3G到0.3G，ViT下从0.2G到0.2G，实际数值变化极小，主要受Batch Size影响）。
- **推理速度**：由于需要运行 $M$ 个分支，训练时间增加（Table 7显示Time从8.0s增至11.6s），但推理时若合并注意力则速度影响不大，或者推理时仍保留多分支逻辑。*注：论文指出STKIM在推理时移除，但未明确说明MBA在推理时是否合并。通常MBA作为架构一部分，推理时需执行所有分支并平均。*
- **论文是否提供效率对比**：是，Table 6和7提供了详细的FLOPs、时间和显存对比。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI弱监督分类。
- **可迁移到的任务/数据集**：任何基于MIL的任务，特别是那些存在长尾分布或多模式判别特征的视觉分类任务（如自然图像细粒度分类、视频动作识别）。
- **迁移所需调整**：调整 $M$ 的大小以适应不同任务的复杂度；可能需要调整多样性损失的权重系数（论文中似乎直接相加，隐含权重为1）。
- **适用条件**：实例数量 $N$ 不宜过小，否则多样性正则化可能不稳定。
- **潜在限制**：超参数 $M$ 和 $K$ 需要针对特定数据集调优；未考虑实例间的空间拓扑关系。

#### 9. 实验与消融证据
- **主要性能结果**：在CAMELYON16 (ViT) AUC达到0.974，BRACS (ViT) AUC达到0.888，LBC (ViT) AUC达到0.901。
- **相对基线的提升**：相比ABMIL，ACMIL在大多数指标上有显著提升（平均提升4.4个点）。
- **相关消融实验**：
    - Table 3b: 移除多样性损失 $L_d$ 导致性能大幅下降（如CAMELYON F1从0.954降至0.901），证明 $L_d$ 至关重要。
    - Fig 7: 验证了 $M, K, p$ 的影响。
- **作者结论**：MBA和STKIM均有效，结合使用效果最好；多样性损失不可或缺。
- **证据是否充分**：充分，包括可视化（UMAP, Heatmap）、定量指标和消融实验。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 提出多样性正则化约束注意力热力图，视角独特；STKIM简单有效。 |
| 技术可行性 | 高 | 基于标准MIL框架修改，易于实现，无需复杂的新算子。 |
| 实现难度 | 低 | 代码逻辑清晰，依赖标准PyTorch操作。 |
| 架构相关性 | 高 | 专门针对MIL的注意力机制缺陷设计。 |
| 可迁移性 | 中 | 主要针对WSI的多模式特性，在其他领域需验证多样性假设的有效性。 |
| 计算成本 | 低 | 仅轻微增加计算和存储开销。 |

#### 11. 一句话总结
ACMIL通过多分支注意力机制配合多样性正则化来捕捉多样化的判别模式，并结合随机Top-K实例掩码技术抑制注意力过度集中，从而有效缓解WSI分类中的过拟合问题。

### 方法 2：Stochastic Top-K Instance Masking (STKIM)

#### 1. 核心思想与解决的问题
- **目标问题**：在ABMIL等模型中，极少数的实例（Top-K）占据了绝大部分的注意力权重（例如Top-10占据>85%的注意力），导致大量次级但仍有判别力的实例被忽略，加剧过拟合。
- **现有方法的局限**：WENO和MHIM-MIL也采用掩码策略，但它们通常需要教师模型预训练、掩码比例过大（如95个实例或1%）或确定性掩码，增加了复杂性和计算负担。
- **核心思想**：在训练阶段，以一定概率 $p$ 随机将Top-K高注意力值的实例的注意力权重置零，然后将这些权重重新归一化分配给剩余实例。推理阶段禁用此操作。
- **创新点**：无需教师模型；随机性避免了模型对特定“硬”实例的依赖；简单的实现方式。

#### 2. 详细结构与数据流
- **输入**：原始注意力权重向量 $a = [a_1, ..., a_N]$。
- **处理流程**：
    1.  对 $a$ 进行排序，找出Top-K索引。
    2.  生成随机掩码：对于Top-K中的每个元素，以概率 $p$ 将其值设为0。
    3.  重新归一化：将剩余非零元素的注意力值除以它们的总和，使总和恢复为1。
- **输出**：修正后的注意力权重向量 $a'$。
- **模块在整体网络中的位置**：位于注意力权重计算之后，特征聚合（$z = \sum a_n h_n$）之前。
- **与其他模块的连接方式**：接收来自MBA或单分支注意力的输出，处理后传递给聚合层。

#### 3. 数学公式

**Masking Operation:**
$$
a'_n = 
\begin{cases} 
0, & \text{with probability } p \text{ and } n \in \text{Top-K indices} \\
a_n, & \text{otherwise}
\end{cases}
$$

**Renormalization:**
$$
\tilde{a}_n = \frac{a'_n}{\sum_{j=1}^N a'_j}
$$
最终使用的注意力权重为 $\tilde{a}$。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | Original Attention $a$ | $(N)$ | 原始注意力分布 |
| 中间 | Masked Attention $a'$ | $(N)$ | 部分Top-K元素被置零 |
| 输出 | Normalized Attention $\tilde{a}$ | $(N)$ | 重新归一化后的注意力分布 |

#### 5. 实现伪代码

```python
def stochastic_top_k_masking(attention_weights, k=10, p=0.6):
    """
    attention_weights: Tensor of shape (N,), summing to 1
    k: int, number of top instances to consider for masking
    p: float, probability of masking a top-k instance
    """
    # 1. Find top-k indices
    # Ensure attention_weights is on the correct device
    top_k_values, top_k_indices = torch.topk(attention_weights, k)
    
    # 2. Create a mask tensor
    # Initialize with zeros, then fill with original values
    masked_weights = attention_weights.clone()
    
    # Generate random numbers for top-k indices
    rand_vals = torch.rand(k, device=attention_weights.device)
    
    # If rand_val < p, set the weight to 0
    # Note: The paper says "randomly set ... to 0 with probability p"
    mask_bool = rand_vals < p
    
    # Apply mask
    masked_weights[top_k_indices[mask_bool]] = 0.0
    
    # 3. Renormalize
    sum_masked = masked_weights.sum()
    if sum_masked > 0:
        normalized_weights = masked_weights / sum_masked
    else:
        # Fallback if all top-k are masked and others are 0 (rare)
        normalized_weights = masked_weights 
        
    return normalized_weights
```

#### 6. 实现提示
- **关键网络组件**：`torch.topk` 用于定位Top-K，`torch.rand` 用于生成随机掩码。
- **重要超参数**：
    - $K$: 默认10。消融实验表明 $K$ 不敏感，10即可。
    - $p$: 默认0.6或0.8。$p=1.0$（全部屏蔽）会导致性能下降。
- **归一化/激活方式**：必须保证输出向量的和为1。
- **实现注意事项**：**仅在训练阶段启用**。推理时直接使用原始注意力权重。可以使用 `torch.no_grad()` 包裹推理部分的注意力计算，或者在Module的 `train()/eval()` 切换中控制。
- **依赖的特殊算子或第三方库**：无特殊依赖，标准PyTorch。

#### 7. 计算与资源开销
- **理论计算复杂度**：排序操作 $O(N \log N)$ 或 $O(N)$ (使用topk)，掩码和归一化 $O(N)$。相对于 $N$ 很大（数千至数万）的WSI Patch数量，这部分开销极小。
- **参数量**：0。纯操作符，无可学习参数。
- **FLOPs/MACs**：极低，Table 6显示STKIM的FLOPs与基线ABMIL完全一致（201M/84M）。
- **显存开销**：极低，Table 6显示显存占用与ABMIL一致。
- **推理速度**：推理时不使用STKIM，因此推理速度与基线相同。
- **论文是否提供效率对比**：是，Table 6对比了STKIM与MHIM-MIL，显示STKIM在速度和显存上远优于MHIM-MIL。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI分类训练阶段。
- **可迁移到的任务/数据集**：任何使用注意力机制且存在“赢家通吃”现象的MIL任务，或Transformer中注意力过于集中的场景。
- **迁移所需调整**：需确定合适的 $K$ 和 $p$ 值。
- **适用条件**：注意力机制能够产生有意义的Top-K区分。
- **潜在限制**：如果Top-K实例确实是唯一的关键信息，强行屏蔽可能会丢失关键信号（尽管论文认为WSI中存在更多判别实例）。

#### 9. 实验与消融证据
- **主要性能结果**：结合STKIM后，模型泛化能力增强。
- **相对基线的提升**：单独使用STKIM也能带来性能提升（Fig 7蓝色虚线高于橙色虚线）。
- **相关消融实验**：
    - Fig 10: 展示STKIM显著降低了Top-K累积注意力值（如从0.87降至0.6）。
    - Table 3a: 测试阶段使用STKIM（T-STKIM）反而略微降低性能，证明其仅适用于训练。
    - Table 5: STKIM的掩码策略优于WENO和MHIM-MIL的策略。
- **作者结论**：STKIM能有效分散注意力，且无需额外训练成本。
- **证据是否充分**：充分，有可视化、定量对比和策略对比。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 中 | 思路类似Dropout/Cutout，但应用于MIL注意力分布，针对性强。 |
| 技术可行性 | 高 | 实现极其简单，无副作用。 |
| 实现难度 | 低 | 几行代码即可实现。 |
| 架构相关性 | 高 | 直接作用于MIL的核心聚合步骤。 |
| 可迁移性 | 高 | 通用性强，可用于多种注意力模型。 |
| 计算成本 | 极低 | 几乎零额外开销。 |

#### 11. 一句话总结
STKIM通过在训练时随机屏蔽并重新分配Top-K高注意力权重，迫使模型关注更多判别性实例，以极低的计算代价提升了模型的鲁棒性和泛化能力。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **多样性正则化 ($L_d$)**：通过约束注意力热力图的余弦相似度来强制多分支学习不同特征，这一思路简洁且有效，可推广到其他多视图或多头注意力融合任务中。
- **STKIM的随机性设计**：相比于确定性的Hard Example Mining或Teacher-Student框架，这种轻量级的随机扰动策略在保持高性能的同时极大降低了工程复杂度。

### 2. 方法之间的关系
- **互补关系**：MBA解决了“广度”问题（捕捉更多样化的模式），STKIM解决了“深度/集中度”问题（防止少数实例垄断注意力）。两者结合形成了完整的注意力挑战机制。
- **层级关系**：MBA是主干架构的扩展，STKIM是训练策略的增强。STKIM可以独立于MBA使用（作用于单分支），也可以与MBA结合使用。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，公式、超参数、预处理细节均在文中或附录中给出。
- **关键配置是否明确**：是，明确了 $M, K, p$ 的设置建议及不同数据集下的推荐值。
- **预计复现难点**：
    - 数据预处理（CLAM风格的阈值分割和Patch提取）可能需要仔细调试。
    - 多样性损失的具体实现需注意梯度流动和数值稳定性（虽然公式简单，但在大规模 $N$ 下计算余弦相似度矩阵需注意内存，尽管 $M$ 很小，所以不是瓶颈）。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：STKIM可作为任何基于注意力的MIL模型的即插即用模块。
- **需要改造的设计**：MBA需要修改网络结构以支持多分支并行和额外的损失计算，需根据具体任务调整分支数 $M$。
- **可能形成的新研究思路**：
    - 探索自适应的 $K$ 或 $p$ 值，而非固定超参数。
    - 将多样性正则化应用于其他类型的注意力机制（如Self-Attention中的Head间多样性）。
    - 结合空间信息（论文提到的局限性）与MBA，构建空间感知的多分支MIL。

### 5. 阅读备注
- 论文强调了**解释性**的提升，通过UMAP和Heatmap展示了ACMIL比ABMIL更均匀地覆盖病灶区域，这符合病理学家对诊断可信度的需求。
- 尽管在分类任务上表现优异，但论文承认在**精确定位**（Localization/FROC）任务上不如全监督方法，这是一个重要的局限性提示。
