# 23_DGR_MIL_Exploring Diverse Global Representation in MIL for WSI Classification 方法总结

> 证据说明：输入为完整论文（含正文及附录），PDF文本提取完整，无页面或公式缺失。

## 一、论文基本信息

- **论文标题**：DGR-MIL: Exploring Diverse Global Representation in Multiple Instance Learning for Whole Slide Image Classification
- **作者**：Wenhui Zhu, Xiwen Chen, Peijie Qiu, Aristeidis Sotiras, Abolfazl Razi, Yalin Wang
- **发表年份**：2024 (arXiv:2407.03575v1)
- **会议/期刊**：未明确标注具体会议/期刊名称（arXiv预印本）
- **论文链接/DOI/arXiv ID**：https://arxiv.org/abs/2407.03575
- **代码仓库**：https://github.com/ChongQingNoSubway/DGR-MIL
- **研究任务**：全切片图像（WSI）分类（弱监督学习）
- **数据模态**：数字病理图像（Whole Slide Images, WSIs）

## 二、论文整体概述

### 1. 核心问题
现有的多实例学习（MIL）方法主要关注实例间的相关性建模，而忽视了实例内在的多样性（Diversity）。在WSI中，即使是同一类别的实例（如肿瘤区域）也存在表型、大小和空间分布上的差异；不同类别的实例也可能因边界效应而相似。现有聚类/原型方法通过注意力分数作为伪标签来建模多样性，存在“鸡生蛋”问题且计算成本高。

### 2. 整体方法
提出了一种基于多样化全局表示（Diverse Global Representation, DGR）的MIL聚合方法。
1.  **全局向量建模**：引入一组可学习的“全局向量”（Global Vectors）作为所有实例的摘要。
2.  **交叉注意力机制**：将实例间的相关性转化为实例嵌入与预定义全局向量之间的相似度，通过交叉注意力（Cross-Attention）实现，降低复杂度。
3.  **Tokenized Global Vector**：添加一个特殊的Token向量来汇总其他全局向量，以捕捉最具判别性的全局上下文并抑制负样本信息。
4.  **多样性学习策略**：
    *   **正例对齐（Positive Instance Alignment）**：利用动量更新的正/负袋中心，通过Triplet Loss迫使全局向量靠近正例中心，远离负例中心。
    *   **多样性损失（Diversity Loss）**：基于行列式点过程（DPP）的理论性质，最小化全局向量协方差矩阵行列式的负对数，理论上保证全局向量正交，从而最大化多样性。

### 3. 主要贡献
1.  从多样性角度重新审视WSI中的实例建模。
2.  提出DGR-MIL模型，通过可学习的全局向量显式建模实例多样性。
3.  设计了正例对齐机制和基于DPP的高效多样性损失。
4.  在CAMELYON-16和TCGA-lung数据集上显著优于SOTA方法。

## 三、方法总结

### 方法 1：DGR-MIL 聚合模块与多样性学习

#### 1. 核心思想与解决的问题
- **目标问题**：解决传统MIL忽略实例多样性导致模型陷入局部最优或错误聚合的问题，同时避免自注意力机制在长序列（WSI包含成千上万patch）下的二次方计算复杂度。
- **现有方法的局限**：AB-MIL假设实例独立同分布；Transformer-based MIL（如TransMIL）虽考虑相关性但计算昂贵且未显式建模多样性；Prototype-based MIL依赖伪标签且原型数量受限。
- **核心思想**：使用少量可学习的“全局向量”代替大量实例进行交互。通过交叉注意力计算实例与全局向量的关系，既保留了实例间的语义关联，又通过全局向量的多样性约束捕捉了数据的异质性。
- **创新点**：
    1.  用全局向量作为Query，实例作为Key/Value的交叉注意力机制替代自注意力。
    2.  引入Tokenized Global Vector专门用于最终分类决策，抑制噪声。
    3.  基于DPP理论推导出的正交性多样性损失，无需迭代采样即可高效优化多样性。

#### 2. 详细结构与数据流
- **输入**：
    - 实例特征集合 $\tilde{X} = \{\tilde{x}_1, \dots, \tilde{x}_n\}$，其中 $\tilde{x}_i \in \mathbb{R}^L$。
    - 可学习的全局向量 $G = [g_1^\top, \dots, g_K^\top]^\top \in \mathbb{R}^{K \times L}$。
    - Tokenized全局向量 $g_{token} \in \mathbb{R}^L$。
- **处理流程**：
    1.  **预处理**：对实例 $\tilde{X}$ 和全局向量 $G$ 分别应用前馈网络（FFN）进行嵌入增强（文中提及，但未给出具体FFN结构，通常指线性投影+激活）。
    2.  **Nystrom Attention (可选前置)**：为了进一步过滤背景和增加全局向量差异，先对 $\tilde{X}$ 和 $G$ 分别应用Nystrom Self-Attention（文中Fig. 2所示，但在公式推导中主要描述Cross-Attention部分）。
    3.  **多头交叉注意力 (MHCA)**：
        - Query ($Q$): 来自全局向量 $G$。
        - Key ($K$), Value ($V$): 来自实例 $\tilde{X}$。
        - 计算多头注意力输出并拼接，经过线性投影得到 $MHCA(G, \tilde{X})$。
    4.  **Tokenized 聚合**：
        - 构建扩展的全局向量集 $\tilde{G} = \{g_{token}, g_1, \dots, g_K\}$。
        - 使用 $g_{token}$ 与每个实例 $\tilde{x}_i$ 的点积计算重要性权重 $\sigma(\tilde{x}_i)$（见公式6）。
        - 加权求和实例特征得到Bag-level表示，或直接使用经过MHCA处理后的 $g_{token}$ 对应的输出进行分类。*注：文中公式(6)显示使用softmax后的注意力分数对实例进行加权，或者直接使用Token向量作为Bag表示。根据Fig 2和Text，最终分类器接收的是经过Cross-Attention层输出的Tokenized Global Vector representation。*
    5.  **多样性学习**：
        - 计算正/负袋中心的动量更新。
        - 计算Triplet Loss ($L_{tri}$)。
        - 计算DPP Diversity Loss ($L_{div}$)。
- **输出**：Bag级别的预测概率 $\hat{Y}$。
- **模块在整体网络中的位置**：位于特征提取器（Feature Extractor）之后，Bag Classifier之前。
- **与其他模块的连接方式**：接收固定维度的实例特征，输出Bag级特征供分类头使用；同时输出全局向量用于Loss计算。

#### 3. 数学公式

**Instance Correlation as Cross Attention:**
$$
\text{head}_h(G, \tilde{X}) = \text{Attention}(Q_h, K_h, V_h)
$$
$$
Q_h = G W_Q^h, \quad K_h = \tilde{X} W_K^h, \quad V_h = \tilde{X} W_V^h
$$
$$
\text{MHCA}(G, \tilde{X}) = \text{concat}(\text{head}_1; \dots; \text{head}_H) W_O
$$
其中 $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{L \times L/H}$ 为可学习参数，$H$ 为头数。复杂度从 $O(n^2)$ 降至 $O(Kn)$。

**Importance Score (for Tokenized Vector):**
$$
\sigma(\tilde{x}_i) = \text{softmax}\left( \frac{(g_{token} W_Q^h)(\tilde{x}_i W_K^h)^\top}{\sqrt{d_k}} \right)
$$
*(注：此处公式6仅展示了单个头的注意力分数计算，实际Bag表示可能由所有头平均或拼接后处理，或直接使用Token向量本身作为Summary)*

**Positive Instance Alignment (Triplet Loss):**
定义正/负袋中心（动量更新）：
$$
\tilde{x}_c^{(pos)} = m \tilde{x}_c^{(pos)} + (1-m) \frac{1}{|I_{pos}|} \sum_{i \in I_{pos}} \tilde{x}_i
$$
$$
\tilde{x}_c^{(neg)} = m \tilde{x}_c^{(neg)} + (1-m) \frac{1}{|I_{neg}|} \sum_{i \in I_{neg}} \tilde{x}_i
$$
Triplet Loss:
$$
L_{tri} = \sum_{k=1}^K [\max(0, d(g_k, \tilde{x}_c^{(pos)}) - d(g_k, \tilde{x}_c^{(neg)}) + \mu)]
$$
其中 $d$ 为余弦距离，$\mu$ 为margin。

**Diversity Learning (DPP-based Loss):**
基于定理1，最大化多样性等价于使全局向量正交。
$$
L_{div} = -\log \det(G G^\top + \epsilon I)
$$
约束条件：$\|g_i\| = 1$。$\epsilon = 1 \times 10^{-10}$ 防止数值不稳定。

**Final Objective:**
$$
L_{final} = L_{ce} + \lambda_{tri} L_{tri} + \lambda_{div} L_{div}
$$

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入实例 | $\tilde{X}$ | $(n, L)$ | $n$为patch数量，$L$为特征维度(如512, 1024) |
| 全局向量 | $G$ | $(K, L)$ | $K$为全局向量数量(实验设3或5)，$L$同输入 |
| Token向量 | $g_{token}$ | $(1, L)$ | 额外的可学习向量 |
| 扩展全局向量 | $\tilde{G}$ | $(K+1, L)$ | 包含Token的全局向量集 |
| Query | $Q_h$ | $(K, L/H)$ | 全局向量投影 |
| Key/Value | $K_h, V_h$ | $(n, L/H)$ | 实例投影 |
| 注意力输出 | MHCA Output | $(K, L)$ | 经Head拼接和线性投影后 |
| Bag表示 | $Z_{bag}$ | $(1, L)$ | 由Token向量或其变换得到 |
| 预测 | $\hat{Y}$ | scalar / prob | 二分类概率 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DGR_MIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_global_vectors=5, num_heads=4):
        super(DGR_MIL, self).__init__()
        self.K = num_global_vectors
        self.H = num_heads
        self.d_k = hidden_dim // num_heads
        
        # 可学习的全局向量 G 和 Token g_token
        self.G = nn.Parameter(torch.randn(self.K, hidden_dim))
        self.g_token = nn.Parameter(torch.randn(1, hidden_dim))
        
        # 线性投影层 (对应 W_Q, W_K, W_V, W_O)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim) # 简化：实际应分头
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim, 1)
        
        # 超参数
        self.momentum = 0.4
        self.margin = 0.5
        self.eps = 1e-10
        
        # 动量中心 (需在训练循环中维护)
        self.register_buffer('pos_center', torch.zeros(hidden_dim))
        self.register_buffer('neg_center', torch.zeros(hidden_dim))
        self.pos_count = 0
        self.neg_count = 0

    def forward(self, X, bag_labels=None):
        """
        X: (B, N, L) - Batch of bags, N instances, L features
        bag_labels: (B,) - Bag labels (0 or 1)
        """
        B, N, L = X.shape
        
        # 1. Cross-Attention Mechanism
        # Q from Global Vectors (K, L) -> expand to (B, K, L) if needed, 
        # but usually G is shared across batch or projected per instance? 
        # Paper implies G is learnable parameters, likely shared or batched.
        # Assuming G shape (K, L) broadcasted or repeated.
        
        # Project Q, K, V
        # Note: Standard Multi-Head Attention implementation details omitted for brevity
        # Q: (B, K, H, d_k)
        # K, V: (B, N, H, d_k)
        
        # Simplified attention calculation assuming single head logic extended
        # Q = G @ W_Q.T -> (K, L) -> (B, K, L)
        # K = X @ W_K.T -> (N, L) -> (B, N, L)
        
        # Compute Attention Scores
        # Sim(Q, K) = Q @ K.T / sqrt(d_k) -> (B, K, N)
        attn_weights = torch.softmax(
            torch.matmul(self.W_Q(self.G.unsqueeze(0).expand(B, -1, -1)), 
                         self.W_K(X).transpose(1, 2)) / (self.d_k ** 0.5), 
            dim=-1
        )
        
        # Weighted Sum of Values
        # Out: (B, K, L)
        out = torch.matmul(attn_weights, self.W_V(X))
        out = self.W_O(out.transpose(1, 2)).transpose(1, 2) # (B, K, L)
        
        # 2. Tokenized Global Vector Processing
        # The paper suggests using g_token to compute importance or as the summary.
        # Eq 6 computes sigma(x_i) using g_token.
        # Let's assume the final bag representation is derived from the token's interaction
        # or simply the output corresponding to the token position if we concatenated it.
        # However, Fig 2 shows Token is part of G. 
        # Let's construct full G_tilde = [g_token, G]
        
        G_tilde = torch.cat([self.g_token, self.G], dim=0) # (K+1, L)
        
        # Re-run attention with G_tilde as Query to get representations for all global vectors
        # This step might be implicit in "output of tokenized global vectors after cross-attention"
        # For simplicity, we take the first element of 'out' if we included token in G initially,
        # or re-calculate specifically for token.
        
        # Recalculating specifically for Token to get Bag Representation Z_bag
        Q_token = self.W_Q(self.g_token.expand(B, -1, -1)) # (B, 1, L)
        K_all = self.W_K(X) # (B, N, L)
        V_all = self.W_V(X) # (B, N, L)
        
        attn_token = torch.softmax(
            torch.matmul(Q_token, K_all.transpose(1, 2)) / (self.d_k ** 0.5), 
            dim=-1
        ) # (B, 1, N)
        
        Z_bag = torch.matmul(attn_token, V_all).squeeze(1) # (B, L)
        
        # Classification
        logits = self.classifier(Z_bag).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        loss = None
        if bag_labels is not None:
            # CE Loss
            ce_loss = F.binary_cross_entropy_with_logits(logits, bag_labels.float())
            
            # Update Centers (Momentum)
            # ... (Implementation of momentum update based on bag_labels) ...
            
            # Triplet Loss
            tri_loss = self.compute_triplet_loss(self.G, self.pos_center, self.neg_center)
            
            # Diversity Loss
            div_loss = self.compute_diversity_loss(self.G)
            
            # Hyperparameters (example values from ablation)
            lambda_tri = 0.1
            lambda_div = 0.1
            
            loss = ce_loss + lambda_tri * tri_loss + lambda_div * div_loss
            
        return probs, loss

    def compute_triplet_loss(self, G, pos_c, neg_c):
        # G: (K, L), pos_c: (L), neg_c: (L)
        dist_pos = F.pairwise_distance(G, pos_c.unsqueeze(0))
        dist_neg = F.pairwise_distance(G, neg_c.unsqueeze(0))
        loss = F.relu(dist_pos - dist_neg + self.margin).mean()
        return loss

    def compute_diversity_loss(self, G):
        # G: (K, L), normalize rows
        G_norm = F.normalize(G, p=2, dim=1)
        Gram = torch.matmul(G_norm, G_norm.t()) # (K, K)
        # Det computation via log-det-sum-log-eigenvalues or direct det
        # Using torch.linalg.det for small K
        det_val = torch.linalg.det(Gram + self.eps * torch.eye(Gram.size(0), device=Gram.device))
        loss = -torch.log(det_val)
        return loss
```

#### 6. 实现提示
- **关键网络组件**：Multi-Head Cross-Attention, Linear Projections, Momentum Buffer for Centers.
- **重要超参数**：
    - `num_global_vectors` ($K$): CAMELYON16设为5，TCGA设为3。
    - `momentum` ($m$): 0.4。
    - `margin` ($\mu$): 0.5 (Triplet Loss)。
    - `lambda_tri`, `lambda_div`: 0.1。
    - `warmup_epochs`: 20 (前20个epoch只训练CE Loss)。
- **归一化/激活方式**：全局向量需保持单位范数 ($\|g_i\|=1$) 以配合DPP Loss；注意力使用Softmax；分类头使用Sigmoid+BCE。
- **维度对齐方式**：所有向量映射到相同的隐藏维度 $L$ (实验中常设为512)。
- **实现注意事项**：
    - DPP Loss中的行列式计算在 $K$ 较小时直接计算即可，复杂度低。
    - 动量中心的更新仅在遇到对应标签的Batch时进行。
    - Warm-up策略至关重要，否则随机初始化的全局向量会导致早期训练不稳定。

#### 7. 计算与资源开销
- **理论计算复杂度**：
    - 交叉注意力：$O(K \cdot n \cdot L)$，其中 $K \ll n$，远优于自注意力的 $O(n^2 L)$。
    - 多样性Loss：$O(L \cdot K^2)$，由于 $K$ 很小（~5），可视为 $O(L)$。
- **参数量**：约 0.642 M (ResNet-18特征下)，少于Trans-MIL (3.04 M) 和 ILRA-MIL (1.05 M)。
- **FLOPs/MACs**：约 1.054 G，显著低于对比方法。
- **显存开销**：较低，因为不需要存储 $n \times n$ 的注意力矩阵。
- **推理速度**：快，得益于线性复杂度的聚合。
- **论文是否提供效率对比**：是，Table 5 提供了Params和MACs对比。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI癌症分类（乳腺癌淋巴结转移、肺癌亚型）。
- **可迁移到的任务/数据集**：任何基于MIL的弱监督视觉任务（视频动作识别、时间序列分类、文档分类），特别是当数据内部存在显著子类型多样性时。
- **迁移所需调整**：调整 $K$ 值以适应不同数据集的多样性程度；调整动量更新率。
- **适用条件**：Bag内实例数量较大（$n$大），且实例间存在语义多样性。
- **潜在限制**：如果实例间完全同质，过多的全局向量可能导致过拟合或优化困难。

#### 9. 实验与消融证据
- **主要性能结果**：
    - CAMELYON16 (ResNet-50): Acc 0.917, AUC 0.957 (SOTA)。
    - TCGA-NSCLC (ResNet-50): Acc 0.908, AUC 0.963 (SOTA)。
- **相对基线的提升**：在CAMELYON16上比第二好方法DTFD-MIL (AFS) 提升 Accuracy 0.9%, AUC 1.1%。
- **相关消融实验**：
    - Table 2: 移除正例对齐或多样性Loss均导致性能下降。
    - Table 3: 移除Tokenized Global Vector导致性能下降（Acc -1.0%）。
    - Fig 4: $K$ 值敏感性分析，过大过小都不好。
- **作者结论**：DGR-MIL在性能和效率上均优于现有方法，多样性建模是关键。
- **证据是否充分**：是，包含多个数据集、多种特征提取器、统计检验和详细的消融实验。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 将DPP理论引入MIL多样性损失，并结合Tokenized Global Vector，视角新颖。 |
| 技术可行性 | 高 | 模块标准，易于集成到现有Transformer/MIL架构中。 |
| 实现难度 | 中 | 需注意动量中心的维护和Warm-up策略，DPP Loss实现简单。 |
| 架构相关性 | 高 | 专为MIL设计，解决了长序列注意力瓶颈。 |
| 可迁移性 | 高 | 不依赖特定病理学知识，通用MIL聚合模块。 |
| 计算成本 | 低 | 参数量和计算量均低于主流Transformer MIL方法。 |

#### 11. 一句话总结
DGR-MIL通过引入可学习的全局向量和基于DPP的正交性多样性损失，结合正例对齐机制，以线性复杂度有效建模了WSI中实例的多样性，显著提升了弱监督分类性能。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **基于DPP的正交性多样性损失**：利用 $\min -\log \det(GG^\top)$ 强制向量正交，提供了一种无需采样的、理论保证的多样性优化手段，比传统的互斥损失或对比学习更直接地作用于表示空间的几何结构。
- **Tokenized Global Vector的设计**：将用于分类的Token放在全局向量端而非实例端，有效地分离了“多样性探索”和“判别性总结”，缓解了边界噪声干扰。

### 2. 方法之间的关系
- **与AB-MIL的关系**：DGR-MIL可以看作是AB-MIL的变体，用全局向量Query替换了标量注意力权重生成，引入了跨序列交互。
- **与Trans-MIL的关系**：Trans-MIL使用Self-Attention建模实例间相关性，DGR-MIL使用Cross-Attention（Global vs Instance）建模，降低了复杂度并显式引入了多样性约束。
- **与PMIL的关系**：PMIL使用聚类原型，DGR-MIL使用可学习向量并通过梯度下降直接优化多样性，避免了伪标签的不稳定性。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，公式、超参数、训练策略（Warm-up）均有详细说明。
- **关键配置是否明确**：是，包括 $K$ 值、动量 $m$、Loss权重等。
- **预计复现难点**：
    1.  **动量中心的同步**：在多GPU训练时，动量中心的更新需要小心处理（是每步更新还是累积？文中暗示每步基于当前Batch更新，需注意EMA的实现细节）。
    2.  **Nystrom Attention的使用**：文中提到在Cross-Attention前对实例和全局向量分别应用了Nystrom Self-Attention，这部分的具体实现细节（如核函数选择、子采样数量）在正文中未展开，需参考附录或代码。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：DPP Diversity Loss 可轻松插入任何基于向量池化的MIL模型中。
- **需要改造的设计**：Tokenized Global Vector 需要修改标准的Transformer Encoder结构，将Token置于Query侧。
- **可能形成的新研究思路**：
    1.  探索动态数量的全局向量（自适应 $K$）。
    2.  将DPP Loss应用于其他需要多样性的领域，如推荐系统、核心集选择。
    3.  结合对比学习，进一步区分不同全局向量所代表的语义子空间。

### 5. 阅读备注
- 论文强调了“病理学驱动的多样性”，即肿瘤内部的异质性是客观存在的，模型应反映这一点。
- 实验中使用ResNet-50特征效果最好，部分原因是DTFD-MIL提供的特征提取策略包含了更多的阳性实例，这提示我们在比较不同MIL方法时，特征提取器的质量对结果影响巨大。
