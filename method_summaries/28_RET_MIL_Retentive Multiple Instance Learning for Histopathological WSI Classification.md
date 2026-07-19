# 28_RET_MIL_Retentive Multiple Instance Learning for Histopathological WSI Classification 方法总结

> 证据说明：输入为完整论文文本（10页），包含摘要、引言、方法论、实验及参考文献。公式提取基本完整，关键符号定义清晰。无明显的页面缺失或公式乱码导致无法理解的情况。

## 一、论文基本信息

- **论文标题**：RetMIL: Retentive Multiple Instance Learning for Histopathological Whole Slide Image Classification
- **作者**：Hongbo Chu, Qiehe Sun, Jiawen Li, Yuxuan Chen, Lizhong Zhang, Tian Guan, Anjia Han, Yonghong He
- **发表年份**：2024 (arXiv:2403.10858v1)
- **会议/期刊**：arXiv预印本 (cs.CV)
- **论文链接/DOI/arXiv ID**：https://arxiv.org/abs/2403.10858
- **代码仓库**：文中提到 "Our code will be accessed shortly"，未提供具体链接。
- **研究任务**：计算病理学中的全切片图像（WSI）分类（弱监督多实例学习）。
- **数据模态**：数字病理切片图像（H&E染色），提取为224x224的Patch特征序列。

## 二、论文整体概述

### 1. 核心问题
传统基于Transformer的多实例学习（MIL）方法在处理WSI时面临两大挑战：
1.  **高内存消耗与慢推理速度**：由于WSI尺寸巨大（Gigapixel级），生成的Patch序列极长，Self-Attention机制的二次方复杂度导致显存爆炸和延迟增加。
2.  **性能瓶颈**：在高异质性特征和超长序列下，现有Transformer基线模型难以兼顾效率与精度。

### 2. 整体方法
提出 **RetMIL**，一种基于**保留机制（Retention Mechanism）**的MIL框架。
-   **局部层级**：将WSI序列分割为多个子序列，并行使用线性保留层更新Token，并通过局部注意力池化聚合。
-   **全局层级**：将局部特征融合为全局序列，串行使用保留层更新，最后通过全局注意力池化得到Slide级表示进行分类。
-   **核心优势**：用线性复杂度的保留机制替代非线性Self-Attention，实现常数级显存占用和更高的吞吐量。

### 3. 主要贡献
1.  提出RetMIL架构，引入保留机制解决WSI分析中的长序列和高计算成本问题。
2.  设计分层保留聚合结构，有效整合局部子序列信息和全局上下文。
3.  在CAMELYON、BRACS和LUNG数据集上达到SOTA性能，同时显著降低GPU显存占用并提升推理速度。

## 三、方法总结

### 方法 1：RetMIL 整体架构与保留机制

#### 1. 核心思想与解决的问题
-   **目标问题**：解决Transformer-based MIL在处理超长WSI序列时的 $O(N^2)$ 计算复杂度和高显存占用问题。
-   **现有方法的局限**：Self-Attention需要计算所有Token对的交互，导致显存随序列长度线性甚至超线性增长，推理速度慢。
-   **核心思想**：借鉴RetNet（Retentive Network）的思想，使用**线性保留机制**替代Self-Attention。该机制允许状态在时间步之间传递，从而以线性复杂度建模长距离依赖。
-   **创新点**：
    1.  将WSI序列划分为子序列，利用Batch并行处理局部子序列。
    2.  结合旋转位置编码（RoPE）和相对距离衰减矩阵，增强位置感知能力。
    3.  构建“局部保留+局部注意力”到“全局保留+全局注意力”的分层聚合范式。

#### 2. 详细结构与数据流
-   **输入**：WSI被裁剪为 $N$ 个 $224 \times 224$ 的Patch，经ViT-S/DINO编码器提取为特征序列 $X = \{x_1, ..., x_N\} \in \mathbb{R}^{N \times d}$。
-   **处理流程**：
    1.  **序列分割**：将 $X$ 分割为 $q+1$ 个子序列 $\{S_1, ..., S_{q+1}\}$，每个子序列长度固定为 $l=512$。若剩余不足 $l$，则填充补齐。
    2.  **局部更新（Parallel）**：对每个子序列 $S_i$ 独立应用多尺度保留层（MSR），输出 $F_i$。
    3.  **局部聚合（Local Attention）**：对每个 $F_i$ 进行加权求和，得到局部特征 $F_{local,i}$。
    4.  **全局更新（Serial）**：将所有 $F_{local,i}$ 拼接成全局序列 $F_{local}$，再次应用MSR得到 $G$。
    5.  **全局聚合（Global Attention）**：对 $G$ 进行加权求和，得到最终Slide级表示 $F_{global}$。
    6.  **分类**：$F_{global}$ 经过线性分类器输出预测分数。
-   **输出**：WSI的分类概率/分数。
-   **模块在整体网络中的位置**：位于特征提取器（ViT）之后，分类头之前。
-   **与其他模块的连接方式**：输入来自ViT输出的Patch Embeddings；输出连接至Linear Classifier。

#### 3. 数学公式

**1. 投影与多头分解：**
$$ Q = XW_Q, \quad K = XW_K, \quad V = XW_V $$
其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ 为可学习权重。将 $Q, K, V$ 拆分为 $h$ 个头 $\{Q_h\}, \{K_h\}, \{V_h\}$。

**2. 旋转位置编码（RoPE）：**
对 $Q_h$ 和 $K_h$ 应用RoPE得到 $\tilde{Q}_h$ 和 $\tilde{K}_h$。

**3. 保留层（Retention Layer）：**
$$ \text{Retention}(h, X) = (\tilde{Q}_h \tilde{K}_h^\top \odot D_h) V_h $$
其中 $\odot$ 为逐元素乘法，$D_h$ 为相对距离衰减矩阵。

**4. 距离衰减矩阵 $D_h$：**
$$ D_{h,nm} = \begin{cases} \gamma^{n-m}, & n \ge m \\ 0, & n < m \end{cases} $$
$\gamma$ 为衰减因子（通常小于1）。

**5. 归一化与门控：**
输出经过 GroupNorm 和 Swish Gate 处理。整个批次的操作记为 $MSR(B; S)$。

**6. 局部注意力池化：**
$$ F_{local,i} = \sum_{k=1}^{l} \alpha_{i,k} F_{i,k} $$
$$ \alpha_{i,k} = \frac{\exp\{\Gamma_l \tanh(W_l F_{i,k}) \odot \text{sigm}(U_l F_{i,k})\}}{\sum_{t=1}^{l} \exp\{\Gamma_l \tanh(W_l F_{i,t}) \odot \text{sigm}(U_l F_{i,t})\}} $$
其中 $W_l, U_l \in \mathbb{R}^{M \times d}$, $\Gamma_l \in \mathbb{R}^{1 \times M}$。

**7. 全局注意力池化：**
$$ F_{global} = \sum_{p=1}^{q+1} \beta_p G_p $$
$$ \beta_p = \frac{\exp\{\Gamma_{global} \tanh(W_{global} G_p) \odot \text{sigm}(U_{global} G_p)\}}{\sum_{t=1}^{q+1} \exp\{\Gamma_{global} \tanh(W_{global} G_t) \odot \text{sigm}(U_{global} G_t)\}} $$

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | Patch Features ($X$) | $\mathbb{R}^{N \times d}$ | $N$ 为Patch总数，$d$ 为特征维度 (ViT输出) |
| 分割后 | Subsequence ($S_i$) | $\mathbb{R}^{l \times d}$ | $l=512$ 为子序列长度 |
| 投影后 | $Q, K, V$ | $\mathbb{R}^{l \times d}$ | 线性变换后 |
| 多头拆分 | $Q_h, K_h, V_h$ | $\mathbb{R}^{l \times d_h}$ | $d_h = d/h$ |
| 保留输出 | $F_i$ | $\mathbb{R}^{l \times d}$ | 经过MSR及Norm/Gate后的子序列特征 |
| 局部聚合 | $F_{local,i}$ | $\mathbb{R}^{d}$ | 单个子序列的聚合向量 |
| 全局输入 | $F_{local}$ | $\mathbb{R}^{(q+1) \times d}$ | 所有局部特征的拼接 |
| 全局保留 | $G$ | $\mathbb{R}^{(q+1) \times d}$ | 全局序列经过MSR后的特征 |
| 最终输出 | $F_{global}$ | $\mathbb{R}^{d}$ | Slide级表示 |
| 预测 | Score | $\mathbb{R}^{1}$ 或 $\mathbb{R}^{C}$ | 线性分类器输出 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 简化版RoPE实现，实际需根据频率生成
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        # 此处省略具体的RoPE旋转矩阵计算细节，假设有一个apply_rope函数
        return apply_rope(x, self.inv_freq)

class MultiScaleRetentionLayer(nn.Module):
    def __init__(self, dim, num_heads, gamma=0.9):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gamma = gamma
        
        # 线性投影层
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        
        # 归一化和激活
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim)
        self.gate = nn.Linear(dim, dim) # 用于Swish Gate
        
        self.rope = RotaryPositionalEncoding(self.head_dim)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape
        
        # 1. 投影
        Q = self.W_q(x) # [B, L, D]
        K = self.W_k(x) # [B, L, D]
        V = self.W_v(x) # [B, L, D]
        
        # 2. 多头拆分 & RoPE
        # 假设 reshape 和 view 操作正确执行
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, Hd]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        Q = self.rope(Q)
        K = self.rope(K)
        
        # 3. 计算 Retention
        # Q @ K^T -> [B, H, L, L]
        # 注意：为了节省显存，实际实现中可能使用循环或分块计算，或者利用递归性质
        # 这里展示矩阵形式： R = Q @ K^T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) # [B, H, L, L]
        
        # 4. 应用距离衰减矩阵 Dh
        # Dh[n, m] = gamma^(n-m) if n>=m else 0
        # 构造三角掩码和指数衰减
        indices = torch.arange(L, device=x.device)
        # 广播形成矩阵 [L, L]
        dist_matrix = indices.unsqueeze(1) - indices.unsqueeze(0) # [L, L]
        mask = (dist_matrix >= 0).float()
        decay = self.gamma ** dist_matrix
        D = mask * decay # [L, L]
        
        # 扩展D到Batch和Head维度 [B, H, L, L]
        D = D.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
        
        # 逐元素乘积
        retention_output = (attn_scores * D) @ V # [B, H, L, Hd]
        
        # 5. 合并头
        retention_output = retention_output.transpose(1, 2).contiguous().view(B, L, D)
        
        # 6. Norm & Gate
        out = self.norm(retention_output)
        out = out * F.silu(self.gate(out)) # Swish gate
        
        return out

class AttentionPool(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.W = nn.Linear(dim, hidden_dim)
        self.U = nn.Linear(dim, hidden_dim)
        self.Gamma = nn.Parameter(torch.randn(1, hidden_dim))
        
    def forward(self, x):
        """
        x: [L, D] or [B, L, D]
        Returns weighted sum: [D] or [B, D]
        """
        # 计算注意力权重 alpha
        # tanh(Wx) * sigmoid(Ux)
        term1 = torch.tanh(self.W(x)) # [L, H]
        term2 = torch.sigmoid(self.U(x)) # [L, H]
        
        # Gamma broadcasting
        # exp(Gamma * term1 * term2)
        logits = self.Gamma * term1 * term2 # [L, H]
        
        # Softmax over sequence length dimension
        alphas = F.softmax(logits.sum(dim=-1), dim=-1) # [L] or [B, L]
        
        # Weighted Sum
        # x: [L, D], alphas: [L] -> [D]
        if x.dim() == 3:
            output = torch.einsum('bl,bd->bd', alphas, x)
        else:
            output = torch.einsum('l,bd->bd', alphas, x)
            
        return output

class RetMIL(nn.Module):
    def __init__(self, input_dim=768, subseq_len=512, num_heads=8, gamma=0.9):
        super().__init__()
        self.subseq_len = subseq_len
        self.retention_layer = MultiScaleRetentionLayer(input_dim, num_heads, gamma)
        self.local_pool = AttentionPool(input_dim)
        self.global_pool = AttentionPool(input_dim)
        self.classifier = nn.Linear(input_dim, 1) # Binary classification example

    def split_subsequences(self, X):
        """
        X: [N, D]
        Returns list of subsequences [S1, ..., Sq+1] each [L, D]
        """
        N, D = X.shape
        l = self.subseq_len
        q = N // l
        r = N % l
        
        subsequences = []
        for j in range(q):
            start = j * l
            end = (j + 1) * l
            subsequences.append(X[start:end])
            
        if r > 0:
            remainder = X[q*l:]
            # Padding logic as described in paper
            # If r < l/2, repeat remainder to fill? 
            # Paper says: extend R to Sq+1 = Concat(R, X_{l-r})
            # This implies padding with existing features or zeros? 
            # Text: "extend R to Sq+1 ... ensuring mapping satisfied"
            # Simplified assumption: Pad with zeros or repeat last token if not specified clearly.
            # Based on "X_{l-r}" description, it seems to use existing tokens.
            # For implementation simplicity, we assume zero-padding or specific filling strategy.
            # Let's assume simple zero padding for the pseudocode structure, 
            # but note the paper has complex filling rules.
            pad_len = l - r
            padded_remainder = F.pad(remainder, (0, 0, 0, pad_len)) # Pad feature dim? No, seq dim.
            # Actually need to pad sequence dimension.
            # Since X is [N, D], we can't easily pad without creating new tensor.
            # Assuming a helper function `pad_sequence` exists.
            pass 
            
        return subsequences

    def forward(self, X):
        """
        X: [N, D] - Single WSI patch features
        """
        # 1. Split
        subs = self.split_subsequences(X)
        
        # 2. Local Processing (Parallel)
        local_features = []
        for S in subs:
            # S: [L, D] -> add batch dim for MSR if needed, or modify MSR to handle single seq
            # MSR expects Batch dim usually. Let's assume MSR handles [L,D] by treating as B=1
            F_i = self.retention_layer(S.unsqueeze(0)).squeeze(0) # [L, D]
            F_local_i = self.local_pool(F_i) # [D]
            local_features.append(F_local_i)
            
        # Stack to form Global Sequence
        F_global_seq = torch.stack(local_features, dim=0) # [q+1, D]
        
        # 3. Global Processing
        G = self.retention_layer(F_global_seq.unsqueeze(0)).squeeze(0) # [q+1, D]
        F_slide = self.global_pool(G) # [D]
        
        # 4. Classification
        score = self.classifier(F_slide)
        return score
```

#### 6. 实现提示
-   **关键网络组件**：`MultiScaleRetentionLayer` (含RoPE, Linear Projections, Decay Mask), `AttentionPool` (含Tanh/Sigmoid门控)。
-   **重要超参数**：
    -   `subseq_len` (l): 512 (固定)。
    -   `num_heads`: 未明确给出，通常取8或16。
    -   `gamma`: 衰减因子，默认0.9左右。
    -   `hidden_dim` (M): 注意力池化中的隐藏层维度，通常为 $d$ 的一部分或相等。
-   **归一化/激活方式**：GroupNorm (Group=8), Swish Gate (SiLU), Tanh, Sigmoid。
-   **维度对齐方式**：线性投影保持维度不变；多头拆分后重组；池化操作沿序列维度求和。
-   **实现注意事项**：
    -   **RoPE实现**：需正确实现旋转位置编码，确保Query和Key在相同空间旋转。
    -   **衰减矩阵 $D$**：对于长序列，直接构建 $L \times L$ 矩阵可能耗显存。虽然RetNet有递归实现，但本文公式显示为矩阵形式。若 $L=512$，$512^2$ 尚可接受。若需优化，可利用其Toeplitz结构或递归特性。
    -   **子序列填充**：论文描述了复杂的填充逻辑（基于剩余长度 $r$ 与 $l/2$ 的关系），复现时需严格遵循此逻辑以保证每个Patch只出现一次且长度一致。
-   **依赖的特殊算子或第三方库**：PyTorch, 可能需要 `einops` 辅助维度操作。

#### 7. 计算与资源开销
-   **理论计算复杂度**：
    -   Retention层：$O(N \cdot d)$，线性复杂度。
    -   Attention Pooling：$O(N \cdot d)$。
    -   总体：相比Transformer的 $O(N^2 \cdot d)$，大幅降低。
-   **参数量**：主要由线性层 $W_Q, W_K, W_V$ 和池化层的 $W, U$ 组成，参数量远小于同等深度的Transformer Encoder。
-   **FLOPs/MACs**：显著低于TransMIL等基线。
-   **显存开销**：论文Figure 3.b显示，随着序列长度增加，RetMIL的显存占用几乎保持恒定，而Transformer模型线性/二次增长。
-   **推理速度**：论文Figure 3.a显示，吞吐量比TransMIL高约1.5倍，比其他Transformer模型更高。
-   **论文是否提供效率对比**：是，提供了Throughput和GPU Memory对比图表。

#### 8. 适用场景与可迁移性
-   **原论文应用场景**：WSI分类（癌症亚型分类、转移检测）。
-   **可迁移到的任务/数据集**：任何具有长序列输入的MIL任务，如基因序列分析、长时间序列信号分类、长文档分类。
-   **迁移所需调整**：调整输入维度 $d$，修改分类头以适应类别数，调整子序列长度 $l$ 以适应不同数据分布。
-   **适用条件**：序列长度较长，显存受限，或对推理速度有要求。
-   **潜在限制**：保留机制对绝对位置的敏感性可能不如Self-Attention强（尽管引入了RoPE），在极短序列上优势不明显。

#### 9. 实验与消融证据
-   **主要性能结果**：
    -   CAMELYON: F1 87.24%, B-Acc 87.53% (SOTA)。
    -   BRACS: F1 68.5%, B-Acc 67.0% (SOTA)。
    -   LUNG: B-Acc 91.56% (接近SOTA TransMIL 91.43%)。
-   **相对基线的提升**：在CAMELYON上超越TransMIL约3.18% (F1)。
-   **相关消融实验**：文中未详细列出针对Retention vs Attention的单独消融，但通过对比Transformer基线和可视化证明了有效性。
-   **作者结论**：RetMIL在保持高性能的同时，显著降低了计算开销。
-   **证据是否充分**：在三个数据集上的表现有力支持了论点，效率对比图提供了直观证据。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 首次将RetNet的保留机制系统性地应用于WSI MIL，解决长序列痛点。 |
| 技术可行性 | 高 | 基于标准线性层和矩阵运算，易于实现，无需复杂自定义算子。 |
| 实现难度 | 中 | 需注意RoPE的正确集成、衰减矩阵的高效计算以及子序列填充逻辑。 |
| 架构相关性 | 高 | 专为WSI长序列特性设计，分层结构符合病理学语义。 |
| 可迁移性 | 高 | 通用序列建模模块，可迁移至其他长序列MIL任务。 |
| 计算成本 | 低 | 线性复杂度，显存占用恒定，适合大规模部署。 |

#### 11. 一句话总结
RetMIL通过引入线性保留机制和分层聚合架构，成功解决了WSI分类中Transformer模型面临的长序列高计算成本问题，在保持SOTA精度的同时实现了高效的推理。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
-   **线性保留机制替代Self-Attention**：在长序列场景下，用 $O(N)$ 的Retention替代 $O(N^2)$ 的Attention，是提升效率的关键。
-   **分层处理策略**：先局部并行处理再全局串行聚合，既利用了并行计算加速，又保留了全局上下文信息。

### 2. 方法之间的关系
-   RetMIL的核心模块（Retention Layer）是基础，上层包裹了局部和全局的Attention Pooling。
-   它与TransMIL等基线方法的主要区别在于核心的序列更新机制（Retention vs Self-Attention）。

### 3. 复现可行性
-   **代码是否公开**：否（文中声明即将公开）。
-   **方法描述是否完整**：是。公式、维度、超参数（如l=512）均给出。
-   **关键配置是否明确**：是。包括预处理步骤、损失函数（Cross-Entropy）、优化器设置。
-   **预计复现难点**：
    1.  **子序列填充逻辑**：论文对剩余部分 $R$ 的处理描述较为文字化，需仔细推导代码逻辑。
    2.  **RoPE的具体实现**：需确保与RetNet或RoFormer中的实现一致。
    3.  **衰减矩阵 $D$ 的计算**：虽然公式简单，但在大规模Batch下的数值稳定性需注意。

### 4. 与当前研究方向的关系
-   **可直接采用的设计**：Retention Layer可以作为通用模块替换Transformer中的Attention Block，用于任何长序列任务。
-   **需要改造的设计**：WSI特定的Patch提取和ViT特征提取器需替换为目标任务的特征提取器。
-   **可能形成的新研究思路**：探索更高效的衰减策略，或将Retention机制与稀疏Attention结合，进一步平衡精度与效率。

### 5. 阅读备注
-   论文强调“Retentive”而非“Recurrent”，虽然数学形式上有相似之处，但其并行训练特性更接近Transformer的改进版。
-   实验部分特别强调了在不同序列长度下的表现，这是验证该方法有效性的关键证据。
