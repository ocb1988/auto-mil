# 41_PSA_MIL_Probabilistic Spatial Attention-Based MIL for Whole Slide Image Classification 方法总结

> 证据说明：输入为完整论文文本（10页），包含摘要、引言、方法、实验及参考文献。公式提取基本完整，关键数学推导清晰。无页面缺失或严重OCR错误影响核心逻辑理解。

## 一、论文基本信息

- **论文标题**：PSA-MIL: A Probabilistic Spatial Attention-Based Multiple Instance Learning for Whole Slide Image Classification
- **作者**：Sharon Peled, Yosef E. Maruvka, Moti Freiman
- **发表年份**：2025 (arXiv:2503.16284v2, Nov 2025)
- **会议/期刊**：arXiv预印本 (尚未见正式会议/期刊录用信息，但格式符合CVPR/ICCV等顶会风格)
- **论文链接/DOI/arXiv ID**：https://arxiv.org/abs/2503.16284
- **代码仓库**：https://github.com/SharonPeled/PSA-MIL
- **研究任务**：全切片图像（WSI）分类（癌症亚型分类、转移检测、生存预测）
- **数据模态**：数字病理学全切片图像（WSIs），提取为Tile patches

## 二、论文整体概述

### 1. 核心问题
传统多实例学习（MIL）将WSI视为无序的Tile集合，忽略了空间结构；现有的上下文MIL方法通常依赖固定的空间假设（如固定核大小的CRF、固定邻域注意力或静态位置编码），缺乏灵活性，难以捕捉复杂的组织空间依赖关系。此外，Transformer自注意力的二次复杂度在长序列WSI分析中计算成本高昂。

### 2. 整体方法
提出概率空间注意力MIL（PSA-MIL）。该方法基于自注意力的概率解释（后验分布），引入可学习的距离衰减先验来动态建模空间关系。具体包括：
1.  **概率空间注意力机制**：放松标准自注意力的归一化和均匀先验假设，使用参数化的距离衰减函数作为先验，结合查询-键相似度计算后验注意力权重。
2.  **多样性损失（Diversity Loss）**：通过熵最大化鼓励不同注意力头学习不同的空间局部性范围，避免冗余。
3.  **空间剪枝策略（Spatial Pruning）**：利用可逆的距离衰减函数确定动态阈值，仅保留高影响力的Tile进行注意力计算，将复杂度从 $O(n^2)$ 降低至 $O(n \cdot K^2)$。

### 3. 主要贡献
- 提出了基于概率解释的自适应上下文MIL框架，动态学习空间依赖而非固定假设。
- 实现了多头空间 specialization，每个头独立推导其空间交互范围。
- 设计了基于熵的多样性损失，促进多头表示的解耦。
- 提出了空间剪枝策略，在保证性能的同时显著降低计算开销。

## 三、方法总结

### 方法 1：概率空间注意力机制 (Probabilistic Spatial Attention)

#### 1. 核心思想与解决的问题
- **目标问题**：解决标准MIL忽略空间结构以及现有上下文MIL方法空间假设僵化的问题。
- **现有方法的局限**：CAM I L使用固定邻域约束，Bayes-MIL使用固定核CRF，位置编码（RPE/PPEG）缺乏理论支撑且改进有限。
- **核心思想**：将自注意力重新解释为高斯混合模型（GMM）的后验推断。通过引入可学习的距离衰减先验 $\pi_j$，使注意力权重不仅取决于特征相似度（似然），还取决于空间距离（先验）。
- **创新点**：放松了标准自注意力中Query/Key的L2归一化假设和均匀先验假设，允许模型根据数据驱动的方式动态调整空间影响力。

#### 2. 详细结构与数据流
- **输入**：
    - 实例嵌入矩阵 $X \in \mathbb{R}^{n \times d}$，其中 $n$ 为Tile数量，$d$ 为特征维度。
    - 成对欧氏距离矩阵 $D \in \mathbb{R}^{n \times n}$，其中 $d_{ij}$ 是Tile $i$ 和 $j$ 之间的物理距离。
- **处理流程**：
    1.  通过线性投影生成 Query ($Q$), Key ($K$), Value ($V$)。
    2.  计算成对距离矩阵 $D$。
    3.  应用可学习的距离衰减函数 $f(d_{ij}|\theta)$ 计算先验概率 $\pi_j$。
    4.  结合未归一化的Query-Key内积（或范数差）与先验的对数，计算后验注意力分布。
    5.  （可选）应用空间剪枝策略过滤低先验权重的交互。
    6.  使用注意力权重加权Value矩阵得到输出。
- **输出**：更新后的Tile嵌入表示，用于后续的池化和分类。
- **模块在整体网络中的位置**：替代标准Transformer中的Self-Attention层，位于特征编码器之后，全局池化之前。
- **与其他模块的连接方式**：接收原始Tile嵌入和坐标信息（用于计算距离），输出增强空间上下文的嵌入，送入Attention-based Pooling。

#### 3. 数学公式

**基础概率解释：**
标准自注意力对应于以下后验概率（在特定假设下）：
$$ p(t_j=1|q_i) = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^\top k_{j'} / \sqrt{d_k})} $$
其中 $t_j$ 是选择Key $k_j$ 的指示变量。

**PSA-MIL 后验公式 (Eq. 6)：**
放松假设后，后验分布定义为：
$$ p(t_j=1|q_i) = \frac{\exp\left( \frac{-\|q_i - k_j\|^2}{\sqrt{d_k}} + \log(f(d_{ij}|\theta)) \right)}{\sum_{j'=1}^N \exp\left( \frac{-\|q_i - k_{j'}\|^2}{\sqrt{d_k}} + \log(f(d_{ij'}|\theta)) \right)} $$
*注：原文公式(6)中分子部分写为 $\|q_i - k_j\|^2$，但在指数前符号需仔细核对。通常注意力分数为正相关时取正号。原文公式(6)显示为 $\frac{\|q_i - k_j\|^2}{\sqrt{d_k}}$，这看起来像是负距离平方？让我们仔细看原文截图/文本。*
*修正阅读*：原文 Eq. (6) 写作：
$$ \exp\left( \frac{\|q_i - k_j\|^2}{\sqrt{d_k}} + \log(f(d_{ij}|\theta)) \right) $$
这里可能存在排版歧义或特定定义。通常 $\|q-k\|^2 = \|q\|^2 + \|k\|^2 - 2q^\top k$。如果 $q,k$ 未归一化，直接相减可能不稳定。但在Eq(3)推导中，使用的是 $q_i^\top k_j$。
再看Eq(3)到Eq(4)的简化过程：
$p(t_j=1|q_i) \propto \pi_j \exp(q_i^\top k_j / \sigma^2)$。
在PSA-MIL中，$\pi_j = f(d_{ij}|\theta)$。
为了数值稳定性，原文将先验放入指数：$\log(\pi_j) = \log(f(d_{ij}|\theta))$。
关于距离项，原文Eq(6)确实写的是 $\|q_i - k_j\|^2$。这可能是一个笔误或者特定的度量方式（例如希望距离越小值越大？如果是这样，前面应该有负号）。
*再次检查原文文本*：
"Relaxing Assumption 1... compute the full l2-norm $\|q_i - k_j\|^2$ ... allowing queries and keys to retain magnitude information".
如果在指数中是正的 $\|q_i - k_j\|^2$，那么距离越远（范数差越大），注意力越高？这与“距离衰减”矛盾。
*合理推测*：原文公式(6)中的第一项应该是负的相关性分数，或者是 $-\|q_i - k_j\|^2$ 以体现相似性（距离小则范数差小，指数大）。或者，原文意指使用余弦相似度类似的结构，但保留了范数。
*严格忠实原文*：原文公式(6)明确写作：
$$ \exp\left( \frac{\|q_i - k_j\|^2}{\sqrt{d_k}} + \log(f(d_{ij}|\theta)) \right) $$
*但是*，考虑到物理意义（Distance-Decayed Priors）和常规注意力逻辑，这极有可能是印刷错误，应为 $-\|q_i - k_j\|^2$ 或者原文定义的 $\|q_i - k_j\|^2$ 实际上是指某种相似度度量（不太可能）。
*另一种可能性*：原文可能在Eq(3)推导中使用了 $\exp(-\|q_i - k_j\|^2/2\sigma^2)$ 的高斯形式，然后简化。
Eq(3)中：$\pi_j N(q_i | k_j, \sigma^2 I) \propto \pi_j \exp(-( \|q_i\|^2 + \|k_j\|^2 )/2\sigma^2) \exp(q_i^\top k_j / \sigma^2)$。
如果忽略常数项，核心项是 $q_i^\top k_j$。
Eq(6)中出现的 $\|q_i - k_j\|^2$ 展开后包含 $-2q_i^\top k_j$。如果公式是 $\exp( - \|q_i - k_j\|^2 / \dots )$，则等价于 $\exp( q_i^\top k_j / \dots )$。
鉴于原文文字描述“distance-decayed priors”，且公式(6)中 $\log(f)$ 是加性的，第一项必须体现特征相似度。**此处标记为：公式文本提取可能存在符号歧义，建议实现时使用 $-\|q_i - k_j\|^2$ 或等效的 $q_i^\top k_j$ 形式以符合“相似度越高注意力越高”的逻辑，同时加上 $\log(f(d_{ij}))$。**
*为了忠实于提供的文本*，我将列出原文公式，并在实现提示中指出这一潜在的不一致性。

**距离衰减函数 (Sec 3.2.2):**
提供三种参数化形式，$\theta$ 为可学习参数：
1.  指数衰减: $f(d|\lambda) = \exp(-\lambda d)$
2.  高斯衰减: $f(d|\sigma) = \exp(-\frac{d^2}{2\sigma^2})$
3.  Cauchy衰减: $f(d|\gamma) = \frac{1}{1 + (\frac{d}{\gamma})^2}$

**多样性损失 (Sec 3.3):**
$$ L_{Diversity} = -H(p) \approx \frac{1}{M} \sum_{m=1}^M \log \hat{p}(\tilde{\theta}_m) $$
其中 $\hat{p}(\theta)$ 是通过KDE估计的参数分布：
$$ \hat{p}(\theta) = \frac{1}{H\sigma} \sum_{h=1}^H K\left(\frac{\theta - \theta_h}{\sigma}\right) $$
总损失：
$$ L = L_{CE} + \alpha L_{Diversity} $$

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | Tile Embeddings $X$ | $(n, d)$ | $n$个patch，$d$维特征 |
| 输入 | Distance Matrix $D$ | $(n, n)$ | Patch间的欧氏距离 |
| 中间 | Query $Q$, Key $K$, Value $V$ | $(n, d_k)$ | 线性投影后，$d_k=d/h$ |
| 中间 | Prior Logits $\log \pi$ | $(n, n)$ | $\log(f(d_{ij}|\theta))$ |
| 中间 | Attention Weights $A$ | $(n, n)$ | 经过Softmax后的后验概率 |
| 输出 | Updated Embeddings | $(n, d)$ | $A \cdot V$ 聚合结果 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, decay_type='gaussian'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # 或使用 sqrt(head_dim) 如原文
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        # Learnable spatial parameters per head
        # For Gaussian: theta is sigma; Exp: lambda; Cauchy: gamma
        if decay_type == 'gaussian':
            self.spatial_params = nn.Parameter(torch.ones(num_heads) * 1.0) # sigma
        elif decay_type == 'exponential':
            self.spatial_params = nn.Parameter(torch.ones(num_heads) * 1.0) # lambda
        elif decay_type == 'cauchy':
            self.spatial_params = nn.Parameter(torch.ones(num_heads) * 1.0) # gamma
            
        self.dropout = nn.Dropout(dropout)
        self.decay_type = decay_type

    def distance_decay(self, dist_matrix, param, head_idx):
        """
        dist_matrix: (B, H, N, N) or broadcasted
        param: scalar or (H,)
        """
        p = param[head_idx] if param.dim() > 0 else param
        
        if self.decay_type == 'exponential':
            # f(d) = exp(-lambda * d)
            return torch.exp(-p * dist_matrix)
        elif self.decay_type == 'gaussian':
            # f(d) = exp(-d^2 / (2 * sigma^2))
            return torch.exp(- (dist_matrix ** 2) / (2 * p ** 2 + 1e-8))
        elif self.decay_type == 'cauchy':
            # f(d) = 1 / (1 + (d/gamma)^2)
            return 1.0 / (1.0 + (dist_matrix / (p + 1e-8)) ** 2)
        else:
            raise ValueError("Unknown decay type")

    def forward(self, x, dist_matrix):
        """
        x: (B, N, D)
        dist_matrix: (B, N, N)
        """
        B, N, D = x.shape
        H = self.num_heads
        
        # Q, K, V projection
        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, head_dim)
        
        # Calculate similarity term
        # Note: Original Eq 6 uses ||q-k||^2. 
        # Standard attention uses q @ k^T. 
        # ||q-k||^2 = ||q||^2 + ||k||^2 - 2*q@k^T.
        # Assuming the intent is similarity maximization, we use -||q-k||^2 or similar.
        # However, strictly following text "exp(||q-k||^2 ...)" implies a potential typo in paper 
        # or specific metric. We implement standard scaled dot-product with prior bias 
        # as it's the standard interpretation of "attention", but add the log-prior.
        
        # Let's assume the term is related to similarity. 
        # If we follow Eq 6 literally with ||q-k||^2, we need to check signs.
        # Given "distance-decayed priors", the feature term should reward similarity.
        # We will use -||q-k||^2 / scale to approximate q@k^T behavior while keeping norms,
        # OR simply use q@k^T if norms are normalized. The paper says "Relaxing L2 norm assumption".
        # So we keep magnitudes. 
        # Implementation choice: Use -||q-k||^2 to ensure closer features get higher score.
        
        # Compute squared Euclidean distance between Q and K
        # ||q_i - k_j||^2 = ||q_i||^2 + ||k_j||^2 - 2 q_i . k_j
        # Efficiently:
        q_sq = (q ** 2).sum(dim=-1, keepdim=True) # (B, H, N, 1)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True) # (B, H, 1, N)
        # dist_sq = q_sq + k_sq.T - 2 * q @ k.transpose(-2, -1)
        # But wait, Eq 6 has ||q-k||^2 in the EXPONENT. 
        # If it's positive, large distance -> large exponent -> high attention? No.
        # It MUST be negative. We assume the paper meant -||q-k||^2 or similar.
        
        attn_sim = -torch.cdist(q.permute(0,1,3,2).reshape(B*H, self.head_dim, N), 
                                k.permute(0,1,3,2).reshape(B*H, self.head_dim, N)).pow(2)
        attn_sim = attn_sim.reshape(B, H, N, N) / (self.head_dim ** 0.5) # Scale by sqrt(dk)
        
        # Compute Spatial Priors
        # dist_matrix is (B, N, N). Expand to (B, H, N, N)
        dist_expanded = dist_matrix.unsqueeze(1).expand(-1, H, -1, -1)
        
        # Calculate priors for each head
        priors = []
        for h in range(H):
            p_val = self.spatial_params[h]
            # Apply decay function
            if self.decay_type == 'exponential':
                p = torch.exp(-p_val * dist_expanded[:, h])
            elif self.decay_type == 'gaussian':
                p = torch.exp(- (dist_expanded[:, h] ** 2) / (2 * (p_val ** 2 + 1e-8)))
            elif self.decay_type == 'cauchy':
                p = 1.0 / (1.0 + (dist_expanded[:, h] / (p_val + 1e-8)) ** 2)
            priors.append(p)
        priors = torch.stack(priors, dim=1) # (B, H, N, N)
        
        # Combine: Log(Prior) + Similarity
        # Eq 6: exp( Sim + log(Prior) ) => Prior * exp(Sim)
        # Numerical stability: Softmax on (Sim + log(Prior))
        log_prior = torch.log(priors + 1e-8)
        logits = attn_sim + log_prior
        
        # Spatial Pruning (Optional but recommended for efficiency)
        # Threshold tau. Keep entries where prior >= tau (or log_prior >= log_tau)
        # This effectively masks out low-probability interactions before softmax
        # Or simply let the small values vanish in softmax. 
        # Paper suggests explicit pruning for O(N*K) complexity.
        # Here we implement the soft version via masking if needed, 
        # but for general implementation, the log-prior acts as a strong gate.
        
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Output
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        return out
```

#### 6. 实现提示
- **关键网络组件**：`nn.Linear` 用于QKV投影；自定义的 `distance_decay` 函数用于生成先验。
- **重要超参数**：
    - `num_heads`: 注意力头数。
    - `decay_type`: 'exponential', 'gaussian', 'cauchy'。
    - `spatial_params`: 每个头的衰减参数（$\lambda, \sigma, \gamma$），初始化为1.0左右，可学习。
    - `tau`: 剪枝阈值（Sec 3.4），文中提到设为常数如 $1e^{-3}$，不微调。
    - `alpha`: 多样性损失权重。
- **归一化/激活方式**：Softmax用于注意力权重；ReLU/GELU用于MLP（如有）；Log用于先验组合以保持数值稳定。
- **维度对齐方式**：距离矩阵 $D$ 需要在Batch和Head维度上广播。
- **实现注意事项**：
    - **公式符号歧义**：如前所述，实现时应确保特征相似度项（$\|q-k\|^2$ 或 $q^\top k$）与距离衰减项协同工作，即“特征相似且空间邻近”获得高注意力。原文公式(6)的正号极可能是负号的笔误，因为 $\|q-k\|^2$ 越大代表越不相似。
    - **距离计算**：需要预先计算Tile中心的欧氏距离矩阵。
    - **剪枝优化**：为了实现 $O(N \cdot K)$ 复杂度，可以在Softmax前，将 $\log(f(d_{ij})) < \log(\tau)$ 的位置Mask掉（设为 $-\infty$）。

#### 7. 计算与资源开销
- **理论计算复杂度**：
    - 无剪枝：$O(N^2 \cdot d)$，同标准Attention。
    - 有剪枝：$O(N \cdot K^2 \cdot d)$，其中 $K$ 是动态学习的局部感受野大小（由 $f^{-1}(\tau)$ 决定）。
- **参数量**：主要增加在每个头的空间衰减参数（极少，每个头1-3个标量）和可能的MLP层。总体参数量低于许多图神经网络基线。
- **FLOPs/MACs**：由于剪枝策略，训练时的平均FLOPs显著低于SM-MIL等基线（Fig 5显示PSA-MIL FLOPs最低）。
- **显存开销**：较低，得益于稀疏注意力计算。
- **推理速度**：快，特别是对于长序列WSI。
- **论文是否提供效率对比**：是，Fig 5展示了AUC vs FLOPs，PSA-MIL在相同或更高精度下拥有最低的FLOPs。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI分类（癌症亚型、转移检测）、生存预测。
- **可迁移到的任务/数据集**：任何具有空间结构的MIL任务，如遥感图像分割/分类、视频动作识别（帧作为Instance）、3D点云分析。
- **迁移所需调整**：需要能够计算Instance间距离的方法（如坐标、拓扑距离）。
- **适用条件**：Instance之间存在有意义的空间或拓扑距离度量。
- **潜在限制**：依赖于预训练编码器；跨域泛化能力未充分验证。

#### 9. 实验与消融证据
- **主要性能结果**：
    - TCGA-CRC Subtyping: AUC 88.97% (Gaussian), 优于所有Contextual和非Contextual基线。
    - TCGA-STAD Subtyping: AUC 82.98%, 最优。
    - CAMELYON16 Metastasis Detection: Slide-level AUC 96.1%, Patch-level Localization AUC-FROC 75.9%, 最优。
    - Survival Prediction (TCGA-CRC): C-index 70.7, 最优。
- **相对基线的提升**：相比MSA-MIL baseline，TCGA-CRC提升+5.5%，TCGA-STAD提升+3.1%。
- **相关消融实验**：
    - 多样性损失 $\alpha$：增加 $\alpha$ 提升性能（Fig 6）。
    - 剪枝阈值 $\tau$：对性能影响较小，但能带来效率增益。
    - 不同衰减函数：Gaussian表现最好，Exponential次之。
    - 多头特异性：无多样性损失时，各头收敛到相似的局部性；有损失时，各头学习不同范围的 $K$（Fig 3）。
- **作者结论**：PSA-MIL在性能和效率上均优于SOTA，动态空间建模优于固定假设。
- **证据是否充分**：是，涵盖多个数据集、多种任务、详细的消融和可视化。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 将自注意力重构为概率后验并引入可学习距离先验，理论框架新颖。 |
| 技术可行性 | 高 | 基于标准Transformer模块修改，易于集成。 |
| 实现难度 | 中 | 需注意距离矩阵计算和公式符号的正确解读（特别是负号问题）。 |
| 架构相关性 | 高 | 专为WSI/MIL设计，但也适用于其他空间MIL任务。 |
| 可迁移性 | 高 | 核心思想（距离先验+动态剪枝）通用性强。 |
| 计算成本 | 低 | 剪枝策略有效降低了计算负担。 |

#### 11. 一句话总结
PSA-MIL通过构建基于可学习距离衰减先验的概率自注意力机制，并结合多样性损失和动态空间剪枝，实现了高效且高精度的全切片图像空间上下文建模。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **概率视角的自注意力重构**：将注意力权重解释为后验分布，并将空间信息作为先验融入，提供了比单纯拼接位置编码更坚实的理论基础。
- **动态局部注意力（Dynamic Local Attention）**：通过可逆衰减函数自动确定每个头的感受野大小 $K$，无需手动设置窗口大小，实现了自适应的计算效率。

### 2. 方法之间的关系
- **PSA-MIL Core** 是基础，包含概率注意力模块。
- **Diversity Loss** 作用于 Multi-head 的 $\theta$ 参数，防止多头坍塌。
- **Spatial Pruning** 是推理/训练加速策略，依赖于 $f(d|\theta)$ 的可逆性，作用于注意力计算前的掩码步骤。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，公式、算法步骤、超参数设置较为详细。
- **关键配置是否明确**：Encoder使用Lunit或UNI，Loss权重 $\alpha$ 和阈值 $\tau$ 有提及，但最佳 $\alpha$ 值需参考附录或调参。
- **预计复现难点**：
    1.  **公式符号确认**：需调试 Eq. 6 中 $\|q-k\|^2$ 的符号，确保其与距离衰减方向一致。
    2.  **距离矩阵计算**：WSI中Tile的数量巨大，计算 $N \times N$ 距离矩阵可能显存溢出，需采用分块计算或近似最近邻搜索（虽然论文声称用剪枝优化，但初始距离矩阵仍需生成或估算）。
    3.  **KDE实现**：多样性损失中的KDE采样和密度估计需小心实现以避免梯度断裂。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：距离衰减先验机制可以很容易地替换到现有的TransMIL、CLAM等架构的Attention层中。
- **需要改造的设计**：剪枝策略需要根据具体的硬件内存限制调整阈值 $\tau$ 的实现方式（是硬掩码还是软权重截断）。
- **可能形成的新研究思路**：
    - 探索非欧几里得距离（如图拓扑距离）下的概率注意力。
    - 将动态局部性扩展到层级式MIL结构中。
    - 结合对比学习进一步解耦多头表示。

### 5. 阅读备注
- 论文发表于2025年，属于最新工作。
- 强调“概率解释”是其区别于其他位置编码方法的核心卖点。
- 实验部分特别强调了与固定局部注意力（K-local）的对比，证明了动态学习的优越性。
