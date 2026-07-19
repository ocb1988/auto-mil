# 04_TRANS_MIL_Transformer based Correlated Multiple Instance Learning for WSI Classification 方法总结

> 证据说明：输入为完整论文文本（15页），包含摘要、引言、方法、实验、附录及参考文献。公式提取基本完整，关键算法步骤清晰。无明显的页面缺失或公式乱码导致无法理解的情况。

## 一、论文基本信息

- **论文标题**：TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification
- **作者**：Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, Yongbing Zhang
- **发表年份**：2021 (NeurIPS 2021)
- **会议/期刊**：35th Conference on Neural Information Processing Systems (NeurIPS 2021)
- **论文链接/DOI/arXiv ID**：arXiv:2106.00908v2
- **代码仓库**：https://github.com/szc19990412/TransMIL
- **研究任务**：全切片图像（WSI）的弱监督分类（包括二分类和多分类）
- **数据模态**：数字病理学全切片图像（WSI），提取为256x256的Patch特征序列

## 二、论文整体概述

### 1. 核心问题
现有的基于深度学习的多实例学习（MIL）方法通常假设Bag中的Instances（即WSI中的Patch）是独立同分布（i.i.d.）的。然而，在病理诊断中，不同区域之间存在强烈的上下文和空间相关性。忽略这种相关性会导致信息利用不充分。此外，传统Transformer因自注意力机制的计算复杂度为 $O(n^2)$，难以直接处理WSI中成千上万个Patch构成的长序列。

### 2. 整体方法
论文提出了一个**相关MIL（Correlated MIL）**框架，并在此基础上设计了**TransMIL**。
1.  **理论层面**：证明了引入实例间相关性可以降低信息熵，从而减少不确定性。
2.  **架构层面**：设计了TPT（Transformer-based Pathology Transformer）模块，通过Nystrom近似降低计算复杂度以处理长序列；设计了PPEG（Pyramid Position Encoding Generator）模块，利用多尺度卷积生成条件位置编码，以捕捉WSI中可变长度的空间结构信息。

### 3. 主要贡献
1.  提出了相关MIL框架，并从理论上证明了其优于i.i.d.假设。
2.  设计了TransMIL模型，首次将Transformer有效应用于WSI分类，同时建模形态学和空间信息。
3.  引入了PPEG模块解决WSI Patch数量可变且缺乏绝对位置信息的问题。
4.  使用Nystrom Method近似自注意力，解决了长序列处理的计算瓶颈。
5.  在CAMELYON16、TCGA-NSCLC和TCGA-RCC数据集上取得了SOTA性能，且具有更快的收敛速度和良好的可解释性。

## 三、方法总结

### 方法 1：Correlated MIL Framework & TransMIL Architecture

#### 1. 核心思想与解决的问题
- **目标问题**：解决现有MIL方法忽略Instance间相关性以及传统Transformer无法处理WSI长序列的问题。
- **现有方法的局限**：ABMIL等Attention MIL方法假设i.i.d.，仅关注单个Instance的重要性而忽略Instance间的交互；标准Transformer计算复杂度随序列长度平方增长。
- **核心思想**：
    1.  构建一个通用的三步相关MIL算法：提取形态/空间信息 -> 通过Pooling Matrix聚合 -> 预测。
    2.  利用Transformer的Self-Attention作为Pooling Matrix，显式建模Instance两两之间的相关性。
    3.  针对WSI特性，优化Transformer的输入表示（PPEG）和注意力计算（Nystrom）。
- **创新点**：
    1.  理论上的相关MIL框架证明。
    2.  TPT模块：结合Nystrom近似和PPEG的高效Transformer变体。
    3.  PPEG：基于卷积的条件位置编码，适应可变长度序列。

#### 2. 详细结构与数据流
- **输入**：
    -   Bag of instances $X_i = \{x_{i,1}, ..., x_{i,n}\}$，其中每个 $x_{i,j}$ 是通过ResNet50提取的Patch特征向量。
    -   Bag-level label $Y_i$。
- **处理流程**：
    1.  **特征提取**：WSI裁剪为256x256 patches，背景剔除。使用ImageNet预训练的ResNet50提取特征，维度为1024。通过全连接层降维至512。得到 $H_i \in \mathbb{R}^{n \times 512}$。
    2.  **TPT模块处理**：
        -   **Sequence Squaring**：添加Class Token，并将序列长度补齐至 $\sqrt{n}$ 的整数倍 $N+1$。
        -   **Correlation Modelling**：第一层Multi-head Self-Attention (MSA)，使用Nystrom近似计算注意力矩阵。
        -   **Conditional Position Encoding**：应用PPEG模块，融合多尺度空间信息。
        -   **Deep Feature Aggregation**：第二层MSA + MLP + Layer Norm。
    3.  **Prediction**：取Class Token的输出，经过MLP映射到类别概率。
- **输出**：Bag-level predicted label $\hat{Y}_i$。
- **模块在整体网络中的位置**：位于特征提取器（ResNet50）之后，分类头之前。
- **与其他模块的连接方式**：接收降维后的Patch特征序列，输出聚合后的全局表示。

#### 3. 数学公式

**Algorithm 1: Generic Three-step Approach**
$$
\begin{aligned}
1) \quad & X_f \leftarrow f(X_i), \quad X_h \leftarrow h(X_i), \quad X_{fh} \leftarrow X_f + X_h \\
2) \quad & X_P \leftarrow P X_{fh} \\
3) \quad & \hat{Y}_i \leftarrow g(X_P)
\end{aligned}
$$
其中 $P \in \mathbb{R}^{n \times n}$ 是Pooling Matrix，在TransMIL中由Self-Attention机制动态生成。

**Theorem 2 (Information Entropy)**:
$$
H(\Theta_1, \dots, \Theta_n) = \sum_{t=2}^{n} H(\Theta_t | \Theta_1, \dots, \Theta_{t-1}) + H(\Theta_1) \leq \sum_{t=1}^{n} H(\Theta_t)
$$
表明相关假设下的信息熵小于或等于独立假设下的信息熵。

**Nystrom Approximation for Self-Attention (Eq. 9)**:
$$
\hat{S} = \text{softmax}\left(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_q}}\right)\left(\text{softmax}\left(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_q}}\right)\right)^+ + \text{softmax}\left(\frac{\tilde{Q}K^T}{\sqrt{d_q}}\right)
$$
其中 $\tilde{Q}, \tilde{K}$ 是从原始 $Q, K$ 中选出的 $m$ 个landmarks，$(\cdot)^+$ 表示Moore-Penrose伪逆。复杂度从 $O(n^2)$ 降至 $O(n)$。

**PPEG Processing (Algorithm 3)**:
1.  Split: $H_S^\ell$ 分为 patch tokens $H_f \in \mathbb{R}^{N \times d}$ 和 class token $H_c \in \mathbb{R}^{1 \times d}$。
2.  Spatial Restore: $H_f$ reshape 为 $H_f^S \in \mathbb{R}^{\sqrt{N} \times \sqrt{N} \times d}$。
3.  Group Convolution: 使用 kernel size $k \in \{3, 5, 7\}$ 的组卷积提取多尺度空间特征 $H_f^t$。
4.  Fusion: $H_F^S = H_f^S + H_f^1 + H_f^2 + H_f^3$。
5.  Flatten & Concat: 展平后与 $H_c$ 拼接得到 $H_P^S$。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| Input Patches | $X_i$ | $n \times 1024$ | ResNet50提取的特征，$n$为patch数量 |
| After FC Layer | $H_i$ | $n \times 512$ | 降维后的instance embeddings |
| Sequence Squaring | $H_S$ | $(N+1) \times 512$ | $N=\lceil\sqrt{n}\rceil$, 含class token |
| After MSA (Layer 1) | $H_S^\ell$ | $(N+1) \times 512$ | 相关性建模后 |
| After PPEG | $H_P^S$ | $(N+1) \times 512$ | 加入位置编码和局部信息 |
| After MSA (Layer 2) | $H_S^{\ell+1}$ | $(N+1) \times 512$ | 深层特征聚合 |
| Output Class Token | $(H_S^{\ell+1})^{(0)}$ | $1 \times 512$ | Class token的最终表示 |
| Prediction | $\hat{Y}_i$ | $1 \times C$ | $C$为类别数，经MLP和Softmax |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import math

class NystromAttention(nn.Module):
    """简化版Nystrom Attention，实际需参考Nyströmformer实现细节"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., landmark_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.landmark_dim = landmark_dim
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Standard QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Select landmarks (simplified: first m tokens)
        m = self.landmark_dim
        q_l, k_l = q[:, :, :m, :], k[:, :, :m, :]
        
        # Compute attention using landmarks (pseudo-code logic based on Eq 9)
        # Note: Full implementation requires careful handling of pseudoinverse and softmax normalization
        # Here we assume a helper function `nystrom_attn` exists
        attn = nystrom_attn(q, k, v, q_l, k_l, self.scale) 
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PPEG(nn.Module):
    """Pyramid Position Encoding Generator"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Using group convolutions with different kernel sizes
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
    def forward(self, x):
        # x shape: (B, N+1, D)
        # Split class token and patch tokens
        cls_token = x[:, 0:1, :] # (B, 1, D)
        patch_tokens = x[:, 1:, :] # (B, N, D)
        
        B, N, D = patch_tokens.shape
        sqrt_N = int(math.sqrt(N))
        
        # Reshape to 2D spatial representation
        patch_tokens_2d = patch_tokens.permute(0, 2, 1).reshape(B, D, sqrt_N, sqrt_N)
        
        # Apply convolutions
        out1 = self.conv1(patch_tokens_2d)
        out2 = self.conv2(patch_tokens_2d)
        out3 = self.conv3(patch_tokens_2d)
        
        # Fuse
        fused_2d = patch_tokens_2d + out1 + out2 + out3
        
        # Flatten back to sequence
        fused_seq = fused_2d.reshape(B, D, N).permute(0, 2, 1)
        
        # Concat with class token
        out = torch.cat([cls_token, fused_seq], dim=1)
        return out

class TPTBlock(nn.Module):
    """Transformer Block with Nystrom Attention and PPEG"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        # PPEG is applied between the two attention layers in TPT module description
        # However, standard Transformer structure is LN -> Attn -> Add -> LN -> MLP -> Add
        # The paper says: 1) Sqauring, 2) Corr Modelling (MSA), 3) Cond Pos Enc (PPEG), 4) Deep Feat Agg (MSA)
        # This implies a specific order: MSA -> PPEG -> MSA? Or PPEG inside?
        # Algorithm 2 Step 3: HP_S <- PPEG(HS_l)
        # Step 4: HS_{l+1} <- MSA(HP_S)
        # So PPEG is inserted between the two MSA blocks of the "TPT Module" which consists of 2 layers.
        self.ppeg = PPEG(dim)

    def forward(self, x):
        # First MSA (Correlation Modelling)
        x = x + self.attn(self.norm1(x))
        
        # PPEG (Conditional Position Encoding)
        x = self.ppeg(x)
        
        # Second MSA (Deep Feature Aggregation) - Note: Paper uses another MSA block here
        # Assuming another attention layer 'attn2' or reusing structure
        # For simplicity, assuming a second identical attention block or residual connection logic
        # Based on Fig 3 and Alg 2, it's two Transformer layers. 
        # Let's assume standard Transformer layer structure but with PPEG injection.
        # Actually, Alg 2 lists: 1. Square, 2. MSA, 3. PPEG, 4. MSA, 5. MLP.
        # This suggests the "TPT Module" contains TWO MSA operations separated by PPEG.
        
        # Re-implementing strictly per Alg 2:
        # Layer 1: MSA
        # Intermediary: PPEG
        # Layer 2: MSA
        # Final: MLP
        
        # Since I defined one TPTBlock above, let's define the full TPT Module below
        pass

class TransMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_classes=2, num_heads=8, depth=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim)) # Max expected patches approx sqrt(10000)=100, squared 10000? No, seq length varies.
        # Actually, pos embed is handled by PPEG mostly, but initial embedding might need something.
        # Paper says: X0_i = [cls; features] + Epos. But then PPEG replaces/adds to this.
        
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (B, N, D_in)
        B, N, _ = x.shape
        
        # 1. Squaring of sequence
        sqrt_N = int(math.ceil(math.sqrt(N)))
        new_N = sqrt_N * sqrt_N
        pad_len = new_N - N
        
        # Pad features
        if pad_len > 0:
            padding = torch.zeros(B, pad_len, x.size(2), device=x.device)
            x = torch.cat([x, padding], dim=1)
            
        # Add Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, D)
        
        # Initial Position Embedding (Sinusoidal or Learnable? Paper mentions Sinusoidal in ablation, but PPEG is main)
        # Appendix B Eq 15: X0 = [cls; f(xi)] + Epos
        # We assume a simple positional encoding is added initially, then refined by PPEG
        
        # Process through TPT (2 MSA blocks + PPEG in between)
        # Layer 1 MSA
        x = x + self.encoder_layers[0].attn(self.encoder_layers[0].norm1(x))
        
        # PPEG
        x = self.encoder_layers[0].ppeg(x)
        
        # Layer 2 MSA
        x = x + self.encoder_layers[1].attn(self.encoder_layers[1].norm1(x))
        
        # Mapping T -> Y
        x = self.norm(x)
        logits = self.head(x[:, 0, :]) # Class token output
        return logits

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = NystromAttention(dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ppeg = PPEG(dim)

    def forward(self, x):
        # Standard Transformer step, but note TPT structure splits this
        # In TPT, we only use the MSA part for the first half, then PPEG, then second MSA
        # This class is just for component definition
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

#### 6. 实现提示
- **关键网络组件**：`NystromAttention` 是实现高效注意力的核心，需仔细实现 Landmark 选择和伪逆计算；`PPEG` 需要正确处理 Tensor 的 Reshape 和 Conv2d 操作。
- **重要超参数**：
    -   Patch特征维度：512 (由1024降维而来)。
    -   Head数量：未明确指定默认值，通常设为8或16。
    -   Nystrom Landmarks数量 ($m$)：文中提到 $m$ selected landmarks，具体数值需在实验中确定，通常远小于 $n$。
    -   Optimizer：Lookahead optimizer, LR=2e-4, Weight Decay=1e-5。
- **归一化/激活方式**：Layer Normalization (LN) 用于Attention和MLP前后；GELU 用于MLP内部。
- **维度对齐方式**：PPEG中Conv2d的Group Convolution保持通道数不变；Reshape操作确保Spatial Restore正确。
- **实现注意事项**：WSI的Patch数量 $n$ 变化很大，必须动态处理Padding到最近的平方数。Background removal是关键预处理步骤。

#### 7. 计算与资源开销
- **理论计算复杂度**：标准Self-Attention为 $O(n^2)$，TransMIL使用Nystrom近似后降至 $O(n)$（假设Landmarks数量固定且较小）。
- **参数量**：约2.6M - 2.7M (见Table 2，不含ResNet50 backbone)。
- **FLOPs/MACs**：未提供具体数值，但相比标准Transformer显著降低。
- **显存开销**：由于避免了完整的 $n \times n$ 注意力矩阵存储，显存占用大幅降低，允许处理数千个Patch。
- **推理速度**：比ABMIL、DSMIL等方法更快收敛，训练epoch数减少2-3倍。
- **论文是否提供效率对比**：提供了收敛速度对比（Figure 7），但未提供具体的FLOPs或推理时间秒数对比表格。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI病理图像分类（二分类如癌症检测，多分类如亚型分类）。
- **可迁移到的任务/数据集**：其他具有长序列依赖关系的视觉任务，如遥感图像分割、视频动作识别、文档OCR等。
- **迁移所需调整**：需重新设计位置编码以适应新数据的空间结构；调整Nystrom Landmarks数量以平衡精度和速度。
- **适用条件**：输入序列长度较长且存在显著的局部或全局相关性。
- **潜在限制**：对于极短序列，Nystrom近似的优势不明显且可能引入误差；对极端不平衡数据的鲁棒性需进一步验证（虽然论文声称适用）。

#### 9. 实验与消融证据
- **主要性能结果**：
    -   CAMELYON16: AUC 0.9309 (SOTA)。
    -   TCGA-NSCLC: AUC 0.9603 (SOTA)。
    -   TCGA-RCC: AUC 0.9882 (SOTA)。
- **相对基线的提升**：在CAMELYON16上比ABMIL高5%以上AUC。
- **相关消融实验**：
    -   PPEG有效性：对比无位置编码、Sinusoidal编码、单一卷积核编码，证明PPEG最佳。
    -   条件位置编码有效性：打乱输入顺序测试，证明顺序信息对性能有显著提升（~0.9% AUC提升）。
- **作者结论**：TransMIL在性能和收敛速度上均优于现有方法，且具有良好的可解释性。
- **证据是否充分**：在三个主流数据集上进行了全面比较和消融，证据充分。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 提出相关MIL理论框架，并结合Nystrom和PPEG解决WSI特定难题。 |
| 技术可行性 | 高 | 基于成熟组件（ResNet, Transformer, Conv）的组合，逻辑清晰。 |
| 实现难度 | 中 | Nystrom Attention的实现较为复杂，需注意数值稳定性。 |
| 架构相关性 | 高 | 专为WSI长序列和相关性建模设计。 |
| 可迁移性 | 中 | 位置编码部分（PPEG）高度依赖2D空间结构，迁移到其他非网格数据需改造。 |
| 计算成本 | 低 | 相比标准Transformer，计算效率显著提升。 |

#### 11. 一句话总结
TransMIL通过构建相关MIL框架，利用Nystrom近似Transformer和金字塔位置编码（PPEG），有效解决了WSI分类中实例间相关性建模难和长序列计算复杂度高的问题，实现了高精度、快收敛和强可解释性的病理图像分析。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **PPEG (Pyramid Position Encoding Generator)**：利用多尺度卷积生成条件位置编码，巧妙解决了WSI Patch数量可变且缺乏绝对坐标的问题，比传统的Sinusoidal或Learnable PE更具适应性。
- **Nystrom Approximation in MIL**：将Nystrom方法引入MIL框架，使得Transformer能够处理WSI级别的长序列，这是一个非常实用的工程技巧。

### 2. 方法之间的关系
- **Correlated MIL** 是理论基础，指导了 **TransMIL** 的设计。
- **TPT Module** 是 TransMIL 的核心执行单元，内部集成了 **Nystrom Attention** 和 **PPEG**。
- **PPEG** 增强了 **Self-Attention** 的空间感知能力，弥补了纯Attention机制在空间结构建模上的不足。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，包含了算法伪代码、公式和详细的实验设置。
- **关键配置是否明确**：是，明确了Backbone (ResNet50)、Embedding Dim (512)、Optimizer (Lookahead) 等。
- **预计复现难点**：Nystrom Attention的具体实现细节（如Landmarks的选择策略、伪逆的计算稳定性）可能需要参考原始的Nyströmformer论文或源码进行微调。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：PPEG模块可以很容易地集成到任何基于Transformer的医学图像分析管道中，特别是那些输入分辨率可变或序列长度不固定的任务。
- **需要改造的设计**：Nystrom Attention需要根据具体任务的序列长度和内存限制调整Landmarks的数量。
- **可能形成的新研究思路**：探索更轻量级的Position Encoding方法；将Correlated MIL的思想应用到图神经网络（GNN）或其他序列建模架构中；研究如何在极低标注数据下利用这种相关性先验。

### 5. 阅读备注
- 论文强调“Correlated”不仅指空间相邻，也指语义上的相互依赖。
- 实验部分特别指出了TransMIL在收敛速度上的优势，这对于医疗AI模型的快速迭代非常重要。
- 可视化结果显示Attention Map与病理专家标注的高度一致性，证明了模型学到了有意义的病理特征。
