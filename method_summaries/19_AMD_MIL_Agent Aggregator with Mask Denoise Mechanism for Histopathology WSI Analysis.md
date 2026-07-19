# 19_AMD_MIL_Agent Aggregator with Mask Denoise Mechanism for Histopathology WSI Analysis 方法总结

> 证据说明：输入为完整论文全文（9页），包含摘要、引言、相关工作、方法论、实验及结论。PDF提取文本基本完整，公式符号清晰，无缺失页面或关键公式残缺情况。

## 一、论文基本信息

- **论文标题**：Agent Aggregator with Mask Denoise Mechanism for Histopathology Whole Slide Image Analysis
- **作者**：Xitong Ling, Minxi Ouyang, Yizhi Wang, Xinrui Chen, Renao Yan, Hongbo Chu, Junru Cheng, Tian Guan, Sufang Tian, Xiaoping Liu, Yonghong He
- **发表年份**：2024
- **会议/期刊**：ACM Multimedia (MM '24)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1145/3664647.3681425 / arXiv:2409.11664v1
- **代码仓库**：未说明
- **研究任务**：全切片图像（WSI）的弱监督分类与感兴趣区域（ROI）定位
- **数据模态**：数字病理学图像（Histopathology WSIs）

## 二、论文整体概述

### 1. 核心问题
传统基于注意力的多实例学习（MIL）聚合器无法捕捉实例间信息；自注意力机制虽然能捕捉长距离依赖，但具有二次计算复杂度 $O(N^2)$，难以处理包含数千个patch的WSI。现有的近似自注意力方法（如TransMIL使用的Nyström Attention）存在采样偏差和信息稀释问题，且固定采样策略难以适应不同Bag大小的变化。此外，标准Agent Attention使用均值池化生成Agent Token，无法适应变长输入且可能丢失重要信息。

### 2. 整体方法
提出 **AMD-MIL** (Agent Aggregator with Mask Denoise Mechanism)。该方法包含两个核心创新：
1.  **可训练Agent聚合器 (Trainable Agent Aggregator)**：将标准Agent Attention中的均值池化Agent替换为可学习的参数矩阵，作为Query和Key之间的中间变量，实现线性复杂度的全局建模。
2.  **掩码去噪机制 (Mask Denoise Mechanism)**：通过从Agent聚合的值中投影生成可学习的掩码矩阵和去噪矩阵。掩码矩阵动态过滤低贡献表示，去噪矩阵修正因二值掩码引入的相对噪声，从而优化注意力分配并提高可解释性。

### 3. 主要贡献
- 提出了基于可训练Agent的聚合器，解决了标准Agent Attention对变长输入适应性差的问题，同时保持了线性时间复杂度。
- 设计了掩码去噪机制，通过动态调整实例表示来优化注意力分数分布，增强模型对微转移灶的捕捉能力和临床可解释性。
- 在CAMELYON-16/17, TCGA-KIDNEY, TCGA-LUNG四个数据集上取得了优于SOTA方法（如TransMIL, CLAM等）的性能。

## 三、方法总结

### 方法 1：AMD-MIL 整体框架与特征提取

#### 1. 核心思想与解决的问题
- **目标问题**：WSI分析中实例数量巨大（N大），直接应用自注意力计算成本高；需要有效的特征聚合方式以区分癌变与非癌变区域。
- **现有方法的局限**：Pooling方法忽略实例关系；Self-Attention计算复杂度高；Nyström Attention采样不均导致信息丢失。
- **核心思想**：采用“特征提取 -> Agent聚合 -> 掩码去噪 -> 分类”的流程。利用可训练Agent矩阵降低复杂度，并通过掩码去噪细化特征表示。
- **创新点**：引入可训练Agent替代池化Agent；引入基于Value投影的动态掩码和去噪模块。

#### 2. 详细结构与数据流
- **输入**：WSI被分割为 $N$ 个不重叠的 patch，经过预训练编码器（ResNet50）提取特征，得到特征矩阵 $H \in \mathbb{R}^{(N+1) \times D}$，其中包含一个嵌入的Class Token。
- **处理流程**：
    1.  线性投影生成 Query ($Q$), Key ($K$), Value ($V$)。
    2.  可训练Agent矩阵 $A$ 参与计算，生成中间变量 $V_A$。
    3.  通过线性层从 $V_A$ 生成掩码阈值 $\tau$ 和掩码矩阵 $M$。
    4.  应用掩码过滤 $V_A$ 得到 $V_M$，并加入去噪项 $D_N$ 得到 $V_{MD}$。
    5.  最终输出加权后的Bag特征。
- **输出**：加权后的Bag特征表示，用于后续分类头预测。
- **模块在整体网络中的位置**：位于特征提取器之后，分类头之前。
- **与其他模块的连接方式**：接收Patch特征 $H$，输出聚合后的特征 $Y$。

#### 3. 数学公式

**特征提取与投影：**
$$ Q = H W_Q, \quad K = H W_K, \quad V = H W_V $$
其中 $W_Q, W_K, W_V$ 为可训练权重矩阵，$H \in \mathbb{R}^{(N+1) \times D}$。

**可训练Agent聚合：**
$$ Q_A = Q A^T \in \mathbb{R}^{(N+1) \times n} $$
$$ K_A = A K^T \in \mathbb{R}^{n \times (N+1)} $$
$$ V_A = \sigma(K_A) V \in \mathbb{R}^{n \times D} $$
其中 $\sigma(\cdot)$ 为Softmax函数，$A \in \mathbb{R}^{n \times D}$ 为可训练Agent矩阵，$n$ 为Agent数量（超参数）。

**掩码去噪机制 (Mask Denoise Mechanism)：**
生成阈值 $\tau$：
$$ \tau = \sigma(p(W_\tau V_A^T)) $$
其中 $W_\tau \in \mathbb{R}^{1 \times D}$，$p$ 为聚合函数（论文实验表明使用Linear层优于Mean/CNN）。

生成掩码矩阵 $M$ 和去噪矩阵 $D_N$：
$$ M = W_M V_A $$
$$ D_N = W_{DN} V_A $$
其中 $W_M, W_{DN}$ 为可学习参数。

应用掩码与去噪：
$$ V_{MD_{ij}} = V_{A_{ij}} \mathbb{I}_{M_{ij} > \tau} + D_{N_{ij}} $$
或者根据Algorithm 1的逻辑，先通过阈值筛选生成二值掩码 $M_t$，再相乘：
$$ M_t = \text{where}(M > \tau, 1, 0) $$
$$ V_M = V_A \odot M_t $$
$$ V_{MD} = V_M + D_N $$

**最终输出：**
$$ Y = \sigma(Q_A) V_{MD} $$
注：Algorithm 1中第15行显示 $Y = \text{matmul}(Q_A, V_{MD})$，结合公式(14) $O = \sigma(Q_A^T) V_{MD}$，这里可能存在维度转置的细节差异，通常Agent Attention最后一步是将Query侧的权重作用于Value侧。根据Eq 9和Eq 14，最终聚合形式为：
$$ Output = \sigma(Q A^T) V_{MD} $$
*(注：Algorithm 1第15行 `torch.matmul(Q_A, V_M_D)` 若 $Q_A$ 为 $(B, N, n)$ 且 $V_{MD}$ 为 $(B, n, D)$，则结果为 $(B, N, D)$，这与公式(14)略有不同，公式(14)暗示 $Q_A$ 可能指代注意力权重部分。依据Algorithm 1的代码逻辑为准进行伪代码编写)*

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入特征 | $H$ | $(B, N+1, D)$ | Batch size, Patch数+1(Class Token), 特征维度 |
| 投影后 | $Q, K, V$ | $(B, N+1, D)$ | 线性变换后 |
| Agent矩阵 | $A$ | $(B, n, D)$ | 可训练参数，$n$为Agent数 |
| 中间变量 | $Q_A$ | $(B, N+1, n)$ | $Q \cdot A^T$ |
| 中间变量 | $K_A$ | $(B, n, N+1)$ | $A \cdot K^T$ |
| 聚合值 | $V_A$ | $(B, n, D)$ | $\text{Softmax}(K_A) \cdot V$ |
| 掩码阈值 | $\tau$ | $(B, 1)$ 或标量 | 由 $V_A$ 投影得到 |
| 掩码矩阵 | $M$ | $(B, n, D)$ | $W_M \cdot V_A$ |
| 去噪矩阵 | $D_N$ | $(B, n, D)$ | $W_{DN} \cdot V_A$ |
| 去噪后值 | $V_{MD}$ | $(B, n, D)$ | 掩码过滤并加上去噪项 |
| 输出特征 | $Y$ | $(B, N+1, D)$ | 加权后的Bag表示 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMD_MIL_Aggregator(nn.Module):
    def __init__(self, input_dim, agent_dim=64, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.agent_dim = agent_dim
        
        # Linear projections for Q, K, V
        self.proj_qkv = nn.Linear(input_dim, 3 * input_dim)
        
        # Trainable Agent Matrix A
        self.A = nn.Parameter(torch.randn(agent_dim, input_dim))
        
        # Mask and Denoise parameters
        # According to Eq 11, 12 and Algorithm 1
        # Threshold generation: linear layer on VA
        self.linear_tau = nn.Linear(input_dim, 1) 
        
        # Mask matrix M = W_M * VA
        self.linear_mask = nn.Linear(input_dim, input_dim)
        
        # Denoise matrix DN = W_DN * VA
        self.linear_denoise = nn.Linear(input_dim, input_dim)
        
        # Softmax function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, H):
        """
        H: (B, N+1, D)
        """
        B, N_plus_1, D = H.shape
        
        # 1. Project Q, K, V
        qkv = self.proj_qkv(H) # (B, N+1, 3*D)
        Q = qkv[:, :, :D]      # (B, N+1, D)
        K = qkv[:, :, D:2*D]   # (B, N+1, D)
        V = qkv[:, :, 2*D:]    # (B, N+1, D)
        
        # 2. Agent Attention Computation
        # A shape: (agent_dim, D) -> needs to be compatible with batch or broadcast
        # In Algo 1, A is treated as trainable parameters, likely shared across batch or expanded
        # Assuming A is (agent_dim, D) or (B, agent_dim, D). 
        # Paper says "A: (B, n, D) <- trainable parameters", implying per-batch or learned embedding.
        # Let's assume A is a parameter of shape (agent_dim, D) and we expand it or treat it as global.
        # However, Algo 1 line 4 suggests A might be batch-dependent or just initialized. 
        # Standard practice for such agents is often global or per-batch. 
        # Given "trainable parameters" and shape (B, n, D) in Algo 1 comment, 
        # let's assume A is a buffer or parameter that matches batch size or is broadcasted.
        # For simplicity in implementation, if A is (n, D), we repeat it.
        
        A = self.A.unsqueeze(0).expand(B, -1, -1) # (B, n, D)
        
        # QA = Q @ A.T -> (B, N+1, n)
        QA = torch.matmul(Q, A.transpose(1, 2))
        
        # KA = A @ K.T -> (B, n, N+1)
        KA = torch.matmul(A, K.transpose(1, 2))
        
        # VA = Softmax(KA) @ V -> (B, n, D)
        # Note: Softmax is applied over the instance dimension (dim=2 for KA which is n x N)
        KA_softmax = self.softmax(KA) 
        VA = torch.matmul(KA_softmax, V) # (B, n, D)
        
        # 3. Mask Denoise Mechanism
        # Generate Threshold tau
        # Algo 1 Line 10: TH = nn.linear(VA).squeeze().mean(-1) ?? 
        # Text Eq 11: tau = sigma(p(W_tau VA^T)). 
        # Let's follow Algo 1 logic more closely for threshold if possible, 
        # but Eq 11 is more descriptive. 
        # Algo 1 Line 10 seems to compute a scalar threshold per batch? 
        # "TH : (B, 1) <- nn.linear(VA).squeeze().mean(-1)"
        # This implies a single threshold value for the whole bag in the batch? 
        # Or per agent? The text says "threshold used in Eq 12". Eq 12 uses I_{M_ij > tau}.
        # If tau is scalar, it applies globally.
        
        # Implementation based on Algo 1 Line 10 interpretation:
        # Linear projection of VA to get scores, then mean over feature dim?
        # Let's stick to the most logical interpretation of Eq 11 + Algo 1:
        # Compute a score for each agent-feature pair or aggregate to a scalar.
        # Let's use the Linear layer approach from Eq 11 for robustness, 
        # but Algo 1 explicitly writes a specific operation.
        # Re-reading Algo 1 Line 10: `nn.linear(VA).squeeze().mean(-1)`
        # VA is (B, n, D). Linear(D, 1) -> (B, n, 1). Squeeze -> (B, n). Mean(-1) -> (B,).
        # So TH is a scalar per batch item.
        
        # However, Eq 12 compares M_ij and tau. If tau is scalar, M must be comparable.
        # M is (B, n, D). Comparing element-wise with scalar tau makes sense.
        
        # Let's implement Algo 1 Line 10 exactly:
        # Note: Algo 1 Line 9 generates M via nn.linear(VA). 
        # Wait, Algo 1 Line 9: M : (B, n, D) <- nn.linear(VA). 
        # This M is the "importance matrix" before thresholding? 
        # And Line 10 computes TH.
        
        M_raw = self.linear_mask(VA) # (B, n, D) -- Using linear_mask as W_M
        # Actually, Algo 1 Line 9 says M comes from nn.linear(VA). 
        # Let's define a specific linear layer for M generation if needed, 
        # or reuse linear_mask.
        
        # Threshold calculation per Algo 1 Line 10
        # We need a linear layer to project VA to 1 dim first? 
        # Algo 1 doesn't specify the linear layer for TH explicitly other than "nn.linear".
        # Let's add a linear layer for TH generation.
        if not hasattr(self, 'linear_th'):
            self.linear_th = nn.Linear(D, 1)
            
        th_scores = self.linear_th(VA) # (B, n, 1)
        TH = th_scores.squeeze(-1).mean(dim=1, keepdim=True) # (B, 1) ? Or (B,)
        # To broadcast with (B, n, D), TH should be (B, 1, 1)
        TH = TH.mean(dim=1, keepdim=True) # Just taking mean over agents too? 
        # Algo 1: `.mean(-1)` usually means last dim. 
        # If TH is (B, n), mean(-1) is (B,). 
        # Let's assume TH is a scalar per batch for global masking.
        TH_val = TH.squeeze() # (B,)
        
        # Generate Binary Mask Mt
        # M_raw is (B, n, D). TH_val is (B,).
        # Broadcasting: M_raw > TH_val.unsqueeze(1).unsqueeze(2)
        Mt = (M_raw > TH_val.unsqueeze(1).unsqueeze(2)).float() # (B, n, D)
        
        # Apply Mask
        VM = VA * Mt # (B, n, D)
        
        # Generate Denoise
        DN = self.linear_denoise(VA) # (B, n, D)
        
        # Combine
        V_MD = VM + DN # (B, n, D)
        
        # 4. Final Output
        # Y = matmul(QA, V_MD)
        # QA is (B, N+1, n), V_MD is (B, n, D)
        Y = torch.matmul(QA, V_MD) # (B, N+1, D)
        
        return Y
```

#### 6. 实现提示
- **关键网络组件**：`nn.Linear` 用于QKV投影、Agent生成、Mask/Denoise生成。`nn.Parameter` 用于初始化Agent矩阵 $A$。
- **重要超参数**：
    - `agent_dim` ($n$): Agent的数量。实验测试了32, 64, 128, 256, 384, 512。
    - `input_dim` ($D$): 特征维度（ResNet50通常为2048）。
- **归一化/激活方式**：Softmax用于注意力权重计算。线性层后未明确提及激活函数，但在生成Mask时通过阈值比较隐含了非线性的选择过程。
- **维度对齐方式**：Agent矩阵 $A$ 需要在Batch维度上进行广播或复制，以匹配输入特征 $H$ 的Batch Size。
- **实现注意事项**：
    - Algorithm 1中阈值 $TH$ 的计算方式较为特殊（对VA投影后取平均），需严格遵循代码逻辑而非仅凭公式直觉。
    - Mask操作是逐元素的二值化（0或1），这可能导致梯度消失问题，但在推理和反向传播中通常通过Straight-Through Estimator或直接使用可微分的近似（如果论文未提及ST estimator，则按硬阈值处理，实际训练中可能需要关注梯度流动）。*注：论文未明确提及ST estimator，但硬阈值通常不可导。考虑到这是MIL聚合器，可能在训练中使用软掩码或假设梯度可以通过其他方式传递，或者在实际实现中使用了可微近似。但在复现时，若遇到梯度问题，建议检查是否应使用Sigmoid代替硬阈值。然而，根据论文描述 "transform into binary matrices via threshold filtering"，应为硬阈值。*
- **依赖的特殊算子或第三方库**：PyTorch基础算子。

#### 7. 计算与资源开销
- **理论计算复杂度**：
    - 标准Self-Attention: $O(N^2 D)$。
    - AMD-MIL: Agent数量为 $n$。
        - $Q_A, K_A$ 计算: $O(N \cdot n \cdot D)$。
        - $V_A$ 计算: $O(n \cdot N \cdot D)$。
        - 最终输出: $O(N \cdot n \cdot D)$。
    - 总体复杂度为线性 $O(N \cdot n \cdot D)$，远小于 $O(N^2 D)$，因为 $n \ll N$。
- **参数量**：主要增加在于可训练Agent矩阵 $A$ ($n \times D$)，以及Mask/Denoise相关的线性层参数 ($3 \times D^2$ 左右)。相对于Transformer巨大的FFN参数，AMD-MIL参数量较小。
- **FLOPs/MACs**：显著低于TransMIL，因为避免了 $N \times N$ 的注意力矩阵计算。
- **显存开销**：由于不需要存储 $N \times N$ 的注意力图，显存占用大幅降低，适合处理大规模WSI。
- **推理速度**：比TransMIL快，接近线性Attention的速度。
- **论文是否提供效率对比**：提供了收敛曲线对比（Figure 7），显示更稳定，但未直接给出FLOPs或FPS数值对比表格。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：计算病理学WSI分类（癌症检测、亚型分类）。
- **可迁移到的任务/数据集**：任何基于MIL的长序列分类任务，如视频分类、音频事件检测、其他领域的文档分类。
- **迁移所需调整**：调整Agent数量 $n$ 以适应不同长度的序列；调整特征维度 $D$。
- **适用条件**：序列长度 $N$ 较大，且存在大量无关背景（需要Mask机制过滤噪声）。
- **潜在限制**：硬阈值掩码可能影响梯度传播；Agent数量 $n$ 的选择对性能敏感（如图5所示，过大或过小均可能影响效果）。

#### 9. 实验与消融证据
- **主要性能结果**：
    - CAMELYON-16: ACC 92.9%, AUC 96.4%, F1 92.7%。
    - TCGA-KIDNEY: ACC 94.4%, AUC 97.3%, F1 92.9%。
    - 均优于CLAM, TransMIL, DSMIL等基线。
- **相对基线的提升**：在CAMELYON-16上比TransMIL (ACC 87.8%) 提升约5.1%。
- **相关消融实验**：
    - **Table 2**: 验证了各组件的有效性。
        - Baseline (Nyström): ACC 87.8%。
        - + Trainable Agent: ACC 89.3%。
        - + Mask: ACC 91.5%。
        - + Denoise: ACC 93.0% (Note: Table 2最后一行是Full Model 92.9%，中间步骤有波动，可能是随机种子或交叉验证差异，但趋势是增加的)。
        - *更正*: Table 2显示 Full Model (All ticks) 为 92.9%。Step-by-step: Nyström(87.8) -> Trainable(89.3) -> +Mask(91.5) -> +Denoise(93.0?? No, last row is 92.9). 实际上Table 2最后一行是全模型。倒数第二行是去掉Denoise？不，Table 2列标题是 Component: Nyström agent, train, mask, denoise.
          - Row 1: Only Nyström (Baseline TransMIL part?) -> 87.8
          - Row 2: Trainable Agent only? -> 89.3
          - Row 3: Trainable + Mask -> 91.5
          - Row 4: Trainable + Mask + Denoise? -> 93.0 (Wait, Table 2 has 5 rows. Last one is Full Model 92.9. The 4th row is 93.0? Let's re-read carefully.)
          - Table 2 Rows:
            1. ✓ (Nyström) -> 87.8
            2. ✓ (Trainable) -> 89.3
            3. ✓ ✓ (Trainable + Mask) -> 91.5
            4. ✓ ✓ ✓ (Trainable + Mask + Denoise?? No, check columns)
            Columns: Nyström agent, train, mask, denoise.
            Row 1: ✓ under Nyström. Others empty.
            Row 2: ✓ under train.
            Row 3: ✓ under train, ✓ under mask.
            Row 4: ✓ under train, ✓ under mask, ✓ under denoise? -> Result 93.0?
            Row 5: All ✓ -> Result 92.9.
            *Discrepancy*: Row 4 shows 93.0, Row 5 shows 92.9. This might be due to different random seeds or cross-validation folds averaging differently, or a typo in the paper. However, the trend generally supports the components.
    - **Table 3**: 阈值选择方法对比。Linear层生成的阈值优于Mean和CNN方法。
    - **Figure 5**: Agent数量敏感性分析。
- **作者结论**：可训练Agent和掩码去噪机制均有效，提升了性能和可解释性。
- **证据是否充分**：在四个公开数据集上进行了广泛实验，消融实验完整，可视化证明了可解释性。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 结合了可训练Agent和动态掩码去噪，针对WSI特性进行了专门设计。 |
| 技术可行性 | 高 | 基于标准Attention修改，易于集成到现有MIL框架中。 |
| 实现难度 | 中 | 需注意Agent维度的广播和阈值的计算逻辑。 |
| 架构相关性 | 高 | 专为MIL和长序列设计，与ViT/Transformer架构兼容性好。 |
| 可迁移性 | 高 | 适用于任何需要高效长序列聚合的任务。 |
| 计算成本 | 低 | 线性复杂度，显存友好。 |

#### 11. 一句话总结
AMD-MIL通过引入可训练Agent矩阵和动态掩码去噪机制，实现了高效、线性复杂度的WSI特征聚合，显著提升了分类精度和模型的可解释性。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **可训练Agent替代池化Agent**：解决了标准Agent Attention在处理变长病理切片时的局限性，同时保留了线性复杂度优势。
- **基于Value投影的动态掩码**：不依赖外部标签或固定规则，而是从数据内部表示中学习重要性并进行过滤，增强了模型的鲁棒性。

### 2. 方法之间的关系
- **与TransMIL的关系**：AMD-MIL可以看作是TransMIL的一种改进版。它用Agent Attention替代了Nyström Approximation，解决了Nyström采样的偏差问题，并通过Mask Denoise进一步细化了特征。
- **与ABMIL/DSMIL的关系**：相比轻量级注意力，AMD-MIL引入了Agent作为中间变量，能够捕捉更复杂的跨实例交互，同时保持较低的计算成本。

### 3. 复现可行性
- **代码是否公开**：未说明（通常此类会议论文代码不一定立即开源，需查看arXiv或作者主页）。
- **方法描述是否完整**：算法步骤（Algorithm 1）和公式描述较为清晰，但关于Threshold的具体计算细节（Algo 1 Line 10 vs Eq 11）存在细微歧义，需结合代码逻辑推断。
- **关键配置是否明确**：超参数如Agent数量、学习率、Batch Size等在实验部分有提及，但具体的网络结构细节（如ResNet50的具体修改）需参考常见做法。
- **预计复现难点**：
    1.  **Threshold计算**：准确复现Algorithm 1中第10行的阈值计算逻辑。
    2.  **Hard Mask的梯度**：硬阈值操作在反向传播中的处理。
    3.  **Agent初始化**：Agent矩阵 $A$ 的初始化方式可能影响收敛。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：可训练Agent模块可以直接嵌入到任何基于Transformer的MIL模型中，替换原有的Self-Attention或Linear Attention模块。
- **需要改造的设计**：Mask Denoise机制需要根据具体任务的噪声特性调整阈值生成策略（例如，是否需要对每个Agent单独设置阈值）。
- **可能形成的新研究思路**：
    - 探索更复杂的去噪矩阵结构（如非线性变换）。
    - 将Mask机制应用于其他视觉任务中的稀疏注意力优化。
    - 结合自监督学习，让Mask机制自动发现语义块。

### 5. 阅读备注
- 论文中Table 2的消融实验结果存在微小的数值倒挂（Row 4高于Row 5），建议在复现时多次运行取平均以确认各组件的确切贡献。
- Figure 3的可视化展示了模型对微转移灶的关注，这是病理诊断中的难点，体现了该方法的临床价值。
