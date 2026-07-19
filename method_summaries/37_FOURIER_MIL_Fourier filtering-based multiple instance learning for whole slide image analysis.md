# 37_FOURIER_MIL_Fourier filtering-based multiple instance learning for whole slide image analysis 方法总结

> 证据说明：输入为完整论文全文（16页），包含标题、摘要、引言、相关工作、方法、实验、结论及参考文献。PDF文本提取完整，关键公式（如DFT定义、APFF算法步骤）清晰可辨，无缺失或严重残缺情况。

## 一、论文基本信息

- **论文标题**：FourierMIL: Fourier Filtering-based Multiple Instance Learning for Whole Slide Image Analysis
- **作者**：Yi Zheng, Harsh Sharma, Margrit Betke, Jonathan D. Cherry, Jesse B. Mez, Jennifer E. Beane, Vijaya B. Kolachalama
- **发表年份**：2025 (Online), 2026 (Journal Issue)
- **会议/期刊**：International Journal of Computer Vision (IJCV)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1007/s11263-025-02679-x
- **代码仓库**：https://github.com/vkola-lab/ijcv2025
- **研究任务**：全切片图像（WSI）分类（包括癌症转移检测、肺癌亚型分类、阿尔茨海默病病理识别）
- **数据模态**：数字病理学图像（H&E染色和免疫组化IHC染色）

## 二、论文整体概述

### 1. 核心问题
传统深度学习模型难以直接处理吉字节级（gigapixel）的全切片图像（WSIs）。现有的多实例学习（MIL）方法存在局限：基于CNN的方法忽略长距离空间依赖；基于Transformer的方法计算复杂度高（二次方复杂度），内存占用大；基于图的方法需要复杂的拓扑构建且难以扩展；现有方法主要面向H&E染色，缺乏对特殊染色（如IHC）的泛化能力。此外，标准频域滤波方法（如低通滤波）可能丢失对病理诊断至关重要的细粒度高频信息。

### 2. 整体方法
提出 **FourierMIL**，一种基于离散傅里叶变换（DFT）的多实例学习框架。该方法将WSI分割为补丁（patches/tokens），提取特征后，通过引入 **自适应令牌填充（Adaptive Token Padding, ATP）** 缓解频谱泄漏，并利用 **全通频域滤波（All-Pass Frequency Filtering, APFF）** 模块在频率域进行高效的令牌混合（Token Mixing）。APFF结合MLP结构实现动态滤波器权重，保留所有频率成分以捕捉全局和局部依赖。最后通过类令牌（Class Token）聚合输出Slide-level预测。

### 3. 主要贡献
1. 构建了FourierMIL框架，利用大动态核的高效令牌混合捕获大量实例间的依赖关系。
2. 引入自适应令牌填充（ATP）策略，解决因令牌非周期性和DFT周期性假设导致的频谱泄漏和边缘效应。
3. 设计了基于MLP的FourierMIL模型，通过全通频域滤波（APFF）显著优于现有的Transformer和基于图的SOTA方法，并验证了其在不同染色类型（H&E和IHC）上的通用性。

## 三、方法总结

### 方法 1：全通频域滤波令牌混合 (All-Pass Frequency Filtering, APFF)

#### 1. 核心思想与解决的问题
- **目标问题**：在MIL中高效地混合大量Patch Tokens的特征，同时保持线性或近线性计算复杂度，避免Transformer自注意力的二次方复杂度，并克服传统CNN局部感受野的限制。
- **现有方法的局限**：
    - Transformer/Attention：$O(L^2)$ 复杂度，无法扩展到数万级的Patch。
    - AFNO/GFNet：通常使用固定大小的滤波器或低通/高通滤波，可能丢失重要的高频细节（对于病理图像至关重要），且AFNO等模型往往针对2D图像设计，不直接适用于1D序列化的Patch Embeddings。
    - FNet：简单的FFT乘法，未考虑数据分布变化，缺乏适应性。
- **核心思想**：利用卷积定理，将空间域的全局卷积转化为频率域的逐元素乘法。通过在频率域应用可学习的MLP滤波器来混合令牌。不同于传统的低通滤波，采用“全通”模式，允许所有频率分量通过，从而保留细粒度的局部特征和宏观的全局上下文。
- **创新点**：
    - 将频域滤波与MLP结合，形成适应任意长度输入的可学习滤波器。
    - 采用块对角结构减少参数量。
    - 明确区分并对比LP/HP/AP滤波效果，证明AP在病理分析中的优越性。

#### 2. 详细结构与数据流
- **输入**：批次大小为 $B$，令牌数量为 $L$，特征维度为 $D$ 的张量 $X \in \mathbb{R}^{B \times L \times D}$。
- **处理流程**：
    1.  **FFT变换**：对输入 $X$ 进行实数快速傅里叶变换（RFFT），得到频域表示 $X_F$。由于输入是实数，输出具有共轭对称性，形状变为 $(B, L//2 + 1, D)$。
    2.  **重塑与分块**：将 $X_F$ 重塑为 $(B, L//2 + 1, h, D/h)$，其中 $h$ 是头数（heads）。
    3.  **MLP滤波**：应用两个全连接层 $W_1, W_2$ 和非线性激活函数 $\sigma$（GELU）。$W_1$ 和 $W_2$ 被约束为块对角矩阵以共享参数并减少计算量。
        $$ \hat{X}_F = W_2 \cdot \sigma(W_1 \cdot X_F + b_1) + b_2 $$
    4.  **滤波模式选择**：根据配置模式（AP/LP/HP）处理频域信号。默认AP模式下恒等映射；LP模式下应用软收缩（Soft Shrinkage）抑制高频；HP模式下保留高频。
    5.  **逆FFT**：执行逆实数快速傅里叶变换（IRFFT），恢复时域特征。
    6.  **残差连接**：加上原始输入（经过LayerNorm后的输入，见公式9）。
- **输出**：变换后的令牌特征 $\hat{X} \in \mathbb{R}^{B \times L \times D}$。
- **模块在整体网络中的位置**：作为FourierMIL的核心Block之一，与Channel Mixing (CM) Block交替堆叠。每个Block内部先进行Token Mixing (APFF)，再进行Channel Mixing (MLP)。
- **与其他模块的连接方式**：输入来自上一层的输出或初始Embedding；输出连接到下一层的LayerNorm或直接进入下一个Block。

#### 3. 数学公式

**离散傅里叶变换 (DFT):**
$$ X_F[m] = \sum_{l=0}^{L-1} X[l] e^{-2\pi i (ml)/L}, \quad m=0,\dots,L-1 \quad (\text{Eq. 5}) $$

**频域令牌混合 (Token Mixing in Frequency Domain):**
$$ \hat{X} = \mathcal{F}^{-1}[W_F \odot \mathcal{F}(X)] \quad (\text{Eq. 6}) $$
其中 $W_F$ 是可学习的频域滤波器。

**可学习滤波器权重 (Adaptive Learnable Filter Weights):**
$$ W_F \odot \mathcal{F}(X) = W_2 \sigma(W_1 \mathcal{F}(X)) \quad (\text{Eq. 8}) $$
这里 $W_1 \in \mathbb{R}^{h \times (D/h) \times (D/h)}$, $W_2 \in \mathbb{R}^{h \times (D/h) \times (D/h)}$ (注：原文Algorithm 1描述$b_1,b_2$维度略有不同，但逻辑为MLP变换)。

**APFF 算法步骤 (Algorithm 1):**
1.  $x \leftarrow \text{RFFT}(x).reshape(B, L//2 + 1, h, D/h)$
2.  $x \leftarrow \text{MatMul}(x, W_1) + b_1$
3.  $x \leftarrow \text{ACT}(x)$  (GELU)
4.  $x \leftarrow \text{MatMul}(x, W_2) + b_2$
5.  $x \leftarrow x.reshape(B, L//2 + 1, D)$
6.  **Passing**:
    - If mode == 'AP': $x \leftarrow Identity(x)$
    - If mode == 'LP': $x \leftarrow SoftShrink(x)$
    - If mode == 'HP': $x \leftarrow (x - SoftShrink(x))$
7.  $x \leftarrow \text{IRFFT}(x)$

**FourierMIL Block 结构 (Eq. 9):**
$$ \hat{X}_b = \text{APFF}(\text{LN}(X_{b-1})) + X_{b-1} $$
$$ X_b = \text{MLP}(\text{LN}(\hat{X}_b)) + \hat{X}_b $$

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| Input | $X$ | $[B, L, D]$ | Batch size, Number of tokens, Feature dimension |
| After RFFT | $X_F$ | $[B, L//2+1, D]$ | Real FFT output |
| Reshape | $X_{block}$ | $[B, L//2+1, h, D/h]$ | Split into heads for block-diagonal MLP |
| After MLP | $\hat{X}_{freq}$ | $[B, L//2+1, D]$ | Filtered frequency representation |
| Output | $\hat{X}$ | $[B, L, D]$ | Inverse FFT result |

*注：实验中 $D=512$ (压缩后), $h$ 未明确给出具体数值，但提及分为 $h$ 个权重块。*

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class APFFBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, act_layer=nn.GELU):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        # Block diagonal structure weights
        # W1 shape: [num_heads, head_dim, head_dim]
        # W2 shape: [num_heads, head_dim, head_dim]
        self.W1 = nn.Parameter(torch.randn(num_heads, head_dim, head_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.W2 = nn.Parameter(torch.randn(num_heads, head_dim, head_dim))
        self.b2 = nn.Parameter(torch.zeros(num_heads, head_dim))
        
        self.act = act_layer()
        self.mode = 'AP' # Default All-Pass

    def forward(self, x):
        """
        x: [B, L, D]
        """
        residual = x
        x = self.norm(x)
        
        # 1. FFT
        # rfft along the last dimension (sequence length is not the last dim here? 
        # Wait, standard ViT/Mixer usually mixes tokens along sequence dim.
        # Eq 6 implies mixing across tokens. 
        # Let's assume input is [B, L, D]. FFT should be applied on L dimension.
        # However, PyTorch rfft defaults to last dim. 
        # Need to permute or specify dim.
        
        # Permute to [B, D, L] to apply FFT on L easily if using default, 
        # or use dim parameter.
        # Let's stick to the paper's logic: Global convolution over tokens.
        
        # Correct approach for [B, L, D]:
        # FFT along dim=1 (L)
        x_freq = torch.fft.rfft(x, dim=1) # Shape: [B, L//2+1, D]
        
        # Reshape for block diagonal processing
        B, L_half, D = x_freq.shape
        H = self.num_heads
        head_dim = D // H
        
        x_freq = x_freq.view(B, L_half, H, head_dim) # [B, L_half, H, head_dim]
        x_freq = x_freq.permute(0, 2, 1, 3).contiguous() # [B, H, L_half, head_dim]
        
        # Apply MLP filters per head
        # MatMul: [B, H, L_half, head_dim] @ [H, head_dim, head_dim] -> [B, H, L_half, head_dim]
        # Note: Broadcasting W1 over batch and L_half dimensions
        
        out = torch.matmul(x_freq, self.W1.unsqueeze(0).unsqueeze(2)) + self.b1.unsqueeze(0).unsqueeze(2)
        out = self.act(out)
        out = torch.matmul(out, self.W2.unsqueeze(0).unsqueeze(2)) + self.b2.unsqueeze(0).unsqueeze(2)
        
        # Restore shape
        out = out.permute(0, 2, 1, 3).contiguous() # [B, L_half, H, head_dim]
        out = out.view(B, L_half, D) # [B, L_half, D]
        
        # Passing Mode
        if self.mode == 'LP':
            # Soft shrinkage example (threshold lambda needs definition, e.g., 0.02)
            # Simple implementation placeholder
            out = self.soft_shrink(out, threshold=0.02)
        elif self.mode == 'HP':
            out = out - self.soft_shrink(out, threshold=0.02)
        else: # AP
            pass
            
        # 2. IFFT
        x_out = torch.fft.irfft(out, n=x.size(1), dim=1) # Recover original L
        
        return x_out + residual

    def soft_shrink(self, x, threshold=0.02):
        # Placeholder for soft shrinkage operation described in paper
        # Typically: sign(x) * max(|x| - threshold, 0)
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)

class FourierMILBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.token_mixing = APFFBlock(dim, num_heads=num_heads)
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x):
        x = self.token_mixing(x)
        x = self.channel_mixing(x)
        return x
```

#### 6. 实现提示
- **关键网络组件**：`torch.fft.rfft` 和 `torch.fft.irfft` 用于频域变换。
- **重要超参数**：
    - Hidden Dimension ($D$): 512 (由1024压缩而来)。
    - Number of APFF Blocks: 2。
    - Learning Rate: $2 \times 10^{-4}$。
    - Weight Decay: $1 \times 10^{-5}$。
    - Optimizer: Lookahead。
    - Filter Mode: 'AP' (All-Pass) 为默认最佳。
- **归一化/激活方式**：Layer Normalization (LN) 用于Token Mixing前和Channel Mixing前；GELU 作为非线性激活函数。
- **维度对齐方式**：频域变换后，实部虚部合并，长度变为 $L//2 + 1$。逆变换时需指定 $n=L$ 以恢复原序列长度。
- **实现注意事项**：
    - 权重 $W_1, W_2$ 需初始化为随机值或零均值分布。
    - 块对角结构通过 `view` 和 `permute` 配合独立的 Head 权重矩阵实现，无需显式构造稀疏大矩阵，节省内存。
    - ATP（自适应令牌填充）需在进入APFF之前执行。

#### 7. 计算与资源开销
- **理论计算复杂度**：FFT/IFFT 为 $O(L \log L)$。MLP滤波部分为 $O(L \cdot D^2 / h)$ 或类似线性复杂度（取决于Head划分）。总体远低于Transformer的 $O(L^2)$。
- **参数量**：相比Transformer显著减少，因为避免了 $L \times L$ 的注意力矩阵。具体参数量文中未列出总数，但强调比AFNO少（通过块对角结构）。
- **FLOPs/MACs**：文中未提供具体FLOPs数值，但指出效率高于Transformer和GCN。
- **显存开销**：Table 6显示，使用ATP时显存约为 0.324 GB (CM16数据集)，低于固定比例填充（如10% TP为0.351 GB），表明其内存效率高。
- **推理速度**：文中未提供具体FPS，但强调“efficiently handles thousands of patches”。
- **论文是否提供效率对比**：提供了显存对比（Table 6）和定性复杂度分析（$O(n \log n)$ vs $O(n)$ Nyströmformer）。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI分类（二进制和多类），特别是小病灶检测（CAMELYON16）和大区域分类（TCGA/CPTAC）以及IHC染色数据（AD）。
- **可迁移到的任务/数据集**：任何基于Patch Embedding的MIL任务，尤其是序列长度较长、需要全局上下文感知的视觉任务（如长视频分类、时间序列分析）。
- **迁移所需调整**：调整Feature Extractor（如ResNet/CTransPath）以适应新数据的预处理；调整MLP隐藏层维度。
- **适用条件**：输入可分割为独立Patch；希望降低计算复杂度同时保持全局感受野。
- **潜在限制**：FFT假设周期性，虽通过ATP缓解，但在边界处仍可能有轻微伪影；对于极短序列，FFT优势不明显。

#### 9. 实验与消融证据
- **主要性能结果**：
    - CAMELYON16 (ResNet50): Acc 0.898, AUC 0.927 (SOTA)。
    - TCGA (ResNet50): Acc 0.893, AUC 0.954 (SOTA)。
    - UNITE (AD, ResNet50): Acc 0.878, AUC 0.955 (SOTA)。
    - CTransPath提取器下同样取得SOTA或接近SOTA成绩。
- **相对基线的提升**：相比ABMIL, CLAM, TransMIL等在Accuracy和AUC上均有显著提升（例如CM16上Acc提升至少1.9%）。
- **相关消融实验**：
    - **w/o APFF**: 性能崩溃（全部预测为负），证明频域混合必要性。
    - **w/o CM**: 性能下降，证明通道混合也重要。
    - **Filter Type**: AP > LP ≈ HP (在大样本UNITE中差异小，但在CM16中小病灶检测中AP最优)。
    - **Padding**: ATP 优于无填充和固定比例填充，且更省内存。
- **作者结论**：APFF模块能有效捕捉全局依赖和细粒度细节，ATP有效缓解频谱泄漏，FourierMIL具有强鲁棒性和泛化性。
- **证据是否充分**：在多个公开和私有数据集、不同染色类型、不同特征提取器下均进行了验证，消融实验全面，证据充分。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 将全通频域滤波与MLP结合用于MIL令牌混合，解决病理图像高频细节保留问题，区别于传统低通滤波。 |
| 技术可行性 | 高 | 基于标准FFT和MLP操作，易于实现，计算稳定。 |
| 实现难度 | 中 | 需注意频域维度的重塑、块对角权重的广播机制以及ATP的具体实现。 |
| 架构相关性 | 高 | 专为WSI这种长序列、高维Patch Embedding场景设计，替代Attention。 |
| 可迁移性 | 高 | 适用于任何基于Token的序列建模任务，不仅限于病理。 |
| 计算成本 | 低 | $O(N \log N)$ 复杂度，显存占用低，适合大规模WSI。 |

#### 11. 一句话总结
FourierMIL通过引入自适应令牌填充和全通频域滤波（APFF）模块，在频率域实现了高效且保留高频细节的令牌混合，解决了WSI分析中Transformer计算成本高和传统方法丢失细粒度特征的问题，在多类病理任务中达到SOTA性能。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **全通频域滤波（APFF）设计**：摒弃了传统频域方法（如AFNO）的低通倾向，明确论证了在视觉/病理任务中保留高频信息的重要性，并通过MLP实现自适应滤波。
- **自适应令牌填充（ATP）**：针对FFT的非周期性误差，提出了一种基于 $\lceil \sqrt{L} \rceil^2 - L$ 的动态填充策略，平衡了性能增益与内存开销。

### 2. 方法之间的关系
- **与MLP-Mixer的关系**：FourierMIL保留了MLP-Mixer的Channel Mixing结构，但用频域全局卷积替换了空间域的Token Mixing。
- **与Transformer的关系**：旨在替代Self-Attention，提供近似线性的全局感受野。
- **与AFNO的关系**：AFNO主要用于2D图像且常隐含低通特性；FourierMIL针对1D Patch序列，显式支持全通模式，更适合MIL场景。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，包含了算法伪代码（Algorithm 1 & 2）、数学公式和详细的超参数设置。
- **关键配置是否明确**：是，明确了LR、Batch Size、Optimizer、Hidden Dim等。
- **预计复现难点**：
    - 确保FFT/IFFT的维度处理正确（特别是Real FFT的共轭对称性处理）。
    - 块对角权重 $W_1, W_2$ 的正确初始化与广播乘法实现。
    - ATP的具体填充内容（零填充还是复制？文中提到“zero-padding or duplicated tokens”，需确认默认行为，通常零填充较多见，但需看代码）。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：APFF模块可作为通用Token Mixer集成到ViT或MIL框架中，以加速训练并降低显存。
- **需要改造的设计**：若应用于纯2D图像而非Patch序列，需调整FFT的轴（沿H/W而非Sequence）。
- **可能形成的新研究思路**：探索其他频域变换（如Wavelet Transform）在MIL中的应用；结合ATP思想处理其他周期性假设敏感的变换。

### 5. 阅读备注
- 论文强调了IHC染色数据的适用性，这是许多Foundation Models（主要基于H&E）所欠缺的，体现了方法的通用性。
- 消融实验中，“w/o FT”模型表现极差，强烈暗示频域变换本身对于这种特定的全局混合机制是关键，而不仅仅是MLP的作用。
