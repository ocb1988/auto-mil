# MSM_MIL 方法总结

> 证据说明：输入为完整论文全文（13页），包含摘要、引言、相关工作、方法、实验及附录。PDF提取文本完整，关键公式和图表描述清晰，无缺失。

## 一、论文基本信息

- **论文标题**：MSMMIL: Multi-scan Mamba-based Multiple Instance Learning for whole slide image classification
- **作者**：Haiqin Zhong, Meidan Ding, Cheng Zhao, Yongtao Zhang, Tianfu Wang, Baiying Lei
- **发表年份**：2025
- **会议/期刊**：Knowledge-Based Systems (Elsevier)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1016/j.knosys.2025.113871
- **代码仓库**：https://github.com/cool-breeze-and-rain/MSMMIL
- **研究任务**：全切片图像（WSI）分类（计算病理学中的多实例学习）
- **数据模态**：数字病理图像（WSI Patches）

## 二、论文整体概述

### 1. 核心问题
在基于多实例学习（MIL）的全切片图像（WSI）分类中，现有方法面临两大挑战：
1. 传统注意力机制假设实例间独立，忽略了临床诊断中上下文信息的重要性。
2. Transformer类模型虽然能建模长程依赖，但自注意力的二次方复杂度导致计算成本过高，难以处理包含数千个Patches的长序列。
3. 现有的Mamba类方法通常仅使用1-2个扫描方向，无法充分提取长序列中的判别性特征，尤其是当有效信息稀疏时；而增加扫描方向（如SS2D的四向扫描）会显著增加序列长度和计算负担。

### 2. 整体方法
提出 **MSMMIL** (Multi-scan Mamba-based Multiple Instance Learning) 框架。
1. **特征提取**：使用预训练ResNet50提取Patch特征，并通过MLP降维。
2. **特征聚合**：核心模块为 **Multi-scan Mamba (MSM)**。该模块并行处理三种扫描策略生成的序列：
   - **Original Scan (OS)**：保持原始顺序，保留全局结构。
   - **Grid Scan (GS)**：新颖策略，将序列划分为四个子序列进行四向扫描，再合并，旨在以固定序列长度实现四向建模。
   - **Layer Scan (LS)**：新颖策略，通过交错遍历缓解网格划分带来的特征不连续性。
   每个分支后接 **Global Context Attention (GCA)** 块，用于增强实例间关联并突出关键实例。
3. **Bag预测**：使用与ABMIL相同的注意力机制聚合MSM输出，最后通过线性层分类。

### 3. 主要贡献
1. 提出两种互补的新颖Mamba扫描策略：**Grid Scan** 和 **Layer Scan**。相比SS2D，它们在单序列内实现多向扫描，减少内存消耗并缓解重排序导致的特征不连续。
2. 设计 **MSM模块**，并行学习三种扫描模式的序列，实现特征互补，挖掘判别性特征。
3. 设计轻量级 **GCA块**，强化实例间关系，突出关键实例，提升特征表示能力。
4. 在Camelyon16和TCGA-Lung数据集上达到SOTA性能，且在参数量和显存占用上优于Transformer和部分Mamba基线。

## 三、方法总结

### 方法 1：Multi-scan Mamba (MSM) 模块

#### 1. 核心思想与解决的问题
- **目标问题**：解决单一或双向扫描无法充分捕捉WSI长序列中空间结构和全局上下文的问题，同时避免多向扫描导致的序列长度膨胀和计算量激增。
- **现有方法的局限**：SS2D等四向扫描方法使序列长度变为4倍，计算成本高；单向/双向扫描遗漏空间交互信息。
- **核心思想**：通过三种并行的扫描分支（Original, Grid, Layer）从不同视角重塑序列，利用Mamba的高效线性复杂度建模长程依赖，并通过残差连接融合多视角特征。
- **创新点**：
    - **Grid Scan**：在一个序列内部模拟四向扫描，不增加总序列长度。
    - **Layer Scan**：通过交错遍历平滑Grid Scan产生的边界断裂。
    - **Original Scan**：作为基准，保留原始空间一致性。

#### 2. 详细结构与数据流
- **输入**：实例特征序列 $X \in \mathbb{R}^{L \times D}$，其中 $L$ 为实例数量，$D$ 为特征维度（文中设为256）。
- **处理流程**：
    1. **重塑**：将 $X$ 重塑为二维形式 $X_{2d} \in \mathbb{R}^{R \times T \times D}$，其中 $T$ 是子序列长度（超参数，设为10），$R = L/T$。若 $L$ 不能被 $T$ 整除，进行零填充。
    2. **分支1 (Original)**：保持 $X$ 的顺序不变，直接输入Mamba Block。
    3. **分支2 (Grid)**：将 $X_{2d}$ 分割为四个方向的子序列，重新排序生成新序列，输入Mamba Block，输出后还原顺序。
    4. **分支3 (Layer)**：采用Layer Scan策略重塑序列，输入Mamba Block，输出后还原顺序。
    5. **GCA处理**：每个分支的Mamba输出经过GCA块处理。
    6. **融合**：三个分支的输出相加，通过线性层投影，并与原始输入 $X$ 进行残差连接。
- **输出**：聚合后的实例特征 $Y^i \in \mathbb{R}^{L \times D}$。
- **模块在整体网络中的位置**：位于特征提取之后，Bag-level聚合之前。由 $H=2$ 个这样的MSM模块堆叠而成。
- **与其他模块的连接方式**：输入来自ResNet50+MLP的特征；输出进入ABMIL风格的Attention Aggregator。

#### 3. 数学公式

**Mamba Block 输出 (Eq. 7):**
$$ X'_{it1} = SSM(\text{SiLU}(\text{Conv1D}(\text{Linear}(X_{it1})))) $$
$$ Y'_{it1} = X'_{it1} \cdot \text{SiLU}(\text{Linear}(X_{it1})) $$
*符号定义*：
- $X_{it1}$: 第 $t$ 个分支在第1阶段的输入序列。
- $\text{Linear}, \text{Conv1D}, \text{SiLU}$: 线性层、因果卷积、SiLU激活函数。
- $SSM$: Mamba的状态空间模型块。
- $Y'_{it1}$: Mamba模块的输出。

**GCA Block 输出 (Eq. 8):**
$$ Y'_{it2} = \text{Linear}(Y'_{it1} \cdot \text{MLP}(\text{AVG}(\text{LN}(Y'_{it1})))) $$
*符号定义*：
- $\text{LN}$: Layer Normalization。
- $\text{AVG}$: 平均操作（计算均值作为权重基础）。
- $\text{MLP}$: 多层感知机，生成重要性权重。
- $Y'_{it2}$: GCA模块的输出。

**MSM Module 最终输出 (Eq. 9):**
$$ Y^i = \text{Linear}\left(\sum_{t=1}^{3} Y'_{it2}\right) + X^i $$
*符号定义*：
- $Y^i$: MSM模块的最终输出。
- $X^i$: 原始输入特征（残差连接）。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | $X$ | $\mathbb{R}^{L \times 256}$ | $L$ 个Patch的特征，维度256 |
| 重塑后 | $X_{2d}$ | $\mathbb{R}^{R \times T \times 256}$ | $R=L/T$, $T=10$ |
| Branch 1 Input | $X_{i1}$ | $\mathbb{R}^{L \times 256}$ | Original Scan |
| Branch 2 Input | $X_{i2}$ | $\mathbb{R}^{L \times 256}$ | Grid Scan (重组后) |
| Branch 3 Input | $X_{i3}$ | $\mathbb{R}^{L \times 256}$ | Layer Scan (重组后) |
| MSM Output | $Y^i$ | $\mathbb{R}^{L \times 256}$ | 融合后的特征 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn

class MSMBuildBlock(nn.Module):
    def __init__(self, dim, t_size=10):
        super().__init__()
        self.t_size = t_size
        # 假设已有 MambaBlock 和 GCABlock 的实现
        self.mamba_branches = nn.ModuleList([
            MambaBlock(dim), 
            MambaBlock(dim), 
            MambaBlock(dim)
        ])
        self.gca_blocks = nn.ModuleList([
            GCABlock(dim), 
            GCABlock(dim), 
            GCABlock(dim)
        ])
        self.proj = nn.Linear(dim, dim)
        self.residual_proj = nn.Linear(dim, dim) # 用于残差连接的维度对齐，若dim相同可省略

    def grid_scan(self, x_2d):
        # x_2d: [R, T, D]
        # 模拟四向扫描并重组为一个长度为 R*T 的序列
        # 具体逻辑需根据论文Fig 1实现：Split into 4 directions, reorder
        pass 

    def layer_scan(self, x_2d):
        # 交错遍历，生成两个方向的子序列并重组
        pass

    def forward(self, x):
        # x: [L, D]
        L, D = x.shape
        R = L // self.t_size
        
        # Reshape to [R, T, D], pad if necessary
        x_2d = x.view(R, self.t_size, D)
        
        outputs = []
        
        # Branch 1: Original
        out_mamba_1 = self.mamba_branches[0](x) # 直接对1D序列操作或reshape后操作
        out_gca_1 = self.gca_blocks[0](out_mamba_1)
        outputs.append(out_gca_1)
        
        # Branch 2: Grid Scan
        # 1. Reorder based on Grid strategy
        x_grid_reordered = self.grid_scan(x_2d) # Returns [L, D]
        out_mamba_2 = self.mamba_branches[1](x_grid_reordered)
        # 2. Reorder back to original spatial order (inverse of grid scan)
        out_mamba_2_original_order = inverse_grid_scan(out_mamba_2, R, self.t_size)
        out_gca_2 = self.gca_blocks[1](out_mamba_2_original_order)
        outputs.append(out_gca_2)
        
        # Branch 3: Layer Scan
        x_layer_reordered = self.layer_scan(x_2d)
        out_mamba_3 = self.mamba_branches[2](x_layer_reordered)
        out_mamba_3_original_order = inverse_layer_scan(out_mamba_3, R, self.t_size)
        out_gca_3 = self.gca_blocks[2](out_mamba_3_original_order)
        outputs.append(out_gca_3)
        
        # Sum and Project
        sum_features = torch.stack(outputs, dim=0).sum(dim=0)
        y = self.proj(sum_features) + self.residual_proj(x)
        return y

class GCABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 对应 AVG()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [L, D]
        normed = self.ln(x)
        # Compute mean across sequence dimension? 
        # Eq 8 says AVG(LN(Y)). If AVG is global average over instances:
        weights = self.avg_pool(normed.transpose(1, 2)).transpose(1, 2) # [L, 1, D] -> broadcast? 
        # Note: Paper says "mean is taken as the weight". Usually implies instance-wise or bag-wise.
        # Given Eq 8 structure Y * MLP(AVG(...)), it likely generates a scalar or vector per instance/bag.
        # Assuming element-wise multiplication with generated weights.
        w = self.mlp(weights) 
        weighted_x = x * w
        out = self.linear(weighted_x)
        return out
```
*注：伪代码中 `grid_scan` 和 `layer_scan` 的具体索引映射逻辑未在文本中给出详细算法步骤，需根据图1示意实现。`AVG` 操作的具体维度缩减方式（是对所有实例求平均还是每通道平均）需结合代码确认，此处假设生成与输入形状兼容的权重。*

#### 6. 实现提示
- **关键网络组件**：Mamba Block (需引入 `mamba_ssm` 库或复现Selective Scan)，GCABlock。
- **重要超参数**：
    - $T$ (子序列长度): 设为 10 (消融实验得出最优)。
    - $H$ (MSM模块数量): 设为 2。
    - 特征维度 $D$: 256 (由ResNet50的1024降维而来)。
- **归一化/激活方式**：Layer Norm (LN), SiLU, GELU (MLP中隐含), Linear。
- **维度对齐方式**：残差连接要求输入输出维度一致 ($L \times 256$)。
- **实现注意事项**：
    - **Padding**: 必须使用 **Zero Padding** (消融实验Table 6证明其最佳)。
    - **Scan Reordering**: Grid和Layer扫描涉及复杂的索引重排，输出后必须执行逆操作以恢复原始Patch的空间顺序，才能正确相加。
- **依赖的特殊算子或第三方库**：Mamba (State Space Model implementation)。

#### 7. 计算与资源开销
- **理论计算复杂度**：Mamba具有线性复杂度 $O(L)$。由于并行三个分支，常数因子约为3倍于单分支Mamba，但仍远低于Transformer的 $O(L^2)$。
- **参数量**：MSMMIL总参数量为 **1.910 M** (Table 1)。
- **FLOPs/MACs**：**33.156 G** (Table 1)。
- **显存开销**：**0.594 GB** (Table 1)。
- **推理速度**：未提供具体FPS，但强调比TransMIL和IAT更高效。
- **论文是否提供效率对比**：是，Table 1对比了Param, FLOPs, Memory。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI癌症分类（乳腺癌转移检测、肺癌亚型分类）。
- **可迁移到的任务/数据集**：其他基于Patch序列的医学图像分类任务（如细胞分类、组织分级），或任何长序列分类任务。
- **迁移所需调整**：调整特征提取器（Backbone）和最终分类头；可能需要调整 $T$ 值以适应不同的序列长度分布。
- **适用条件**：序列长度较长，且存在空间结构信息的任务。
- **潜在限制**：训练时间较慢（Mamba的不稳定性导致需多次运行取平均）；对噪声有一定敏感性（Camelyon16上噪声影响较大）。

#### 9. 实验与消融证据
- **主要性能结果**：
    - Camelyon16: ACC 92.87%, AUC 96.41% (SOTA)。
    - TCGA-Lung: ACC 89.81%, AUC 95.44% (SOTA)。
- **相对基线的提升**：优于Mamba2MIL (ACC +0.93%) 和 RRTMIL。
- **相关消融实验**：
    - **扫描策略**：OS+GS+LS 组合效果最好 (Table 2, Table 3)。
    - **GCA块**：加入GCA后ACC/F1/Spe显著提升，尽管部分情况AUC略降，但综合性能更好 (Table 4, Fig 8, Fig 9)。
    - **Padding**：Zero-padding 最佳 (Table 6)。
    - **Hyperparameter T**：T=10 最佳 (Table 7)。
- **作者结论**：多方向扫描能有效捕捉判别特征，尤其在有效信息稀疏时；GCA能抑制假阳性，聚焦关键区域。
- **证据是否充分**：是，提供了详细的消融、可视化（Heatmap, t-SNE）和统计检验（Violin plot）。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 提出了Grid/Layer两种新颖扫描策略，解决了Mamba在视觉应用中方向性受限的问题。 |
| 技术可行性 | 高 | 基于成熟的Mamba架构，模块设计清晰，易于集成。 |
| 实现难度 | 中 | 难点在于Scan策略的索引重排逻辑及逆操作的准确实现。 |
| 架构相关性 | 高 | 专为WSI长序列特性设计，紧密贴合Mamba的序列建模优势。 |
| 可迁移性 | 中 | 依赖于Patch级别的特征提取和序列假设，对其他非序列数据不适用。 |
| 计算成本 | 低 | 参数量和FLOPs均低于主流Transformer和部分Mamba变体。 |

#### 11. 一句话总结
MSMMIL通过引入Grid和Layer两种互补的多向扫描策略以及轻量级GCA块，在保持Mamba线性复杂度的前提下，显著提升了WSI长序列中判别性特征的提取能力和模型性能。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **Multi-scan Strategy (Grid & Layer)**：在不增加序列长度的情况下，通过内部重排模拟多方向扫描，平衡了感受野扩展与计算效率，这是改进Vision Mamba的一个有效思路。
- **Lightweight GCA**：用简单的均值+MLP替代复杂的Self-Attention来生成Instance权重，极大地降低了计算开销，适合大规模WSI分析。

### 2. 方法之间的关系
- **Feature Extraction** 与 **Aggregation** 解耦：使用离线ResNet50提取特征，MSM模块专注于序列间的依赖建模，最后复用ABMIL的聚合方式。这种模块化设计使得MSM可以作为一个通用的Sequence Encoder插入到其他MIL框架中。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：基本完整。公式给出了核心逻辑，但具体的 `Grid Scan` 和 `Layer Scan` 的索引映射算法（即如何将 $R \times T$ 矩阵展平并重组）仅通过图示展示，文字描述较为简略。复现时需仔细对照 Figure 1 推导索引。
- **关键配置是否明确**：明确（$T=10, H=2, D=256$, Zero Padding）。
- **预计复现难点**：
    1. Mamba Selective Scan 的具体实现细节（如硬件优化部分）。
    2. Scan策略的精确索引重建（Inverse Scanning）。
    3. Mamba训练的不稳定性（论文提到需训练5次取平均）。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：GCA块的权重生成机制；Zero-padding策略。
- **需要改造的设计**：Scan策略需要根据具体的序列长度和数据结构调整 $T$ 值和重组逻辑。
- **可能形成的新研究思路**：探索更多种互补的扫描模式（如螺旋扫描）；将GCA应用于其他状态空间模型中以增强其注意力机制；研究如何在保持线性复杂度的同时进一步降低Mamba的训练不稳定性。

### 5. 阅读备注
- 论文中提到的 "Original Scan" 实际上就是标准的从左到右（或按Patch ID顺序）的Mamba扫描，并未改变数据顺序，主要作用是作为Baseline和残差来源。
- 实验部分特别强调了在Camelyon16（癌症区域少）上的优势，暗示该方法在处理“信号稀疏”的长序列时尤为有效。
