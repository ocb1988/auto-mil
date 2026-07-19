# MO_MIL 方法总结

> 证据说明：输入为完整论文文本（10页），包含摘要、引言、方法、实验及结论。公式提取完整，无缺失页面。代码仓库链接已提供。

## 一、论文基本信息

- **论文标题**：MoMIL: Multi-order enhanced multiple instance learning for computational pathology
- **作者**：Yuqi Zhang, Xiaoqian Zhang, Jiakai Wang, Baoyu Liang, Yuancheng Yang, Chao Tong
- **发表年份**：2026 (Online 28 January 2026)
- **会议/期刊**：Image and Vision Computing (Elsevier)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1016/j.imavis.2026.105918
- **代码仓库**：https://github.com/YuqiZhang-Buaa/MoMIL
- **研究任务**：计算病理学中的全切片图像（WSI）分类与生存预测（弱监督学习）
- **数据模态**：数字病理图像（Whole Slide Images, WSIs）

## 二、论文整体概述

### 1. 核心问题
传统多实例学习（MIL）在处理WSI时面临两个主要挑战：
1. **结构固定性与灵活性不足**：基于CNN或Attention的框架难以适应不同大小的WSI，且缺乏对空间结构的显式建模。
2. **信息利用不完整**：现有的Mamba/SSM架构虽然能处理长序列，但往往忽略了序列顺序对特征利用的影响，或者缺乏能够充分利用多方向上下文信息的机制，导致在跨数据集和不同癌症类型上性能不一致。

### 2. 整体方法
提出 **MoMIL (Multi-order MIL)** 框架。核心思路包括：
1. **序列变换（Sequence Transformation）**：将不规则的一维特征序列通过“平方化”和“重排序”操作，生成原始顺序、翻转顺序和转置顺序三种结构化序列，以模拟WSI的多方向扫描。
2. **状态空间对偶模型（SSD）**：使用Mamba-2中的SSD模块对多序列表征进行长程依赖建模，捕捉全局上下文。
3. **轻量级特征融合（Lightweight Feature Fusion）**：通过可学习的注意力权重自适应地融合多序列表征，最终通过MLP输出预测结果。

### 3. 主要贡献
1. 提出了一种针对WSI的序列变换方法，增强了序列信息的利用率，同时保持对不同尺寸WSI的效率。
2. 设计了一个灵活可扩展的框架，可根据需求调整使用的序列阶数。
3. 在五个数据集上的广泛实验表明，MoMIL在癌症亚型分类和生存预测任务上均优于最先进的方法。

## 三、方法总结

### 方法 1：MoMIL 整体框架

#### 1. 核心思想与解决的问题
- **目标问题**：解决WSI中实例（Patch）数量巨大且不规则，以及传统MIL方法无法有效捕捉空间连续性和多方向上下文依赖的问题。
- **现有方法的局限**：Transformer计算复杂度高；传统Mamba仅单向扫描，存在历史衰减问题且忽略横向连续性；现有位置编码方法固定且计算昂贵。
- **核心思想**：通过人为构造多种扫描顺序（纵向、反向纵向、横向）来丰富WSI的空间表征，并利用SSD的高效线性复杂度特性进行建模，最后融合多视角特征。
- **创新点**：
    - 无需真实坐标即可通过数学变换近似多方向扫描。
    - SSD结合选择性状态更新机制，高效处理长序列。
    - 轻量级融合模块避免额外计算负担。

#### 2. 详细结构与数据流
- **输入**：经过背景去除后的WSI Patch序列 $S = \{x_1, x_2, ..., x_N\}$，其中 $x_i \in \mathbb{R}^D$ 是通过ResNet50提取的特征。
- **处理流程**：
    1. **特征降维与平方化**：线性层降维后，填充至正方形长度 $L = (\lceil\sqrt{N}\rceil)^2$。
    2. **序列重排序**：生成三个序列：
       - $S_o$: 原始顺序（纵向扫描）。
       - $S_f$: 翻转顺序（反向纵向扫描，缓解历史衰减）。
       - $S_t$: 转置顺序（横向扫描，捕捉水平连续性）。
    3. **SSD建模**：三个序列分别通过堆叠的SSD Block，得到特征 $F_o, F_f, F_t$。
    4. **特征融合**：拼接 $F_o, F_f, F_t$，通过Layer Norm、线性层+Sigmoid、Softmax计算权重 $\alpha$，加权求和得到 $h_{fused}$。
    5. **分类头**：MLP输出Bag-level预测 $\hat{Y}$。
- **输出**：Bag级别的预测标签 $\hat{Y}$（分类概率或生存风险）。
- **模块在整体网络中的位置**：位于特征提取器（ResNet50）之后，分类器之前。
- **与其他模块的连接方式**：接收降维后的Patch序列，输出融合后的全局表示给MLP。

#### 3. 数学公式

**序列平方化与填充：**
$$ S' = \text{Concat}(S_l, x_1, x_2, \dots, x_M) $$
$$ M = L - N, \quad L = (\lceil\sqrt{N}\rceil)^2 $$
其中 $S_l$ 是降维后的序列，$N$ 是原始Patch数量，$L$ 是填充后的总长度。前 $M$ 个token通过复制 $S_l$ 的前 $M$ 个元素实现。

**转置序列生成：**
$$ S_t = \text{Flatten}(\text{Transpose}(\text{Reshape}(S', \lceil\sqrt{N}\rceil, \lceil\sqrt{N}\rceil))) $$

**SSD分支输出：**
$$ F = \text{Linear}(\text{SSD}(\text{Conv1D}(\text{Linear}(S)))) $$
$$ F \in \{F_o, F_f, F_t\}, \quad S \in \{S_o, S_f, S_t\} $$
注：此处公式(5)暗示每个分支内部可能包含一个小的Conv1D预处理和一个Linear投影，具体细节需参考代码，但逻辑上是SSD处理序列。

**特征融合：**
1. 拼接：$H \in \mathbb{R}^{3L \times D'}$，其中 $D'$ 是降维后的特征维度。
2. 归一化：$\tilde{H} = \text{LN}(H)$
3. 分数计算：$a = \text{Linear}_2(\sigma(\text{Linear}_1(\tilde{H})))$, $a \in \mathbb{R}^{3L \times 1}$，$\sigma$ 为Sigmoid。
4. 权重归一化：$\alpha = \text{Softmax}(a^\top)$, $\alpha \in \mathbb{R}^{1 \times 3L}$
5. 加权聚合：$h_{fused} = \alpha \tilde{H}$, $h_{fused} \in \mathbb{R}^{D'}$
6. 预测：$\hat{Y} = \text{MLP}(h_{fused})$

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | $S$ | $(N, D)$ | $N$为Patch数，$D$为ResNet50输出维度(通常2048) |
| 降维后 | $S_l$ | $(N, D')$ | $D'$由Linear层决定，文中未明确具体数值，通常为较小值如256或512 |
| 平方化后 | $S'$ | $(L, D')$ | $L=(\lceil\sqrt{N}\rceil)^2$ |
| 重排序后 | $S_o, S_f, S_t$ | $(L, D')$ | 三个独立的序列 |
| SSD输出 | $F_o, F_f, F_t$ | $(L, D')$ | 经过SSD块处理后的特征 |
| 拼接后 | $H$ | $(3L, D')$ | 三个分支特征沿序列维度拼接 |
| 融合权重 | $\alpha$ | $(1, 3L)$ | 每个token的融合注意力权重 |
| 融合特征 | $h_{fused}$ | $(D')$ | 全局池化后的向量 |
| 输出 | $\hat{Y}$ | $(C)$ 或 $(1)$ | $C$为类别数，生存预测时为标量风险 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba # 假设使用mamba_ssm库实现SSD/Mamba-2

class MoMIL(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=2, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特征降维
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # SSD Blocks (Mamba-2 based)
        # 注意：实际SSD实现可能更复杂，这里简化为Mamba Block
        self.ssd_block = Mamba(d_model=hidden_dim, layer_idx=0) 
        # 为了并行处理三个分支，可以共享权重或独立初始化
        self.ssd_branches = nn.ModuleList([
            Mamba(d_model=hidden_dim, layer_idx=i) for i in range(n_layers)
        ])
        
        # 轻量级融合模块
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim, 1)
        
        # 分类头
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def square_and_pad(self, x):
        """
        x: (N, D)
        """
        N = x.size(0)
        L = int(torch.ceil(torch.sqrt(torch.tensor(N))).item()) ** 2
        pad_len = L - N
        
        if pad_len > 0:
            # 复制前pad_len个token进行填充
            padding_tokens = x[:pad_len].repeat(pad_len // x.size(0) + 1, 1)[:pad_len]
            x_padded = torch.cat([x, padding_tokens], dim=0)
        else:
            x_padded = x
            
        return x_padded, L

    def reorder_sequences(self, x_sq, L):
        """
        x_sq: (L, D)
        Returns: So, Sf, St
        """
        side = int(L ** 0.5)
        
        # So: Original order (already flattened from row-major scan usually)
        So = x_sq 
        
        # Sf: Flipped sequence
        Sf = torch.flip(x_sq, dims=[0])
        
        # St: Transposed sequence
        # Reshape to (side, side), Transpose, Flatten
        x_matrix = x_sq.view(side, side, -1)
        x_transposed = x_matrix.transpose(0, 1).contiguous()
        St = x_transposed.view(-1, x_sq.size(1))
        
        return So, Sf, St

    def forward(self, x):
        # x: (N, D)
        
        # 1. Projection & Squaring
        x_proj = self.projection(x) # (N, D')
        x_sq, L = self.square_and_pad(x_proj) # (L, D')
        
        # 2. Reordering
        So, Sf, St = self.reorder_sequences(x_sq, L)
        
        # 3. SSD Modeling (Parallel branches)
        # 假设SSD块支持批量处理，这里分别处理三个序列
        Fo = self.process_with_ssd(So)
        Ff = self.process_with_ssd(Sf)
        Ft = self.process_with_ssd(St)
        
        # 4. Fusion
        H = torch.cat([Fo, Ff, Ft], dim=0) # (3L, D')
        H_norm = self.norm(H) # (3L, D')
        
        # Calculate attention scores
        a = self.linear2(self.sigmoid(self.linear1(H_norm))) # (3L, 1)
        alpha = torch.softmax(a.squeeze(-1), dim=0) # (3L,)
        
        # Weighted sum
        h_fused = torch.sum(alpha.unsqueeze(-1) * H_norm, dim=0) # (D')
        
        # 5. Classification
        out = self.mlp(h_fused)
        return out

    def process_with_ssd(self, seq):
        # 简化：直接过SSD层，实际可能有Conv1D预处理
        # seq: (L, D')
        out = seq
        for block in self.ssd_branches:
            out = block(out) # 假设block返回相同形状
        return out
```

#### 6. 实现提示
- **关键网络组件**：Mamba-2 (SSD) 模块是核心。需要确保使用支持SSD格式的Mamba实现（如 `mamba_ssm` 库中的特定配置）。
- **重要超参数**：
    - `hidden_dim` ($D'$): 降维后的特征维度，影响计算量和表达能力。
    - `n_layers`: SSD块的堆叠层数。
    - 填充策略：必须严格遵循“复制前M个token”的规则，而非零填充。
- **归一化/激活方式**：融合模块中使用 LayerNorm，中间激活使用 Sigmoid，权重归一化使用 Softmax。消融实验显示 Sigmoid 优于 ReLU/Tanh。
- **维度对齐方式**：所有分支输出维度必须一致，拼接后沿序列维度（dim=0）操作。
- **实现注意事项**：WSI的Patch数量 $N$ 变化很大，动态计算 $L$ 和填充是必须的。转置操作需注意内存布局（Contiguous）。

#### 7. 计算与资源开销
- **理论计算复杂度**：SSD具有线性复杂度 $O(L)$，相比Transformer的 $O(L^2)$ 有显著优势。但由于引入了3个分支，计算量约为单分支的3倍。
- **参数量**：主要取决于SSD层的深度和宽度，以及融合模块的小型MLP。相比大型Transformer，参数量较小。
- **FLOPs/MACs**：未提供具体数值，但得益于SSD的线性特性和低秩投影，应低于TransMIL等基于Attention的方法。
- **显存开销**：由于需要存储3个序列的状态和中间激活，显存占用比单分支Mamba高，但远低于同等长度的Transformer。
- **推理速度**：SSD支持并行训练和快速推理（非自回归），速度较快。
- **论文是否提供效率对比**：未在表格中直接列出FLOPs，但在讨论中提到Mamba架构解决了长序列的计算瓶颈。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：计算病理学WSI分析（癌症分类、亚型分类、生存预测）。
- **可迁移到的任务/数据集**：任何具有长序列输入且隐含空间结构的数据，如遥感图像分割、时间序列分析（若时间具有周期性或双向依赖性）、文档理解。
- **迁移所需调整**：可能需要调整“平方化”和“转置”的逻辑以适应不同的数据结构（例如，对于纯时间序列，转置可能无意义，需改为其他扰动方式）。
- **适用条件**：输入序列长度较大，且数据中存在潜在的空间或结构相关性。
- **潜在限制**：多分支结构增加了约3倍的计算和显存开销；对于极短序列，平方化带来的填充比例过高，可能引入噪声。

#### 9. 实验与消融证据
- **主要性能结果**：
    - BRACS: AUC 0.8085 (+0.027 vs SOTA)。
    - TCGA-NSCLC: AUC 0.9461 (+0.0136 vs SOTA)。
    - CAMELYON-16: AUC 0.8196 (+0.0273 vs SOTA)。
    - LUAD/LUSC Survival: C-index 0.6615 / 0.6347。
- **相对基线的提升**：在所有指标上均优于CLAM, TransMIL, MambaMIL, LongMIL等。
- **相关消融实验**：
    - 单序列对比：转置序列($S_t$)表现最好，证明横向连续性重要。
    - 融合策略：移除融合模块导致AUC大幅下降至0.9218，证明融合必要性。
    - 激活函数：Sigmoid最佳。
- **作者结论**：多序列表征和融合机制共同贡献了性能提升。
- **证据是否充分**：在5个数据集、3类任务上进行了全面验证，消融实验覆盖了核心组件，证据充分。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 将序列重排序与SSM结合，提出多方向扫描近似，新颖性强。 |
| 技术可行性 | 高 | 基于成熟的Mamba-2架构，模块设计简单，易于实现。 |
| 实现难度 | 中 | 需正确处理动态序列长度、填充及转置逻辑，调试SSD配置。 |
| 架构相关性 | 高 | 专为WSI的大规模、不规则实例特性设计。 |
| 可迁移性 | 中 | 序列变换逻辑依赖于数据的“网格状”或“有序”假设，通用性受限。 |
| 计算成本 | 中 | 3倍分支带来额外开销，但SSD保证了线性扩展性。 |

#### 11. 一句话总结
MoMIL通过构建原始、翻转和转置三种序列顺序，利用SSD模型高效捕捉WSI的多方向空间上下文，并通过轻量级融合模块实现高性能的病理图像分析。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **序列变换策略**：不依赖昂贵的绝对位置编码，而是通过简单的矩阵重塑和转置来隐式编码空间关系，既高效又有效。
- **翻转序列缓解历史衰减**：明确指出了单向SSM的历史衰减问题，并通过反向扫描互补信息，这是一个简单而有效的技巧。

### 2. 方法之间的关系
- **基础**：建立在MIL范式之上，使用ResNet50作为Backbone。
- **核心引擎**：采用Mamba-2 (SSD) 替代传统的Attention或CNN进行序列建模。
- **增强模块**：引入多序列表征和自适应融合，是对单一SSM建模能力的增强。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，提供了详细的公式、流程图和超参数设置（如5-fold CV, 80/10/10 split）。
- **关键配置是否明确**：是，明确了Patch大小(512x512)、放大倍数(x20)、背景去除步骤。
- **预计复现难点**：
    1. **SSD/Mamba的具体配置**：Mamba-2的实现细节较多，需确保使用正确的Selective Scan机制。
    2. **数据预处理**：WSI的背景去除和Patch提取流程需要特定的病理图像处理库（如OpenSlide, PyVIPS）。
    3. **填充逻辑**：必须精确复现“复制前M个token”的填充方式，否则转置后的语义会错乱。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：SSD用于长序列建模；多分支并行处理不同视角的特征。
- **需要改造的设计**：如果应用于非图像数据（如文本），转置操作需重新定义；如果应用于实时性要求极高的场景，3倍分支的开销可能需要通过知识蒸馏或剪枝来优化。
- **可能形成的新研究思路**：探索更多样化的序列变换（如螺旋形扫描、分块打乱）；将这种多序列表征思想应用到其他视觉大模型（ViT）中，作为位置编码的替代或补充。

### 5. 阅读备注
- 论文发表于2026年，属于较新的工作，反映了State Space Models在医疗影像领域的最新应用趋势。
- 实验部分非常详尽，涵盖了分类和生存预测两种主流MIL任务。
- 局限性部分诚实指出了计算负担问题，未来工作方向明确。
