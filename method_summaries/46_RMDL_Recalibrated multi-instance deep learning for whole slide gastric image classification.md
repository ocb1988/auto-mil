# 46_RMDL_Recalibrated multi-instance deep learning for whole slide gastric image classification 方法总结

> 证据说明：输入为完整论文文本（12页），包含标题、摘要、引言、数据集描述、方法论、实验结果及结论。公式提取完整，无缺失页面或关键信息。

## 一、论文基本信息

- **论文标题**：RMDL: Recalibrated Multi-instance Deep Learning for Whole Slide Gastric Image Classification
- **作者**：Shujun Wang, Yaxi Zhu, Lequan Yu, Hao Chen, Huangjing Lin, Xiangbo Wan, Xinjuan Fan, Pheng-Ann Heng
- **发表年份**：2019 (Published in Medical Image Analysis) / arXiv:2010.06440v1 (2020)
- **会议/期刊**：Medical Image Analysis (MedIA)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1016/j.media.2019.101549；arXiv:2010.06440
- **代码仓库**：未提供
- **研究任务**：全切片胃组织病理图像（WSI）分类（正常、异型增生、癌症）
- **数据模态**：全切片数字病理图像（WSI），H&E染色

## 二、论文整体概述

### 1. 核心问题
针对全切片胃病理图像中异常区域占比小、类内差异大、类间差异小的挑战，解决如何从海量背景中有效选择具有判别力的实例（Patches），并克服传统多实例学习（MIL）忽略实例间依赖关系及正常区域特征抑制异常特征的问题。

### 2. 整体方法
提出一个两阶段框架：
1.  **判别性实例选择（Discriminative Instance Selection）**：使用基于Inception-ResNet v2改进的全卷积定位网络生成概率图，通过非极大值抑制（NMS）筛选出最具判别力的Top-K个Patch作为实例。
2.  **图像级预测（Image-level Prediction）**：设计重新校准的多实例深度学习网络（RMDL）。该网络包含局部-全局特征融合模块和实例重新校准模块，利用注意力机制动态调整每个实例的特征权重，最后通过平均池化进行聚合分类。

### 3. 主要贡献
1.  提出了包含判别性实例选择和RMDL的两阶段高效框架。
2.  设计了RMDL网络，通过结合局部和全局特征来计算实例的重要性系数，自动重新校准实例特征，捕捉实例间的相互依赖关系。
3.  构建了一个包含608张WSI的大规模胃病理数据集（WSGI），并在该数据集上验证了方法的有效性。

## 三、方法总结

### 方法 1：判别性实例选择与定位网络

#### 1. 核心思想与解决的问题
- **目标问题**：WSI尺寸巨大，直接处理计算成本高；且异常区域占比小，普通CNN容易受大量正常背景干扰。
- **现有方法的局限**：EM算法计算成本高；基于病理医生查看日志的方法主观性强且难泛化；传统分割网络耗时耗力。
- **核心思想**：不追求精确的像素级分割轮廓，而是训练一个轻量级的全卷积定位网络快速生成异常概率图，利用NMS筛选出高概率的Patch作为后续处理的输入实例。
- **创新点**：将预训练的Inception-ResNet v2转化为全卷积网络，采用“大块（Block）”推理策略加速概率图生成，并结合像素级标注进行Hard Negative Mining。

#### 2. 详细结构与数据流
- **输入**：WSI图像块（Block），尺寸 $I \times I$ ($1899 \times 1899$)。
- **处理流程**：
    1.  **定位网络架构**：基于Inception-ResNet v2，移除最后的Global Average Pooling和FC层。添加一个Kernel=8, Stride=1的平均池化层，随后是特征提取卷积层（FConv, 1024通道）和分类卷积层（CConv, 3通道输出Normal/Dysplasia/Cancer概率）。
    2.  **推理加速**：将WSI划分为重叠的大块（Block Size $1899 \times 1899$, Stride $1632$），输入定位网络得到小块概率图（$51 \times 51$），然后拼接成整张WSI的概率图。
    3.  **实例筛选**：在概率图上应用NMS（重叠阈值0.68），选取Top $m'$个最显著Patch。最终每个WSI选取 $m = 3m'$ 个实例。
    4.  **特征提取**：从定位网络的FConv层（无激活）提取实例特征向量 $U$。
- **输出**：一组判别性实例的特征集合 $\{u_1, u_2, ..., u_m\}$ 及其对应的原始Patch位置信息。
- **模块在整体网络中的位置**：第一阶段，位于RMDL网络之前。
- **与其他模块的连接方式**：输出的实例特征 $U$ 作为RMDL网络的输入。

#### 3. 数学公式
定位网络输出概率图 $P = f(X)$。
实例数量计算依赖于NMS过程，文中未给出显式解析公式，但逻辑为：
$$ m = 3 \times m' $$
其中 $m'$ 是通过NMS选出的单类最高分Patch数量。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 定位网络输入 | Block Input | $1899 \times 1899 \times 3$ | RGB图像块 |
| 定位网络中间层 | FConv Output | $H_{loc} \times W_{loc} \times 1024$ | 特征图，具体尺寸取决于下采样步长(S=32) |
| 定位网络输出 | Probability Map | $O \times O \times 3$ | $O = \lceil(I-N)/S\rceil + 1$，此处为 $51 \times 51 \times 3$ |
| 实例特征 | Instance Feature $u_i$ | $1 \times 1024$ | 从FConv层对应位置展平或池化得到（文中暗示为向量） |

*注：文中提到从FConv层提取特征 $U$，通常对于Patch级特征，可能是对FConv输出进行Global Average Pooling或直接取中心点特征，文中表述为 "extract features U from the FConv layer... as the discriminative instance representation"，结合后续FC层输入，推测 $u_i$ 为固定维度的向量。*

#### 5. 实现伪代码

```python
class LocalizationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Base Inception-ResNet v2 backbone (pretrained on ImageNet)
        self.backbone = InceptionResNetV2(pretrained=True)
        # Remove final GAP and FC
        # Add custom layers
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fconv = nn.Conv2d(1536, 1024, kernel_size=1) # Assuming input dim to last block
        self.cconv = nn.Conv2d(1024, 3, kernel_size=1)
        
    def forward(self, x):
        # x shape: [Batch, 3, 1899, 1899]
        feats = self.backbone.stem(x)
        feats = self.backbone.block1(feats)
        # ... pass through inception-resnet blocks ...
        # Assume 'feats' is the output before original GAP
        feats = self.avg_pool(feats)
        feats = self.fconv(feats)
        prob_map = self.cconv(feats) # [B, 3, H_out, W_out]
        return prob_map

def select_discriminative_instances(wsi_prob_map, top_k_per_class=100, overlap_thresh=0.68):
    """
    wsi_prob_map: [H, W, 3] probability map for Normal, Dysplasia, Cancer
    Returns: List of patch indices or coordinates
    """
    selected_patches = []
    for class_idx in range(3):
        # Extract probability map for specific class
        p_map = wsi_prob_map[:, :, class_idx]
        # Apply NMS logic
        patches = nms(p_map, threshold=overlap_thresh, k=top_k_per_class)
        selected_patches.extend(patches)
    return selected_patches
```

#### 6. 实现提示
- **关键网络组件**：Inception-ResNet v2 Backbone, Conv2d (1x1), AvgPool2d。
- **重要超参数**：Block Size $1899 \times 1899$, Block Stride $1632$, Patch Size $299 \times 299$ (训练时裁剪), NMS Overlap Threshold $0.68$, Top-$k$ per class $100$ (即总共300个实例)。
- **归一化/激活方式**：定位网络内部未详细说明BN/LN，但RMDL中使用Instance Norm。激活函数主要为Leaky ReLU (alpha=0.2) 和 Softmax。
- **维度对齐方式**：定位网络输出概率图后，通过坐标映射找到对应Patch在原始WSI上的位置，再提取特征。
- **实现注意事项**：定位网络训练时需平衡正负样本比例（手动控制batch中各类Patch数量相等）。

#### 7. 计算与资源开销
- **理论计算复杂度**：定位网络基于Inception-ResNet v2，参数量较大。
- **参数量**：未明确给出，但基于Inception-ResNet v2。
- **FLOPs/MACs**：未提供。
- **显存开销**：训练时使用4块NVIDIA TITAN Xp GPU，Batch Size 80。
- **推理速度**：平均WSI总耗时约93.79秒。其中实例选择（定位网络推理）占绝大部分时间（~93.78s），RMDL预测仅占0.01s。
- **论文是否提供效率对比**：提供了不同大小WSI的处理时间对比（Table 5）。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：胃癌WSI分类（三分类：Normal, Dysplasia, Cancer）。
- **可迁移到的任务/数据集**：其他癌症类型的WSI分类（如乳腺癌、前列腺癌），特别是病灶区域较小、背景复杂的场景。
- **迁移所需调整**：需重新训练定位网络以适应新数据的病理特征；调整NMS阈值和Top-k数量。
- **适用条件**：拥有像素级或至少Patch级标注数据以训练定位网络。
- **潜在限制**：两阶段分离训练可能导致特征不一致；计算速度慢，尤其是定位阶段。

#### 9. 实验与消融证据
- **主要性能结果**：Accuracy 86.5%, Average Score 0.923。优于Attention-MIP (82.0%) 和 MAXMIN-Layer (79.5%)。
- **相对基线的提升**：相比Attention-MIP提升4.5% Accuracy。
- **相关消融实验**：
    - 去除Local-Global Fusion (LG)：Accuracy下降至84.0% (-2.5%)。
    - 去除Instance Recalibration (IR)：Accuracy下降至82.0% (-4.5%)。
    - 两者结合达到最佳86.5%。
- **作者结论**：两个模块均对性能有显著提升，证明了局部-全局融合和实例重新校准的有效性。
- **证据是否充分**：在自建数据集上进行了充分的对比和消融，但在公开数据集上未验证泛化性（讨论部分提及未来工作）。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 中 | 结合了现有的Attention MIL和特征融合思想，重点在于针对WSI小病灶特性的工程化设计（定位+重校准）。 |
| 技术可行性 | 高 | 基于成熟的CNN架构，模块清晰，易于复现。 |
| 实现难度 | 中 | 需要处理WSI的大文件读取、分块推理及NMS逻辑，数据预处理较繁琐。 |
| 架构相关性 | 高 | 专为WSI分析设计，解决了尺度问题和实例间依赖问题。 |
| 可迁移性 | 中 | 依赖像素级标注训练定位网络，限制了在无标注数据上的直接迁移。 |
| 计算成本 | 低 | 推理速度较慢（分钟级），不适合实时临床辅助。 |

#### 11. 一句话总结
RMDL通过两阶段框架，先利用全卷积定位网络筛选高判别力Patch，再通过融合局部全局特征的注意力机制重新校准实例权重，实现了高精度的胃病理WSI分类。

### 方法 2：RMDL网络（局部-全局特征融合与实例重新校准）

#### 1. 核心思想与解决的问题
- **目标问题**：传统MIL方法（如Attention-MIP）将实例视为独立同分布，忽略了实例之间的上下文依赖关系；且单个Patch的局部特征可能不足以代表整个WSI的全局语义。
- **现有方法的局限**：SENet等通道注意力机制作用于特征图内部，而本文需要在实例级别（Instance-level）进行加权；单纯的最大池化或平均池化无法自适应地突出关键实例。
- **核心思想**：
    1.  **局部-全局融合**：不仅使用当前实例的特征，还引入由所有实例聚合而成的全局特征，使每个实例都能感知整体上下文。
    2.  **实例重新校准**：基于融合后的特征计算每个实例的重要性系数（Attention Weight），对原始实例特征进行逐元素乘法加权，从而抑制噪声实例，增强关键实例的贡献。
- **创新点**：在实例级别应用类似SENet的注意力机制，但输入是“局部+全局”融合特征，而非单纯的局部特征；通过这种方式隐式地建模实例间的相互依赖性。

#### 2. 详细结构与数据流
- **输入**：$m$ 个判别性实例的特征向量 $\{u_1, u_2, ..., u_m\}$，每个维度为 $L$（例如1024）。
- **处理流程**：
    1.  **局部特征提取**：每个实例 $u_i$ 经过三个连续的FC层（$fc1, fc2, fc3$）。每层后接Instance Norm, Leaky ReLU(0.2), Dropout(0.5)。
    2.  **全局特征生成**：对 $fc1$ 和 $fc2$ 的输出分别进行Max Pooling（沿实例维度），得到两个子全局特征，Concat后形成最终全局特征 $G$。
    3.  **局部-全局融合**：将全局特征 $G$ Tile（复制）到每个实例的 $fc3$ 输出之后，拼接得到组合特征 $H_i = [fc3(u_i); G]$。
    4.  **重要性系数计算**：对 $H_i$ 进行线性变换（FC层，权重 $W$, 偏置 $b$），并通过Softmax归一化得到系数 $\alpha_i$。
    5.  **特征重新校准**：$\hat{u}_i = \alpha_i \cdot u_i$ （注意：原文公式(2)写的是 $\hat{u}_i = \alpha_i \cdot u_i$，但前文提到是对combined feature $H$ 计算系数，这里可能存在表述歧义，通常Attention是将权重乘回原始特征或融合特征。根据公式(2) $\hat{u}_i = \alpha_i \cdot u_i$，权重是直接乘在原实例特征 $u_i$ 上，或者原文意指乘在某个特定表示上。细读原文："The recalibrated instance feature $\hat{U}$ is the element-wise multiplication of the coefficients and the original instance features $U$." 确认是乘在原特征 $U$ 上）。
    6.  **多实例池化**：对所有 $\hat{u}_i$ 进行Average Pooling，得到全局图像级特征 $z$。
    7.  **分类**：$z$ 经过FC层和Softmax得到最终类别概率。
- **输出**：WSI的分类标签（Normal/Dysplasia/Cancer）。
- **模块在整体网络中的位置**：第二阶段，接收定位网络输出的实例特征。
- **与其他模块的连接方式**：输入来自定位网络的FConv特征；输出连接至分类头。

#### 3. 数学公式
**重要性系数计算：**
$$ \alpha_i = \frac{\exp(W^T h_i + b)}{\sum_{j=1}^{m} \exp(W^T h_j + b)}, \quad i \in \{1, 2, ..., m\} \quad (1) $$
其中 $h_i$ 是第 $i$ 个实例的组合特征（局部+全局），$W \in \mathbb{R}^{L \times 1}, b \in \mathbb{R}$。

**重新校准特征：**
$$ \hat{u}_i = \alpha_i \cdot u_i \quad (2) $$
其中 $u_i$ 是原始实例特征，$\hat{u}_i$ 是校准后的特征。

**最终预测：**
$$ z = \text{AvgPool}(\{\hat{u}_1, ..., \hat{u}_m\}) $$
$$ Y = \text{Softmax}(W_{out} z + b_{out}) $$

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| RMDL输入 | Instance Features $U$ | $[m, L]$ | $m$个实例，每个维度$L$ (如1024) |
| FC层输出 | $fc1, fc2, fc3$ | $[m, D_{fc}]$ | 隐藏层维度，文中未明确指定D_fc，通常为降维或保持 |
| 全局特征 | $G$ | $[D_{global}]$ | 由fc1/fc2的MaxPool后Concat得到 |
| 融合特征 | $H_i$ | $[D_{fc} + D_{global}]$ | 拼接后的特征 |
| 重要性系数 | $\alpha_i$ | $[1]$ | 标量，Softmax输出 |
| 校准特征 | $\hat{u}_i$ | $[L]$ | 原始特征乘以系数 |
| 池化后特征 | $z$ | $[L]$ | 所有校准特征的平均 |
| 最终输出 | Class Probabilities | $[3]$ | 三类概率 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalGlobalFusion(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        # Three consecutive FC layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        
        # Norm and Activation
        self.norm1 = nn.InstanceNorm1d(hidden_dims[0])
        self.norm2 = nn.InstanceNorm1d(hidden_dims[1])
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.5)
        
        # Global feature dimension calculation
        # Max pool over instances reduces dim to hidden_dims[0] and hidden_dims[1]
        # Concatenated global dim = hidden_dims[0] + hidden_dims[1]
        self.global_dim = hidden_dims[0] + hidden_dims[1]
        
        # Final projection for attention if needed, but paper says FC layer for alpha
        # The combined feature H is [fc3(u_i); G]
        # Let's assume we project H to a scalar logit via a linear layer
        
    def forward(self, u):
        # u: [m, input_dim]
        m = u.size(0)
        
        # Local features
        h1 = self.dropout(self.relu(self.norm1(self.fc1(u))))
        h2 = self.dropout(self.relu(self.norm2(self.fc2(h1))))
        h3 = self.fc3(h2) # No activation mentioned for last fc before concat? 
                          # Text says "except the last one with softmax", likely referring to final classifier.
                          # For fusion module, it just produces features.
        
        # Global features
        # Max pooling over the instance dimension (dim=0)
        g1 = torch.max(h1, dim=0)[0] # [hidden_dims[0]]
        g2 = torch.max(h2, dim=0)[0] # [hidden_dims[1]]
        G = torch.cat([g1, g2], dim=0) # [global_dim]
        
        # Tile G for each instance
        G_tiled = G.unsqueeze(0).expand(m, -1) # [m, global_dim]
        
        # Combine local and global
        H = torch.cat([h3, G_tiled], dim=1) # [m, hidden_dims[2] + global_dim]
        
        return H, u # Return combined features and original features for recalibration

class InstanceRecalibration(nn.Module):
    def __init__(self, combined_dim):
        super().__init__()
        # Linear layer to squeeze to 1 dimension (scalar logit)
        self.attention_layer = nn.Linear(combined_dim, 1)
        
    def forward(self, H, u):
        # H: [m, combined_dim]
        logits = self.attention_layer(H).squeeze(-1) # [m]
        alphas = F.softmax(logits, dim=0) # [m]
        
        # Recalibrate: element-wise multiply alpha with original instance feature u
        # u: [m, L]
        # alphas needs to be expanded to match u's dimensions
        alphas_expanded = alphas.unsqueeze(1).expand_as(u) # [m, L]
        
        u_hat = alphas_expanded * u # [m, L]
        return u_hat

class RMDLNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.fusion = LocalGlobalFusion(input_dim, hidden_dims)
        combined_dim = hidden_dims[-1] + (hidden_dims[0] + hidden_dims[1])
        self.recalibration = InstanceRecalibration(combined_dim)
        
        # Multi-instance pooling (Average) and Classifier
        self.pool = nn.AdaptiveAvgPool1d(1) # Or simple mean over dim 0
        self.classifier = nn.Linear(input_dim, 3) # Output 3 classes
        
    def forward(self, u):
        # u: [m, input_dim]
        H, u_orig = self.fusion(u)
        u_calibrated = self.recalibration(H, u_orig)
        
        # Average Pooling over instances
        # u_calibrated: [m, L] -> Mean over dim 0 -> [L]
        z = torch.mean(u_calibrated, dim=0) 
        
        # Classification
        out = self.classifier(z)
        return out
```

#### 6. 实现提示
- **关键网络组件**：Linear Layers, InstanceNorm1d, LeakyReLU, Dropout, Softmax, AdaptiveAvgPool/MeanPool。
- **重要超参数**：FC层维度（文中未明确给出具体数值，需根据经验设定，如512, 256等），Dropout Rate 0.5, Leaky ReLU Alpha 0.2。
- **归一化/激活方式**：FC层后接Instance Norm和Leaky ReLU；注意力权重使用Softmax；最终分类器使用Softmax。
- **维度对齐方式**：全局特征 $G$ 通过Tile操作扩展到与实例数 $m$ 相同的维度，以便与每个实例的局部特征拼接。
- **实现注意事项**：
    - 确保 `torch.max` 在实例维度（通常是Dim 0）上进行。
    - Attention权重的广播机制要正确应用到原始特征 $u$ 上。
    - 训练RMDL时，实例的顺序不应影响结果（Permutation Invariant），因此使用Mean Pooling是合适的。

#### 7. 计算与资源开销
- **理论计算复杂度**：主要在于FC层的矩阵乘法。由于实例数 $m$ 较小（300），计算量远小于定位网络。
- **参数量**：较少，主要由几个FC层组成。
- **FLOPs/MACs**：极低，推理仅需0.01秒。
- **显存开销**：低，Batch Size 256时可轻松运行于单GPU。
- **推理速度**：极快（毫秒级）。
- **论文是否提供效率对比**：表5显示预测阶段耗时几乎可以忽略不计。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI图像级分类。
- **可迁移到的任务/数据集**：任何基于MIL框架的任务，如细胞计数、组织亚型分类、生存预测等。
- **迁移所需调整**：调整FC层维度以适应不同的特征嵌入大小；调整类别数。
- **适用条件**：已提取好实例特征。
- **潜在限制**：如果实例数 $m$ 非常大，全局特征的Max Pooling可能会丢失一些细节，但文中证明其有效性。

#### 9. 实验与消融证据
- **主要性能结果**：见上文方法1。
- **相对基线的提升**：消融实验中，加入IR模块带来2.5%提升，加入LG模块带来4.5%提升。
- **相关消融实验**：Table 3展示了MIP, IR, LG三种组合的效果。
- **作者结论**：局部-全局融合有助于捕捉实例间依赖，重新校准模块能更鲁棒地处理无关实例。
- **证据是否充分**：消融实验设计合理，结果支持论点。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 中 | 将SENet的思想迁移到实例级别，并结合全局上下文，思路清晰但非颠覆性创新。 |
| 技术可行性 | 高 | 标准深度学习组件堆叠，无复杂算子。 |
| 实现难度 | 低 | 代码结构简单，易于实现。 |
| 架构相关性 | 高 | 专门针对MIL痛点设计。 |
| 可迁移性 | 高 | 模块通用性强。 |
| 计算成本 | 低 | 计算开销极小。 |

#### 11. 一句话总结
RMDL通过构建局部与全局特征的融合表示，并利用Softmax注意力机制动态加权实例特征，有效提升了多实例学习中关键信息的提取能力。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **局部-全局特征融合用于MIL**：在计算实例注意力权重时，不仅仅依赖实例自身的局部特征，而是引入了由所有实例聚合而成的全局上下文信息。这解决了孤立实例注意力可能产生的偏差，是一个简单但有效的改进。
- **两阶段解耦策略**：将“哪里有问题”（定位）和“是什么问题”（分类）分开处理，并使用全卷积网络加速定位，这种工程化的流水线设计对于处理GB级WSI非常实用。

### 2. 方法之间的关系
- **定位网络**为**RMDL**提供高质量的输入实例。定位网络的准确性直接影响RMDL的上限。
- **RMDL内部**，**局部-全局融合**是**实例重新校准**的前提。没有全局特征，注意力系数就无法反映实例在整个WSI语境下的相对重要性。
- **多实例池化**是最后一步，将校准后的实例特征聚合为图像级表示。

### 3. 复现可行性
- **代码是否公开**：否。
- **方法描述是否完整**：是。网络结构、超参数（如NMS阈值、学习率、Batch Size）、损失函数（隐含交叉熵）均有描述。
- **关键配置是否明确**：是。包括输入尺寸、步长、优化器等。
- **预计复现难点**：
    1.  **数据预处理**：WSI的分块、金字塔层级处理、以及基于像素级标注的Patch采样策略需要仔细实现。
    2.  **定位网络的全卷积转换**：需要准确地将预训练的Inception-ResNet v2转换为FCN，并确保输出概率图的分辨率计算正确。
    3.  **Hard Negative Mining**：文中提到的利用假阳性扩展训练集的具体流程可能需要根据实际代码逻辑推断。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：实例级别的Attention机制结合全局上下文的思路，可用于改进当前的TransMIL、CLAM等模型中的注意力模块。
- **需要改造的设计**：定位网络部分过于依赖像素级标注，若要在无标注数据上使用，需替换为弱监督定位或自监督预训练方法。
- **可能形成的新研究思路**：探索端到端的联合训练（文中提到因显存限制未做），或者用知识蒸馏压缩定位网络以提升速度。

### 5. 阅读备注
- 论文发表于2019年，当时Attention-based MIL（如Ilse et al. 2018）刚兴起，RMDL提出的“局部+全局”融合视角在当时具有前瞻性。
- 数据集WSGI是私有数据集，虽然规模尚可（608张），但缺乏在Camelyon16/17等公共基准上的验证，限制了其广泛的可比性。
- 计算效率是其主要短板，近1分钟的推理时间在临床场景中难以接受，这也是后续许多WSI研究致力于优化的方向。
