# 53_MIXED_SUPERVISION_MIL_Multiple instance learning with mixed supervision in Gleason grading 方法总结

> 证据说明：输入为完整论文文本（arXiv:2206.12798v1），包含摘要、引言、方法、实验及结论。部分数学符号在 PDF 转文本时可能存在上下标等格式丢失；无法确认的符号按“提取不完整”处理，不自行补全。无页面缺失。

## 一、论文基本信息

- **论文标题**：Multiple Instance Learning with Mixed Supervision in Gleason Grading
- **作者**：Hao Bian, Zhuchen Shao, Yang Chen, Yifeng Wang, Haoqian Wang, Jian Zhang, Yongbing Zhang
- **发表年份**：2022 (arXiv preprint)
- **会议/期刊**：Medical Image Computing and Computer Assisted Intervention (MICCAI 2022)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1007/978-3-031-16452-1_20；https://arxiv.org/abs/2206.12798
- **代码仓库**：https://github.com/bianhao123/Mixed_supervision
- **研究任务**：前列腺癌Gleason分级（WSI分类）
- **数据模态**：全切片图像 (WSIs)，包含Slide-level标签和有限的Pixel-level标签

## 二、论文整体概述

### 1. 核心问题
在计算病理学的Gleason分级任务中，WSI通常只有Slide-level标签或有限的Pixel-level标签。现有方法存在两个主要局限：
1. 仅使用Slide-level标签的MIL方法忽略了Pixel-level标签中包含的丰富局部信息。
2. 同时使用两种标签的方法未考虑Pixel-level标签可能存在的重叠和不准确性，导致模型性能受损。

### 2. 整体方法
提出了一种基于多实例学习框架的混合监督Transformer（Mixed Supervision Transformer）。该方法分为两步：
1. **实例特征与标签生成**：利用SLIC超像素算法将不准确的Pixel-level标签转化为更可靠的Instance-level标签，并提取实例特征。
2. **混合监督训练**：设计一个包含Class Token（对应Slide级）和Instance Token（对应Instance级）的Transformer。引入随机掩码策略（Random Masking）以减少不准确Instance-level标签的影响，并结合2D正弦位置编码利用空间信息。

### 3. 主要贡献
1. 提出了从Pixel-level标签生成可靠Instance-level标签的流程，解决了标签噪声问题。
2. 设计了混合监督Transformer，同时优化Slide级和Instance级损失。
3. 引入了受MAE启发的随机掩码策略，有效缓解了标签不准确带来的负面影响，并降低了计算开销。
4. 在SICAPv2数据集上达到了SOTA性能。

## 三、方法总结

### 方法 1：混合监督Transformer (Mixed Supervision Transformer)

#### 1. 核心思想与解决的问题
- **目标问题**：如何有效利用WSI中同时存在的Slide-level标签和有限且可能不准确的Pixel-level标签进行Gleason分级。
- **现有方法的局限**：传统MIL忽略局部细节；直接混合监督易受标签噪声干扰。
- **核心思想**：构建一个双分支Transformer，分别处理Slide级分类（通过Class Token）和Instance级分类（通过Instance Tokens）。通过随机掩码策略动态屏蔽部分Instance Token及其标签，迫使模型依赖更鲁棒的Slide级监督信号或学习更通用的特征表示，从而抑制噪声标签的影响。
- **创新点**：
    - 结合SLIC超像素生成实例，比矩形Patch更符合组织结构。
    - 引入随机掩码机制处理混合监督中的标签不一致性。
    - 在Transformer中显式融合2D空间位置编码。

#### 2. 详细结构与数据流
- **输入**：
    - WSI被分割为 $N$ 个超像素区域（Instances）。
    - 每个实例提取出 $d=1280$ 维的特征向量 $z_i$。
    - Slide-level标签 $Y$（多标签二进制向量）。
    - Instance-level标签 $y_i$（多类别标签，由Pixel-level标签多数投票得到）。
- **处理流程**：
    1. **实例生成**：SLIC聚类 -> 裁剪224x224 Patch -> MobileNetV2提取特征 -> 平均池化得到 $z_i$。
    2. **标签映射**：统计超像素内Pixel-level标签比例，取最大比例标签作为 $y_i$。
    3. **随机掩码**：以概率 $m$ 随机选择部分Instance Token进行掩码（不参与后续Transformer计算，也不计算Instance Loss）。
    4. **位置编码**：对保留的Unmasked Instances计算2D正弦位置编码，并与特征相加。
    5. **Token拼接**：添加可学习的Class Token，与所有Unmasked Instance Tokens拼接。
    6. **Transformer编码**：经过 $L$ 层Transformer Block，捕捉实例间相关性。
    7. **输出头**：
        - Class Token输出 $\tilde{Y}$ 进入MLP + Sigmoid，计算Slide级Loss。
        - Instance Tokens输出 $\tilde{y}_i$ 进入MLP + Softmax，计算Instance级Loss。
- **输出**：
    - Slide级预测概率 $\hat{Y}$。
    - Instance级预测概率 $\hat{y}_i$。
- **模块在整体网络中的位置**：核心骨干网络。
- **与其他模块的连接方式**：接收预处理后的实例特征和标签；输出用于计算总损失 $L_{total}$。

#### 3. 数学公式

**位置编码 (Algorithm 1):**
$$ PE(i, pos, 2j) = \sin\left(\frac{pos}{10000^{2j/d_{half}}}\right), \quad PE(i, pos, 2j+1) = \cos\left(\frac{pos}{10000^{2j/d_{half}}}\right) $$
其中 $pos \in \{\frac{p_{i,h}}{100}, \frac{p_{i,w}}{100}\}$，$d_{half} = d/2$，$j \in [0, d_{half}-1]$。
最终位置嵌入 $s_i = \text{CONCAT}[PE_{i,h}, PE_{i,w}]$。
输入Token $h_i = z_i + w \cdot s_i$，其中 $w=0.1$ 是缩放因子。

**Slide-level Loss:**
$$ L_{slide} = L_1(Y, \text{sigmoid}(\hat{Y})) $$
其中 $L_1$ 是多标签加权交叉熵损失，$\hat{Y}$ 来自Class Token的输出经过MLP。

**Instance-level Loss:**
$$ L_{instance} = L_2(y_i, \text{softmax}(\hat{y}_i)) $$
其中 $L_2$ 是多类别加权交叉熵损失，$\hat{y}_i$ 来自第 $i$ 个Instance Token的输出经过MLP。注意：此损失仅对Unmasked的实例计算。

**Total Loss:**
$$ L_{total} = \lambda L_{slide} + (1 - \lambda) \sum_{k} L_{instance}^{(k)} $$
其中 $\lambda = 0.5$，求和遍历所有Unmasked的实例 $k$。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入特征 | $Z$ | $(N, 1280)$ | $N$ 为实例总数，1280为MobileNetV2输出维度 |
| 位置编码 | $S$ | $(N, 1280)$ | 2D坐标编码后拼接 |
| 输入Token | $H^{(0)}$ | $(N_{un}+1, 1280)$ | $N_{un}$ 为Unmasked实例数，+1为Class Token |
| Transformer输出 | $H^{(L)}$ | $(N_{un}+1, 1280)$ | 经过L层Transformer后的Embedding |
| Slide预测 | $\hat{Y}$ | $(C_{slide})$ | $C_{slide}$ 为Gleason等级数量 (如NC, GG3, GG4, GG5等组合) |
| Instance预测 | $\hat{y}_i$ | $(C_{inst})$ | $C_{inst}$ 为单类Gleason等级数量 (NC, GG3, GG4, GG5) |

*注：具体类别数取决于数据集定义，SICAPv2通常涉及非癌、GG3、GG4、GG5的组合。*

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedSupervisionTransformer(nn.Module):
    def __init__(self, input_dim=1280, num_heads=6, num_layers=2, 
                 num_classes_slide=4, num_classes_instance=4, 
                 mask_ratio=0.5, lambda_weight=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.lambda_weight = lambda_weight
        
        # Position Encoding parameters
        self.pos_scale = 0.1
        
        # Learnable Class Token
        self.class_token = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Heads
        self.slide_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes_slide)
        )
        
        self.instance_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes_instance)
        )

    def generate_position_encoding(self, centroids_h, centroids_w, d):
        """
        Generate 2D sinusoidal position encoding based on centroid coordinates.
        centroids_h, centroids_w: Tensor of shape (N,) or (Batch, N)
        """
        # Normalize coordinates to [0, 1] roughly by dividing by 100 as per paper
        h_norm = centroids_h / 100.0
        w_norm = centroids_w / 100.0
        
        # Reshape for broadcasting if necessary
        if h_norm.dim() == 1:
            h_norm = h_norm.unsqueeze(1)
            w_norm = w_norm.unsqueeze(1)
            
        # Calculate PE dimensions
        half_d = d // 2
        div_term = torch.exp(torch.arange(0, half_d, 2).float() * (-math.log(10000.0) / half_d))
        
        pe_h = torch.zeros_like(h_norm)
        pe_w = torch.zeros_like(w_norm)
        
        # Sinusoidal encoding logic similar to standard ViT but adapted for 2D coords
        # Paper formula: sin(pos / 10000^(2j/d_half))
        # Note: The paper implementation details might vary slightly in normalization, 
        # assuming standard sinusoidal application here.
        
        # Simplified reconstruction based on Algorithm 1 description
        # We need to concatenate H and W encodings
        # Let's assume a helper function or direct calculation
        
        # For pseudocode clarity, we assume a function `get_2d_pe` exists that returns (N, d)
        pe_h = self._get_1d_pe(h_norm.squeeze(), half_d)
        pe_w = self._get_1d_pe(w_norm.squeeze(), half_d)
        
        # Concatenate along feature dim
        pe = torch.cat([pe_h, pe_w], dim=-1)
        return pe

    def _get_1d_pe(self, pos, half_d):
        # Standard sinusoidal PE for 1D coordinate
        device = pos.device
        pe = torch.zeros((pos.shape[0], half_d), device=device)
        div_term = torch.exp(torch.arange(0, half_d, 2, dtype=torch.float32, device=device) * 
                             -(math.log(10000.0) / half_d))
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)
        return pe

    def forward(self, x, centroids_h, centroids_w, y_slide, y_inst=None, training=True):
        """
        x: (B, N, D) Instance features
        centroids_h, centroids_w: (B, N) Centroid coordinates
        y_slide: (B, C_slide) Slide labels
        y_inst: (B, N, C_inst) Instance labels (optional, used for loss)
        """
        B, N, D = x.shape
        
        # 1. Random Masking
        if training:
            num_masked = int(N * self.mask_ratio)
            indices = torch.randperm(N, device=x.device)[:num_masked]
            mask = torch.ones(B, N, device=x.device)
            mask[:, indices] = 0 # 0 means masked, 1 means unmasked
            
            # Create unmask indices
            unmask_indices = torch.where(mask == 1)[1]
            # Gather unmasked tokens
            x_unmasked = torch.gather(x, 1, unmask_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, D))
            # Also gather corresponding positions and labels if needed
            c_h_unmasked = torch.gather(centroids_h, 1, unmask_indices.unsqueeze(1))
            c_w_unmasked = torch.gather(centroids_w, 1, unmask_indices.unsqueeze(1))
            
            N_un = x_unmasked.size(1)
        else:
            x_unmasked = x
            c_h_unmasked = centroids_h
            c_w_unmasked = centroids_w
            N_un = N
            mask = None # No masking during inference usually, or full mask

        # 2. Position Encoding & Addition
        # Assuming centroids are normalized appropriately before this step or handled inside
        pe = self.generate_position_encoding(c_h_unmasked, c_w_unmasked, D)
        x_with_pos = x_unmasked + self.pos_scale * pe
        
        # 3. Add Class Token
        class_tokens = self.class_token.expand(B, -1, -1)
        h_input = torch.cat([class_tokens, x_with_pos], dim=1) # Shape: (B, N_un + 1, D)
        
        # 4. Transformer Forward
        h_out = self.transformer_encoder(h_input)
        
        # 5. Extract Outputs
        class_token_out = h_out[:, 0, :] # (B, D)
        inst_token_out = h_out[:, 1:, :] # (B, N_un, D)
        
        # 6. Prediction Heads
        slide_pred = self.slide_head(class_token_out) # (B, C_slide)
        inst_pred = self.instance_head(inst_token_out) # (B, N_un, C_inst)
        
        # 7. Loss Calculation (only if labels provided and training)
        loss = None
        if training and y_slide is not None:
            # Slide Loss
            l_slide = F.cross_entropy(slide_pred, y_slide.argmax(dim=1)) # Assuming single label per slide for CE, or BCE for multi-label
            # Note: Paper says "multi-label weighted cross entropy", likely BCEWithLogitsLoss
            # But SICAPv2 slide level is often treated as specific grade assignment. 
            # Let's stick to the paper's L1 definition conceptually.
            
            total_loss = l_slide
            
            if y_inst is not None:
                # Gather instance labels corresponding to unmasked instances
                y_inst_unmasked = torch.gather(y_inst, 1, unmask_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, y_inst.size(-1)))
                
                # Instance Loss
                l_inst = F.cross_entropy(inst_pred.view(-1, inst_pred.size(-1)), 
                                         y_inst_unmasked.view(-1))
                
                # Weighted Sum
                total_loss = self.lambda_weight * l_slide + (1 - self.lambda_weight) * l_inst
                
            loss = total_loss
            
        return slide_pred, inst_pred, loss
```

#### 6. 实现提示
- **关键网络组件**：`nn.TransformerEncoder`，MobileNetV2 (预训练权重)，SLIC算法库 (如 `skimage.segmentation.slic`)。
- **重要超参数**：
    - Embedding Dimension ($d$): 1280 (来自MobileNetV2)。
    - Masking Ratio ($m$): 实验中测试了 0%, 10%, 25%, 50%，最佳为 50%。
    - Lambda ($\lambda$): 0.5。
    - Transformer Layers: 2。
    - Heads: 6。
    - Learning Rate: 2e-4。
    - Optimizer: Ranger (RAdam + Lookahead)。
- **归一化/激活方式**：Slide分支使用Sigmoid + CrossEntropy/BCE；Instance分支使用Softmax + CrossEntropy。内部Transformer使用标准LayerNorm和GeLU/ReLU（取决于PyTorch默认实现，通常ViT用GeLU，但文中未明确指定Transformer内部激活，参考TransMIL通常用GeLU）。
- **维度对齐方式**：位置编码维度必须与特征维度 $d$ 一致，通过将2D坐标分别编码后拼接实现。
- **实现注意事项**：
    - SLIC生成的超像素大小不一，需统一裁剪为224x224或平均池化多个Patch特征。
    - 随机掩码是在Epoch级别或Batch级别进行的？文中提到"In each training epoch... sampled... without replacement"，暗示可能在每个Epoch开始时确定掩码模式，或者在Batch内动态采样。伪代码中按Batch内动态采样实现更常见，若按Epoch则需维护状态。
    - 坐标归一化：文中提到 $pos \in \{p_{i,h}/100, p_{i,w}/100\}$，需注意坐标值的范围。

#### 7. 计算与资源开销
- **理论计算复杂度**：Transformer复杂度为 $O(L \cdot N^2 \cdot d)$。由于使用了Masking，实际参与计算的 $N$ 减少为 $(1-m)N$。
- **参数量**：主要参数来自MobileNetV2（冻结或微调？）和Transformer。文中未给出具体参数量，但MobileNetV2较大，Transformer较小（2层，6头）。
- **FLOPs/MACs**：未提供具体数值。
- **显存开销**：Masking策略显著减少了显存占用，因为不需要存储和处理所有Instance Token的梯度。
- **推理速度**：推理时不使用Masking（或Masking率为0），速度取决于 $N$（超像素数量）。
- **论文是否提供效率对比**：未提供详细的FLOPs对比，但声称Masking减少了计算和内存占用。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI的Gleason分级（多标签分类）。
- **可迁移到的任务/数据集**：任何具有Slide-level标签和部分噪声Instance/Pixel-level标签的弱监督/半监督视觉分类任务（如组织学亚型分类、肿瘤分级）。
- **迁移所需调整**：
    - 特征提取器可能需要更换（如ResNet, Swin Transformer）。
    - 类别数和损失函数需根据新任务调整。
    - 位置编码需适配新的空间结构（如Grid vs Superpixel）。
- **适用条件**：数据集中存在一定比例的局部标注，且这些标注可能存在噪声。
- **潜在限制**：严重依赖SLIC生成的超像素质量；对于没有局部标注的数据集退化为普通MIL。

#### 9. 实验与消融证据
- **主要性能结果**：在SICAPv2上AUC达到 $0.9429 \pm 0.0094$，优于TransMIL ($0.9152$) 和 ATMIL ($0.9373$)。
- **相对基线的提升**：相比纯Slide级监督的SOTA方法提升约2-3% AUC；相比其他混合监督方法（SegGini）提升显著。
- **相关消融实验**：
    - **Masking比率**：50% Masking效果最好 (0.9429)，优于0% (0.9273)。
    - **位置编码**：加入2D位置编码在所有Masking设置下均有提升。
    - **混合监督有效性**：混合监督优于仅Slide级监督。
- **作者结论**：随机掩码有效缓解了标签不准确的影响，混合监督提升了性能。
- **证据是否充分**：在单一公开数据集上进行了充分的对比和消融，证据较为充分。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 中 | 混合监督MIL并非全新概念，但结合随机掩码处理噪声标签的思路具有新意。 |
| 技术可行性 | 高 | 基于标准Transformer和常见损失函数，易于复现。 |
| 实现难度 | 中 | 需要处理SLIC超像素生成、坐标编码以及复杂的Masking逻辑。 |
| 架构相关性 | 高 | 专为WSI的大规模实例和混合监督特性设计。 |
| 可迁移性 | 高 | 核心思想（噪声标签下的混合监督+掩码）可应用于其他领域。 |
| 计算成本 | 低/中 | Masking降低了训练时的计算负担。 |

#### 11. 一句话总结
该论文提出了一种结合随机掩码策略的混合监督Transformer，通过同时利用Slide级和经SLIC处理的Instance级标签，有效克服了病理图像中标签噪声问题，实现了高精度的Gleason分级。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **随机掩码策略用于混合监督**：借鉴MAE的思想，通过随机丢弃部分带有噪声标签的Instance Token，防止模型过拟合错误标签，这是一个简单而有效的正则化手段。
- **SLIC超像素实例化**：相比于传统的网格Patch，基于超像素的实例划分更符合病理组织的语义边界，提高了实例标签的可靠性。

### 2. 方法之间的关系
- **实例生成模块**是预处理步骤，为Transformer提供高质量的输入。
- **Transformer编码器**是核心，负责特征交互。
- **双分支输出头**分别对应两种监督信号。
- **损失函数**将两者结合，Masking机制作用于前向传播过程中，影响梯度的来源。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，包括算法步骤、超参数和损失函数定义。
- **关键配置是否明确**：是，如MobileNetV2维度、Transformer层数、Masking率等。
- **预计复现难点**：
    - SLIC参数的调优以获得合适的超像素数量。
    - 坐标归一化的具体细节（文中提到除以100，需确认原始坐标单位）。
    - Ranger优化器的具体实现（虽然开源，但需确保版本一致）。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：混合监督框架、随机掩码去噪机制。
- **需要改造的设计**：位置编码方式（若数据不是WSI而是自然图像，可能不需要2D坐标或需改为绝对位置编码）。
- **可能形成的新研究思路**：
    - 探索自适应掩码策略（而非均匀随机）。
    - 结合自监督学习（如MAE的重建任务）进一步利用未掩码部分的特征。
    - 应用于其他具有层级标签结构的医学影像分析任务。

### 5. 阅读备注
- 论文中提到的“Multi-label classification”对于Slide-level，但在SICAPv2中，每个Slide通常有一个主要的Gleason Score（如3+4=7），这可能是一个多分类问题或者是特定的多标签定义（如是否存在GG3, GG4等）。文中公式(1)使用了Sigmoid，暗示可能是多标签（Binary Multi-label），即判断每种Grade是否存在。需仔细核对SICAPv2的标签定义。
- 公式(3)中的求和 $\sum_k$ 仅针对Unmasked的实例，这一点在实现时需特别注意，避免对Masked实例计算梯度。
