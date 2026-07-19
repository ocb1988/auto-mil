# 56_VINO_MIL_Transformer-based video-structure MIL for WSI classification 方法总结

> 证据说明：输入为完整论文全文（9页），包含摘要、引言、方法、实验及参考文献。公式提取基本完整，关键符号定义清晰。无明显的页面或公式提取缺失。

## 一、论文基本信息

- **论文标题**：Transformer-Based Video-Structure Multi-Instance Learning for Whole Slide Image Classification
- **作者**：Yingfan Ma, Xiaoyuan Luo, Kexue Fu, Manning Wang
- **发表年份**：2024
- **会议/期刊**：The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1609/aaai.v38i13.29338
- **代码仓库**：未说明
- **研究任务**：全切片图像（WSI）分类（Bag-level）和阳性区域检测/实例分类（Instance-level）
- **数据模态**：数字病理学全切片图像（WSIs）

## 二、论文整体概述

### 1. 核心问题
现有WSI分析方法主要面临两个挑战：
1.  **内存限制与端到端学习的矛盾**：WSI尺寸巨大，无法直接输入网络。主流两阶段方法先提取Patch特征再聚合，导致无法进行端到端训练，且预训练特征存在域间隙（Domain Gap）和不恰当的归纳偏置。
2.  **上下文信息的缺失**：传统MIL方法假设实例独立分布，忽略了病理诊断中重要的空间上下文信息；而基于图或Transformer的方法虽然能建模上下文，但因计算复杂度高，通常也依赖冻结的特征提取器，难以实现真正的端到端联合优化。

### 2. 整体方法
提出 **VINO** (Video-structure Instance Network) 框架。
1.  **视频结构构建**：将WSI划分为非重叠的小块（Patches），根据空间位置关系将其组织成一系列合成的“视频片段”（Video Clips）。每个Clip包含时间跨度 $T$ 内的相邻Patches。
2.  **分治策略与共享参数**：采用分治策略，将所有Clip输入到参数共享的Transformer模型中。这降低了空间复杂度，使得端到端训练成为可能。
3.  **类特定Clip Token**：借鉴MCT-Former，在每个Clip中附加 $n$ 个类特定的Clip Token（Class-specific Clip Tokens），用于学习不同类别的语义信息。
4.  **双分支架构**：
    *   **Bag级分类分支**：通过全局平均池化（GAP）聚合所有Clip的类特定Token，输入MLP得到Slide级预测。
    *   **Instance级分类分支**：利用Bag分支训练好的类特定Token与经过MHSA交互后的Clip特征计算点积，生成高质量伪标签，用于训练实例分类器，实现阳性区域定位。

### 3. 主要贡献
1.  提出将WSI建模为合成视频片段，保留局部空间关系，便于计算机模拟病理学家观察过程，并支持上下文感知学习。
2.  引入类特定Clip Token，通过其与实例特征的相似度生成伪标签，训练实例级分类器，实现端到端的实例级分类。
3.  在三个公开数据集（CAMELYON16, PANDA, TCGA-NSCLC）上验证了该方法在Bag和Instance分类任务上均优于SOTA基线。

## 三、方法总结

### 方法 1：VINO 框架 (Bag-level & Instance-level Classification)

#### 1. 核心思想与解决的问题
- **目标问题**：解决WSI分析中端到端训练困难、上下文信息利用不足以及实例级伪标签质量低的问题。
- **现有方法的局限**：两阶段方法存在域间隙；纯Transformer/GNN方法计算开销大，难以端到端处理海量Patches。
- **核心思想**：利用“视频”概念组织空间相邻的Patches，通过参数共享的Transformer处理这些短序列（Clips），既保留了局部上下文，又控制了计算复杂度。利用Bag分支学到的类语义指导Instance分支的伪标签生成。
- **创新点**：
    - 将WSI视为病理视频，使用Divide-and-Conquer策略。
    - 引入类特定Clip Token（Class-specific Clip Tokens）作为语义锚点。
    - 基于Clip Token与实例特征相似度的伪标签生成机制。

#### 2. 详细结构与数据流
- **输入**：
    - WSI $X_i$，被分割为 $M_i$ 个非重叠 Patch $x_{i,j}$。
    - Bag标签 $Y_i \in \{0, 1\}$（二分类）或多类标签。
- **处理流程**：
    1.  **Patch编码**：使用ResNet18 backbone $F(\cdot)$ 提取Patch特征。
    2.  **Video Clip构建**：将空间相邻的 $T$ 个Patch组成一个Clip $Clip_{it} \in \mathbb{R}^{T \times dim}$。
    3.  **Transformer处理**：
        - 向每个Clip添加 $n$ 个可学习的类特定Token $Clip\_token_{itc}$ ($c=1...n$)。
        - 输入共享参数的Transformer（含Multi-Head Self-Attention, MHSA）。
        - 输出交互后的Clip特征 $Clip'_{it}$ 和更新后的类特定Token。
    4.  **Bag级分类**：
        - 对每个类别 $c$ 的所有Clip对应的Token进行全局平均池化（GAP），得到 $Class\_feat_c$。
        - 输入MLP Head得到视频得分 $Video\_score_i$。
        - 计算Cross-Entropy Loss $\mathcal{L}_{bag}$。
    5.  **Instance级分类**：
        - 计算类特定Token与交互后Clip特征 $Clip'_{it}$ 的点积，得到伪标签 $\hat{y}_{i,j}$。
        - 使用伪标签训练Instance Classifier Head $H$，计算 $\mathcal{L}_{ins}$。
- **输出**：
    - Bag级预测概率。
    - Instance级预测概率（用于热力图可视化/阳性区域定位）。
- **模块在整体网络中的位置**：核心主干网络。
- **与其他模块的连接方式**：ResNet18提取特征 -> Transformer处理 -> 并行连接Bag Head和Instance Head。

#### 3. 数学公式

**Patch特征提取：**
$$ Clip_{it} = F(X_{it}), \quad X_{it} \in \mathbb{R}^{T \times H \times W \times C} \quad (1) $$
其中 $F$ 为ResNet18编码器，$dim$ 为特征维度。

**Self-Attention:**
$$ Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_{mk}}}\right)V \quad (2) $$
其中 $Q, K, V$ 由可学习矩阵 $W_q, W_k, W_v$ 投影得到。

**Multi-Head Self-Attention (MHSA):**
$$ Z_h = softmax\left(\frac{Q_h K_h^T}{\sqrt{d_{mk}}}\right)V_h, \quad h=1,\dots,heads \quad (3) $$
$$ MHSA(Q, K, V) = W_o \cdot Concat(Z_1, \dots, Z_{heads}) \quad (4) $$
其中 $d_{mk} = d_{mv} = \frac{dim}{heads}$。

**Bag-level Classification:**
$$ Class\_feat_{ic} = GAP(Clip\_token_{itc}) \quad (5) $$
$$ Video\_score_i = softmax(MLP_i(Class\_feat_c)) \quad (6) $$
$$ \mathcal{L}_{bag} = CrossEntropy(Video\_score_i, Y_i) \quad (7) $$

**Instance-level Classification (Pseudo-labels):**
$$ \hat{y}_{i,j} = dot(Clip\_token, Clip'_{itj}) \quad (8) $$
$$ y_j = H(Clip'_{itj}), \quad j=1,\dots,k \quad (9) $$
$$ \mathcal{L}_{ins} = CrossEntropy(\hat{y}, y) \quad (10) $$
*注：公式(10)中的 $y$ 指代伪标签 $\hat{y}$ 作为监督信号训练Head $H$ 的输出预测值与伪标签之间的损失，原文表述略有歧义，通常理解为用伪标签训练Instance Classifier。*

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | Patch Block | $T \times H \times W \times C$ | 单个Video Clip的空间输入 |
| 特征 | $Clip_{it}$ | $T \times dim$ | ResNet18输出的Patch特征序列 |
| Token | $Clip\_token_{itc}$ | $n \times dim$ | 附加的类特定Token数量等于类别数 |
| Attention | $Q, K, V$ | $(T+n) \times d_{mk}$ | Query, Key, Value 矩阵 |
| Output | $Clip'_{it}$ | $(T+n) \times dim$ | MHSA输出，包含交互后的Patch和Token特征 |
| Bag Pred | $Video\_score_i$ | $n \times 1$ | Bag级分类概率 |
| Inst Pred | $\hat{y}_{i,j}$ | Scalar | 第j个Patch属于某类的伪标签分数 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VINO(nn.Module):
    def __init__(self, patch_dim, hidden_dim, num_classes, num_heads, clip_length, dropout=0.1):
        super(VINO, self).__init__()
        # 1. Feature Extractor (ResNet18)
        self.encoder = ResNet18() 
        # 假设 encoder 输出维度为 patch_dim
        
        # 2. Transformer Encoder with Shared Parameters
        self.transformer_encoder = TransformerEncoder(
            d_model=hidden_dim, 
            nhead=num_heads, 
            num_layers=1, # 论文未明确层数，通常少量即可
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        
        # 3. Class-specific Clip Tokens
        # n classes, each token is 1 x hidden_dim
        self.clip_tokens = nn.Parameter(torch.randn(num_classes, hidden_dim))
        
        # 4. MLP Heads
        self.bag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.instance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.clip_length = clip_length

    def forward(self, patches_batch):
        """
        patches_batch: List of WSIs. Each WSI is a list of patches.
        For simplicity, assume we process one WSI at a time or batched clips.
        Let's assume input is a tensor of shape [Batch_Size, Num_Clips, Clip_Length, Patch_H, Patch_W, Channels]
        """
        B, N_clips, T, H, W, C = patches_batch.shape
        
        # Step 1: Extract Features using Shared Encoder
        # Reshape to process all patches in the batch
        patches_flat = patches_batch.view(B * N_clips * T, H, W, C)
        features = self.encoder(patches_flat) # Shape: [B*N_clips*T, patch_dim]
        
        # Reshape back to clips
        features = features.view(B, N_clips, T, -1) # Shape: [B, N_clips, T, patch_dim]
        
        # Project to hidden_dim if necessary (assuming patch_dim == hidden_dim for simplicity)
        # If not, add a linear projection layer here
        
        # Step 2: Add Class-Specific Tokens
        # Expand tokens to match batch and clip count
        # clip_tokens shape: [num_classes, hidden_dim]
        # We need to append these to the feature sequence along the time dimension
        # New sequence length: T + num_classes
        
        # Repeat tokens for each clip in the batch
        # tokens_expanded: [B, N_clips, num_classes, hidden_dim]
        tokens_expanded = self.clip_tokens.unsqueeze(0).unsqueeze(0).expand(B, N_clips, -1, -1)
        
        # Concatenate features and tokens along time dimension
        # features: [B, N_clips, T, hidden_dim]
        # tokens_expanded: [B, N_clips, num_classes, hidden_dim]
        transformer_input = torch.cat([features, tokens_expanded], dim=2) 
        # Shape: [B, N_clips, T + num_classes, hidden_dim]
        
        # Permute for Transformer: [Seq_Len, Batch, Hidden]
        transformer_input = transformer_input.permute(2, 0, 1, 3).reshape(-1, B, self.hidden_dim)
        
        # Step 3: Transformer Forward
        output = self.transformer_encoder(transformer_input)
        # output shape: [(T+num_classes)*N_clips, B, hidden_dim]
        
        # Reshape back
        seq_len = T + self.num_classes
        output = output.reshape(seq_len, B, N_clips, self.hidden_dim)
        
        # Separate Patch Outputs and Token Outputs
        # Patch outputs are first T positions
        patch_outputs = output[:T, :, :, :] # [T, B, N_clips, hidden_dim]
        # Token outputs are last num_classes positions
        token_outputs = output[T:, :, :, :] # [num_classes, B, N_clips, hidden_dim]
        
        # Transpose for easier indexing: [B, N_clips, num_classes, hidden_dim]
        token_outputs = token_outputs.permute(1, 2, 0, 3)
        # Transpose patch outputs: [B, N_clips, T, hidden_dim]
        patch_outputs = patch_outputs.permute(1, 2, 0, 3)
        
        bag_predictions = []
        instance_predictions = []
        pseudo_labels_list = []
        
        for b in range(B):
            # --- Bag Level Classification ---
            # Average Pooling over clips for each class token
            # token_outputs[b]: [N_clips, num_classes, hidden_dim]
            avg_tokens = token_outputs[b].mean(dim=0) # [num_classes, hidden_dim]
            
            # Apply MLP Head
            bag_logits = self.bag_head(avg_tokens) # [num_classes]
            bag_prob = F.softmax(bag_logits, dim=0)
            bag_predictions.append(bag_prob)
            
            # --- Instance Level Classification ---
            # Compute pseudo-labels using similarity between Class Tokens and Patch Features
            # token_outputs[b]: [N_clips, num_classes, hidden_dim]
            # patch_outputs[b]: [N_clips, T, hidden_dim]
            
            # Dot product: [N_clips, num_classes, T]
            # Note: The paper says "dot product between class-specific tokens ... and instance features"
            # It implies comparing each patch in a clip against the class token of that clip
            
            # Similarity calculation
            # token_outputs[b] has shape [N_clips, num_classes, hidden_dim]
            # patch_outputs[b] has shape [N_clips, T, hidden_dim]
            
            # Expand for broadcasting: [N_clips, num_classes, 1, hidden_dim] vs [N_clips, 1, T, hidden_dim]
            sim_matrix = torch.einsum('ncd,nid->nci', token_outputs[b], patch_outputs[b])
            # sim_matrix: [N_clips, num_classes, T]
            
            # Normalize or use directly? Paper uses dot product scores as pseudo-labels.
            # Usually sigmoid or softmax is applied to get probabilities.
            # Let's assume sigmoid for binary/multi-label or softmax for multi-class mutually exclusive per patch?
            # Given it's MIL, usually soft labels. Let's use Sigmoid for general case.
            pseudo_labels = torch.sigmoid(sim_matrix) # [N_clips, num_classes, T]
            pseudo_labels_list.append(pseudo_labels)
            
            # Train Instance Classifier Head using Pseudo Labels
            # Reshape patches and pseudo labels for training
            # patches_flat_inst: [N_clips * T, hidden_dim]
            # p_labels_flat: [N_clips * T, num_classes]
            
            patches_flat_inst = patch_outputs[b].reshape(-1, self.hidden_dim)
            p_labels_flat = pseudo_labels.reshape(-1, self.num_classes)
            
            inst_logits = self.instance_head(patches_flat_inst)
            inst_prob = F.softmax(inst_logits, dim=1)
            instance_predictions.append((inst_prob, p_labels_flat))
            
        return bag_predictions, instance_predictions

    def compute_losses(self, bag_preds, inst_preds, bag_labels):
        # Bag Loss
        bag_loss = 0
        for pred, label in zip(bag_preds, bag_labels):
            bag_loss += F.cross_entropy(pred.unsqueeze(0), label.unsqueeze(0))
        bag_loss /= len(bag_preds)
        
        # Instance Loss
        inst_loss = 0
        for prob, p_label in inst_preds:
            # Cross entropy between predicted probability and pseudo-label (soft target)
            # Using KLDivLoss or BCEWithLogitsLoss depending on activation
            # Here assuming p_label is probability distribution
            inst_loss += F.kl_div(F.log_softmax(prob, dim=1), p_label, reduction='batchmean')
        inst_loss /= len(inst_preds)
        
        return bag_loss, inst_loss
```

#### 6. 实现提示
- **关键网络组件**：ResNet18 Backbone, Transformer Encoder (MHSA), MLP Heads.
- **重要超参数**：
    - `clip_length` ($T$): 视频片段长度，论文提到“subset”，具体数值需查实验设置部分（文中Table 5附近提到overlapping rate，但未明确T值，通常设为较小整数如4, 8, 16）。
    - `num_classes`: 类别数。
    - `hidden_dim`: Transformer隐藏层维度。
    - `num_heads`: 注意力头数。
- **归一化/激活方式**：Transformer内部通常使用LayerNorm和GELU/ReLU；Bag Head使用Softmax；Instance Head使用Softmax或Sigmoid（取决于任务是多分类互斥还是多标签）。
- **维度对齐方式**：Patch特征维度需映射到Transformer的$d_{model}$；Clip Token维度必须与$d_{model}$一致。
- **实现注意事项**：
    - **参数共享**：所有Clip共享同一个Transformer权重。
    - **伪标签生成**：注意点积结果的尺度，可能需要温度系数缩放后再做Softmax/Sigmoid以稳定梯度。
    - **内存管理**：由于是End-to-End，需小心显存占用，可使用梯度累积或减小Batch Size。

#### 7. 计算与资源开销
- **理论计算复杂度**：Transformer的复杂度为 $O(N \cdot L^2 \cdot d)$，其中 $N$ 是Clip数量，$L=T+n$ 是序列长度，$d$ 是维度。由于 $L$ 很小（$T$小，$n$为类别数），复杂度远低于处理整个WSI所有Patches的Transformer。
- **参数量**：主要由ResNet18和Transformer构成。Transformer参数量相对较小（单层或少层）。
- **FLOPs/MACs**：论文未提供具体FLOPs对比，但强调比全图Transformer更轻量。
- **显存开销**：显著低于全图Transformer，因为每次只处理一个小Clip。
- **推理速度**：较快，适合大规模WSI筛查。
- **论文是否提供效率对比**：未提供具体的FLOPs或秒数对比，仅定性描述“reduce spatial complexity”。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：WSI癌症分类（乳腺癌淋巴结转移、前列腺癌分级、肺癌亚型）。
- **可迁移到的任务/数据集**：任何具有空间结构的弱监督学习任务，如遥感图像分类、显微细胞图像分析、长序列时间序列异常检测。
- **迁移所需调整**：调整Backbone以适应新数据模态；调整Clip长度$T$以适应新的空间相关性范围。
- **适用条件**：数据具有明显的局部空间连续性或序列依赖性。
- **潜在限制**：如果WSI中的病理模式跨越很大的空间距离（Long-range dependency），仅靠局部Clip可能丢失全局上下文，需要多层堆叠或层级结构。

#### 9. 实验与消融证据
- **主要性能结果**：
    - **CAMELYON16**: Bag AUC 0.9466, Instance AUC 0.9213 (VINO-E2E)，优于TransMIL (0.8360) 等。
    - **PANDA**: Bag AUC ~0.91-0.94, Instance AUC ~0.94-0.96，接近Fully Supervised。
    - **TCGA-NSCLC**: Bag AUC 0.9853 (VINO-Feature)。
- **相对基线的提升**：在CAMELYON16上Bag AUC提升约10%以上。
- **相关消融实验**：
    - w/o contextual information (随机采样Patches): 性能下降。
    - w/o end-to-end training (VINO-Feature): Bag AUC从0.9466降至0.9085，证明端到端的重要性。
    - SVM评估特征质量：VINO提取的特征线性可分性最高。
- **作者结论**：端到端训练、上下文建模、类特定Token均有效。
- **证据是否充分**：在三个数据集上均有验证，消融实验覆盖了核心组件，证据充分。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 将WSI建模为视频Clip并结合类特定Token进行伪标签生成，视角新颖。 |
| 技术可行性 | 高 | 基于标准Transformer和ResNet，易于实现，逻辑自洽。 |
| 实现难度 | 中 | 需注意伪标签生成的细节和显存优化，但无极端复杂的算子。 |
| 架构相关性 | 高 | 专门针对WSI的大尺寸和弱监督特性设计。 |
| 可迁移性 | 中 | 依赖于空间局部性假设，对其他非空间序列数据需调整。 |
| 计算成本 | 低 | 相比全图Transformer，计算效率高。 |

#### 11. 一句话总结
VINO通过将WSI划分为空间相邻的视频片段并利用参数共享的Transformer和类特定Token，实现了兼具上下文感知能力和端到端训练效率的全切片图像分类与阳性区域定位。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **Video Clip Construction Strategy**: 将高维空间数据降维为一维序列（视频帧）的处理方式，有效平衡了局部上下文捕获与计算复杂度。
- **Class-Specific Clip Tokens for Pseudo-labeling**: 利用Bag分支学到的全局类语义（Token）来指导Instance分支的伪标签生成，解决了弱监督下实例标签噪声大的问题。

### 2. 方法之间的关系
- **VINO vs TransMIL**: TransMIL处理所有Patches的自注意力，计算量大且非端到端；VINO处理局部Clip，参数共享，端到端。
- **VINO vs GNN-MIL**: GNN需要构建邻接图，VINO隐式地通过空间顺序构建了“图”（链式结构），无需显式建图。

### 3. 复现可行性
- **代码是否公开**：未说明。
- **方法描述是否完整**：较为完整，给出了公式和架构图。但Transformer的具体层数、Dropout率、Learning Rate schedule等超参数未在正文详细描述（仅在Implementation Details提到LR=1e-4, Weight Decay=1e-4, Adam optimizer）。
- **关键配置是否明确**：ResNet18作为Backbone明确。Clip长度$T$未明确给出具体数值，需通过实验猜测或查阅补充材料（如有）。
- **预计复现难点**：伪标签的计算细节（如是否需要温度缩放、Softmax还是Sigmoid）以及Bag和Instance Loss的加权比例。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：类特定Token的设计思路可用于其他多实例学习任务。
- **需要改造的设计**：对于非网格状数据（如分子图），需重新定义“Clip”的概念。
- **可能形成的新研究思路**：结合层级视频结构（Hierarchical Video Structure），即先在小尺度Clip内聚合，再在大尺度Clip间聚合，以捕捉长程依赖。

### 5. 阅读备注
- 论文中提到的“Video”并非真实的时间序列视频，而是基于空间坐标构造的合成序列，这一点在理解“Temporal”相关术语时需特别注意。
- 实验部分提到RTFM（传统视频异常检测方法）效果不佳，原因是病理视频帧之间没有自然的时间连续性，这反向证明了VINO中“位置对应时间”这一假设的合理性及其局限性（即仅捕捉局部，不捕捉跨远距离的全局时序）。
