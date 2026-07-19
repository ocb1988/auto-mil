# 50_SMMILE_MIL_Accurate spatial quantification in computational pathology with MIL 方法总结

> 证据说明：输入为完整论文全文（含正文及附录）。PDF提取内容完整，关键数学公式、符号定义、算法步骤及超参数均可从正文中确认。无缺失内容。

## 一、论文基本信息

- **论文标题**：Accurate spatial quantification in computational pathology with multiple instance learning
- **作者**：Zeyu Gao, Anyu Mao, Yuxing Dong, Jialun Wu, Jiashuai Liu, ChunBao Wang, Kai He, Tieliang Gong, Chen Li, Mireia Crispin-Ortuzar
- **发表年份**：2024 (Preprint on medRxiv)
- **会议/期刊**：medRxiv Preprint (未注明最终发表期刊/会议，但已提交同行评审前版本)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1101/2024.04.25.24306364
- **代码仓库**：https://github.com/ZeyuGaoAi/SMMILe（预印本文本当时声明将在正式发表后发布；该仓库现已公开）
- **研究任务**：全切片图像（WSI）分类与空间量化（Patch-level classification/spatial quantification）
- **数据模态**：数字病理切片（H&E染色），多实例学习（MIL）设置，仅使用患者级标签监督。

## 二、论文整体概述

### 1. 核心问题
现有的基于表示的多实例学习（RAMIL，如CLAM等）在WSI分类上表现优异，但其注意力图往往不精确，难以用于准确的**空间量化**（即识别具体的肿瘤区域或亚型位置）。而基于实例的MIL（IAMIL）虽然具有更好的空间定位潜力，但存在两个主要缺陷：1) WSI级别的分类性能通常低于RAMIL；2) 注意力分布极度偏斜，只关注高判别力实例，导致召回率低（漏检低判别力阳性区域）。

### 2. 整体方法
论文提出 **SMMILe** (Superpatch-based Measurable Multiple Instance Learning)。该方法基于IAMIL框架，旨在结合IAMIL的空间感知能力和RAMIL的分类性能。SMMILe通过以下机制解决IAMIL的局限：
1.  **NIC特征压缩与卷积层**：利用神经图像压缩（NIC）将实例嵌入重组为压缩WSI，并通过卷积层扩大局部感受野，增强特征表达。
2.  **一致性约束（Consistency Constraint）**：针对负样本袋，强制所有实例获得相同的注意力分数，防止负样本被错误赋予高注意力。
3.  **无参数实例Dropout**：根据实例得分动态丢弃高判别力实例，迫使模型关注低判别力实例，提升召回率。
4.  **去中心化实例采样（Delocalised Instance Sampling）**：基于超块（Superpatches）进行多轮随机采样，生成伪袋，作为数据增强并打破高判别力实例的主导地位。
5.  **MRF基实例细化网络**：训练一个辅助的多阶段细化网络，利用伪标签自训练，并结合马尔可夫随机场（MRF）能量约束，学习统一的实例级决策边界并保证空间平滑性。

### 3. 主要贡献
1.  理论上证明了RAMIL等价于Logit-based IAMIL，并推导了两者注意力分配差异的数学原因（激活函数性质导致IAMIL更倾向于给高判别力实例更高分数）。
2.  提出了SMMILe框架，包含五个新颖模块，同时实现了SOTA的WSI分类性能和卓越的空间量化能力。
3.  在8个数据集（涵盖6种癌症类型和3类任务：二元、多类、多标签）上进行了广泛评估，显著优于9种现有SOTA方法。

## 三、方法总结

### 方法 1：SMMILe 整体架构与核心模块

#### 1. 核心思想与解决的问题
- **目标问题**：解决传统MIL方法在WSI分类精度与空间定位精度之间的权衡问题，特别是解决IAML召回率低和RAMIL空间模糊的问题。
- **现有方法的局限**：RAMIL注意力分散且不准确；IAMIL注意力过于集中，忽略弱阳性区域，且Bag级预测不稳定。
- **核心思想**：保留IAMIL的“先映射后聚合”结构以获取实例级分数，引入NIC卷积增强局部上下文，并通过一系列注意力增强模块（一致性、Dropout、采样）和细化网络来平衡高/低判别力实例的关注度，最终实现高精度的Patch级预测。
- **创新点**：
    - 理论层面揭示了RAMIL与IAMIL注意力分配差异的本质。
    - 设计了参数无关的实例Dropout和基于SLIC超块的采样策略。
    - 引入了MRF约束的实例细化网络，统一不同WSI间的实例决策边界。

#### 2. 详细结构与数据流
- **输入**：WSI $X \in \mathbb{R}^{W \times H \times 3}$，患者级标签 $Y$。
- **处理流程**：
    1.  **切片与编码**：WSI切分为不重叠的Patches $\{x_1, ..., x_K\}$。使用预训练的ResNet-50（冻结）提取第3残差块的特征，经全局平均池化得到实例嵌入 $h_k \in \mathbb{R}^{1024}$。
    2.  **NIC压缩与卷积**：将嵌入按空间位置排列成压缩WSI $H_{nic}$，填充零值。使用 $3\times3$ 卷积核（Camelyon16用 $1\times1$）进行卷积 $cov(\cdot)$，得到压缩嵌入 $h'_k$。
    3.  **实例检测器与分类器**：
        -   检测器 $g(\cdot)$：Gated Attention机制，输出原始注意力分数 $z^c_k$，Softmax归一化为检测分数 $a^c_k$。
        -   分类器 $f(\cdot)$：线性层，输出Logits $l^c_k$，经Softmax/Sigmoid得到分类分数 $p^c_k$。
    4.  **综合注意力模块**：
        -   **一致性约束**：对负样本Bag，计算所有实例注意力分数的MSE损失。
        -   **实例Dropout**：对实例得分 $I^c_k = a^c_k \cdot p^c_k$ 进行Min-Max归一化，与随机数比较生成Mask，丢弃高分实例。
        -   **超块采样**：使用SLIC对压缩WSI进行超分割生成Superpatches。每轮从每个Superpatch采样一个实例组成伪Bag，重复T轮。
    5.  **实例细化网络**：
        -   基于主网络生成的实例得分，选取Top-$\theta$%作为正类伪标签，Bottom-$\theta$%作为负类伪标签。
        -   训练N个线性层 $v_n(\cdot)$ 进行自训练。
        -   应用MRF约束损失，惩罚相邻Superpatch间预测分数的差异。
    6.  **输出**：Bag级预测 $P_c$ 和 Patch级预测 $p^N_k$（来自最后一个细化层）。
- **输出**：WSI类别概率向量，以及每个Patch的类别概率向量。
- **模块在整体网络中的位置**：位于特征提取之后，Bag聚合之前（对于主网络）；细化网络并行运行，利用主网络的中间结果进行迭代优化。

#### 3. 数学公式

**Bag级预测 (Eq. 4):**
$$ P_c = \sum_{k=1}^{K} a^c_k \cdot p^c_k $$
其中 $a^c_k$ 是第 $k$ 个实例在第 $c$ 类的注意力分数，$p^c_k$ 是分类分数。

**一致性约束损失 (Eq. 10):**
$$ L_{cons} = \frac{1}{CK} \sum_{c=1}^{C} \sum_{i=1}^{K} (a^c_i - \bar{a}^c)^2 $$
其中 $\bar{a}^c = \frac{1}{K}\sum_{k=1}^{K} a^c_k$。仅应用于负样本Bag。

**实例Dropout Bag级预测 (Eq. 11):**
$$ P^{dp}_c = \sum_{k=1}^{K} O^c_k \cdot I^c_k $$
其中 $I^c_k = a^c_k \cdot p^c_k$，$O^c_k = [\check{I}^c_k < \eta^c_k]$ 是掩码，$\check{I}$ 是归一化后的实例得分，$\eta$ 是 $[0,1]$ 间的随机数。

**超块采样Bag级预测 (Eq. 12 & 13):**
单轮采样预测：
$$ P^{sp}_{c,t} = \sum_{s=1}^{S} \tilde{I}^c_{s,t} $$
带Dropout的采样预测：
$$ P^{sdp}_{c,t} = \sum_{s=1}^{S} O^c_s \cdot \tilde{I}^c_{s,t} $$

**总分类损失 (Eq. 14):**
$$ L_{cls} = \frac{1}{C} \sum_{c=1}^{C} \left( BCE(P_c, Y_c) + BCE(P^{dp}_c, Y_c) \right) + \frac{1}{CT} \sum_{t=1}^{T} \sum_{c=1}^{C} \left( BCE(P^{sp}_{c,t}, Y_c) + BCE(P^{sdp}_{c,t}, Y_c) \right) $$

**实例细化损失 (Eq. 15):**
$$ L_{ref} = \frac{1}{NJ} \sum_{n=1}^{N} \sum_{j=1}^{J} CE(p^n_j, \breve{y}^n_j) $$
其中 $p^n_j$ 是第 $n$ 个细化层的预测，$\breve{y}^n_j$ 是伪标签。

**MRF约束损失 (Eq. 16):**
$$ L_{mrf} = \frac{1}{N} \sum_{n=1}^{N} \left( \lambda_1 \sum_{i=1}^{|SP_s|} \| p^n_i - \bar{p}^n_s \|^2 + \lambda_2 \sum_{m=1}^{M_s} \| \bar{p}^n_{s,m} - \bar{p}^n_s \|^2 \right) $$
第一项为Superpatch内部的一阶能量（平滑），第二项为相邻Superpatch间的二阶能量。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | WSI Patch $x_k$ | $D \times D \times 3$ | 例如 $512 \times 512 \times 3$ 或 $1024 \times 1024 \times 3$ |
| 实例嵌入 $h_k$ | $\mathbb{R}^{1024}$ | ResNet-50第3残差块 GAP后 |
| 压缩WSI $H_{nic}$ | $\frac{W}{D} \times \frac{H}{D} \times 1024$ | 稀疏矩阵，非实例位置补0 |
| 卷积后嵌入 $h'_k$ | $\mathbb{R}^{128}$ | 经过 $3\times3$ Conv (128 kernels) |
| 注意力分数 $a^c_k$ | Scalar ($\in [0,1]$) | Softmax over instances |
| 分类Logits $l^c_k$ | Scalar | Linear projection to C classes |
| 分类分数 $p^c_k$ | Scalar ($\in [0,1]$) | Softmax/Sigmoid |
| 实例得分 $I^c_k$ | Scalar | $a^c_k \cdot p^c_k$ |
| 细化层输出 $p^n_j$ | Vector ($C+1$) | 包含负类，Softmax输出概率 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic

class SMMILe(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, num_classes=C, 
                 conv_kernel_size=3, superpatch_size=(5,5), sampling_rounds=10, 
                 refinement_layers=3, theta_percent=10):
        super().__init__()
        # 1. Feature Encoder (Frozen ResNet-50 part implied)
        self.encoder = get_resnet50_features() 
        
        # 2. NIC Compression & Convolution
        self.conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=conv_kernel_size, padding=1)
        
        # 3. Instance Detector (Attention)
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes) # Raw attention logits z
        )
        
        # 4. Instance Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes) # Logits l
        
        # 5. Refinement Network (N linear layers)
        self.refiners = nn.ModuleList([
            nn.Linear(num_classes, num_classes + 1) for _ in range(refinement_layers)
        ])
        
        self.num_classes = num_classes
        self.sampling_rounds = sampling_rounds
        self.theta_percent = theta_percent
        self.superpatch_size = superpatch_size

    def forward(self, patches, wsi_coords, bag_label, is_negative_bag=False):
        """
        patches: [K, D, D, 3]
        wsi_coords: [K, 2] (row, col indices in the grid)
        bag_label: [C] or scalar
        """
        K = patches.size(0)
        
        # Step 1: Embedding
        h = self.encoder(patches) # [K, 1024]
        
        # Step 2: NIC Compression & Conv
        # Create sparse grid tensor based on wsi_coords
        H_nic = create_sparse_wsi(h, wsi_coords, grid_size=(W//D, H//D)) 
        H_conv = self.conv(H_nic) # [Grid_H, Grid_W, 128]
        # Gather back to instance order
        h_prime = gather_from_grid(H_conv, wsi_coords) # [K, 128]
        
        # Step 3: Detection & Classification Scores
        z = self.detector(h_prime) # [K, C] raw attention logits
        a = F.softmax(z, dim=0)    # [K, C] attention scores per class
        l = self.classifier(h_prime) # [K, C] classification logits
        
        if self.num_classes == 1: # Binary case handling
             p = torch.sigmoid(l)
        else:
             p = F.softmax(l, dim=1) # [K, C] classification scores
            
        # Instance Score I = a * p
        I = a * p # [K, C]
        
        # --- Main Bag Prediction ---
        P_main = torch.sum(I, dim=0) # [C]
        
        # --- Augmentation 1: Consistency Loss (Negative Bags Only) ---
        L_cons = 0
        if is_negative_bag:
            mean_a = torch.mean(a, dim=0)
            L_cons = torch.mean((a - mean_a)**2)
            
        # --- Augmentation 2: Parameter-Free Dropout ---
        # Normalize I per class
        I_min = I.min(dim=0, keepdim=True)[0]
        I_max = I.max(dim=0, keepdim=True)[0]
        I_norm = (I - I_min) / (I_max - I_min + 1e-8)
        
        # Random floats
        eta = torch.rand_like(I_norm)
        mask = (I_norm < eta).float()
        
        I_dropped = I * mask
        P_dropout = torch.sum(I_dropped, dim=0)
        
        # --- Augmentation 3: Superpatch Sampling ---
        # Generate Superpatches using SLIC on H_conv (or coordinates)
        # Note: In practice, SLIC is run on the compressed feature map or coordinate space
        superpatches = generate_superpatches(wsi_coords, self.superpatch_size)
        
        P_sampling_list = []
        P_sampling_dropout_list = []
        
        for t in range(self.sampling_rounds):
            sampled_indices = []
            for sp_id in range(len(superpatches)):
                # Sample one index from this superpatch
                idx = random.choice(superpatches[sp_id])
                sampled_indices.append(idx)
            
            sampled_I = I[:, sampled_indices] # [S, C]
            P_sp = torch.sum(sampled_I, dim=0) # [C]
            P_sampling_list.append(P_sp)
            
            # Apply dropout mask logic again for sampled instances if needed
            # Simplified: assume similar logic applies
            P_sampling_dropout_list.append(P_sp) 
            
        # --- Loss Calculation ---
        loss_cls = bce_loss(P_main, bag_label) + bce_loss(P_dropout, bag_label)
        for ps in P_sampling_list:
            loss_cls += bce_loss(ps, bag_label)
        for psd in P_sampling_dropout_list:
            loss_cls += bce_loss(psd, bag_label)
            
        # --- Refinement Network ---
        # Online Label Generation
        pseudo_labels = generate_pseudo_labels(I, bag_label, self.theta_percent)
        
        L_ref = 0
        L_mrf = 0
        
        current_scores = I # Start with main instance scores or logits? Text says "compressed embeddings... fed into vn"
        # Actually text says: "compressed embeddings of all instances... fed into vn... yielding prediction scores pn"
        # So we pass h_prime through refiners sequentially
        
        h_for_ref = h_prime # [K, 128] -> needs to match refiner input? 
        # Text: "Each layer is implemented with (C+1) hidden nodes". 
        # Wait, Refiner takes embeddings? Or Scores?
        # Text: "vn(h'k) is a (C+1)-dimensional vector". So Refiner takes h'.
        
        current_h = h_prime 
        
        for n, v_n in enumerate(self.refiners):
            # Predict probabilities for C+1 classes
            logits_ref = v_n(current_h) # [K, C+1]
            probs_ref = F.softmax(logits_ref, dim=1) # [K, C+1]
            
            # Update pseudo labels for next stage based on current probs
            # (Self-training strategy)
            if n > 0:
                 # Re-select top/bottom based on previous refinement output
                 pseudo_labels = update_pseudo_labels(probs_ref, bag_label, self.theta_percent)
                 
            # Compute Refinement Loss
            # Map bag_label to C+1 format (add negative class indicator)
            target_one_hot = label_to_c_plus_1(bag_label, self.num_classes)
            # Select only labeled instances for loss
            selected_mask = get_selected_instances(pseudo_labels)
            L_ref += cross_entropy(probs_ref[selected_mask], pseudo_labels[selected_mask])
            
            # MRF Constraint
            # Calculate mean score per superpatch
            sp_means = compute_superpatch_means(probs_ref, superpatches)
            # First order energy (within SP)
            # Second order energy (between adjacent SPs)
            L_mrf += compute_mrf_loss(sp_means, superpatches, lambda1=0.8, lambda2=0.2)
            
            # Prepare for next layer? Text implies parallel training or sequential?
            # "concurrently trained... each epoch". 
            # But self-training uses output of vn to train v(n+1).
            # For inference, we use the last layer vN.
            
        total_loss = loss_cls + L_cons + L_ref + L_mrf
        
        # Inference outputs
        final_patch_preds = probs_ref[:, :-1] # Exclude negative class prob if needed, or argmax
        final_bag_pred = P_main
        
        return total_loss, final_bag_pred, final_patch_preds
```

#### 6. 实现提示
- **关键网络组件**：
    - `Conv2d` for NIC compression (Kernel size 3x3 default, 1x1 for Camelyon16).
    - `Gated Attention` for detector (Linear-Tanh-Linear-Tanh-Linear).
    - `SLIC` segmentation for Superpatches (`skimage.segmentation.slic`).
- **重要超参数**：
    - Superpatch initial size: 5x5 for multi-class, 3x3 for others.
    - Sampling rounds $T$: 10.
    - MRF weights $\lambda_1=0.8, \lambda_2=0.2$.
    - Refinement layers $N$: 3.
    - Selection rate $\theta$: 10%.
    - Learning Rate: $2 \times 10^{-5}$ (Adam).
- **归一化/激活方式**：
    - Attention: Softmax over instances.
    - Classification: Softmax (Multi-class) or Sigmoid (Binary/Multi-label).
    - Dropout: Min-Max normalization on instance scores.
- **维度对齐方式**：
    - NIC压缩时，需根据Patch坐标构建稀疏张量，卷积后再根据坐标gather回实例顺序。
- **实现注意事项**：
    - 两阶段训练：第一阶段仅训练主网络（Cons + Cls Loss），第二阶段加入Refinement网络（Ref + MRF Loss）。
    - 伪标签生成需严格遵循Top-$\theta$和Bottom-$\theta$策略，且负样本选择基于跨类别的平均得分。

#### 7. 计算与资源开销
- **理论计算复杂度**：由于引入了SLIC聚类、多轮采样（T=10）和多层细化网络，推理和训练时间显著高于标准CLAM。采样增加了前向传播的次数。
- **参数量**：主要增加在于Refinement Network（3个线性层，每层约 $(C+1)\times 128$ 参数，相对较小）和Detector/Classifier。NIC卷积层参数较少。
- **FLOPs/MACs**：未明确提供具体数值，但相比单路径MIL，因采样和细化网络，计算量增加。
- **显存开销**：需存储压缩WSI特征图和多个采样批次的中间状态。
- **推理速度**：较慢，因为需要运行Refinement网络的所有层级以获得最终Patch预测，且SLIC预处理耗时。
- **论文是否提供效率对比**：未在正文表格中直接列出FPS或FLOPs对比，主要对比Accuracy/AUC。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：计算病理学中的WSI分类及空间表型量化（转移灶检测、亚型预测、分级）。
- **可迁移到的任务/数据集**：任何需要弱监督下细粒度定位的多实例学习任务，如医学影像中的病灶分割、遥感图像中的地物分类。
- **迁移所需调整**：调整SLIC参数以适应不同分辨率；调整Refinement网络的层数和选择比例。
- **适用条件**：拥有大量WSI数据，且希望在不增加标注成本的情况下获得高精度定位。
- **潜在限制**：对小数据集（如Prostate，153张WSI）效果受限，因为Refinement网络需要足够的实例进行伪标签训练。

#### 9. 实验与消融证据
- **主要性能结果**：在8个数据集上，SMMILe在WSI分类AUC上达到或接近SOTA，在空间量化AUC上大幅领先（如Lung数据集空间AUC 85.84%，远超第二名的61.79%）。
- **相对基线的提升**：空间量化方面提升巨大（10%-20%+）。
- **相关消融实验**：
    - 移除所有模块（Index 0）仍有不错基准。
    - Cons提升负样本准确性。
    - InD和InS提升召回率。
    - InR（Refinement）对Precision/Recall/F1提升最大。
    - MRF进一步提升空间平滑性。
- **作者结论**：所有模块均对空间量化有显著贡献，尤其是Refinement网络。
- **证据是否充分**：充分，涵盖了多种癌症类型和任务类型，且有详细的可视化对比。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 理论推导RAMIL/IAMIL差异，提出独特的无参Dropout和SLIC采样，结合MRF细化。 |
| 技术可行性 | 高 | 基于成熟MIL框架，模块均为标准操作，易于集成。 |
| 实现难度 | 中 | 需处理稀疏张量、SLIC聚类逻辑及两阶段训练流程，细节较多。 |
| 架构相关性 | 高 | 专为WSI的大规模实例特性设计。 |
| 可迁移性 | 高 | 核心思想（注意力平衡、空间平滑）适用于其他MIL场景。 |
| 计算成本 | 中 | 比标准MIL高，但在可接受范围内。 |

#### 11. 一句话总结
SMMILe通过理论分析揭示IAMIL优势，并利用NIC卷积、无参实例Dropout、超块采样及MRF细化网络，成功解决了IAMIL召回率低的问题，实现了WSI分类与高精度空间量化的双重突破。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **理论洞察**：通过梯度下降过程证明RAMIL与IAMIL注意力分配差异源于激活函数的凸凹性，这为理解MIL的可解释性提供了新的视角。
- **无参数实例Dropout**：一种简单有效的正则化手段，通过动态屏蔽高置信度实例来强迫模型学习难例，无需手动调节Dropout率。
- **MRF约束的细化网络**：将空间拓扑信息（通过Superpatches近似）融入实例级分类器的训练中，有效提升了预测的空间连贯性。

### 2. 方法之间的关系
- **基础**：建立在IAMIL（Instance-based MIL）之上。
- **增强**：NIC卷积增强了局部上下文（类似CNN的感受野）；Consistency Constraint修正了负样本的注意力偏差；Dropout和Sampling构成了Bag级的数据增强策略；Refinement Network则是后处理的校正模块。

### 3. 复现可行性
- **代码是否公开**：否（承诺发表后公开）。
- **方法描述是否完整**：是。提供了详细的公式、架构图、超参数设置和训练阶段划分。
- **关键配置是否明确**：是。包括卷积核大小、SLIC参数、采样轮数、学习率等。
- **预计复现难点**：
    1.  **NIC压缩的实现**：如何高效地将离散Patch嵌入重组为连续网格并进行卷积，再取回，需要高效的PyTorch索引操作。
    2.  **SLIC超块生成**：需要在特征空间或坐标空间正确执行SLIC，并确保后续采样和MRF计算能正确映射回Patch索引。
    3.  **伪标签的动态更新**：在多阶段细化中，伪标签的生成逻辑需严格对应每一层的输出。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：无参数实例Dropout和基于Superpatch的采样策略可以很容易地移植到其他MIL变体中以提高鲁棒性。
- **需要改造的设计**：MRF约束依赖于Superpatch的空间邻接关系，若应用于非网格化数据（如图神经网络MIL），需重新定义邻接能量项。
- **可能形成的新研究思路**：探索其他形式的空间平滑约束（如CRF），或将此框架应用于多模态病理数据（结合基因组学）的空间关联分析。

### 5. 阅读备注
- 论文强调了多标签分类任务的扩展，这是许多早期MIL工作所缺乏的。
- 实验部分特别指出了在小样本数据集（如Prostate）上Refinement网络效果受限的现象，提示在实际应用中需注意数据规模。
- 可视化结果（Fig. 4）直观展示了SMMILe在复杂多标签场景下的优越性，是评估其空间量化能力的关键证据。
