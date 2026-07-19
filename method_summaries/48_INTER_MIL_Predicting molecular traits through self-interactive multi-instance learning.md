# 48_INTER_MIL_Predicting molecular traits through self-interactive multi-instance learning 方法总结

> 证据说明：输入为同一作者团队 2025 年 Medical Image Analysis 期刊扩展版全文（17页），不是综述引用的 MICCAI 2022 原始章节；该扩展版完整覆盖 Inter-MIL / adInter-MIL，并增加了实验与分析。PDF 提取包含算法、公式、表格及实验细节，无明显页面缺失。

## 一、论文基本信息

- **论文标题**：Self-interactive learning: Fusion and evolution of multi-scale histomorphology features for molecular traits prediction in computational pathology
- **作者**：Yang Hu, Korsuk Sirinukunwattana, Bin Li, Kezia Gaitskell, Enric Domingo, Willem Bonnaffé, Marta Wojciechowska, Ruby Wood, Nasullah Khalid Alham, Stefano Malacrino, Dan J Woodcock, Clare Verrill, Ahmed Ahmed, Jens Rittscher
- **发表年份**：2025 (Online Jan 2025, Received Dec 2024)
- **会议/期刊**：Medical Image Analysis (MedIA)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1016/j.media.2024.103437
- **代码仓库**：https://github.com/superhy/LCSB-MIL
- **研究任务**：基于全切片图像（WSI）的癌症分子亚型预测（弱监督多实例学习）
- **数据模态**：H&E染色组织病理学全切片图像（WSI），标签为分子特征状态（如EMT, KRAS, EGFR, HER2）

## 二、论文整体概述

### 1. 核心问题
在计算病理学中，从H&E图像预测分子性状面临两大挑战：(1) 缺乏细粒度的tile-level标注，通常只有slide-level标签；(2) 分子亚型在形态学上差异细微，且不同尺度（细胞级到组织级）的特征均可能具有判别力。现有的MIL方法通常使用预训练的、与任务无关的tile编码器，限制了模型捕捉细粒度判别特征的能力，且在小样本数据集上表现不佳。

### 2. 整体方法
提出 **Inter-MIL**（Self-interactive Multi-Instance Learning）框架。该方法通过迭代优化策略，实现全局slide-level特征与细粒度tile-level特征的自交互。主要包含两个可学习模块：**Gated-AttPool聚合器**（Module-1）和**可训练的tile级特征编码器**（Module-2）。
训练流程为交替优化：
1. 固定编码器，训练聚合器以获取每个tile的注意力分数。
2. 根据注意力分数构建tile级特征池（正样本+负样本），微调编码器以学习更具判别力的细粒度特征。
3. 使用更新后的编码器重新生成tile嵌入，进入下一轮迭代，直到收敛。
此外，还引入了**对比预训练**模块用于初始化聚合器，以及可选的**对抗训练**模块用于抑制低注意力噪声tile的影响。

### 3. 主要贡献
1. 提出了Inter-MIL框架，通过局部与大规模特征间的通信进行迭代优化，提高了小样本病理数据集上的MIL学习效率。
2. 实现了多尺度代表性特征的搜索，既识别细胞病理特征，又改进粗粒度特征搜索。
3. 重塑了tile级特征空间，使不同分子亚型的视觉特征更具区分度。
4. 在四个挑战性分子亚型任务（卵巢癌EMT, 结直肠癌KRAS, 肺癌EGFR/KRAS, 乳腺癌HER2）中验证了方法的鲁棒性和优越性，并在外部队列FOCUS上进行了验证。

## 三、方法总结

### 方法 1：Inter-MIL 自交互迭代优化框架

#### 1. 核心思想与解决的问题
- **目标问题**：解决传统MIL方法中tile编码器仅作为静态特征提取器、无法针对下游分类任务进行端到端优化的问题，特别是在小样本场景下。
- **现有方法的局限**：大多数MIL方法依赖ImageNet或自监督预训练的编码器，这些编码器对特定的分子亚型任务可能是“任务无关”的，导致聚合器难以感知细粒度信息，需要大量数据补偿。
- **核心思想**：模拟病理学家调整显微镜倍率观察不同尺度特征的过程。通过“伪标签传播”策略，利用slide-level标签指导tile编码器的优化。高注意力tile被视为具有代表性的正样本，低注意力tile被视为噪声负样本，通过交替训练聚合器和编码器，形成正反馈循环。
- **创新点**：
    - 引入自交互机制，联合优化tile编码器和slide聚合器。
    - 动态构建tile级训练池，而非使用所有tiles或随机采样。
    - 提供细粒度的tile级解释性（不仅是注意力权重，还有伪分类分数）。

#### 2. 详细结构与数据流
- **输入**：
    - WSI被切割为 $L$ 个不重叠的tile $\{x_i\}_{i=1}^L$。
    - Slide-level标签 $y_s$。
    - 初始编码器 $f_{cnn}$（通常为ImageNet预训练的ResNet-18）。
- **处理流程**：
    1. **初始化**：使用预训练编码器提取初始tile嵌入 $E^{t-1}$。
    2. **聚合器训练 (Module-1)**：固定编码器，使用Gated-AttPool聚合器对当前嵌入进行训练，输出slide预测 $y_{cls}$ 和每个tile的注意力分数 $a_{att}(E_i)$。
    3. **构建Tile池 (Algorithm 1)**：
       - 按注意力分数降序排列tiles。
       - 从Top-K高注意力tiles中随机采样 $k_1$ 个作为正样本集 $S_{top}$。
       - 从剩余tiles中随机采样 $k_2$ 个作为补充样本集 $S_{sup}$。
       - 构造正样本池 $S_{pos} = S_{top} \cup S_{sup}$。
       - （可选）从Bottom-N低注意力tiles中随机采样 $n$ 个作为负样本集 $S_{neg}$。
    4. **编码器微调 (Module-2)**：
       - 使用 $S_{pos}$ 中的tiles及其继承的slide标签 $y_{x_i}$ 训练编码器，最小化交叉熵损失。
       - （可选）如果存在 $S_{neg}$，使用对抗训练策略（梯度反转层）更新编码器，使其尽可能错误分类这些噪声tile。
    5. **更新嵌入**：使用微调后的编码器 $f_{cnn}^t$ 重新提取所有tiles的嵌入 $E^t$。
    6. **迭代**：重复步骤2-5，直到达到最大轮数或损失收敛。
- **输出**：最终的slide预测结果 $Y_{cls}$ 和经过优化的编码器/聚合器参数。
- **模块在整体网络中的位置**：核心训练循环。
- **与其他模块的连接方式**：
    - **对比预训练模块**：在Inter-MIL第一轮迭代前，对聚合器进行对比学习预训练，加速收敛。
    - **对抗训练模块**：作为Inter-MIL的一个变体（adInter-MIL），在编码器更新阶段加入负样本惩罚项。

#### 3. 数学公式

**Slide-level Prediction (Aggregator):**
$$ y_{cls} = \text{softmax}\left(f_{cls}\left(\sum_{i=1}^{L} a_{att}(E_i) \cdot E_i\right)\right) $$
其中 $a_{att}(E_i) \in [0,1]$ 是第 $i$ 个tile的注意力分数，$\sum a_{att}(E_i) = 1$。

**Aggregator Update:**
$$ \theta_{f_{cls}}^t = \theta_{f_{cls}}^{t-1} + \nabla \mathcal{L}_{E \in S_{train}}(y_E, y_{cls}) $$
其中 $\mathcal{L}$ 为加权交叉熵损失。

**Tile Embedding Update:**
$$ E_i^t = f_{cnn}^t(x_i) $$

**Positive Tile Pool Construction:**
$$ S_{pos} = S_{top} \cup S_{sup} $$
$$ S_{top} \subseteq \{x_{d_i}\}_{i=1}^K, \quad S_{sup} \subseteq \{x_{d_j}\}_{j=K+1}^L $$
其中 $K = 0.05 \times L$。

**Negative Tile Pool Construction:**
$$ S_{neg} = \{x_i\}_{i=1}^n \subseteq \{x_{d_i}\}_{i=(L-N)+1}^L $$
其中 $N = 0.2 \times L$。

**Encoder Update (Standard Inter-MIL):**
$$ \theta_{f_{cnn}}^t = \theta_{f_{cnn}}^{t-1} + \gamma \cdot \nabla \mathcal{L}_{x_i \in S_{pos}}(y_{x_i}, f_{cnn}^{t-1}(x_i)) $$

**Encoder Update (Adversarial adInter-MIL):**
$$ \theta_{f_{res}}^t = \theta_{f_{res}}^{t-1} + \gamma \cdot \nabla \mathcal{L}_{x_i \in S_{pos}}(y_{x_i}, f_{res}^{t-1}(x_i)) - \gamma_{neg} \cdot \nabla \mathcal{L}_{x_j \in S_{neg}}(y_{x_j}, f_{res}^{t-1}(x_j)) $$
注意：这里使用了梯度反转的思想，即对负样本的损失取负梯度（或反向传播时翻转符号），迫使编码器忽略这些样本。

**Contrastive Pre-training Loss (Aggregator):**
$$ \mathcal{L}_{pt} = -\log \frac{\exp(|||E_q \cdot E_k - y_c(q \cdot k)||| / \tau)}{\sum_{i \in C} \exp(||E_q \cdot E_k - y_c(i)|| / \tau)} $$
其中 $E_q$ 是查询bag，$E_k$ 是关键bag（同类为正，异类为负），$\tau=1$。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | WSI Tiles $\{x_i\}$ | $(L, 3, 256, 256)$ | $L$为tile数量，分辨率256x256 |
| 中间 | Tile Embeddings $E_i$ | $(D_{emb}=512)$ | ResNet-18输出维度 |
| 中间 | Attention Scores $a_{att}$ | $(L, 1)$ | 归一化后的注意力权重 |
| 中间 | Slide Embedding | $(D_{hidden})$ | 聚合后的全局特征 |
| 输出 | Slide Prediction $y_{cls}$ | $(C=2)$ | 二分类概率分布 |
| 超参 | $K$ | $0.05 \times L$ | Top-K范围 |
| 超参 | $N$ | $0.2 \times L$ | Bottom-N范围 |
| 超参 | $k_1$ | 50 | Top采样数 |
| 超参 | $k_2$ | $0.4 \times k_1$ | Sup采样数 |
| 超参 | $n$ | $0.2 \times k_1$ | Neg采样数 |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.optim as optim

class InterMIL:
    def __init__(self, encoder, aggregator, hyperparams):
        self.encoder = encoder # e.g., ResNet-18
        self.aggregator = aggregator # Gated-AttPool
        self.k1 = hyperparams['k1']
        self.k2 = int(0.4 * self.k1)
        self.n_neg = int(0.2 * self.k1)
        self.lr_enc = 0.0001
        self.lr_agg = 0.0001
        self.gamma_neg = 1.0 # Adversarial weight if used
        
    def train_round(self, slides_data, round_idx, max_epochs_agg, max_epochs_enc):
        """
        slides_data: List of dicts, each containing 'tiles' (tensor), 'label' (int)
        """
        # Step 1: Train Aggregator (Module 1)
        # Freeze encoder gradients temporarily or just use current embeddings
        # In practice, we re-extract embeddings at start of round using current encoder
        
        optimizer_agg = optim.Adam(self.aggregator.parameters(), lr=self.lr_agg)
        
        # Initial embedding extraction
        all_embeddings = []
        for slide in slides_data:
            tiles = slide['tiles'] # Shape: (L, 3, 256, 256)
            with torch.no_grad():
                feats = self.encoder(tiles) # Shape: (L, 512)
            all_embeddings.append(feats)
            
        # Train Aggregator for max_epochs_agg epochs
        for ep in range(max_epochs_agg):
            loss_agg = 0
            for i, slide in enumerate(slides_data):
                emb = all_embeddings[i]
                logits = self.aggregator(emb) # Returns slide-level logit
                label = slide['label']
                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), label.unsqueeze(0))
                
                optimizer_agg.zero_grad()
                loss.backward()
                optimizer_agg.step()
                loss_agg += loss.item()
            
            # Check convergence or early stop logic here if needed
            
        # Get attention scores from the trained aggregator
        attention_scores_list = []
        for emb in all_embeddings:
            # Assuming aggregator returns attention weights as part of forward pass or via a helper
            # Ilse et al. AttPool usually computes attention internally
            # Here we assume we can get attention weights a_att
            a_att = self.aggregator.get_attention_weights(emb) 
            attention_scores_list.append(a_att)

        # Step 2: Construct Tile Pools (Algorithm 1)
        pos_tiles_batch = []
        pos_labels_batch = []
        neg_tiles_batch = []
        neg_labels_batch = [] # For adversarial training, labels might be dummy or inverted
        
        for i, slide in enumerate(slides_data):
            tiles = slide['tiles']
            scores = attention_scores_list[i].squeeze() # Shape: (L,)
            
            # Sort indices by score descending
            sorted_indices = torch.argsort(scores, descending=True)
            sorted_tiles = tiles[sorted_indices]
            
            L = len(tiles)
            K = int(0.05 * L)
            N = int(0.2 * L)
            
            # Top K tiles
            top_k_tiles = sorted_tiles[:K]
            # Sample k1 from top K
            if K >= self.k1:
                idx_top = torch.randperm(K)[:self.k1]
                s_top = top_k_tiles[idx_top]
            else:
                s_top = top_k_tiles # Fallback if K < k1
                
            # Remaining tiles
            remaining = sorted_tiles[K:]
            # Sample k2 from remaining
            rem_len = len(remaining)
            if rem_len >= self.k2:
                idx_sup = torch.randperm(rem_len)[:self.k2]
                s_sup = remaining[idx_sup]
            else:
                s_sup = remaining
                
            # Positive pool
            s_pos = torch.cat([s_top, s_sup], dim=0)
            label = slide['label']
            pos_tiles_batch.append(s_pos)
            pos_labels_batch.append(torch.full((s_pos.shape[0],), label))
            
            # Negative pool (Optional for adInter-MIL)
            if N > 0 and L - N > 0:
                bottom_n_tiles = sorted_tiles[-N:]
                if N >= self.n_neg:
                    idx_neg = torch.randperm(N)[:self.n_neg]
                    s_neg = bottom_n_tiles[idx_neg]
                else:
                    s_neg = bottom_n_tiles
                    
                neg_tiles_batch.append(s_neg)
                # Labels for negative tiles are same as slide label, but gradient is reversed
        
        # Stack batches
        pos_tiles_batch = torch.cat(pos_tiles_batch, dim=0)
        pos_labels_batch = torch.cat(pos_labels_batch, dim=0).long()
        
        # Step 3: Train Encoder (Module 2)
        optimizer_enc = optim.Adam(self.encoder.parameters(), lr=self.lr_enc)
        
        # Forward pass on positive tiles
        pos_feats = self.encoder(pos_tiles_batch)
        # We need a classifier head for tile-level prediction to compute loss
        # The paper implies using the slide label as pseudo-label for tiles
        # Usually, this requires a tile-level classifier head attached to encoder
        # Or simply minimizing distance to class centroids? 
        # Paper Eq 6: L_xi(y_xi, f_cnn(xi)). This implies a tile-level classification head exists.
        # Let's assume a simple FC layer for tile classification
        tile_logits = self.tile_classifier_head(pos_feats) 
        loss_pos = nn.CrossEntropyLoss()(tile_logits, pos_labels_batch)
        
        loss_total = loss_pos
        
        if neg_tiles_batch:
            neg_tiles_batch = torch.cat(neg_tiles_batch, dim=0)
            neg_feats = self.encoder(neg_tiles_batch)
            neg_logits = self.tile_classifier_head(neg_feats)
            neg_labels = torch.full((neg_tiles_batch.shape[0],), slide_label_dummy) # Same label structure
            
            loss_neg = nn.CrossEntropyLoss()(neg_logits, neg_labels)
            
            # Adversarial: subtract gradient of negative loss
            # In PyTorch, this is done by detaching or manual gradient manipulation
            # Simplified: loss_total = loss_pos - gamma_neg * loss_neg
            # Note: Standard implementation uses Gradient Reversal Layer (GRL)
            loss_total = loss_pos - self.gamma_neg * loss_neg
            
        optimizer_enc.zero_grad()
        loss_total.backward()
        optimizer_enc.step()
        
        return loss_agg, loss_total

    def fit(self, slides_data, max_rounds=5, threshold_loss=None):
        for t in range(max_rounds):
            l_agg, l_enc = self.train_round(slides_data, t)
            if threshold_loss and l_agg < threshold_loss:
                break
```

#### 6. 实现提示
- **关键网络组件**：
    - **Encoder**: ResNet-18 (ImageNet pretrained)，移除最后的全连接层，保留512维输出。
    - **Aggregator**: Gated-AttPool (Lu et al., 2021)。包含一个线性层+Tanh+Linear+Sigmoid结构来计算注意力，然后加权求和，再接一个分类头。
    - **Tile Classifier Head**: 在编码器后需要一个小型MLP将512维映射到类别数，用于计算tile级的伪标签损失。
- **重要超参数**：
    - $k_1=50$: 从Top-K中选出的正样本数。
    - $k_2$: 补充样本数，约为 $k_1$ 的40%。
    - $n$: 负样本数，约为 $k_1$ 的20%。
    - $K$: Top-K范围，为总tile数的5%。
    - $N$: Bottom-N范围，为总tile数的20%。
    - 学习率：Adam, 0.0001。
- **归一化/激活方式**：
    - Attention权重使用Softmax归一化。
    - Gated-AttPool中使用Tanh和Sigmoid激活。
- **维度对齐方式**：
    - Tile嵌入维度固定为512。
    - 注意力分数标量化后用于排序。
- **实现注意事项**：
    - **对抗训练实现**：PyTorch中没有直接的“负梯度”操作符。通常使用 `torch.autograd.grad` 手动计算梯度并乘以-1，或者使用Gradient Reversal Layer (GRL) 插件。论文提到使用 "back propagating the negative gradients"，这对应于Domain Adversarial Training的思路。
    - **伪标签**：Tile级的标签直接继承自其所属Slide的标签。这是一种强假设，但通过迭代优化和高注意力筛选来缓解噪声。
    - **对比预训练**：需要在主循环前单独运行。需要构建Query Bag和Key Bag。

#### 7. 计算与资源开销
- **理论计算复杂度**：每轮迭代包括一次完整的Aggregator训练（涉及所有Tiles的Attention计算）和一次Encoder微调（涉及采样的少量Tiles）。由于Encoder微调只涉及几百个Tiles（$k_1+k_2+n \approx 100$左右），相比全图训练大大降低了计算量。
- **参数量**：取决于Encoder（ResNet-18约11M）和Aggregator（较小）。
- **FLOPs/MACs**：未明确给出，但相比端到端训练所有Tiles，显著降低。
- **显存开销**：较低，因为Encoder微调时Batch Size仅为采样的Tiles数量（~100-200），而Aggregator训练时Batch Size为Slides数量（如8-128）。
- **推理速度**：测试阶段仅需一次前向传播（Encoder + Aggregator），速度与标准MIL相当。
- **论文是否提供效率对比**：提供了收敛速度的对比（图3-b, d），显示预训练和Inter-MIL能加速收敛，但未直接对比FLOPs。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：小样本、弱监督下的癌症分子亚型预测（EMT, KRAS, EGFR, HER2）。
- **可迁移到的任务/数据集**：其他需要WSI分析的弱监督任务，如肿瘤分级、生存分析、免疫分型等。也可迁移到其他医学影像的多实例学习任务。
- **迁移所需调整**：
    - 调整Encoder backbone以适应不同分辨率或模态。
    - 调整超参数 $k_1, K$ 等以适应不同数量的Tiles。
    - 若任务不是二分类，需调整分类头。
- **适用条件**：适合训练数据较少（<100 slides）的场景，此时预训练编码器的泛化能力不足，需要任务特定的微调。
- **潜在限制**：对超参数敏感（如 $k_1$）；依赖第一轮的注意力质量，若首轮失败可能导致后续偏差。

#### 9. 实验与消融证据
- **主要性能结果**：
    - OV-EMT: AUC 77.00% (PT-adInter-MIL) vs Baseline 62.46%。
    - COLU-KRAS: AUC 71.38% (PT-adInter-MIL) vs Baseline 59.94%。
    - LU-EGFR: AUC 71.33% (PT-adInter-MIL) vs Baseline 64.15%。
    - BR-HER2: AUC 64.02% (PT-adInter-MIL) vs Baseline 54.84%。
- **相对基线的提升**：在所有任务上均显著优于Gated-AttPool基线，平均提升超过10% AUC。
- **相关消融实验**：
    - **Inter-MIL vs Inter-MIL-b**: 证明随机补充tile的重要性。
    - **adInter-MIL vs Inter-MIL**: 证明对抗训练在去噪方面的作用（尤其在噪声较多的数据上）。
    - **PT (Pre-training) 的影响**: 证明对比预训练能加速收敛并略微提升性能。
    - **Hyperparameter $k_1$ 敏感性**: 图7显示性能随 $k_1$ 波动，但总体优于基线。
- **作者结论**：Inter-MIL在小样本下鲁棒性强，特征空间更具判别力，注意力图更符合病理专家认知。
- **证据是否充分**：充分，包含内部验证、外部验证（FOCUS队列）、多种消融和可视化分析。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 高 | 提出了自交互迭代优化机制，解决了小样本下编码器任务适配问题。 |
| 技术可行性 | 高 | 基于成熟的MIL架构，仅增加简单的采样和交替训练逻辑，易于实现。 |
| 实现难度 | 中 | 需注意对抗梯度的正确实现和超参数的调优。 |
| 架构相关性 | 高 | 专门针对WSI的大尺寸和多实例特性设计。 |
| 可迁移性 | 高 | 模块化设计，可替换Encoder和Aggregator。 |
| 计算成本 | 低 | 相比全图端到端训练，计算开销可控，尤其在小样本下优势明显。 |

#### 11. 一句话总结
Inter-MIL通过迭代地利用slide级注意力引导tile级编码器的微调，并辅以对抗去噪和对比预训练，实现了小样本病理图像中细粒度与全局特征的深度融合与高效学习。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **自交互迭代优化范式**：将“编码器-聚合器”视为一个闭环系统，通过伪标签传播不断精炼特征表示，这一思路对于弱监督学习非常有效。
- **基于注意力的动态采样策略**：根据注意力分数动态构建正负样本池，比固定采样或随机采样更能聚焦于信息量大的区域。

### 2. 方法之间的关系
- **Inter-MIL** 是核心框架。
- **adInter-MIL** 是 Inter-MIL 加上对抗训练模块的变体，用于增强鲁棒性。
- **PT-Inter-MIL** 是 Inter-MIL 加上聚合器对比预训练模块的变体，用于加速收敛。
- 三者可以组合使用（如 PT-adInter-MIL）。

### 3. 复现可行性
- **代码是否公开**：是，GitHub链接已提供。
- **方法描述是否完整**：是，包含了算法伪代码、公式、超参数设置和预处理细节。
- **关键配置是否明确**：是，Table 1列出了所有关键超参数。
- **预计复现难点**：
    - 对抗训练中负梯度的具体实现细节（虽然提到了Ganin et al. 2016，但代码层面需自行处理梯度反转）。
    - 对比预训练的具体Bag构建逻辑（Algorithm 3）。
    - 数据预处理中背景去除和Tile切分的细节（虽提及OpenCV，但具体阈值需参考代码或补充材料）。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：Gated-AttPool聚合器、ResNet-18作为Backbone、加权交叉熵损失。
- **需要改造的设计**：对抗训练部分可能需要适配具体的深度学习框架；对比预训练部分需要根据具体任务调整正负样本对的构建逻辑。
- **可能形成的新研究思路**：
    - 将Transformer作为Encoder或Aggregator集成到Inter-MIL框架中。
    - 探索更复杂的伪标签生成策略，例如结合聚类或一致性正则化。
    - 将该自交互思想应用于多模态融合（如图像+基因组）。

### 5. 阅读备注
- 论文强调了小样本场景下的有效性，这是其相对于CLAM等方法的主要优势领域。
- 外部验证队列FOCUS的使用增强了结论的可信度，表明模型具有一定的泛化能力。
- 可视化分析（Grad-CAM, t-SNE）有力地支持了方法在可解释性和特征判别力上的改进。
