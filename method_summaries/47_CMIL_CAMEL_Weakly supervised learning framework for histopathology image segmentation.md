# 47_CMIL_CAMEL_Weakly supervised learning framework for histopathology image segmentation 方法总结

> 证据说明：输入为完整论文文本（10页），包含摘要、引言、方法、实验及参考文献。公式提取完整，无缺失页面。

## 一、论文基本信息

- **论文标题**：CAMEL: A Weakly Supervised Learning Framework for Histopathology Image Segmentation
- **作者**：Gang Xu, Zhigang Song, Zhuo Sun, Calvin Ku, Zhe Yang, Cancheng Liu, Shuhao Wang, Jianpeng Ma, Wei Xu
- **发表年份**：2019
- **会议/期刊**：IEEE/CVF International Conference on Computer Vision (ICCV 2019)
- **论文链接/DOI/arXiv ID**：https://doi.org/10.1109/ICCV.2019.01078；arXiv:1908.10555
- **代码仓库**：未发现作者公开的官方方法实现；`ThoroughImages/CAMEL` 仅关联数据资源，不能视为论文代码
- **研究任务**：弱监督病理图像分割（仅使用图像级标签）
- **数据模态**：数字病理切片（H&E染色WSI裁剪的1280x1280 patches）

## 二、论文整体概述

### 1. 核心问题
全监督病理图像分割需要昂贵的像素级标注。现有的弱监督方法（如基于MIL的方法）往往精度较低，且引入人工约束（如边界框）成本依然较高。如何仅利用图像级标签自动生成高质量的实例级或像素级标签以训练分割模型是关键问题。

### 2. 整体方法
提出 **CAMEL** 框架，分为两个主要步骤：
1.  **标签富集 (Label Enrichment)**：将图像划分为网格化实例，利用组合多实例学习 (**cMIL**) 自动生成实例级标签。结合级联数据增强和图像级约束优化标签质量。
2.  **分割 (Segmentation)**：将生成的实例级标签分配给对应像素，得到近似像素级掩码，使用标准全监督分割模型（DeepLabv2, U-Net）进行训练。

### 3. 主要贡献
- 提出 CAMEL 框架，实现仅用图像级标签达到接近全监督的性能。
- 提出 cMIL 方法，结合 Max-Max 和 Max-Min 两种实例选择标准，平衡数据分布。
- 提出级联数据增强方法和图像级约束损失，进一步提升标签质量和模型性能。
- 公开一个结直肠腺瘤数据集。

## 三、方法总结

### 方法 1：组合多实例学习 (Combined Multiple Instance Learning, cMIL)

#### 1. 核心思想与解决的问题
- **目标问题**：从仅有图像级标签的数据集中构建高质量的实例级分类数据集，解决弱监督下实例标签缺失问题。
- **现有方法的局限**：传统 MIL 方法可能依赖预定义特征或单一的选择策略（如只选最显著实例），导致负样本选取偏差（Max-Max在NC图像中会选中最像CA的NC实例作为负样本，导致决策边界偏移）。
- **核心思想**：并行训练两个 MIL 分类器，分别采用 **Max-Max** 和 **Max-Min** 准则选择代表性实例。Max-Max倾向于选择响应最高的实例（无论正负），Max-Min对正样本选最高响应，对负样本选最低响应。两者结合可抵消偏差，获得更均衡的实例数据集。
- **创新点**：通过互补的选择准则缓解数据分布偏差；无需预定义特征，端到端训练。

#### 2. 详细结构与数据流
- **输入**：图像级标签的数据集 $\{(I_j, y_j)\}$，其中 $y_j \in \{0, 1\}$ 表示非癌(CA)或癌(NC)。
- **处理流程**：
    1.  将每张图像 $I_j$ 划分为 $N \times N$ 个等大小网格实例 $\{b_i\}$。
    2.  **阶段1 (cMIL训练)**：
        -   初始化 ResNet-50 分类器 $f$。
        -   前向传播：计算每个实例的预测分数 $f(b_i)$。
        -   实例选择：
            -   对于 Max-Max 分支：$S_{Max-Max} = \max_i \{f(b_i)\}$。
            -   对于 Max-Min 分支：若 $y=1$, $S_{Max-Min} = \max_i \{f(b_i)\}$; 若 $y=0$, $S_{Max-Min} = \min_i \{f(b_i)\}$。
        -   反向传播：使用交叉熵损失，比较选定实例的预测与图像级标签 $y_j$。
    3.  **阶段2 (重训练 Relabel)**：
        -   使用上述两个分支选出的实例构建实例级数据集。
        -   丢弃预测标签与图像级标签不一致的混淆样本。
        -   在此新数据集上全监督重新训练分类器（Retrain）。
    4.  **阶段3 (生成标签)**：
        -   用 Retrain 后的模型对所有原始图像的实例进行分类，生成实例级标签。
- **输出**：带有实例级标签的网格化数据集。
- **模块在整体网络中的位置**：位于 CAMEL 框架的第一步“标签富集”阶段，为后续分割提供监督信号。

#### 3. 数学公式

**总损失函数 (单个分类器):**
$$ Loss = - \sum_{j} (y_j \log \hat{p}_j + (1-y_j) \log(1-\hat{p}_j)) $$
其中 $\hat{p}_j = S_{criterion}(\{f(b_i)\})$，$b_i$ 是图像 $j$ 中的实例，$f$ 是分类器，$S_{criterion} \in \{Max-Max, Max-Min\}$。

**Max-Max 选择准则:**
$$ S_{Max-Max}(\{f(b_i)\}) = \max_{i} \{f(b_i)\} $$

**Max-Min 选择准则:**
$$ S_{Max-Min}(\{f(b_i)\}) = \begin{cases} \max_{i} \{f(b_i)\} & \text{if } y = 1 \\ \min_{i} \{f(b_i)\} & \text{if } y = 0 \end{cases} $$

**带图像级约束的总损失 (Retrain阶段):**
$$ Loss = w_1 \cdot Loss_{constrain} + w_2 \cdot Loss_{retrain} $$
其中 $w_1=w_2$。

**约束损失 ($Loss_{constrain}$):**
$$ Loss_{constrain} = - \sum_{S_{criterion}} (y \log \hat{p} + (1-y) \log(1-\hat{p})) $$
其中 $\hat{p} = S_{criterion}(\{f(b_i)\})$，$f$ 为图像级约束路径的网络（与Retrain共享权重）。

**重训练损失 ($Loss_{retrain}$):**
$$ Loss_{retrain} = - \sum_{j} (y_j \log \hat{y}_j + (1-y_j) \log(1-\hat{y}_j)) $$
其中 $\hat{y}_j = g(n_j)$，$n_j$ 是输入的实例，$g$ 是重训练路径的网络。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 输入 | 原始图像 Patch | $(B, H, W, C)$ | 例如 $1280 \times 1280 \times 3$ |
| 实例划分 | 实例集合 | $N^2$ 个实例 | $N=M/m$，例如 $N=4$ 时 $4 \times 4=16$ 个实例 |
| 实例尺寸 | 单个实例 | $(m, m, C)$ | 例如 $320 \times 320$ |
| 分类器输入 | 实例批次 | $(BatchSize, m, m, C)$ | ResNet-50 输入 |
| 分类器输出 | 实例预测 | $(BatchSize, 1)$ | 癌症概率分数 |
| 选定实例索引 | Index | $(BatchSize, 1)$ | 由 Max-Max/Min 选出 |
| 最终标签 | 实例标签 | $(NumInstances, 1)$ | 0 (NC) 或 1 (CA) |

#### 5. 实现伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CMILClassifier(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.f = get_backbone(backbone) # 返回特征提取+分类头
        
    def forward(self, instances, labels, criterion_type='max-max'):
        """
        instances: Tensor of shape (B, N_instances, C, H, W)
        labels: Tensor of shape (B,) containing image-level labels (0 or 1)
        criterion_type: 'max-max' or 'max-min'
        """
        B, N, C, H, W = instances.shape
        
        # 1. 获取所有实例的预测分数
        # f(instances) -> (B, N, 1)
        scores = self.f(instances.view(-1, C, H, W)).view(B, N, 1)
        
        selected_scores = []
        
        if criterion_type == 'max-max':
            # 总是选最大分数的实例
            max_scores, _ = torch.max(scores, dim=1) # (B, 1)
            selected_scores.append(max_scores)
            
        elif criterion_type == 'max-min':
            # 正样本选最大，负样本选最小
            max_scores, _ = torch.max(scores, dim=1) # (B, 1)
            min_scores, _ = torch.min(scores, dim=1) # (B, 1)
            
            # 根据标签mask选择
            mask_pos = (labels == 1).unsqueeze(1) # (B, 1)
            selected_scores.append(torch.where(mask_pos, max_scores, min_scores))
            
        selected_scores = torch.cat(selected_scores, dim=1) # (B, num_criteria)
        
        # 2. 计算损失
        # 假设我们要同时训练两个分支或者分别训练，这里演示单分支逻辑
        # 实际训练中需分别为 Max-Max 和 Max-Min 维护参数或共享参数但不同前向逻辑
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(selected_scores.squeeze(), labels.float())
        
        return loss, selected_scores

# 注意：论文中提到 cMIL 是两个独立的分类器训练，然后合并数据
# Retrain 阶段则是全监督训练
```

#### 6. 实现提示
- **关键网络组件**：ResNet-50 作为特征提取器和分类器。
- **重要超参数**：
    - 学习率：cMIL 和 Retrain 阶段均为 0.0001 (Adam)。
    - Batch Size：cMIL 为 4 (每张GPU一张图)，Retrain 为 40 (每张GPU十个实例)。
    - 损失权重 $w_1, w_2$：设为相等。
- **归一化/激活方式**：BCE Loss 隐含 Sigmoid 激活。
- **维度对齐方式**：图像被均匀切分为 $N \times N$ 网格。
- **实现注意事项**：
    - 在 cMIL 训练阶段，梯度回传仅经过被选中的那个实例对应的网络路径（或通过索引操作实现）。
    - 在生成最终实例标签时，需过滤掉预测标签与图像级标签冲突的样本（即 NC 图像中被预测为 CA 的实例，或反之，具体取决于论文描述的“discard potentially confusing samples”）。
- **依赖的特殊算子**：`torch.max`, `torch.min`, `torch.where`。

#### 7. 计算与资源开销
- **理论计算复杂度**：取决于 ResNet-50 的复杂度，约为 $4 \times 10^9$ FLOPs 每张图片（不含后续分割）。
- **参数量**：ResNet-50 约 25M 参数。
- **显存开销**：受限于 GPU 内存，cMIL 阶段 batch size 很小 (4)，因为每张图切成多个实例后仍需保留上下文信息或单独处理，实际显存占用适中。
- **推理速度**：未提供具体 FPS，但相比全监督像素级推理，实例级推理较快。
- **论文是否提供效率对比**：未提供详细的 FLOPs 或时间对比表格，主要关注精度提升。

#### 8. 适用场景与可迁移性
- **原论文应用场景**：乳腺癌淋巴结转移检测 (CAMELYON16)、结直肠腺瘤检测。
- **可迁移到的任务/数据集**：任何具有图像级标签的医学图像分类/分割任务，特别是前景背景差异不明显、病灶分散的场景。
- **迁移所需调整**：调整网格大小 $N$ 以适应不同分辨率和病灶尺度；调整 Backbone。
- **适用条件**：需要有足够的图像级标签数据；病灶在图像中有空间分布。
- **潜在限制**：网格划分假设病灶与网格有一定对齐关系；过小的实例可能导致信息丢失，过大的实例导致标签粗糙。

#### 9. 实验与消融证据
- **主要性能结果**：
    - 实例分类：Retrain (cMIL) 在 320x320 上 Accuracy 92.3%，接近 FSB (94.5%)。
    - 像素分割：CAMEL (160) DeepLabv2 IoU 85.4%，接近 Pixel-Level FSB (86.3%)。
- **相对基线的提升**：显著优于其他弱监督方法 (WILD CAT, DWS-MIL, CDWS-MIL)。
- **相关消融实验**：
    - 对比 Max-Max, Max-Min, cMIL (组合), Cascade, Constrained。
    - 证明 Max-Max 特异性高敏感性低，Max-Min 反之，组合效果最佳。
    - 证明 Cascade 和 Image-level Constraint 能进一步提升性能。
- **作者结论**：cMIL 能有效平衡数据分布，级联和约束能恢复信息并提高鲁棒性。
- **证据是否充分**：在两个数据集上验证，有充分的消融实验支持各模块有效性。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 中 | cMIL 结合了经典 MIL 策略，思路清晰但非全新架构，属于巧妙应用。 |
| 技术可行性 | 高 | 基于标准 ResNet 和 PyTorch/TensorFlow 易实现。 |
| 实现难度 | 低 | 逻辑简单，主要是数据预处理和损失函数的定制。 |
| 架构相关性 | 高 | 适用于任何基于 CNN 的分类/分割流水线。 |
| 可迁移性 | 高 | 通用性强，适用于多种 WSIs 分析任务。 |
| 计算成本 | 中 | 需要额外的 MIL 训练步骤，但总体可控。 |

#### 11. 一句话总结
CAMEL 通过组合多实例学习（cMIL）自动从图像级标签生成高质量实例级标签，并结合级联增强和约束机制，实现了仅需图像级监督的高精度病理图像分割。

### 方法 2：级联数据增强与图像级约束 (Cascade Data Enhancement & Image-Level Constraints)

#### 1. 核心思想与解决的问题
- **目标问题**：单一尺度的网格划分会导致信息损失（大网格）或标签噪声/过标记（小网格）。此外，仅使用实例级标签忽略了原始图像的全局监督信息。
- **现有方法的局限**：固定尺度的 MIL 难以兼顾局部细节和全局一致性。
- **核心思想**：
    1.  **级联数据增强**：通过两级 MIL 过程（先粗粒度再细粒度，或并行不同粒度）生成更多样化的实例数据，恢复信息。
    2.  **图像级约束**：在 Retrain 阶段，除了使用生成的实例标签外，还将原始图像作为输入，施加图像级的一致性约束，确保实例级预测与图像级标签一致。
- **创新点**：将 MIL 的过程视为一种数据增强手段；在多任务/多损失框架下融合不同粒度的监督信号。

#### 2. 详细结构与数据流
- **输入**：原始图像级数据集。
- **处理流程**：
    1.  **级联生成**：
        -   路线 A：直接使用 cMIL($N$) 生成 $N \times N$ 实例。
        -   路线 B：先使用 cMIL($N_1$) 生成中间层实例，再对每个中间实例使用 cMIL($N_2$) 生成最终实例，其中 $N = N_1 \times N_2$。
        -   合并两条路线产生的实例数据集。
    2.  **带约束的训练 (Retrain with Constraints)**：
        -   网络结构共享：图像级路径 $f$ 和实例级路径 $g$ 共享 ResNet-50 权重。
        -   前向传播：
            -   实例输入 $n_j$ -> $g(n_j)$ -> 计算 $Loss_{retrain}$ (基于实例标签)。
            -   图像输入 $I$ -> 划分为实例 -> $S_{criterion}(\{f(b_i)\})$ -> 计算 $Loss_{constrain}$ (基于图像标签)。
        -   总损失加权求和。
- **输出**：经过双重监督优化的实例级标签和分类器。
- **模块在整体网络中的位置**：标签富集阶段的优化模块。

#### 3. 数学公式

**级联关系**：
$$ N = N_1 \times N_2 $$
其中 $N$ 是最终网格数，$N_1$ 是第一级网格数，$N_2$ 是第二级网格数。

**总损失函数**:
$$ Loss = w_1 \cdot Loss_{constrain} + w_2 \cdot Loss_{retrain} $$

**约束损失**:
$$ Loss_{constrain} = - \sum_{S_{criterion}} (y \log \hat{p} + (1-y) \log(1-\hat{p})) $$
$\hat{p}$ 是通过图像级路径选定的实例预测值。

**重训练损失**:
$$ Loss_{retrain} = - \sum_{j} (y_j \log \hat{y}_j + (1-y_j) \log(1-\hat{y}_j)) $$
$\hat{y}_j$ 是实例级路径的预测值。

#### 4. 输入输出维度

| 阶段 | 张量/变量 | 维度 | 说明 |
|---|---|---|---|
| 级联输入 | 原始图像 | $(B, M, M, C)$ | 大尺寸 Patch |
| 中间实例 | 第一级实例 | $(B, N_1^2, m_1, m_1, C)$ | 中等尺寸 |
| 最终实例 | 第二级实例 | $(B, N_2^2, m_2, m_2, C)$ | 小尺寸 |
| 约束输入 | 原始图像 | $(B, M, M, C)$ | 同原始输入 |
| 约束输出 | 图像级预测 | $(B, 1)$ | 经 MIL 聚合后的分数 |

#### 5. 实现伪代码

```python
def train_with_constraints(model, instance_loader, image_loader, optimizer):
    model.train()
    total_loss = 0
    
    # 假设 instance_loader 提供已标记的实例
    # image_loader 提供原始图像及其标签
    
    for instances, instance_labels in instance_loader:
        # 1. Instance Loss (Retrain)
        pred_inst = model(instances) # Shape: (B, 1)
        loss_retrain = F.binary_cross_entropy(pred_inst, instance_labels)
        
        # 2. Constraint Loss (Image Level)
        # 需要从 image_loader 获取对应的原始图像
        images, image_labels = next(iter(image_loader)) 
        # 注意：实际实现中需确保 batch 对齐
        
        # 将图像切分为实例
        insts_from_img = split_image_into_patches(images) # (B, N*N, C, H, W)
        
        # 前向传播获取实例分数
        scores = model(insts_from_img.view(-1, *insts_from_img.shape[2:]))
        scores = scores.view(images.size(0), -1, 1)
        
        # 应用 Max-Max 或 Max-Min 选择
        # 此处简化为 Max-Max 示例
        selected_scores, _ = torch.max(scores, dim=1) 
        
        loss_constrain = F.binary_cross_entropy(selected_scores, image_labels)
        
        # 3. Total Loss
        loss = 0.5 * loss_constrain + 0.5 * loss_retrain # w1=w2=0.5
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss
```

#### 6. 实现提示
- **关键网络组件**：共享权重的 ResNet-50。
- **重要超参数**：$w_1, w_2$ 均设为 1 (论文说 set $w_1=w_2$，未指定具体值，通常取1或0.5)。
- **维度对齐方式**：图像切分需严格对应。
- **实现注意事项**：
    - 级联过程中，中间层的标签也是通过 MIL 生成的，存在误差传递风险，但论文认为这增加了多样性。
    - 图像级约束路径和实例级路径共享权重，这意味着在计算约束损失时，梯度也会更新用于实例分类的参数，起到正则化作用。

#### 7. 计算与资源开销
- **计算复杂度**：增加了一倍的前向传播（如果并行处理）或串行处理的时间开销。
- **显存开销**：由于需要同时存储图像和其实例，以及两个损失的反向传播，显存需求略高于普通 Retrain。

#### 8. 适用场景与可迁移性
- **适用场景**：对标签噪声敏感的任务，或希望利用全局上下文信息的任务。
- **可迁移性**：可应用于任何 MIL 框架中，作为额外的正则化项。

#### 9. 实验与消融证据
- **主要性能结果**：
    - Table 1 显示 "Cascade (constrained)" 在 320x320 上 Accuracy 91.7%，优于 "Retrain (cMIL)" 的 92.3%? 不，Table 1 中 Retrain(cMIL) 320x320 Acc 92.3%, Cascade(constrained) 91.7%。等等，看 Table 1:
        - Retrain (cMIL) 320x320: Acc 92.3
        - Retrain (constrained) 320x320: Acc 92.9
        - Cascade 320x320: Acc 90.4
        - Cascade (constrained) 320x320: Acc 91.7
    - 看起来 "Retrain (constrained)" 比 "Retrain (cMIL)" 好。
    - 在 160x160 上:
        - Retrain (cMIL): 88.4
        - Retrain (constrained): 89.9
        - Cascade (constrained): 91.7 (这是最好的!)
    - 结论：级联+约束在小粒度（160x160）下效果最好，提升了 3.3% 相对于基础 cMIL。
- **作者结论**：级联恢复了信息，约束利用了全局监督，两者结合在细粒度下优势明显。

#### 10. 方法评估

| 维度 | 评价 | 依据 |
|---|---|---|
| 创新性 | 中 | 约束损失是常见的正则化手段，级联 MIL 是一种数据增强策略。 |
| 技术可行性 | 高 | 易于集成到现有 MIL 代码中。 |
| 实现难度 | 中 | 需要处理不同粒度的数据加载和对齐。 |
| 架构相关性 | 高 | 通用模块。 |
| 可迁移性 | 高 | 适用于各种弱监督场景。 |
| 计算成本 | 中 | 增加了少量计算。 |

#### 11. 一句话总结
通过级联多实例学习和图像级一致性约束，进一步提升了弱监督标签的质量和分割模型的鲁棒性，特别是在细粒度分割任务中。

## 四、论文级综合评价

### 1. 最值得借鉴的方法
- **cMIL 的双分支设计**：利用 Max-Max 和 Max-Min 的互补性来校正 MIL 中的选择偏差，这是一个非常直观且有效的技巧，解决了负样本选取不当的问题。
- **弱监督到全监督的转换范式**：将 MIL 视为一个“标签生成器”，先生成实例级标签，再训练分割模型。这种解耦的思路降低了分割模型训练的门槛。

### 2. 方法之间的关系
- **cMIL** 是核心引擎，负责生成初始的高质量实例标签。
- **Retrain** 是基于 cMIL 生成数据的标准化全监督微调。
- **Cascade & Constraints** 是对 cMIL/Retrain 过程的增强和优化，旨在解决信息丢失和监督信号不足的问题。
- 三者共同构成了 CAMEL 的标签富集模块，随后接入标准的分割网络。

### 3. 复现可行性
- **代码是否公开**：数据集公开，代码未明确公开（GitHub 链接指向数据集）。
- **方法描述是否完整**：非常完整。给出了具体的公式、网络结构（ResNet-50）、超参数（LR, Batch Size）、损失函数细节。
- **关键配置是否明确**：明确。包括网格大小、优化器、损失权重等。
- **预计复现难点**：
    - 级联数据增强的具体实现细节（如中间层标签如何处理、如何合并数据）。
    - 图像级约束路径中，如何将图像切分并与实例路径共享权重的具体代码实现（TensorFlow 实现需注意变量作用域）。

### 4. 与当前研究方向的关系
- **可直接采用的设计**：cMIL 的选择策略可以直接嵌入到任何基于 Attention/MIL 的弱监督分类器中。
- **需要改造的设计**：级联部分需要根据具体的分辨率和算力进行调整。
- **可能形成的新研究思路**：
    - 探索更复杂的实例选择准则（如 Top-K）。
    - 将这种标签富集方法应用到其他模态（如自然图像、遥感图像）的弱监督分割中。
    - 结合 Transformer 架构，利用其全局注意力特性改进 MIL 的表现。

### 5. 阅读备注
- 论文中提到的 "over-labeling issue"（过标记）是由于实例只要包含任何癌细胞就被标记为 CA，这在像素级分割时会引入噪声。虽然论文提到未来工作会用 Mask Boundary Refinement 解决，但在当前实验中，通过减小实例尺寸（160x160 vs 320x320）在一定程度上缓解了该问题。
- CAMEL 的性能在 160x160 粒度下表现更好（IoU 85.4 vs 84.3），证明了细粒度标签的优势，但也暗示了计算成本的增加。
