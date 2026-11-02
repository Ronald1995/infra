# LoRA Without Regret：文章技术总结与分析

> **原文**：John Schulman、Thinking Machines Lab，*LoRA Without Regret*，2025-09-29  
> **链接**：<https://thinkingmachines.ai/blog/lora/>  
> **文档性质**：基于原文的中文技术总结，补充必要的公式解释、适用边界和工程启示。

---

## 1. 核心结论

文章研究的问题是：在大模型后训练（post-training）中，LoRA 是否一定会因为低秩约束而牺牲训练效率或最终效果？

作者的主要结论是：

**在 LoRA 覆盖合适的网络层、可训练参数容量足以承载任务所需信息、并使用适合 LoRA 的优化超参数时，LoRA 可以达到 Full Fine-Tuning（FullFT）相近的样本效率和最终性能。**

文章识别出一个可以称为 **low-regret regime（低遗憾区间）** 的工作范围。在这个范围内，使用 LoRA 相对于全参数微调，并不一定需要付出明显的模型效果代价。

主要发现如下：

1. 对小到中等规模的指令微调和推理微调数据集，容量足够的 LoRA 可以与 FullFT 获得相近的学习曲线和最终 Loss。
2. 当训练数据所需的信息量超过 LoRA 的有效容量时，LoRA 会逐渐落后。其表现不一定是 Loss 到达一个固定下限，而可能是训练效率逐渐下降。
3. 在部分实验中，LoRA 比 FullFT 更不耐受大 Batch Size；增大 Rank 并不能消除这种差距。
4. LoRA 的目标层选择很重要。只作用于 Attention 的 LoRA 明显弱于作用于 MLP/MoE 的 LoRA；在作者测试的配置中，MLP-only 与 MLP+Attention 的表现大致相当。
5. 在作者测试的 Policy Gradient 数学推理 RL 场景中，即使每个目标矩阵使用 Rank-1 LoRA，也能达到与 FullFT 相近的学习表现。
6. 使用标准的 \(\alpha/r\) 缩放时，LoRA 最优学习率在实验中约为 FullFT 的 10 倍；最优学习率对 Rank 的依赖相对较弱。
7. 对单个大矩阵的简化 FLOPs 分析表明，当 Rank 远小于矩阵维度时，LoRA 每个训练 Pass 的计算量略高于 FullFT 的 \(2/3\)，同时能显著减少梯度和优化器状态所需的内存。

这些结论是有实验范围和前提的，并不意味着所有任务、所有 RL 算法或所有模型都可以无条件用 Rank-1 LoRA 替代 FullFT。

## 2. 背景：为什么后训练可能不需要更新全部参数？

预训练和后训练的数据规模及目标不同：

- **预训练**：需要从海量 Token 中学习广泛的语言、知识和行为模式。
- **后训练**：通常使用规模更小的数据，针对指令遵循、推理、领域适配或特定行为进行调整。

如果后训练数据只包含有限的信息，却允许更新整个模型的全部权重，那么从参数自由度角度看，可能存在明显的过度配置。

LoRA 的基本形式是将原始权重矩阵 \(W\) 替换为：

\[
W' = W + \gamma BA
\]

其中：

\[
A \in \mathbb{R}^{r \times d_{\mathrm{in}}}, \qquad
B \in \mathbb{R}^{d_{\mathrm{out}} \times r}
\]

因此：

\[
\operatorname{rank}(BA) \leq r
\]

原始矩阵的参数量为：

\[
P_W = d_{\mathrm{out}}d_{\mathrm{in}}
\]

LoRA 的可训练参数量为：

\[
P_{\mathrm{LoRA}} =
r(d_{\mathrm{in}}+d_{\mathrm{out}})
\]

当 \(r\) 远小于输入、输出维度时，可训练参数量显著减少。

文章还强调 LoRA 的几项系统优势：

- **多租户推理**：基础模型权重保持不变，一个推理服务可以管理多个 Adapter，并让不同请求使用不同 Adapter。
- **训练内存布局更轻**：FullFT 需要为全部权重保存梯度和优化器状态；LoRA 只为 Adapter 保存这些训练状态。
- **权重加载与传输更轻**：Adapter 体积小，跨机器传输和加载更容易。

这些优势使 LoRA 不仅是一种参数高效微调方法，也是一种有利于训练与推理系统解耦的模型更新形式。

## 3. 实验设计

作者希望尽量区分 LoRA 容量、优化超参数和网络层选择的影响。

### 3.1 模型与数据集

| 实验类型 | 模型或数据集 | 主要用途 |
|---|---|---|
| SFT | Llama 3 系列 | Dense 模型实验 |
| SFT | Qwen3 系列，包含 MoE | Dense 与 MoE 结构验证 |
| 指令微调 | Tulu 3 | 评估指令数据上的 Log Loss |
| 推理微调 | OpenThoughts3 | 评估推理数据上的 Log Loss |
| RL | MATH、GSM | 数学任务的 Outcome-based 奖励 |
| RL | DeepMath | 更大规模、更困难的数学推理实验 |
| 泛化评估 | AIME 2024、AIME 2025 | 评估训练集之外的推理表现 |

在部分 RL 实验中，作者使用 Llama-3.1-8B Base，是因为 Qwen2.5 和 Qwen3 的预训练数据已经包含较强数学能力，可能使得单独衡量 RL 阶段学到的内容更加困难。

### 3.2 控制变量与评价方式

- LoRA Rank 覆盖 1 到 512，跨度达到三个数量级。
- 对每种实验配置分别进行 Learning Rate Sweep，避免因使用不合适的学习率而低估某种方法。
- 使用 Constant Learning Rate，不使用 Warmup 或 Cooldown。
- SFT 主要观察 Log Loss，而不是只依赖采样生成后的任务分数。
- RL 主要观察数学任务的 Reward / Accuracy。
- 部分学习曲线是在每个训练阶段，从多个学习率实验中取最低 Loss 得到的点对点最优包络。因此，这类曲线不应被误读为单一固定学习率运行的结果。

## 4. LoRA Rank 与数据规模：容量不足如何体现？

### 4.1 实验观察

在 Tulu3 和 OpenThoughts3 上，作者比较了不同 Rank 的 LoRA 与 FullFT：

- 高 Rank LoRA 与 FullFT 的学习曲线相近。
- 中、低 Rank LoRA 在训练早期也可能表现接近。
- 随着训练继续，低 Rank LoRA 会在某个阶段逐渐偏离最优学习曲线。
- Rank 越低，越容易较早出现这种偏离。

因此，容量不足不一定表现为模型完全无法继续降低 Loss，也可能表现为学习速度下降、达到同一 Loss 需要更多训练步骤。

### 4.2 数据规模与容量的关系

可以将问题概括为：

\[
\text{LoRA 的有效容量}
\quad \text{是否足以承载} \quad
\text{任务需要学习的信息}
\]

当数据规模和任务复杂度较小时，低秩更新可能已经足够；随着训练数据增多或任务要求更复杂，LoRA 可能逐渐受到容量约束。

需要区分：

- **记忆训练集所需的信息量**；
- **学习可泛化规律所需的信息量**。

二者不相同。即使模型不需要记住训练集的每个细节，也可能通过较少的参数学到有用的泛化规律。

### 4.3 用 Log Loss 粗略估计监督数据的信息量

文章讨论了一个估计思路：在第一个 Epoch 中累计的 Log Loss，可以作为数据集描述长度的一个估计，也可以看作完美记忆数据集所需信息量的上界。

若使用自然对数单位（nats）：

\[
L = -\sum_{i=1}^{N}\log p_\theta(y_i \mid x_i)
\]

转换为 bits：

\[
I(D) \approx \frac{L}{\ln 2}
\]

文章指出，LLM 数据集的 Loss 通常约为每 Token 1 bit（约 0.69 nats），但具体数值会随数据集和模型规模变化。

这只是粗略估计。它偏向估计记忆整个数据集所需的容量，可能高估真正学会可泛化规律所需的容量。如何准确预测真实任务的容量需求，仍是开放问题。

## 5. Batch Size：LoRA 对大 Batch 可能更敏感

作者在 OpenThoughts3 的一个 10,000 条样本子集上比较不同 Batch Size。

主要观察：

1. 小 Batch（例如实验中的 Batch Size 32）下，LoRA 与 FullFT 的差距较小，并可能随训练逐渐缩小。
2. Batch Size 增大时，LoRA 相对 FullFT 的 Loss 差距会扩大。
3. 这种差距看起来并不依赖 LoRA Rank；单纯提高 Rank 无法消除。
4. LoRA 和 FullFT 在该实验中都更倾向于在较小 Batch Size 下取得更好的 Loss。

作者认为，可能原因是 LoRA 使用矩阵乘积参数化 \(BA\)，其优化动力学不同于直接优化原始矩阵 \(W\)。在某些数据和训练配置下，这种参数化对大 Batch 的优化不够有利。

**实践含义**：比较 LoRA 和 FullFT 时，不能只对齐 Rank 或学习率，还应系统检查 Batch Size、梯度累积和训练步数。文章的 Batch Size 结果来自特定 SFT 实验，不应直接推断所有 RL 算法都会出现同样程度的问题。

## 6. LoRA 应该作用于哪些层？

这是文章最重要的工程发现之一。

### 6.1 Attention-only 并不理想

作者比较了：

- Attention-only LoRA；
- MLP-only LoRA；
- MLP + Attention LoRA。

在 Llama-3.1-8B Dense 模型和 Qwen3-30B-A3B MoE 模型上，作者观察到：

- Attention-only 明显弱于 MLP-only。
- MLP-only 与 MLP+Attention 的表现大致相当。
- 在作者测试的设置中，给 MLP LoRA 额外加入 Attention LoRA，没有带来明显的额外收益。

作者也指出，QLoRA 的相关实验同样观察到 Attention-only 较弱，但其结果中 MLP+Attention 优于 MLP-only；Thinking Machines 的实验则发现后两者大致相当。因此，不能把不同实验的细微排序视为普遍规律。

### 6.2 这不是简单的参数量差异

文章给出了 Llama-3.1-8B 的参数量对照：

| LoRA 目标层 | Rank | LoRA 参数量 |
|---|---:|---:|
| MLP | 256 | 0.49B |
| Attention | 256 | 0.25B |
| All | 256 | 0.70B |
| MLP | 128 | 0.24B |

Attention Rank-256 与 MLP Rank-128 的参数量接近，但 Attention-only 仍明显更弱。这说明结果不能简单归因于 Attention-only 的可训练参数更少。

### 6.3 作者对原因的解释：Empirical Neural Tangent Kernel

文章引用 Empirical Neural Tangent Kernel（eNTK）解释目标层选择。

对于样本 Token 的模型输出，可以定义梯度：

\[
g_i =
\nabla_\theta \log p_\theta(
\text{token}_i \mid \text{prefix}_i
)
\]

eNTK 可写为：

\[
K(i,j)=g_i^\top g_j
\]

它描述不同样本在参数空间中的梯度相关性。

如果只在 Attention 层加入 LoRA，就相当于排除了大量 MLP 参数对梯度内积的贡献。由于 MLP 通常包含模型中很大一部分参数，这可能改变整体 Kernel 结构。

作者引用的理论工作指出，当 LoRA 覆盖模型中对梯度内积贡献较大的层时，LoRA 的 eNTK 可以近似 FullFT 的 eNTK，从而解释二者在训练初期具有相近的学习动力学。

这是一种理论解释和近似，不应理解为对所有模型、所有任务的严格等价证明。

### 6.4 MoE 中的 LoRA 设置

在 Qwen3-30B-A3B 的 MoE 实验中，作者为每个 Expert 单独训练 LoRA。实验中激活 Expert 数为 8，并将每个 Expert 的 Rank 设置为总 Rank 除以激活 Expert 数：

\[
r_{\mathrm{expert}} = \frac{R}{8}
\]

这样可以让 MoE 层的 LoRA 参数量与其他层保持相近的相对比例。

这证明了在该 MoE 实验设置中，MLP/Expert LoRA 是重要的适配位置。但它没有完整解决大规模 MoE 中 LoRA 的 TP、EP 分片、通信和动态加载问题。

## 7. RL：为什么低 Rank LoRA 也可能足够？

### 7.1 SFT 与 Outcome-based Policy Gradient 的监督信号不同

SFT 通常在每个 Token 位置提供监督：

\[
\mathcal{L}_{\mathrm{SFT}}
=
-\sum_{t=1}^{T}
\log p_\theta(y_t \mid y_{<t},x)
\]

一个包含数千 Token 的回答，可以在多个 Token 位置提供学习信号。

在典型 Outcome-based RL 中，模型先生成完整轨迹：

\[
\tau=(y_1,y_2,\ldots,y_T)
\]

然后获得一个最终奖励或优势值。一个简化的 REINFORCE 梯度估计为：

\[
G = \nabla_\theta \log p_\theta(\tau)\cdot Adv(\tau)
\]

其中：

\[
S=\nabla_\theta \log p_\theta(\tau)
\]

\[
G=S\cdot Adv
\]

作者的核心直觉是：在这种策略梯度设置中，整条轨迹的奖励反馈最终通过一个标量 Advantage 进入梯度估计。因此，相对于 Token 级 SFT 监督，每个 Episode 的有效反馈信息可能少得多。

### 7.2 信息论论证

在给定当前策略和训练历史的条件下，作者认为 \(S\) 与未知奖励函数 \(R\) 独立，奖励信息通过 \(Adv\) 进入更新。

由 Data Processing Inequality：

\[
I(G;R\mid history)
\leq
I((S,Adv);R\mid history)
\]

在上述条件下：

\[
I((S,Adv);R\mid history)
=
I(Adv;R\mid S,history)
\leq H(Adv)
\]

如果将 Advantage 量化为 \(B\) 个区间：

\[
H(Adv)\leq \log_2 B
\]

因此，每个 Episode 的奖励信息量上界是 \(O(1)\)，而不是随生成 Token 数量线性增长。

**适用边界非常重要**：这不是“所有 RL 每个 Episode 只有 1 bit”的普遍定律。它是针对标量 Advantage 的典型 Policy Gradient 信息结构所作的分析。Process Reward、Dense Reward、Model-based RL 或额外辅助目标可能从一个 Episode 中提取更多信息。

### 7.3 MATH 实验中的容量估算

作者给出的估算：

- 约 10,000 道数学问题；
- 每道问题采样 32 个 Completion；
- 总 Completion 数约为 320,000；
- 假设每个 Completion 提供 1 bit 信息。

则训练过程所需吸收的信息量粗略估计为：

\[
I_{\mathrm{RL}} \approx 10{,}000 \times 32
= 320{,}000\ \text{bits}
\]

Llama-3.1-8B 的 Rank-1 LoRA，在所有目标权重矩阵上累加 A、B 参数后，约有 3M 个可训练参数：

\[
P_{\mathrm{LoRA}}
=
\sum_j r_j(d_{\mathrm{in},j}+d_{\mathrm{out},j})
\]

当每个目标矩阵 Rank 为 1 时：

\[
P_{\mathrm{Rank1}}
=
\sum_j(d_{\mathrm{in},j}+d_{\mathrm{out},j})
\approx 3M
\]

作者据此认为 Rank-1 LoRA 的参数数量已经明显超过这个粗略的信息量估计。

这里的“3M 参数”并不表示整个模型只有 3M 参数，而是所有目标矩阵的 LoRA A/B 参数总数。参数数量与可编码 bits 也不是严格相同的量，因此该比较应视为数量级估算，而非严格的容量定理。

作者还提到 DeepSeek-R1-Zero 约有 5.3M Episodes，并预测低秩 LoRA 可能复现类似结果。这是作者根据容量估算提出的预测，并非该文章已经完成的 LoRA 复现结果。

### 7.4 RL 实验结果

作者进行了两组主要验证。

#### MATH 与 GSM

- 使用 Llama-3.1-8B Base。
- 使用 Policy Gradient，并加入 Importance Sampling Correction。
- 使用类似 GRPO 的 Group Centering：同一道题采样多个 Completion，再减去组内平均 Reward。
- 比较不同 Rank 的 LoRA 与 FullFT。

结果显示，在作者测试的噪声精度范围内，LoRA 可以达到与 FullFT 相近的最高 Reward，并且表现良好的学习率范围更宽。

#### DeepMath 与 Qwen3-8B Base

作者进一步在规模更大、难度更高的 DeepMath 数据集上验证，并将训练和评估样本长度限制为 8192 Token，以控制实验成本。

观察结果：

- 不同 Rank 的 LoRA 与 FullFT 在各自最优学习率下，学习曲线相近。
- 在未参与训练的 AIME 2024 和 AIME 2025 上也观察到相似趋势。
- LoRA 和 FullFT 都出现了类似的推理行为变化，例如 Backtracking、Self-verification 和 In-context Exploration。
- CoT 长度随训练过程增长，可作为模型推理行为发生变化的一个观察信号。

这些结果支持低 Rank LoRA 在作者测试的数学 Outcome-based RL 场景中的有效性，但不能直接推广到所有 RL 任务。

## 8. LoRA 超参数与学习率

### 8.1 标准参数化

文章采用：

\[
W' = W + \frac{\alpha}{r}BA
\]

其中：

- \(r\)：LoRA Rank；
- \(\alpha\)：Scaling Factor；
- \(A\)、\(B\)：低秩矩阵。

实验中使用 \(\alpha=32\)，并采用常见初始化：

- \(A\) 使用随机初始化，尺度约为 \(1/\sqrt{d_{\mathrm{in}}}\)；
- \(B\) 初始化为 0；
- A、B 使用相同学习率。

### 8.2 为什么初期最优学习率对 Rank 不敏感？

将矩阵乘积拆成 Rank 个秩一外积：

\[
BA = \sum_{i=1}^{r} b_i a_i^\top
= \sum_{i=1}^{r}\Delta_i
\]

其中：

\[
\Delta_i=b_i a_i^\top
\]

在作者讨论的初始化条件下，各个秩一分量具有相同的统计分布。第一步更新时，每个分量的期望更新相同，因此：

\[
\mathbb{E}\left[
\frac{1}{r}\sum_{i=1}^{r}\Delta_i
\right]
=
\mathbb{E}[\Delta_i]
\]

这解释了为什么使用 \(1/r\) 缩放时，不同 Rank 的 LoRA 在训练初期可以具有相同或非常接近的期望更新规模。

实验中，不同 Rank 的早期学习曲线确实非常接近。但随着训练继续，Rank 对长期优化轨迹和最优学习率仍会产生一定影响。作者观察到 Rank 4 到 Rank 512 的最优学习率变化小于 2 倍。

### 8.3 四个超参数可归纳为两个有效自由度

文章讨论的四个超参数是：

1. Scaling Factor：\(\alpha\)；
2. A 的学习率：\(LR_A\)；
3. B 的学习率：\(LR_B\)；
4. A 的初始化尺度：\(\mathrm{init}_A\)。

在理想化的 Adam 条件下，作者给出一组不改变学习行为的参数变换。对 \(p,q>0\)：

\[
\alpha \rightarrow \frac{\alpha}{pq}
\]

\[
\mathrm{init}_A \rightarrow p\cdot \mathrm{init}_A
\]

\[
LR_A \rightarrow p\cdot LR_A
\]

\[
LR_B \rightarrow q\cdot LR_B
\]

因此，四个表面参数中有两个冗余自由度。作者给出两个更容易解释的有效组合：

**初始更新幅度：**

\[
S = \alpha \cdot \mathrm{init}_A \cdot LR_B
\]

它决定初始更新的规模，也可以理解为学习曲线初始下降斜率相关的量。

**A 矩阵变化时间尺度：**

\[
T = \frac{\mathrm{init}_A}{LR_A}
\]

它决定 A 从初始化状态发生显著变化所需的大致训练步数。

该不变性依赖特定优化器假设。不能直接认为任意 Adam 实现、权重衰减、梯度裁剪和其他训练细节下都严格成立。

### 8.4 LoRA 最优学习率约为 FullFT 的 10 倍

这是文章非常实用的经验结论：

\[
LR_{\mathrm{LoRA}} \approx 10 \times LR_{\mathrm{FullFT}}
\]

作者在 SFT 和 RL 实验中都观察到类似关系，并对 14 个 Llama、Qwen 模型的学习率 Sweep 结果拟合，得到 LoRA 倍率约为 9.8。

实验还发现，LoRA 与 FullFT 的最优学习率随模型 Hidden Size 的变化趋势相近，LoRA 并没有表现出额外的 Hidden Size 指数依赖。

但作者明确表示，目前还没有充分的理论解释说明为什么倍率会稳定在约 10。

**实践建议**：将 FullFT 最优学习率的约 10 倍作为 LoRA 学习率搜索的起点，而不是直接复用 FullFT 的学习率；仍需针对模型、任务和训练长度做 Sweep。

### 8.5 短训练与长训练

LoRA 通常将 B 初始化为 0：

\[
B_0=0
\]

训练初期，B 很小时，A 的变化对 \(BA\) 的影响较小；随着 B 增大，A 的更新对最终 Adapter 的影响逐渐增强。

因此，LoRA 的有效更新速度可能随训练进程发生变化。

作者观察到：

- 较长训练中的经验倍率约为 10 倍；
- 对不足约 100 Steps 的短训练，初步经验表明 15 倍左右可能更合适。

15 倍属于作者的初步经验观察，不应当作通用固定规则。

## 9. 计算效率与内存优势

### 9.1 单个矩阵的 FLOPs 分析

考虑：

\[
W\in\mathbb{R}^{N\times N},\qquad
x\in\mathbb{R}^{N},\qquad
y=Wx
\]

忽略 Batch 维度和其他模型算子。

#### FullFT

Forward：

\[
y=Wx
\]

计算量约为：

\[
N^2
\]

Backward 包含：

\[
\bar{x}=W^\top\bar{y}
\]

计算量约为：

\[
N^2
\]

以及权重梯度：

\[
\bar{W} \mathrel{+}= x\bar{y}^{\top}
\]

计算量约为：

\[
N^2
\]

总计：

\[
F_{\mathrm{FullFT}}\approx 3N^2
\]

#### LoRA

令：

\[
W'=W+BA
\]

其中：

\[
B\in\mathbb{R}^{N\times R},\qquad
A\in\mathbb{R}^{R\times N},\qquad R\ll N
\]

原始 W 冻结，因此仍需要计算：

- \(Wx\) 的 Forward；
- \(W^\top\bar y\) 的输入梯度。

这两项约为：

\[
2N^2
\]

A、B 的前向和反向计算量合计约为：

\[
6NR
\]

因此：

\[
F_{\mathrm{LoRA}}\approx 2N^2+6NR
\]

当 \(R\ll N\) 时：

\[
\frac{F_{\mathrm{LoRA}}}{F_{\mathrm{FullFT}}}
=
\frac{2N^2+6NR}{3N^2}
\approx \frac{2}{3}
\]

所以，LoRA 的单矩阵训练 FLOPs 不是 FullFT 的 \(1/R\)，而是在 Rank 远小于矩阵维度时，略高于 FullFT 的 \(2/3\)。

该估算没有包含所有模型算子；文章特别指出，长上下文场景中的 Attention FLOPs 可能相当重要。

### 9.2 为什么 LoRA 的系统成本仍然可能更低？

LoRA 的主要资源优势还包括：

- 冻结的基础权重不需要保存梯度；
- 不需要为基础权重保存 Adam 一阶、二阶动量；
- 训练状态只覆盖较小的 Adapter 参数；
- Adapter 检查点和传输数据更小；
- 训练所需的硬件布局可以更接近推理布局。

因此，LoRA 的优势不应简单理解为“矩阵计算量减少到 \(1/R\)”，而应综合考虑训练状态内存、优化器状态、权重传输和系统部署。

## 10. 文章明确指出的局限与开放问题

### 10.1 容量预测仍不精确

作者尚不能仅凭数据集大小和 LoRA 参数量，准确预测 LoRA 在某个任务上何时会落后于 FullFT。

每参数约 2 bits 的容量结果来自精心构造的合成数据集，不能直接当作真实 LLM 任务的通用容量定律。

### 10.2 RL 的 1 bit/Episode 论证范围有限

作者明确指出，这个论证主要适用于典型 Policy Gradient 算法。Model-based RL 可能从 Episode 中学习环境预测和世界模型，提取更多信息。Process Reward、Dense Reward 等设置也可能与单个终端标量奖励不同。

### 10.3 学习率理论仍不完整

实验得到 LoRA 最优学习率约为 FullFT 的 10 倍，但作者尚未找到能充分解释这一稳定倍率的理论。

### 10.4 LoRA 变体尚未系统比较

作者提出应使用本文相同的实验方法，进一步评估 PiSSA 等 LoRA 变体。

### 10.5 MoE 的分布式兼容性仍待研究

MoE 中不同 Expert 的 LoRA 放置方式很多，如何与 Tensor Parallelism（TP）和 Expert Parallelism（EP）高效兼容，仍是开放问题。

## 11. 对 vLLM、verl 和 RL 权重更新架构的启示

以下属于基于文章结论的工程推论，并非文章已经实现或验证的系统方案。

### 11.1 LoRA 可以成为 RL 权重更新的一种轻量载体

传统 FullFT 更新流程通常需要传输大量基础模型权重：

```text
Trainer 更新完整模型参数
        |
        v
Weight Transfer 传输完整权重
        |
        v
Inference Engine 更新模型参数
```

LoRA 更新可以简化为：

```text
Trainer 更新 LoRA A/B
        |
        v
Weight Transfer 传输 Adapter 参数
        |
        v
Inference Engine 更新对应 Adapter
```

传输数据量可以从：

\[
S_{\mathrm{FullFT}}
\propto
\sum_j d_{\mathrm{in},j}d_{\mathrm{out},j}
\]

降低为：

\[
S_{\mathrm{LoRA}}
\propto
\sum_j r_j(d_{\mathrm{in},j}+d_{\mathrm{out},j})
\]

其中还需要乘以对应的数据类型字节数。

当 Rank 较小时，Adapter 更新的数据量可能远小于完整权重更新。

### 11.2 对 vLLM 的潜在价值

vLLM 已支持多 Adapter 推理。结合文章结论，可以探索：

- 基础模型权重保持冻结；
- Trainer 只更新 Adapter；
- 通过 Weight Transfer 传输 Adapter 参数；
- 推理端更新指定 Adapter，而不重载整个基础模型；
- 利用 Adapter 版本管理支持训练版本切换和多租户请求。

这可能减少 RL 训练和推理之间的权重传输压力，也能降低动态更新基础模型参数的复杂度。

### 11.3 需要重点解决的工程问题

| 方向 | 需要解决的问题 |
|---|---|
| Adapter 动态更新 | 如何更新 A/B，同时保证正在执行的请求不读取到不一致版本 |
| Adapter 版本管理 | 新旧 Adapter 如何并存、切换和回收 |
| Weight Transfer | 如何只传输发生变化的 Adapter 参数 |
| TP | A/B 如何按照张量并行布局切分和聚合 |
| EP / MoE | Expert Adapter 如何与 Expert 分布、路由和迁移对应 |
| Quantization / Packed | Adapter 与量化权重、融合权重的兼容性 |
| 并发请求 | 如何保证请求生命周期内 Adapter 版本一致 |
| 训练与推理一致性 | Trainer 与推理端如何确认 Adapter 的参数版本和更新完成状态 |

尤其是 MoE，文章验证了 MoE Expert LoRA 的模型效果，但没有解决大规模 TP/EP 下的完整部署问题。

## 12. 建议的实际验证方案

如果要验证文章结论在具体 RL 系统中的适用性，可以使用同一个 Base Model、同一训练数据和尽量一致的采样配置，对比：

| 维度 | 建议对照 |
|---|---|
| 训练方式 | FullFT、Rank-1、Rank-4、Rank-16 LoRA |
| LoRA 目标层 | Attention-only、MLP-only、MLP+Attention |
| 学习率 | FullFT 最优 LR、LoRA 约 10 倍 LR，并做 Sweep |
| Batch Size | 小、中、大 Batch |
| 训练效果 | 训练 Reward、验证 Reward、AIME 等 held-out 评测 |
| 推理行为 | CoT 长度、Backtracking、自我验证等行为变化 |
| 训练成本 | GPU 小时、训练 FLOPs、吞吐 |
| 显存 | 基础权重、梯度、优化器状态、Adapter 占用 |
| 更新成本 | Trainer 到 vLLM 的传输字节数和更新时间 |
| 分布式正确性 | TP、EP、DP 下 Adapter 参数切分与更新正确性 |

重点不是只验证 Rank-1 能否跑通，而是确认在目标任务上，LoRA 是否同时满足：

1. Reward 和泛化能力接近 FullFT；
2. 在相同训练预算下具有相近的学习效率；
3. 训练显存和优化器状态显著减少；
4. Adapter 权重更新确实降低了端到端同步成本；
5. 并行切分和在线更新不会引入参数错配或请求版本不一致。

## 13. 总结

这篇文章最重要的观点，是重新审视后训练阶段对参数更新自由度的需求。

传统理解容易将 LoRA 看成一种以牺牲模型效果换取训练资源节省的折中方案。Thinking Machines Lab 的实验则表明，在目标层覆盖合理、容量充足、学习率经过调整的条件下，LoRA 可以在多种 SFT 和数学推理 Policy Gradient RL 场景中达到与 FullFT 相近的效果。

最值得记住的结论：

1. **容量决定适用范围**：数据和任务所需信息超过 LoRA 容量后，LoRA 会逐渐落后。
2. **目标层比单纯提高 Rank 更重要**：Attention-only 可能明显弱于 MLP/MoE LoRA。
3. **RL 可能更适合低 Rank LoRA**：典型 Outcome-based Policy Gradient 的奖励反馈信息相对稀疏，但该结论不能推广到所有 RL。
4. **学习率不能照搬 FullFT**：文章实验中 LoRA 最优 LR 约为 FullFT 的 10 倍。
5. **系统价值不仅在训练显存**：Adapter 更小，也意味着更轻量的检查点、权重传输和多租户推理管理。

对 vLLM 生态而言，一个值得探索的方向是将 LoRA Adapter 作为 RL Trainer 与 Inference Engine 之间的标准化权重更新载体。但这需要进一步解决 TP/EP 分片、MoE Expert Adapter、在线更新原子性和版本一致性等系统问题。

---

## 参考资料

1. John Schulman, Thinking Machines Lab. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), 2025.
2. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
3. Malladi et al. [A Kernel-Based View of Language Model Fine-Tuning](https://arxiv.org/abs/2210.05643), 2022.
4. Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.
5. [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547), 2023.

## 原文引用信息

```bibtex
@article{schulman2025lora,
  author  = {John Schulman and Thinking Machines Lab},
  title   = {LoRA Without Regret},
  journal = {Thinking Machines Lab: Connectionism},
  year    = {2025},
  note    = {https://thinkingmachines.ai/blog/lora/},
  doi     = {10.64434/tml.20250929}
}
```
