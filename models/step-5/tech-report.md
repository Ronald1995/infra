# Step 5 Preview 技术报告解析：模型架构、Sparse GQA 与 Agentic RL 系统设计

> **技术报告：** [Inside Step 5 Preview](https://platform.stepfun.ai/blog/inside-step-5-preview)
>
> **核心主题：** Narrow-but-Deep、Gated SWA、Block-wise Sparse GQA、训练数值稳定性、Agentic RL、训推一致性与推理系统优化。

## 1. 概述

Step 5 Preview 是阶跃星辰推出的面向长程 Agent 任务的 MoE 大模型。

与单纯扩大模型参数量不同，Step 5 的主要设计目标是：在有限激活计算预算下，通过更深的网络、稀疏注意力以及训练与推理系统协同优化，提升模型处理复杂长上下文任务的能力。

报告中值得重点关注的创新包括：

1. **Narrow-but-Deep 架构：** 使用 92 层 Transformer，提高长 Prompt 中隐式多跳推理的信息处理深度。
2. **Gated SWA + Sparse GQA：** 将局部滑动窗口注意力与全局稀疏注意力结合，降低长上下文计算成本。
3. **Block-wise Indexer：** 将连续 8 个 Token 压缩为一个检索 Block，降低稀疏索引开销。
4. **Prefill Top-k Union：** 通过合并相邻 Query 的候选 KV 集合，提高 Sparse Attention 的硬件利用率。
5. **高精度关键状态：** 使用 FP32 Hidden State、FP32 RMSNorm Scale 等技术改善深网络训练稳定性。
6. **Zero Training-Inference Mismatch：** 通过确定性计算和 Trace Replay 降低 RL 训练与推理之间的偏差。
7. **Agentic RL 系统优化：** 引入 MTP、KV Offload、Session Affinity、Context Compaction 等机制，提高长程 Rollout 效率。

从 AI Infra 的角度来看，Step 5 的创新并不局限于某一个算子，而是围绕长上下文 Agent 构建了一套模型、训练和推理协同设计方案。

---

## 2. Step 5 Preview 模型架构

### 2.1 模型核心参数

| 参数 | Step 5 Preview |
|---|---|
| 总参数量 | 600B |
| 激活参数量 | 27B |
| Transformer 层数 | 92 |
| Hidden Size | 4096 |
| Q Heads | 64 |
| KV Heads | 4 |
| Attention Head Dimension | 192 |
| GQA Group Size | 16 |
| Gated SWA 层数 | 69 |
| Sparse GQA 层数 | 23 |
| SWA Window | 512 Tokens |
| Routed Experts | 352 |
| 每 Token 激活 Routed Experts | 8 |
| Shared Experts | 1 |
| MTP | 3 层 |
| 最大上下文 | 1M Tokens |

模型采用 MoE 架构，92 层 Transformer 中大部分使用 MoE FFN，少量层保留 Dense FFN。

Attention 采用约 3:1 的交错布局：

- 三层 Gated Sliding Window Attention（SWA）。
- 一层 Sparse Grouped Query Attention（Sparse GQA）。

这意味着 69 层使用局部 Attention，23 层负责稀疏的全局历史检索。

### 2.2 Narrow-but-Deep 的设计目的

传统大模型扩展通常可以从两个方向进行：

- **Wider：** 增大 Hidden Size、FFN Dimension 或专家规模。
- **Deeper：** 增加 Transformer 层数。

Step 5 选择相对窄而深的网络结构。

设计依据是：复杂 Agent 任务不仅需要记忆长上下文，还需要在生成下一个 Token 前，对上下文中的信息进行多跳组合、条件推理和依赖分析。

Transformer 的每一层都可以执行一次非线性信息变换，因此增加网络深度可能提高单次 Forward 内隐式信息整合的能力。

例如代码 Agent 的任务可能包含：

1. 阅读大型代码仓库。
2. 识别模块之间的依赖关系。
3. 推断某个 Bug 的传播路径。
4. 综合多个文件中的约束。
5. 生成满足约束的代码修改。

这些过程不一定全部通过显式 Chain-of-Thought 完成，部分推理发生在 Transformer 的层间计算中。

不过，增加深度也会引入额外开销：

- Decode 阶段串行执行的层数增加。
- Kernel Launch 和层间同步次数增加。
- Pipeline Parallel 的调度更加复杂。
- 深层网络的数值误差可能进一步累积。

因此，Step 5 的主要优化目标并不是最低单 Token 延迟，而是模型能力与整体训练、推理成本之间的综合平衡。

### 2.3 为什么结合 SWA 和 Sparse GQA？

对于百万 Token 上下文，所有层都使用 Full Attention 的计算和存储成本过高。

Step 5 将 Attention 分为两类：

**Gated SWA：**

- 每个 Query 只访问最近 512 个 Token。
- 保持标准 Softmax Attention 语义。
- KV Cache 可以只保留有限长度的历史。
- 更容易实现 MTP 验证和 Cache 回滚。

**Sparse GQA：**

- 能够访问更早的历史 Token。
- 使用 Indexer 选择重要的历史位置。
- 主 Attention 只对选中的历史 Token 执行计算。
- 保留长距离信息检索能力。

两者结合，使模型在大多数层使用低成本局部 Attention，同时利用少量全局稀疏层访问完整上下文。

---

## 3. Sparse GQA 的核心设计

Sparse GQA 是 Step 5 最值得关注的架构创新之一。

它的基本思路是：先使用轻量级 Indexer 确定哪些历史 Token 值得关注，再对这些位置执行真正的 GQA。

### 3.1 Block-wise Indexer

传统 Token-wise Sparse Attention 需要为每个历史 Token 计算 Index Score。

如果上下文长度为 1M，就意味着需要面对约 100 万个候选位置。

Step 5 引入 Block-wise Compression，将连续 8 个 Token 压缩为一个 Index Block。

关键参数：

| 参数 | 数值 |
|---|---|
| Index Block Size | 8 Tokens |
| Sparse Token Budget | 4096 Tokens |
| 对应选中 Block 数量 | 512 |
| Indexer Proxy Dimension | 256 |
| Index Query Heads | 16 |
| Index Key Heads | 1 |
| Provider Groups | 4 |

Indexer 的压缩并非简单平均池化，而是使用 Channel-wise Softmax Pooling 聚合 Block 中的表示。

假设上下文长度为 1,048,576：

- 原始候选 Token 数：1,048,576
- Block Size：8
- 压缩后候选 Block 数：131,072

因此，Indexer 的候选位置规模降低约 8 倍。

**这里压缩的是 Indexer 的检索表示，不是主 Attention 使用的原始 K/V。**

### 3.2 Group-wise Top-k

Step 5 的主 Attention 使用 64 个 Query Heads 和 4 个 KV Heads。

按照 GQA 结构，每个 KV Head 对应 16 个 Query Heads。

Sparse Indexer 为不同 KV Group 独立生成 Top-k Block 索引。

例如：

| KV Group | 选中 Block 示例 |
|---|---|
| Group 0 | 1、5、18、29 |
| Group 1 | 2、5、9、21 |
| Group 2 | 1、9、17、31 |
| Group 3 | 3、9、18、25 |

不同 KV Group 可以关注不同历史区域，提高检索的表达能力。

因此，在算子实现中，不能默认所有 KV Head 共用一份 Sparse Block Index。

### 3.3 从 Block Index 映射到真实 KV

Sparse Indexer 输出的是 Block ID，而不是直接提供主 Attention 使用的物理 KV 地址。

例如选中：

`block_id = 123`

并且：

`block_size = 8`

那么对应的原始逻辑 Token 位置为：

`token_position = block_id × block_size + offset`

其中 `offset` 的取值为 0～7。

因此，该 Block 对应的 Token 位置是：

`[984, 985, 986, 987, 988, 989, 990, 991]`

主 Attention 需要根据这些位置读取真正的 K/V 数据。

如果底层使用 Paged KV Cache，还需要进行逻辑 Token 到物理 Cache Slot 的映射。

完整过程可以理解为：

1. Query 输入 Indexer。
2. Indexer 计算 Block Score。
3. 每个 KV Group 独立选择 Top-k Block。
4. 将 Block ID 转换为原始 Token Position。
5. 根据 Paged KV Cache 的 Block Table 获取实际物理位置。
6. Gather 相应 K/V。
7. 执行 Sparse GQA 和输出投影。

此外，需要保留当前 Query 所在 Block 内因果可见的 Token，避免遗漏本地信息或访问未来 Token。

### 3.4 为什么采用 Block Size=8？

扩大 Index Block Size 可以减少候选索引数量，但也会降低选择粒度。

报告比较了不同配置的检索覆盖率：

| 配置 | 主 Attention Token Budget | Token Coverage |
|---|---:|---:|
| Block Size=1 | 2048 | 79.09% |
| Block Size=8 | 4096 | 78.06% |

选择 Block Size=8 后，Indexer 搜索空间大幅缩小，同时通过扩大主 Attention 的 Token Budget，维持接近原先的检索覆盖率。

这个设计体现的是一种成本转移：

- 减少 Indexer 的搜索成本。
- 适当增加主 Attention 的 Token 预算。
- 保持整体检索质量。

需要注意，Token Coverage 是报告使用的检索指标，并不能直接等同于最终任务准确率。

### 3.5 Prefill Top-k Union 优化

独立处理每个 Query 的 Top-k Sparse Attention，可能使 Prefill 阶段形成大量规模很小的矩阵计算。

例如相邻三个 Query 的索引：

| Query | 选中 Block |
|---|---|
| Q0 | A、B、C、D |
| Q1 | A、B、C、E |
| Q2 | A、B、D、E |

三个 Query 的候选历史有大量重叠。

Step 5 使用 Top-k Union，将候选 Block 合并为：

`Union = [A, B, C, D, E]`

多个 Query 可以基于共享候选集合组织更大的矩阵计算，减少重复 Gather 与碎片化计算。

不过，Union 之后仍必须保留各 Query 的有效选择 Mask。

也就是说，Q0 不能因为 Union 包含 E，就额外访问原本没有选中的 E。

Top-k Union 主要优化：

- Prefill Attention 的 GPU 利用率。
- 相邻 Query 的数据复用。
- 小矩阵计算与 Kernel 调度开销。
- 稀疏 K/V Gather 的组织方式。

这一点对 vLLM-Ascend 的 Sparse Attention Kernel 设计尤其重要。

### 3.6 Sparse GQA 是否减少 KV Cache 显存？

这是一个容易混淆的问题。

Sparse GQA 主要减少 Attention 计算时读取的历史 KV 数量，并不意味着可以只保留 4096 个 Token 的历史 KV Cache。

因为下一次 Decode 可能选中完全不同的历史位置，必须保留可被检索的完整历史。

按 Step 5 的 BF16 GQA 参数估算：

单层、单 Token 的 KV 存储量为：

`KV Bytes = 2 × KV Heads × Head Dimension × Bytes Per Element`

代入配置：

`KV Bytes = 2 × 4 × 192 × 2 = 3072 Bytes`

23 层 Sparse GQA、1,048,576 Token 的理论 KV 存储量：

`23 × 1,048,576 × 3072 / 1024³ = 69 GiB`

69 层 SWA 如果每层保留约 512 Token，则理论 KV 量约为：

`69 × 512 × 3072 / 1024² ≈ 103.5 MiB`

因此：

- Sparse GQA 层需要保存可供全局检索的完整历史。
- SWA 层可以只保存滑动窗口范围内的历史。
- 真正显著减少总 KV Cache 容量的是 GQA、SWA 交错布局，以及其他 Cache 压缩和 Offload 方案。

以上只是 BF16 文本 Attention KV 的理论估算，没有计入 Indexer 状态、内存对齐、分页碎片和运行时缓冲区。

---

## 4. 训练稳定性设计

Step 5 的深层架构增加了训练数值稳定性方面的挑战。

报告重点介绍三项优化。

### 4.1 FP32 Hidden State

Step 5 发现，在较深网络中，某些 Hidden State 的 Massive Activations 会随着层数增加而放大。

即使模型使用 FP8 或 BF16 进行主要矩阵运算，低精度 Residual Stream 也可能引入较大的数值误差。

因此，Step 5 将 Hidden State 及其对应梯度保持为 FP32。

这不意味着整个模型都使用 FP32：

- GEMM 仍可采用 FP8 或其他低精度类型。
- Residual Stream 使用 FP32。
- 特定算子边界执行精度转换。

报告称，该方案在额外开销有限的情况下改善了训练稳定性，并降低了异常激活峰值。

### 4.2 FP32 Zero-centered RMSNorm Scale

Step 5 采用 Zero-centered RMSNorm。

其计算可以表示为：

`output = input / RMS(input) × (1 + gamma)`

其中 `gamma` 是相对于 1 的可学习偏移。

当某个通道出现异常激活时，模型可能通过将 `gamma` 调整至接近 -1 来抑制该通道。

但 BF16 在接近 -1 时的表示精度有限，可能将接近 -1 的值舍入为精确的 -1。

这会使相应通道突然完全关闭，而后续参数更新可能又将其打开，造成训练不稳定。

Step 5 将 RMSNorm 的 Scale 参数保存为 FP32，减少这种数值问题。

需要注意，**FP32 Scale 参数与在 RMSNorm 内部使用 FP32 累加并不是同一件事。**

### 4.3 Fine-grained Gradient Clamp

报告发现，深层训练中可能出现少量异常梯度，这些梯度经过多个网络层传播后造成明显的 Gradient Spikes。

Step 5 采用 Fine-grained Gradient Clamp，仅针对极少量异常梯度进行处理。

报告提到每层被 Clamp 的梯度比例约为：

`0.00004%`

这说明造成不稳定的梯度可能极其稀疏，并不需要裁剪所有梯度。

与传统 Global Gradient Clipping 相比，该方案更加细粒度，目的是尽量保留正常梯度信息。

不过，公开报告尚未提供完整阈值选择及 Kernel 实现细节。

---

## 5. Agentic 数据合成系统

Step 5 的另一个研究重点是构建可规模化生产高难度 Agent 任务的数据系统。

传统数据合成主要依赖人工设计问题、脚本生成以及过滤。

Step 5 则引入 Agent 驱动的数据生成流程：

1. 领域专家定义高质量任务原则。
2. Agent 根据知识来源挖掘任务候选。
3. Agent 组织复杂任务和执行环境。
4. 通过确定性工具运行构建、测试与验证。
5. 评估任务难度、覆盖率和多样性。
6. 将成功及失败经验反馈至共享 Context。

其中一个重要原则是：

**由 Agent 决定任务生成策略，由确定性工具负责执行和验证。**

例如代码任务的构建、单元测试、Patch 验证、覆盖率统计，应尽量由稳定的程序完成，而不是完全依赖模型判断。

报告称，该基础设施生成了百万级可验证的复杂任务环境，覆盖软件工程、科学研究和机器学习研发等领域。

其意义不仅是扩大数据规模，还在于提升任务的真实性、可验证性和难度分布。

---

## 6. Agentic RL：训练与推理一致性

Step 5 的 RL 系统设计值得重点研究，尤其是 Zero Training-Inference Mismatch 和 Trace Replay。

### 6.1 两类训推不一致问题

报告将训练与推理的偏差分为两类。

**第一类：相同权重下的数值不一致。**

即：

`Rollout Weights = Training Weights`

但由于不同算子、并行策略或浮点规约顺序，训练和推理得到的结果仍然不完全相同。

可能的来源包括：

- FP8 算子。
- MoE Expert Routing。
- Sparse Attention Top-k。
- 不同 TP Size 的并行规约。
- Softmax 和其他数值运算。

**第二类：Policy Staleness。**

Rollout 使用旧权重，而训练阶段的模型已经更新。

例如：

`Rollout: W_t`

`Training: W_(t+1)`

这时即使计算完全确定，两个 Policy 的输出也可能不同。

Step 5 分别设计了对应的解决方案。

### 6.2 Staleness=0：Bitwise Alignment

针对相同权重下的数值偏差，Step 5 优化了训练和推理算子的计算行为。

其中一个重要技术是确定性的 Tree Reduction。

即使训练和推理采用不同的 TP Size，也尽量保证计算规约顺序一致。

报告称，在 Staleness=0 的条件下可以达到 Bitwise Identical，并使 Logprob Difference 为 0，额外开销低于 10%。

这对 RL 非常重要。

因为即使浮点误差很小，也可能造成：

- MoE Router 选择不同 Expert。
- Sparse Indexer 选择不同 Block。
- Top-p 采样候选集合变化。
- 长序列 Logprob 偏差积累。

因此，对 RL 来说，数值接近并不一定代表执行路径完全一致。

### 6.3 Staleness>0：Trace Replay

对于训练权重和 Rollout 权重不完全一致的场景，Step 5 采用 Trace Replay。

报告主要介绍两类 Replay。

**MoE Routing Replay**

在 Rollout 时记录：

- Expert Index。
- Routing Coefficient。

训练时复用这些离散路由决策，避免训练侧重新选择不同 Expert。

例如 Rollout 使用：

| Expert | Routing Weight |
|---|---:|
| Expert 2 | 0.48 |
| Expert 7 | 0.32 |
| Expert 9 | 0.20 |

如果训练重新路由到 Expert 3、7、9，那么训练与推理实际经过了不同的计算路径。

MoE Routing Replay 通过复用原有路由状态降低这一偏差。

**Top-p Sampling Replay**

Top-p 采样首先确定一个累计概率达到阈值的候选 Token 集合，然后在该集合内采样。

如果训练侧重新计算 Top-p 集合，权重和数值变化可能使候选集合发生变化。

Step 5 保存 Rollout 阶段的 Top-p 候选集合，并在训练阶段基于该集合重新计算 Logprob。

报告称，结合 Replay 等技术，长轨迹中的 Logprob Difference 能够保持在 `10^-2` 以下。

需要特别区分：该报告明确讨论的是 MoE Routing 和 Top-p Candidate Replay，并没有说明会保存、重放整个模型的 Hidden States。

### 6.4 Logit Gradient Zero-center Projection

Step 5 还引入了 Logit Gradient Zero-center Projection。

其目的是抑制 Logit 的共同模式漂移，改善长程 RL 中的 Entropy 稳定性。

对于长轨迹训练，这类数值优化可以降低策略在大量更新之后出现异常收缩或漂移的风险。

---

## 7. RL 推理系统性能优化

Step 5 不仅修改模型和训练算法，还对 Rollout 基础设施进行了优化。

### 7.1 MTP-3 和 FP8 MoE

报告公布了以下系统性能结果：

| 优化项目 | 报告中的收益 |
|---|---|
| MTP-3 | 约 30% 端到端加速 |
| Native FP8 MoE | 约 20% 额外效率提升 |
| KV Cache 综合优化 | 有效容量提升约 8～12 倍 |
| Rollout Lifecycle 管理 | 样本丢失率降低至 1% 以下 |

其中，KV Cache 容量提升来自多种技术组合，包括：

- CPU KV Offload。
- FP8 KV Cache QAT。
- MXFP4 权重存储。
- EP 协调和内存管理优化。

这些数字属于报告公布的特定系统结果，不能直接视为所有硬件、模型并行度和 Batch 配置下的通用收益。

### 7.2 Session Affinity

长程 Agent 通常存在大量 Tool Calling。

一个 Agent Session 可能连续执行：

- 模型推理。
- 工具调用。
- 工具结果处理。
- 再次模型推理。

如果这些 Request 被不断分配到不同推理实例，可能影响缓存局部性与状态复用。

Step 5 使用 Session Affinity，尽可能将同一 Session 的请求调度到相同实例。

这种设计可以提升长程 Agent 的系统效率，但也要求调度器处理实例负载不均、排队和失败重试等问题。

### 7.3 Rollout Lifecycle Ledger

长程 RL 的一个隐藏问题是：复杂任务往往需要更多工具调用和更长执行时间，因此更容易发生超时、异常中断和数据丢失。

如果这些任务的 Rollout 被系统性遗漏，训练数据会逐渐偏向简单任务。

Step 5 为每个 Rollout 建立生命周期记录，确保任务最终进入明确的完成或失败状态。

这使系统能够更准确地统计：

- 实际完成的 Rollout。
- 超时或失败的 Rollout。
- 需要重新调度的任务。
- 不同难度任务的样本覆盖情况。

### 7.4 Context Compaction

百万 Token 上下文依然存在长度上限。

对于执行数小时甚至更长时间的 Agent 任务，单纯扩大 Context Window 并不是最终解决方案。

Step 5 引入 Context Compaction：

1. Agent 执行当前任务。
2. 上下文接近长度上限。
3. 将当前任务状态压缩为紧凑表示。
4. 开启新的上下文。
5. 利用压缩状态恢复任务并继续执行。

区别于简单的外部摘要方案，Step 5 将 Context Compaction 作为 RL Policy 的一部分，通过任务 Reward 训练模型保留和恢复关键状态的能力。

这使长程 Agent 能够跨越固定 Context Window，继续处理同一任务。

---

## 8. Step 5 与 DeepSeek DSA 的架构差异

Step 5 的 Sparse GQA 与 DeepSeek V3.2 DSA 在高层思路上类似：利用学习式索引器选择重要历史位置，然后执行稀疏 Attention。

但底层结构存在明显区别。

| 维度 | DeepSeek V3.2 DSA | Step 5 Sparse GQA |
|---|---|---|
| 主 Attention | MLA | Gated GQA |
| 历史选择 | 学习式 Indexer | 学习式 Indexer |
| 主要检索粒度 | Token 级 | 8-Token Block |
| 主 Attention 输入 | MLA 对应的历史表示 | 原始 GQA K/V |
| Indexer 优化重点 | 历史 Token 检索 | Block 压缩与分组检索 |
| Prefill 重点 | 稀疏 Attention 计算 | Top-k Union 与数据复用 |

需要注意，实际实现中的 Kernel Fusion 和 Block 化方案可能随版本演进，因此不能仅根据架构表判断两者的真实性能优劣。

Step 5 的特点是围绕标准 GQA 构建了独立的稀疏检索流程，并重点优化长上下文 Prefill 的计算利用率。

---

## 9. 对 vLLM / vLLM-Ascend 的技术启示

如果从 vLLM-Ascend 的模型适配和算子实现角度分析，Step 5 的技术主要涉及以下模块。

### 9.1 Sparse GQA 算子实现

建议按功能拆分实现。

| 模块 | 主要职责 |
|---|---|
| Index Compression | 将连续 Token 聚合为 Index Block |
| Indexer Scoring | 计算 Query 与候选 Block 的相关性 |
| Group-wise Top-k | 各 KV Group 独立选择 Block |
| Sparse KV Mapping | 将逻辑 Block 映射到原始 KV Cache |
| Sparse GQA Kernel | 执行稀疏 Attention 与输出计算 |
| Prefill Top-k Union | 合并相邻 Query 候选并复用计算 |
| KV Cache Manager | 分别管理 SWA 和全历史 Sparse GQA 的 KV |

需要重点关注三个 Correctness 问题。

**第一，Indexer Block 和 KV Cache Page 不是同一个概念。**

Indexer 的 Block Size=8，但 Paged KV Cache 的 Page Size 可能更大。

必须保证逻辑 Block、Token Position 和物理 Slot 的映射正确。

**第二，Group-wise Top-k 不能随意共享索引。**

不同 KV Group 的 Top-k 选择可能不同，Kernel 应保留相应的索引维度和 Mask 语义。

**第三，Prefill Top-k Union 不能改变原始稀疏注意力语义。**

多个 Query 可以共享合并后的数据加载，但仍应只访问各自原本允许的历史位置。

### 9.2 Prefill 与 Decode 的优化重点不同

对于 Decode：

- Query 数量少。
- 重点降低 Indexer 和 Top-k 延迟。
- 重点优化随机 KV Gather。
- 需要控制 Kernel Launch 数量。

对于 Prefill：

- Query 数量大。
- 相邻 Query 的 Top-k 存在重叠。
- 可以利用 Top-k Union 和较大的 Attention Tile。
- 更需要关注吞吐与计算利用率。

因此，Step 5 的 Sparse GQA 不适合仅用一个通用 Kernel 解决所有场景。

### 9.3 RL 训推一致性

如果 Step 5 在 vLLM-Ascend 中用于 RL Rollout，还需要关注：

- 训练和推理的 MoE Routing 是否一致。
- Sparse Indexer 的 Top-k 是否具有稳定排序行为。
- 不同 TP / EP 配置是否改变数值规约顺序。
- FP8 GEMM 是否引入不同误差。
- Top-p Candidate 是否可以从 Rollout 侧传递至训练侧。
- Expert Index 和 Routing Weight 是否需要保存用于 Replay。

这些问题不仅影响模型 Forward 的数值误差，也可能改变实际执行的稀疏计算路径。

### 9.4 KV Cache 与 Offload

由于 Sparse GQA 必须保留可检索的历史 KV，而 SWA 只需要有限窗口，KV Cache Manager 应当区分两类层的生命周期。

在长上下文场景下，还需要结合：

- CPU KV Offload。
- FP8 KV Cache。
- Paged KV Cache 内存管理。
- 多实例 Session Affinity。
- 长程 Rollout 的 Cache 回收策略。

不能将 Sparse Attention 的计算稀疏度，直接等同于 KV Cache 的存储压缩比例。

---

## 10. 总结

Step 5 Preview 体现的是一种面向长程 Agent 的系统级模型设计思路。

它的技术价值主要体现在四个方面。

**第一，模型架构更重视深度与长程信息整合。**

92 层 Narrow-but-Deep 结构试图增强复杂上下文内的隐式推理能力，再通过 SWA 和 Sparse GQA 控制 Attention 成本。

**第二，Sparse GQA 不只是减少主 Attention 的计算量。**

Step 5 将 Block-wise Index Compression、Group-wise Top-k 和 Prefill Top-k Union 结合起来，同时优化索引成本、检索质量和计算利用率。

**第三，RL 训推一致性需要系统级解决。**

Bitwise Alignment 负责尽可能消除相同权重下的数值差异，Trace Replay 则用于降低 Policy Staleness 引起的离散计算路径偏差。

**第四，长程 Agent 的基础设施同样决定模型能力。**

MTP、KV Offload、Session Affinity、Rollout Lifecycle、Context Compaction 和细粒度 Reward，都是支撑长任务训练与推理的重要机制。

对于 vLLM 和 vLLM-Ascend 的后续研究，建议优先关注：

1. Sparse GQA Indexer、Block Mapping 与 Prefill Top-k Union。
2. 训练与推理跨 TP 配置的确定性计算。
3. MoE Routing 与 Top-p Sampling Trace Replay。
4. 长程 RL Rollout 的缓存、会话和生命周期管理。

**最终结论：Step 5 最值得借鉴的不是某个孤立的模型结构，而是将长上下文稀疏计算、训练数值稳定性、训推一致性和 Agentic RL 基础设施统一设计的方法。**

---

## 参考资料

- [Inside Step 5 Preview — StepFun 技术报告](https://platform.stepfun.ai/blog/inside-step-5-preview)
- [StepFun 开发者平台](https://platform.stepfun.ai/)
- [vLLM 项目](https://github.com/vllm-project/vllm)
- [vLLM-Ascend 项目](https://github.com/vllm-project/vllm-ascend)
