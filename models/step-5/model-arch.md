# Step-5-Preview 模型架构与 Sparse GQA 实现技术总结

> **版本**：2026-10-19  
> **范围**：模型结构、Sparse GQA、MoE、数值稳定性、MTP、多模态、与 Qwen3.8-Flash-Next 对比，以及 vLLM / vLLM-Ascend 适配关注点。  
> **资料级别**：官方博客、官方公开 checkpoint 配置、开源参考实现与 vLLM 代码。下文将明确区分「模型算法」「公开推理实现」和「特定 NPU 优化实现」。

## 0. 摘要

Step-5-Preview 是一款面向长上下文 Agent 工作负载的 **600B 总参数、约 27B 激活参数**的稀疏 MoE 模型。最核心的架构取舍是：

1. **Narrow-but-Deep**：92 层 Transformer、Hidden Size 4096；增加深度以提升长 Prompt 的多层信息整合能力。
2. **混合 Attention**：按 **3 层 Gated Sliding Window Attention（SWA）+ 1 层 Sparse GQA** 重复，共 69 层 SWA、23 层 Sparse GQA；窗口为 512 Token，最长上下文配置为 1,048,576 Token。
3. **Block Sparse GQA**：独立 Indexer 将 **8 个历史 Token 压缩为 1 个 Region Key**，对 **4 个 KV Provider Group 分别进行 Top-512 Region 选择**；主 Attention 仍读取对应的**原始 Token 级 K/V**，而不是压缩的 Index Key。
4. **细粒度 MoE**：88 个 MoE 层（编号 3–90），每层 352 Routed Experts、Top-8，并有 Shared Expert；首 3 层和最后 1 层是 Dense FFN。
5. **稳定性与系统设计**：FP32 Residual、FP32 Zero-centered RMSNorm Scale、Clamped SwiGLU、MTP-3、视觉编码器，以及训练/推理对齐优化。

**一句话概括**：Step 5 并非仅增加参数，而是将深层 MoE、局部窗口、块级稀疏长程检索与高精度残差流组合，以服务长期 Agent 任务；其中 Sparse GQA 是推理框架需要重点新增的算子路径。[1][2]

## 1. 模型全局结构和配置

| 模块 | 公开配置 | 说明 |
|---|---|---|
| 模型规模 | 约 600B / 激活 27B | 官方总体口径，不等同于简单的 Top-K 专家矩阵之和 |
| Decoder 深度 | 92 层 | Narrow-but-Deep |
| Hidden Size | 4096 | 较深但非超宽 |
| Attention | Q Heads = 64, KV Heads = 4, Head Dim = 192 | 标准 GQA 主 K/V 组织 |
| Attention 层排布 | SWA ×3 + Sparse GQA ×1 | 69 SWA / 23 Sparse GQA |
| Gated SWA | Window = 512 | 近邻 Token 的局部注意力 |
| Sparse GQA | 每 KV Group Top-512 Regions，Region Size = 8 | 每组至多约 4096 个**选中历史 Token**，外加当前 Region 的因果可见部分 |
| Sparse Indexer | Q Heads = 16，K Heads = 1，proxy_dim = 256 | 4 个 Provider Groups；索引网络独立于主 Q/K/V |
| RoPE | SWA θ=1e4；Sparse GQA θ=1e7 | SWA 通常 192 旋转维；Sparse GQA 64 旋转维；Indexer RoPE 另设 32 维 |
| Dense/MoE | Dense 0–2、91；MoE 3–90 | 4 Dense 层 + 88 MoE 层 |
| Experts | 352 Routed + 1 Shared；Top-8 | Intermediate = 1536，FP32 Router |
| Dense FFN | Intermediate = 13824 | Clamped SwiGLU 路径 |
| 数值设计 | FP32 Residual、FP32 Norm、Zero-centered RMSNorm | 深度稳定性 |
| Speculative Decoding | 3 个 MTP 预测层 | 不计入 92 层主干 |
| Vision | 47 层 Perception Encoder，Width 1536 | 728×728、Patch14、16 Heads、Quick GELU |
| 最大序列长度 | 1,048,576 | 这是模型配置上限，不保证所有设备配置均可实际部署 |

**一组 Transformer 层的抽象结构**：

```text
Token / Image Embedding
        |
Transformer Blocks × 92
        |
        +-- 每四层：[SWA, SWA, SWA, Sparse GQA]
        |       +-- Attention Head-wise Gate
        |
        +-- FFN：[0..2、91] Dense；[3..90] MoE
        |       +-- FP32 Router -> Top-8 Routed Experts
        |       +-- Shared Expert
        |
        +-- FP32 Residual + Zero-centered RMSNorm
        |
Final Norm -> LM Head
        +-- MTP-3（辅助候选生成与验证）
```

> 注意：此图刻画模块归属，省略实际 Norm/Residual 的精确前后顺序、视觉输入细节及并行切分。[2]

## 2. 混合 Attention：Gated SWA + Sparse GQA

### 2.1 为什么不是所有层都做长程 Attention

长达 1M Token 时，全量 Attention 的计算和内存访问很昂贵。Step 5 按 3:1 交织两种 Attention：

- **69 层 Gated SWA**：关注近邻窗口（512 Token），控制局部计算和长期 KV 保存需求。
- **23 层 Sparse GQA**：从整个历史上下文中检索有价值的 Region，补足跨工具调用、跨轮对话的长程信息访问。

官方报告说明在效率和准确率权衡后采用 GQA，而没有在主路径使用 MLA；GQA 的代价是主 K/V 较大，因而需要 Indexer 压缩与 Sparse Kernel 优化。[1]

### 2.2 不同位置编码与门控

SWA 和 Sparse GQA 采用不同的 RoPE 配置：前者 `rope_theta=10000`、完整 Head 旋转；后者 `rope_theta=10000000`、`partial_rotary_factor≈1/3`，即 192 维 Head 中大约 64 维参与 RoPE。两类层还启用了 `use_head_wise_attn_gate=true`。[2]

**实现注意**：不能给全部 92 层使用同一个 RoPE 参数，不能省略 Head-wise Attention Gate。

## 3. Sparse GQA 的完整算子链路

### 3.1 两套 K/V 语义

Sparse GQA 实际存在两套数据路径：

```text
Hidden States
   |
   +---- Indexer Q/K/Z/W Projection
   |       |
   |       +-- Q Norm / RoPE
   |       +-- K Norm / RoPE + Z
   |       +-- CSA：8 Token -> 1 Region Key
   |       +-- Group-wise Score -> Top-512 Region IDs
   |
   +---- Main Q/K/V Projection
           |
           +-- Main Paged KV Cache：保存原始 Token 的 K/V
           |
           +-- 按 TopK Region / 当前 Region 因果尾部读取 K/V
           +-- Sparse GQA(Q,K,V) -> Attention Gate / O Projection
```

**重点**：CSA 的 8:1 是 **Indexer Key 表示的压缩比**，**不表示主 GQA K/V Cache 也缩小 8 倍**。[1]

### 3.2 CSA：Content-adaptive Sparse Attention Block Compression

公开参考实现的关键思想是用额外的 `z` 特征对 Region 内 K 特征进行内容自适应加权，而不是平均池化。可抽象为逐特征维度的：

\[
\alpha_{t,d}=\frac{\exp(z_{t,d})}{\sum_{j\in\mathcal R_b}\exp(z_{j,d})},\qquad
c_{b,d}=\sum_{t\in\mathcal R_b}\alpha_{t,d}k_{t,d}.
\]

其中 `\mathcal R_b` 有最多 8 个 Token，`d` 为 256 维代理特征坐标。Indexer 使用压缩 `c_b` 参与评分；主 Attention 不直接使用 `c_b` 作为 Value。实际实现的 Norm/RoPE、BF16/FP8 舍入顺序需要以对应 checkpoint 的推理代码为准。[1][3]

以最长 1,048,576 Token 为例：

\[
N_{\rm region}=1,048,576/8=131,072.
\]

因此评分候选数量理论上减少约 8 倍；这并不意味着 TopK 执行或整体推理延迟恰好减少 8 倍。

### 3.3 Group-wise Indexer Scoring 和 TopK

公开模型配置：Indexer Q Heads = 16，K Heads = 1，4 个 Provider Groups。与主 GQA 每组 16 个 Q Heads、1 个 KV Head 对齐，每个 Provider Group 可使用 4 个 Indexer Query Heads。

可用下式表征独立评分：

\[
 s_{q,g,b}=\sum_{h=1}^{4} w_{q,g,h}\,\operatorname{ReLU}\big(\langle q^I_{q,g,h},c_b\rangle\big),
 \qquad I_{q,g}=\operatorname{TopK}_{b}(s_{q,g,b},512).
\]

这里的 `w` 是动态 Indexer Head 权重；实际比例因子、归一化、特殊的 `ssmax` 配置和 FP8 数值处理按参考实现核对。**4 个 KV Group 可选不同的历史 Region**，这与多个 Group 共享单套 TopK 不同。[2][3]

### 3.4 TopK Region Index 如何查找主 KV Cache

必须区分：

- **Indexer 的逻辑 Region**：每 8 个 Token 的选择单位。
- **Paged KV Cache 的逻辑 Page**：按 KV Page Size 管理 Token。
- **物理 KV Page**：Allocator 在设备内存中分配的 Cache 块。

设选中 Region 为 `b`，Region Size 为 `R=8`，其内部偏移为 `j∈[0,7]`，主 KV Cache Page Size 为 `P`，请求编号为 `r`。则：

\[
 t=bR+j;
 \quad p_{\rm logical}=\lfloor t/P\rfloor;
 \quad o=t\bmod P;
 \quad p_{\rm physical}=\operatorname{block\_table}[r,p_{\rm logical}].
\]

访问 K/V：

```python
key   = k_cache[physical_page, offset, kv_head, :]
value = v_cache[physical_page, offset, kv_head, :]
```

上述表达是假设 K/V Cache 逻辑布局为 `[num_pages, page_size, kv_heads, head_dim]`；真实 NPU Kernel 可以改用 Region Packing 或自定义 Stride。**Region Size 不要求等于 KV Page Size，两个 Cache 可以拥有不同的物理 Block Table；通过请求内的逻辑 Token 坐标关联**。

### 3.5 Prefill 与 Decode

**Prefill**：大量 Query 各自选择 Region 后，数据访问容易碎片化。Step 5 官方提出相邻 Query 的 **Union of Selected Regions + Per-Query Mask**：将重复选中的 K/V Region 归并以提高批量利用率，同时为各 Query 保留原本的稀疏可见性和因果 Mask。

**Decode**：主要瓶颈是长历史的 Indexer Score/TopK、非连续 K/V 读取和频繁 Kernel Launch。需关注评分融合、Region ID 打包、持久化 Buffer、Graph Replay、MTP 多 Query 以及分片 Attention 的 Softmax 正确归并。不能把生产级方案和独立参考代码视为完全相同的 Kernel。[1][3]

## 4. 与 Qwen3.8-Flash-Next QSA 对比

| 对比项 | Qwen3.8-Flash-Next QSA | Step 5 Sparse GQA |
|---|---|---|
| 混合结构 | 36 Gated DeltaNet + 12 QSA（48 层） | 69 Gated SWA + 23 Sparse GQA（92 层） |
| 主 Attention | 24 Q / 2 KV / Head Dim 256 | 64 Q / 4 KV / Head Dim 192 |
| Indexer | 4 Q Heads / 1 K Head / Dim128 | 16 Q Heads / 1 K Head / Dim256 |
| Index Block Size | 4 | 8 |
| Index Compression | 4-token 平均池化 | 8-token CSA 学习型加权池化 |
| Indexer Score | 对 Head 的 ReLU dot-score 求和 | 每 KV Group 独立的带权多 Head ReLU 评分 |
| 选择结果 | 一套 Top-512 Blocks | 4 组各自 Top-512 Regions |
| 主 KV 访问预算 | 每组约 2048 历史 Tokens | 每组约 4096 历史 Tokens |
| 公共 vLLM NVIDIA 实现 | Block ID 展成 Token ID 后，Sparse Triton Kernel 查主 `block_table` | 参考实现含 Physical Region Metadata Packing；950DT 生产镜像具体实现仍需单独确认 |
| 其他模型设计 | GDN 状态机制、PLE、MTP | 深层混合 SWA、FP32 Residual、MTP-3 |

**最根本差异是 Indexer 的数学结构，而不是 4 vs 8 的常量修改。** Qwen 的索引选择更简洁；Step 5 通过内容自适应压缩和独立 KV Provider Group 提升选择灵活性，代价是 Indexer 参数、评分与分组 Cache 的复杂性。[4][5]

### 4.1 Qwen 代码中可验证的 KV 映射

在 `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`，展开逻辑等价于：

```python
block_rank = columns // compress_ratio
offset = columns % compress_ratio
logical_token = selected_block * compress_ratio + offset
```

在 `vllm/models/qwen4_exp/nvidia/ops/qsa.py`，主 Sparse Attention Kernel 执行：

```python
logical_page = logical_token // page_size
page_offset = logical_token % page_size
physical_page = block_table[request, logical_page]
# 从 main K/V 的 physical_page + page_offset 加载
```

该实现**直接在 Sparse Kernel 中加载非连续 Paged K/V**，并不强制先复制成紧凑的 K/V Tensor。[4][5]

## 5. MoE：容量与深度扩展

Step 5 的 MoE 为常见的 `Router + Routed Experts + Shared Expert` 模式，而非全新路由范式。

- 层编号 3–90 为 MoE；层 0–2、91 为 Dense FFN。
- 每层 352 个 Routed Experts，Top-8；另有 Shared Expert。
- Routed Expert Intermediate Size = 1536；Dense FFN Intermediate Size = 13824。
- Router Bias、FP32 Gate、专家输出归一化和 Scaling 的数值语义需要保留。
- SwiGLU 路径采用 Clamp，`swiglu_limits=7`、`swiglu_limits_shared=7`。

粗略估算每个三线性层专家的主体参数数目：

\[
3\times4096\times1536\approx18.87\text{M},
\]

88 层、每层 352 Routed Experts 合计约 **585B Routed Expert 主矩阵参数**。这只是估算项，不包括共享专家、Embedding、Attention、Router、投影与特殊模块，不能拿来替代官方总参数精确统计。[2]

## 6. 深层网络的数值稳定性

### 6.1 FP32 Residual

公开配置启用 `fp32_residual_connection=true`，并设置 `norm_dtype=float32`。这是为了抑制 92 层网络中少量异常大激活不断放大的问题。计算层可在 BF16/FP8 等低精度执行，但残差流保留 FP32 语义。[1][2]

### 6.2 Zero-centered RMSNorm

\[
 y_i=\frac{x_i}{\mathrm{RMS}(x)}(1+\gamma_i).
\]

其中 `γ` 为可学习参数，重要的是 `(1+γ)` 而不是 `γ`。官方指出当 γ 接近 -1 时，BF16 精度可能导致通道关闭/再开启振荡，因而使用 FP32 Scale。[1]

### 6.3 Clamped SwiGLU 与细粒度梯度截断

Clamped SwiGLU 限制极端中间激活，以改善数值稳定性；在推理端 **Clamped SwiGLU 和普通 SwiGLU 并非严格等价**。训练端还引入 fine-grained gradient clamp，这属于训练数值机制，不应误称为新增的推理计算层。[1][2]

## 7. MTP、多模态与训练/系统机制

- **MTP-3**：三个额外的多 Token Prediction 层，服务 Speculative Decoding，主干仍为 92 层；官方给出特定 RL Rollout 系统约 30% 端到端加速，不应泛化为所有服务场景。[1][2]
- **视觉**：Perception Encoder 47 层、Width 1536、16 Heads、728×728、Patch14，Projector 映射至语言模型；服务层的视频输入能力需看具体部署接口。[2]
- **训练与推理对齐**：官方介绍训练/推理稀疏选择、MoE Router Replay、Top-p Candidate Replay 和跨 TP 拓扑的 Logprob 一致性工作；属于系统及 RL 训练设计，不应混为主网络新增结构。[1]

## 8. KV Cache / 性能量级分析

以每层 `KV Heads=4`、`Head Dim=192`、BF16 K/V 为例，每个 Token 每层主 K/V 的存储理论值：

\[
B_{KV}=2\times4\times192\times2=3072\ \mathrm{bytes}.
\]

在单请求 `L=1,048,576`、Sparse GQA 23 层保留全部历史 K/V、SWA 69 层仅保留 512 Token 的理想条件下：

\[
M_{\rm sparse}=23\times L\times3072\approx69\ \mathrm{GiB},
\]

\[
M_{\rm swa}=69\times512\times3072\approx0.101\ \mathrm{GiB},
\]

\[
M_{\rm total}\approx69.1\ \mathrm{GiB}\ \text{（约 74.2 GB）}.
\]

若 92 层都保存完整 1M 历史，理论主 KV 为 276 GiB。以上为**单请求、未考虑 TP 分片、Index Cache、KV Cache Block Padding、分配器及其他运行时内存的理论值**。

由此可见：

- Sparse GQA 的 TopK 主要降低 **Attention 计算/读取集合**，不直接按 8 倍降低主 KV 的存储需求。
- 3:1 SWA 结构则将需要保留完整长历史主 KV 的层数从 92 减少到 23。
- 长上下文瓶颈会落到 Indexer Score/TopK、随机 KV 访存、Prefix Cache、PD 分离与缓存容量管理上。

## 9. vLLM / vLLM-Ascend 适配分析

### 9.1 当前支持状态（截至 2026-10-19）

vLLM-Ascend 的官方 Step 5 部署教程声明：目前通过 `quay.io/ascend/vllm-ascend:step5-a5-openeuler` 专用镜像提供 **950DT** 实验支持；该适配代码尚未并入可直接从源码安装的公共仓库。因此，**不能把当前公共仓库 MiniMax-M3 / DSA 的算子实现视为 Step 5 的真实逐函数调用链**。[6]

公开的 Ascend MiniMax-M3 `npu_sparse_attention_score` 提供 `select_idx` 与 `block_table` 两个输入，这从接口上证明了“逻辑索引 + Paged 物理映射”的路径可行，但不表示它能不经修改就支持 Step 5 的 Head Dim=192、Region=8、Provider Group=4 和 CSA Scoring。[7]

### 9.2 适配模块优先级

| 优先级 | 模块 | 关键检查点 |
|---|---|---|
| P0 | 模型加载 | 92 层、Attention 层类型、4 Dense + 88 MoE、RoPE 差异、MTP / Vision 权重路径 |
| P0 | CSA Indexer | K/Z 压缩、Norm/RoPE 顺序、FP8 数值路径、跨 Chunk 的尾块状态 |
| P0 | Group-wise Score/TopK | Provider Group 映射、TopK / Padding / Tie-breaking、FP8 排序一致性 |
| P0 | Sparse GQA Kernel | Region → Token → 物理 KV 地址、Tail Causal Mask、Per-Group 索引 |
| P0 | 非 Attention 算子 | FP32 Residual、Zero-centered RMSNorm、Clamped SwiGLU、MoE Routing |
| P1 | KV Cache Manager | SWA/Sparse 双类型 Cache、Prefix Cache、Page Reuse、Index Cache 生命周期 |
| P1 | TP/EP/MTP | TP=4/8 下 KV/Indexer 组关系、Expert Dispatch、MTP 候选验证 |
| P1 | NPU Graph | 固定输出缓冲、TopK 变长有效数、Capture/Replay 地址稳定性 |
| P1 | 性能 | Prefill Union、Decode Random Gather、Indexer Workspace 与 Score Buffer 内存峰值 |

### 9.3 必须覆盖的 Correctness 测试

1. **Indexer 数值一致性**：CSA 压缩输出、Score、TopK ID、FP8/BF16 差异，尤其临界分数附近的 TopK 翻转。
2. **主 Attention 一致性**：选中 Region 展开、因果尾部、不同 KV Group、物理 KV Page 非连续排列、Page Size != Region Size。
3. **缓存生命周期**：Chunked Prefill 边界（包含不足 8 Token 的尾块）、Prefix Reuse、请求释放/重新分配、Decode 与 MTP 多步。
4. **并行一致性**：TP1/4/8、多个 DP、MoE EP；注意 KV Provider Group 不等价于任意切分的 Indexer Query Heads。
5. **非 Attention**：FP32 Residual 保真、`1+γ` 的 RMSNorm 语义、Clamped SwiGLU、Router TopK / Shared Expert。
6. **权重更新**：若推理平台支持在线 Reload，验证 Indexer `K/Z/W`、Gate/Norm、MoE 和任何压缩/派生 Buffer 的刷新与地址稳定性；不能只测试冷启动。

> 本节为**建议的工程审查清单**，不是声称上述全部功能已经通过 vLLM-Ascend 公开 CI。

## 10. 结论与架构判断

1. **Step 5 的最大单点创新是 Sparse GQA**：在 GQA 基础上加 8-token CSA Region 压缩、独立 Provider Group 评分与 Block Sparse KV 访问。
2. **第二条主线是深层网络稳定性**：92 层 Narrow-but-Deep 与 FP32 Residual、Zero-centered RMSNorm、Clamped SwiGLU 协同设计。
3. **MoE 是容量扩展而非新路由范式**：88 层、352 Routed + Shared、Top-8，是更大的细粒度稀疏 FFN。
4. **与 Qwen QSA 最本质的不同是 Indexer 数学和分组方式**：Qwen 4-token 平均池化、共享选择结果；Step 5 8-token CSA 加权压缩、4 Group 各自 TopK。
5. **推理框架抽象应优先统一地址映射和 Cache 生命周期，而不是强行统一评分算子**：`selected_indices + index_coordinate_type + region_size + block_table + valid_count + kv_group_mapping` 可以作为候选通用接口，但具体 Kernel Tiling 仍应模型/硬件专门优化。

---

## 参考资料

[1] StepFun 官方报告：<https://platform.stepfun.ai/blog/inside-step-5-preview>  
[2] Step-5-Preview BF16 官方配置：<https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/blob/main/config.json>  
[3] Step-5-Preview 官方参考推理代码目录：<https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/tree/main/inference>  
[4] vLLM Qwen QSA Indexer：<https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py>  
[5] vLLM Qwen Sparse Paged Attention：<https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/nvidia/ops/qsa.py>  
[6] vLLM-Ascend Step 5 部署说明：<https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Step-5-Preview.html>  
[7] vLLM-Ascend SparseAttentionScore 接口：<https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/sparse_attention_score/README.md>  
[8] vLLM-Ascend Qwen3.8-Flash-Next 部署说明：<https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3.8-Flash-Next.md>

*本文的代码示例是算法与接口语义的简化展示，不代表未经公开的 950DT Docker 镜像中存在同名函数或相同 Kernel Tiling。*
