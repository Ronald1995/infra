# Qwen3.8-Flash-Next QSA 结构与 Forward 流程

本文基于 vLLM PR [#53899](https://github.com/vllm-project/vllm/pull/53899) 的 NVIDIA 实现，分析 Qwen3.8-Flash-Next 中 QSA（Qwen Sparse Attention）的模块结构、数据流、张量 shape、缓存组织以及 prefill/decode 行为。

分析对应的 vLLM PR head commit 为：

```text
a5530b90cab09b187463396a99612a486ba91d6f
```

主要源码：

- [`nvidia/model.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/model.py)
- [`nvidia/qsa.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/qsa.py)
- [`nvidia/indexer_qsa.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/indexer_qsa.py)
- [`nvidia/ops/qsa.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/ops/qsa.py)
- [`nvidia/ops/qsa_pre_indexer.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py)
- [`common/qsa_cache.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/common/qsa_cache.py)

---

## 1. 核心结论

QSA 的核心不是直接从全部历史 token 中选择 2048 个独立 token，而是一个两级过程：

```text
第一级：轻量 Indexer
        每 4 个历史 token 压缩成一个 micro-block key
        从历史 micro-block 中选出最多 512 个相关 block

第二级：主 Sparse Attention
        把 512 个 block 展开成最多 2048 个真实 token
        再使用主 Attention 的 Q/K/V 做精确 GQA + Softmax
```

可以概括为：

```text
轻量级粗筛选
    +
主 Q/K/V 精确稀疏 Attention
```

QSA 有两套不同用途的 Q/K：

| 分支 | Q/K 维度 | 作用 |
|---|---:|---|
| 主 Attention | Q: `24×256`，KV: `2×256` | 对选中的真实 token 计算最终 Attention |
| Indexer | Q: `4×128`，K: `1×128` | 低成本选择相关 micro-block |

最重要的边界是：

> Indexer 只输出 token 索引；最终 value 聚合始终使用主 Attention 的精确 K/V。

---

## 2. 模型中的真实配置

Qwen3.8-Flash-Next 的文本模型配置如下：

| 配置项 | 数值 |
|---|---:|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 48 |
| `full_attention_interval` | 4 |
| 主 `num_attention_heads` | 24 |
| 主 `num_key_value_heads` | 2 |
| 主 `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 |
| 主 RoPE dimension | 64 |
| `indexer_n_heads` | 4 |
| `indexer_kv_heads` | 1 |
| `indexer_head_dim` | 128 |
| `indexer_budget` | 2048 token |
| `indexer_compress_ratio` | 4 token/block |
| block Top-K | 512 blocks |
| 最大 selection width | 2051 token indices |

48 层的布局是：

```text
12 × (
    3 × (Gated DeltaNet → MoE)
    1 × (QSA            → MoE)
)
```

也就是说，每 4 层出现一次 full-attention/QSA，总共 12 个 QSA 层。

---

## 3. QSA 与 HyperConnection 的边界

Qwen3.8-Flash-Next 层间保存的是四流 HyperConnection residual state：

```text
[T,4,2560]
```

物理上通常 flatten 为：

```text
[T,10240]
```

但 QSA 不直接处理 10240 维状态。进入 QSA 前，Attention HyperConnection 先执行 mix：

```text
四流 residual state [T,4,2560]
             │
             │ grouped RMSNorm + gated mix
             ▼
QSA block input      [T,2560]
```

QSA 的接口始终是：

```text
[T,2560] → QSA → [T,2560]
```

QSA 输出由下一个 HyperConnection 边界写回四流：

```text
QSA output [T,2560]
        × HC injection [T,4]
               │
               ▼
新四流状态 [T,4,2560]
```

因此不能把以下两种宽度混为一谈：

```text
10240：HyperConnection 层间 residual state 宽度
2560： QSA block 的输入和输出宽度
```

---

## 4. QSA 的整体模块结构

QSA 从同一个 `hidden_states` 分出两条支路：

```text
                         hidden_states [T,2560]
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
          主 Attention 分支                   Indexer 分支
          主 Q/K/V + gate                    检索 Q/K
                 │                                 │
                 │                       4-token micro-block 压缩
                 │                                 │
                 │                         选择最多 512 个块
                 │                                 │
                 │                       展开为最多 2048 个 token
                 │                       加入最多 3 个 causal tail
                 │                                 │
                 └───────────────┬─────────────────┘
                                 ▼
                  selected indices [T,2051]
                                 │
                                 ▼
                从主 paged KV cache 读取精确 K/V
                                 │
                                 ▼
                    Sparse Paged GQA + Softmax
                                 │
                                 ▼
                    Attention output gate + O proj
                                 │
                                 ▼
                         QSA output [T,2560]
```

两条支路的职责严格分离：

```text
Indexer：决定“看哪些位置”
主 Q/K/V：决定“这些位置如何参与 Attention”
```

---

## 5. Decoder Layer 中的调用入口

在 `Qwen4ExpDecoderLayer.forward()` 中，full-attention 层调用：

```python
attn_out = self.self_attn(
    hidden_states=block_input,
    positions=positions,
)
```

其中：

```text
block_input: [T,2560]
positions:   [T] 或 MRoPE 的 [3,T]
```

如果配置了 Indexer，`self.self_attn` 是：

```python
Qwen4ExpQSAAttention
```

否则才会退化为普通的：

```python
Qwen3NextAttention
```

QSA 返回：

```text
attn_out: [T,2560]
```

接下来：

```python
hidden_states, block_input, injection = (
    mlp_hc.combine_and_mix(hidden_states, attn_out, injection)
)
```

也就是将 QSA output 写回四流，并生成 `[T,2560]` 的 MLP input。

---

## 6. 主 QKV 投影

QSA forward 首先执行：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v, gate = self._project_qkv_gate(qkv, positions)
```

这两行并不是连续做两次 Linear projection。

真正的线性投影只有：

```python
self.qkv_proj(hidden_states)
```

`_project_qkv_gate()` 的职责是：

```text
split
+ reshape
+ Q/K RMSNorm
+ Q/K RoPE
```

函数名中的 `project` 容易产生误解，但它不会再次执行 Q/K/V Linear。

### 6.1 为什么把 Q、K、V、Gate 合成一次投影

数学上可以写成四个独立投影：

\[
Q=XW_Q
\]

\[
G=XW_G
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

真实实现将权重沿输出维拼接：

\[
W_{QGKV}=[W_Q\mid W_G\mid W_K\mid W_V]
\]

然后只执行一次 GEMM：

\[
Y=XW_{QGKV}
\]

这在数学上等价于：

```python
packed = torch.cat(
    [
        hidden_states @ W_q,
        hidden_states @ W_gate,
        hidden_states @ W_k,
        hidden_states @ W_v,
    ],
    dim=-1,
)
```

但真实执行只启动一个大 GEMM，而不是四个 GEMM。

收益包括：

- `hidden_states` 只需要作为 GEMM 输入读取一次；
- 减少 kernel launch；
- 大 GEMM 通常比多个小 GEMM 更容易充分利用 GPU；
- 后续 split 大多只是 view/索引，不是昂贵矩阵运算。

### 6.2 主 QKV 的逻辑 shape

主 Q 的宽度：

```text
24 × 256 = 6144
```

Attention output gate 与每个 Q channel 一一对应：

```text
24 × 256 = 6144
```

K/V 的宽度分别为：

```text
2 × 256 = 512
```

因此 packed projection 的逻辑输出宽度是：

```text
Q       6144
Gate    6144
K        512
V        512
----------------
总计   13312
```

完整过程：

```text
hidden_states [T,2560]
        │
        ▼ qkv_proj：唯一一次主 QKV Linear
packed qkv    [T,13312]
```

### 6.3 `_project_qkv_gate()` 的拆分

第一步拆出 `q_gate`、K 和 V：

```text
q_gate [T,12288]
K      [T,512]
V      [T,512]
```

`q_gate` 恢复 head 维度：

```text
[T,12288]
    ↓ reshape
[T,24,512]
```

每个 512 维 head 包含：

```text
前 256：Q
后 256：Gate
```

拆分后得到：

```text
Q:    [T,24,256]
Gate: [T,24,256]
K:    [T,2,256]
V:    [T,2,256]
```

---

## 7. 主 Q/K Norm 与 partial RoPE

主 Q/K 分别执行逐 head Gemma RMSNorm：

```python
q = self.q_norm(q)
k = self.k_norm(k)
```

shape 不变：

```text
Q [T,24,256]
K [T,2,256]
```

V 不执行 Q/K RMSNorm：

```text
V [T,2,256]
```

模型配置：

```text
head_dim = 256
partial_rotary_factor = 0.25
```

因此：

```text
rotary_dim = 256 × 0.25 = 64
```

每个 Q/K head：

```text
前 64 维：应用 RoPE 或 MRoPE
后 192 维：保持不变
```

最终：

```text
Q:    [T,24,256]  # RMSNorm + partial RoPE
K:    [T,2,256]   # RMSNorm + partial RoPE
V:    [T,2,256]   # projection output
Gate: [T,24,256]  # pre-sigmoid gate
```

NVIDIA 路径在条件满足时会使用 fused QK-Norm-RoPE-Gate kernel，但逻辑结果相同。

---

## 8. Indexer 分支的投影结构

Indexer 直接从同一个 QSA input 投影，而不是从主 Q/K 再投影：

```python
projected_qk, _ = self.index_qk_proj(hidden_states)
```

配置：

```text
indexer_n_heads   = 4
indexer_kv_heads  = 1
indexer_head_dim  = 128
```

Indexer Q 宽度：

```text
4 × 128 = 512
```

Indexer K 宽度：

```text
1 × 128 = 128
```

所以：

```text
hidden_states [T,2560]
        │
        ▼ ReplicatedLinear
projected_qk [T,640]
        │
        ├── index Q [T,4,128]
        └── raw K   [T,1,128]
```

### 8.1 为什么使用 ReplicatedLinear

主 QKV 按 Tensor Parallel 切分，但 Indexer projection 在每个 TP rank 上复制。

因此所有 TP rank 都能获得完整的：

```text
index Q [T,4,128]
index K [T,1,128]
```

并独立生成相同的 selected indices，避免为了 Top-K 结果引入额外跨 rank 通信。

### 8.2 Indexer Q 的后处理

Indexer Q 执行：

```text
Gemma RMSNorm
    +
partial RoPE/MRoPE
```

Indexer head dim 是 128，但复用主 Attention 的旋转定义：

```text
前 64 维：旋转
后 64 维：保持不变
```

结果：

```text
index Q [T,4,128]
```

---

## 9. Raw Index K 与 compressed key

Indexer 投影产生：

```text
raw K [T,1,128]
```

`raw` 的含义是它还没有完成最终的 K RMSNorm 和 RoPE。

QSA 的顺序是：

```text
4 个 raw K
    ↓ FP32 累加并求平均
1 个 pooled K
    ↓ Gemma RMSNorm
    ↓ 使用该组第一个 token 的位置做 RoPE/MRoPE
1 个 compressed K
```

而不是：

```text
每个 token 的 K 先 Norm/RoPE
    ↓
再把四个结果平均
```

这两个计算顺序并不等价。

设第 `g` 组包含 token：

```text
4g, 4g+1, 4g+2, 4g+3
```

平均池化为：

\[
\bar{k}_g=
\frac{k_{4g}+k_{4g+1}+k_{4g+2}+k_{4g+3}}{4}
\]

然后：

\[
\hat{k}_g=
\operatorname{RoPE}_{4g}
\left(
\operatorname{RMSNorm}(\bar{k}_g)
\right)
\]

其中 RoPE position 使用该组第一个 token 的位置 `4g`。

对于多模态 MRoPE，如果一个 group 横跨多个 forward，raw cache 还需要保存三个轴的精确位置，确保能够恢复该组第一个 token 的 MRoPE position。

---

## 10. 4-token micro-block

配置：

```text
indexer_compress_ratio = 4
```

表示连续 4 个 token 形成一个检索 micro-block：

```text
block 0 = token 0,1,2,3
block 1 = token 4,5,6,7
block 2 = token 8,9,10,11
...
```

每个完整 block 在 compressed key cache 中只有一行：

```text
[1,128]
```

所以长度为 `L` 的历史上下文，Indexer 需要扫描的 compressed key 数量约为：

\[
L/4
\]

例如：

```text
262144 个历史 token
        ↓ ratio=4
65536 个 compressed micro-block keys
```

QSA 官方所说的“在 micro-block level 选择”就是这个含义。

---

## 11. Indexer 如何计算 block score

对当前 query token，Indexer 有四个 Q head：

```text
q₀, q₁, q₂, q₃
```

每个历史 block 只有一个共享 compressed K：

```text
k̂g
```

先计算四个点积：

\[
s_{g,h}=q_h^\mathsf{T}\hat{k}_g
\]

每个 head 的分数经过 ReLU：

\[
s'_{g,h}=\max(s_{g,h},0)
\]

然后跨四个 head 求和，并除以 `sqrt(128)`：

\[
S_g=
\frac{1}{\sqrt{128}}
\sum_{h=0}^{3}
\max(q_h^\mathsf{T}\hat{k}_g,0)
\]

代码逻辑可以简化成：

```python
per_head_scores = compressed_keys @ index_queries
per_head_scores = relu(per_head_scores)
block_scores = per_head_scores.sum(dim="index_head") / sqrt(128)
```

这个阶段不做 softmax，只根据 `block_scores` 执行 Top-K 排序。

### 11.1 “weight-free selection”是什么意思

vLLM 将其描述为 weight-free QSA selection，但不代表整个 Indexer 没有参数。

下列模块有训练参数：

```text
index_qk_proj
q_layernorm
k_layernorm
```

“weight-free selection”更准确的含义是：

```text
投影得到 Q/K 后，排序阶段没有额外的 router MLP 或 learned scoring network；
直接使用 Q/K 相似度进行打分和 Top-K。
```

---

## 12. 从 512 个 block 展开为 2048 个 token

配置：

```text
indexer_budget         = 2048
indexer_compress_ratio = 4
```

实际 block budget 为：

```text
block_topk = 2048 / 4 = 512
```

Indexer 先选择：

```text
512 个 compressed blocks
```

然后每个 block 展开成四个真实 token index：

```text
block 17 → token 68,69,70,71
block 93 → token 372,373,374,375
```

因此：

```text
512 blocks × 4 token/block = 2048 token indices
```

主 Sparse Attention 最终看到的是 token index，而不是 compressed block index。

---

## 13. 为什么 selection width 是 2051

代码中：

```python
output_width = token_topk + compress_ratio - 1
```

代入：

```text
2048 + 4 - 1 = 2051
```

额外三个位置用于当前尚未组成完整 block 的 causal tail。

### 13.1 position=10 的具体案例

假设 position 从 0 开始，当前 query 位于 token 10：

```text
已经可见 11 个 token：token 0～10
```

完整 block：

```text
block 0 = token 0～3
block 1 = token 4～7
```

尚未完成的 tail：

```text
token 8、9、10
```

完整 block 数：

```text
floor((10 + 1) / 4) = 2
```

tail 长度：

```text
(10 + 1) % 4 = 3
```

Indexer 对完整 block 做 Top-K，然后无条件追加当前 tail：

```text
Top-K blocks 展开：最多 2048 token
causal tail：        最多 3 token
---------------------------------
最大 selection：     2051 token
```

不足 2051 的位置填 `-1`，Sparse Attention kernel 会将它们视为无效索引。

### 13.2 block 边界上的细节

如果当前位置正好闭合一个 block，例如 position=11：

```text
token 8～11 已经形成完整 block
tail 长度 = 0
```

这个新 block 会进入可见 block 候选集合，并参与 Top-K，而不再作为 tail 强制追加。

在短上下文中，可见 block 数小于 512，因此所有可见 block 都会保留；在超长上下文中，新闭合的 block 是否入选取决于 Indexer score。

---

## 14. 三套状态缓存

QSA 为不同目的维护三类缓存。

### 14.1 主 paged K/V cache

主 K/V cache 保存每个 token 的精确主 K/V：

```text
K：2 heads × 256
V：2 heads × 256
```

逻辑 cache shape：

```text
K cache [pages,page_size,num_kv_heads,256]
V cache [pages,page_size,num_kv_heads,256]
```

主 Sparse Attention 根据 selected indices 从这里读取真实 K/V。

因此：

> QSA 降低主 Attention 扫描和计算量，但仍需保留完整的主 K/V cache。

### 14.2 Raw Index Key Circular Buffer

raw key ring 保存尚未组成完整 micro-block 的 raw index K：

```text
每 token：1 × 128 BF16
```

例如 decode 已经生成：

```text
token 8、9、10
```

它们还不能形成完整 4-token block，需要暂存在 raw ring 中。

token 11 到来后：

```text
token 8、9、10、11
        │
        ▼ average
compressed block 2
```

Circular buffer 只需要保存开放 group 以及 speculative rows 所需的小段状态，不需要保存全部历史 raw index K。

其容量逻辑为：

\[
R\times
\left\lceil
\frac{R+N_{spec}}{R}
\right\rceil
\]

其中：

```text
R      = compress ratio
N_spec = speculative token 数
```

这样 rejected draft token 不会覆盖下一个 forward 仍需使用的 committed raw key。

### 14.3 Compressed Index Key Cache

每个完整 4-token block 保存一个归一化、带 RoPE 的 compressed K：

```text
[1,128]
```

Indexer 扫描的是这套 cache：

```text
历史 token 数：        L
compressed key 行数：  约 L/4
```

### 14.4 Top-K buffer 不是持久 cache

QSA owner 还注册：

```text
topk_indices_buffer
```

shape 为：

```text
[max_num_batched_tokens,2051]
```

它是 forward workspace，用来避免每次 forward 动态分配 selected indices，不是跨请求长期保存语义状态的 KV cache。

---

## 15. 主 Sparse Paged GQA

Indexer 输出：

```text
logical_indices [T,2051]
```

然后调用：

```python
qsa_sparse_paged_attention(
    query,
    key_cache,
    value_cache,
    logical_indices,
    block_table,
    token_to_req,
    output,
)
```

逻辑输入：

```text
主 Q：             [T,24,256]
主 K cache：       [pages,page_size,2,256]
主 V cache：       [pages,page_size,2,256]
selected indices： [T,2051]
```

### 15.1 GQA head 映射

主 Attention 有：

```text
24 个 Q heads
2 个 KV heads
```

所以每个 KV head 服务：

```text
24 / 2 = 12 个 Q heads
```

这是标准 Grouped Query Attention：

```text
KV head 0 → Q head 0～11
KV head 1 → Q head 12～23
```

Tensor Parallel 下执行本地等价映射。

### 15.2 精确 Attention 公式

设当前 token 的选择集合为 `S_t`，对每个主 Q head：

\[
a_{t,h,j}=
\frac{q_{t,h}^{\mathsf T}k_j}{\sqrt{256}},
\quad j\in S_t
\]

只在选中 token 上执行 softmax：

\[
p_{t,h,j}=
\operatorname{softmax}_{j\in S_t}(a_{t,h,j})
\]

然后聚合主 V：

\[
o_{t,h}=
\sum_{j\in S_t}p_{t,h,j}v_j
\]

输出：

```text
attn_output [T,24,256]
```

### 15.3 为什么不能使用 compressed index K 做最终 Attention

compressed index K：

- 维度只有 128；
- 四个 token 被平均为一行；
- 只有一个共享 K head；
- 不包含主 V；
- 训练目标是低成本检索，不是替代主 Attention 表示。

主 Attention K/V：

- 每 token 保存精确值；
- K/V head dim 为 256；
- 有两个 KV heads；
- V 是最终聚合的信息载体。

所以：

```text
compressed index K → 只能确定相关 block/index
主 K/V              → 才能完成精确 Attention
```

---

## 16. Split-K Sparse Attention

一个 query 最多需要处理 2051 个 selected indices。

为提高 GPU 并行度，NVIDIA Triton kernel 将选择集合分成多个 split：

```text
selected indices
    ├── split 0
    ├── split 1
    ├── split 2
    └── ...
```

每个 split 独立计算：

```text
partial output
partial log-sum-exp
```

最后用 log-sum-exp 权重合并：

\[
O=
\frac{
\sum_s e^{L_s-L_{max}}O_s
}{
\sum_s e^{L_s-L_{max}}
}
\]

其中：

```text
O_s：split s 内部归一化后的局部 output
L_s：split s 的 log-sum-exp
```

这样能够在增加并行度的同时，保持与对全部 selected tokens 做一次全局 softmax 相同的数学语义。

当只需要一个 split 时，kernel 直接写最终 output，省略 partial workspace 和 merge kernel。

---

## 17. Attention output gate 与 O projection

Sparse Attention 得到：

```text
attn_output [T,24,256]
```

QKV projection 同时产生的 gate 为：

```text
gate [T,24,256]
```

逐元素执行：

\[
\tilde{o}_{t,h,d}
=
o_{t,h,d}\cdot\sigma(g_{t,h,d})
\]

代码：

```python
flat_output = attn_output.view(num_tokens, -1)
flat_output = flat_output * torch.sigmoid(gate)
output, _ = self.o_proj(flat_output)
```

shape：

```text
attn_output       [T,24,256]
gate              [T,24,256]
逐元素 gated       [T,24,256]
flatten           [T,6144]
o_proj            [T,2560]
```

### 17.1 不要和 HyperConnection gate 混淆

| Gate | Shape | 作用 |
|---|---:|---|
| QSA output gate | `[T,24,256]` | 控制每个 Attention head 的每个 channel |
| HC mixing gate | `[T,4,2560]` | 控制四流如何读取为 block input |
| HC injection gate | `[T,4]` | 控制 QSA output 写入四条 residual stream 的强度 |

三者位于完全不同的层级。

---

## 18. `_run_qsa()` 的实际执行顺序

`Qwen4ExpQSAAttention.forward()` 的完整顺序是：

```text
1. 主 qkv_proj
2. split 主 Q/K/V/Gate
3. 主 Q/K RMSNorm + RoPE
4. 调用 _run_qsa()
5. Indexer projection
6. 更新 raw/compressed index caches
7. 对 compressed keys 打分并生成 selected indices
8. 更新主 paged K/V cache
9. 根据 selected indices 执行 Sparse Paged GQA
10. 乘 sigmoid(output gate)
11. o_proj
12. 返回 [T,2560]
```

对应伪代码：

```python
def forward(positions, hidden_states):
    packed = qkv_proj(hidden_states)

    q, k, v, gate = split_norm_rope(
        packed,
        positions,
    )

    selected_indices = indexer(
        hidden_states,
        positions,
    )

    update_main_kv_cache(k, v)

    attn_output = sparse_paged_gqa(
        q,
        main_k_cache,
        main_v_cache,
        selected_indices,
    )

    attn_output *= sigmoid(gate)
    return o_proj(attn_output.flatten(-2))
```

注意真实代码通过一个 custom op 包装完整 QSA transaction，以支持编译、opaque attention op 和输出 buffer 管理。

---

## 19. Prefill 流程

Prefill 时，一次 forward 可能包含多个 request 的大量 token，vLLM 将它们展平：

```text
hidden_states [T_total,2560]
```

metadata 提供：

```text
token_to_req
query_start_loc
logical_positions
seq_lens
block_table
slot_mapping
```

每个 query token 根据自己的 logical position 计算可见完整 block 数：

\[
N_{visible}=
\min
\left(
\left\lfloor\frac{position+1}{4}\right\rfloor,
\left\lfloor\frac{sequence\_length}{4}\right\rfloor
\right)
\]

因此同一个 prefill batch 中，不同 query row 拥有不同的因果可见范围。

对于每个 query row：

```text
1. 只对当前 query 可见的完整 compressed blocks 打分
2. 选择最多 512 个 block
3. 展开为最多 2048 个历史 token
4. 追加当前未完成 group 的 causal tail
5. 在最终 selected tokens 上做主 Sparse Attention
```

### 19.1 group 跨 forward 时如何处理

一个 group 可能部分位于之前的 raw ring，部分位于当前 prefill chunk：

```text
上一轮 raw ring：token 8、9
当前 forward：   token 10、11、12、...
```

构造 block 2 时，compression kernel 会读取：

```text
token 8、9：  来自 raw ring
token 10、11：来自当前 forward 的 raw rows
```

因此 QSA 不要求每次 prefill/decode 的 chunk 边界与 4-token group 边界对齐。

---

## 20. Decode 流程

普通 decode 中，每个 request 每轮通常只有一个新 token。

假设当前已经生成到 position 10：

```text
compressed cache：block 0、block 1
raw ring：        token 8、9、10
```

position 11 到来后：

```text
token 8、9、10、11
        │
        ▼ average + K norm + group-first RoPE
compressed block 2
```

该 block 会在同一 forward 中写入 compressed cache，并进入当前 query 的候选集合。

Decode 的顺序可以理解为：

```text
新 hidden state
    │
    ├── 生成主 Q/K/V/Gate
    │
    └── 生成 index Q/raw K
              │
              ├── 更新 raw ring
              ├── group 完成时更新 compressed cache
              └── 生成 selected indices
                         │
主 K/V 更新到 paged cache │
              └──────────┘
                         │
                         ▼
               Sparse Attention
```

主 K/V 在 sparse kernel 前写入 cache，所以当前 token 的精确 K/V 已经可用；它是否被读取取决于其索引是否位于 tail 或入选 block 中。

---

## 21. MTP 下的 Top-K 复用

Indexer 中存在：

```python
self.skip_topk = False
```

MTP step 0 正常选择与 target 对齐的 Top-K rows。

后续 MTP steps：

```text
继续更新 QSA side caches
但复用 step 0 已保存的 selected indices
```

这样可以避免每个 speculative step 都重新扫描全部 compressed index cache。

复用的是 Top-K 结果，不是停止更新 raw/compressed QSA 状态。

---

## 22. Tensor Parallel 下的 shape

本文前面的 shape 是全模型逻辑 shape。

以 TP=4 为例：

### 22.1 主 Q

```text
总 Q heads：24
每 rank：   24 / 4 = 6 heads

本地 Q：    [T,6,256]
本地 gate： [T,6,256]
```

### 22.2 主 KV

总 KV heads 只有 2，小于 TP size 4。

vLLM 在 rank 之间复制 KV heads：

```text
每个 rank 本地 num_kv_heads = 1
部分 rank 使用同一个逻辑 KV head 的副本
```

本地主 KV shape：

```text
K [T,1,256]
V [T,1,256]
```

### 22.3 Indexer

Indexer 使用 `ReplicatedLinear`，每个 rank 都保留完整：

```text
index Q [T,4,128]
index K [T,1,128]
```

所以每个 rank 的 selected indices 都是：

```text
[T,2051]
```

### 22.4 O projection

每个 rank 的本地 Sparse Attention output：

```text
[T,6,256] → flatten → [T,1536]
```

`RowParallelLinear` 将各 rank 的局部 head output 投影并归约，最终接口仍为：

```text
[T,2560]
```

---

## 23. 为什么 QSA 能降低长上下文成本

标准 full attention 的主要工作量随历史长度 `L` 线性增长：

\[
O(L\cdot H_q\cdot D_{main})
\]

QSA 将它拆成两部分。

Indexer 粗筛：

\[
O
\left(
\frac{L}{4}
\cdot 4
\cdot 128
\right)
\]

主精确 Attention：

\[
O
\left(
2051
\cdot 24
\cdot 256
\right)
\]

以 262144 token 上下文为例：

```text
普通 full attention：
    每个 query 扫描约 262144 个主 K/V token

QSA：
    Indexer 扫描约 65536 个 128 维 compressed keys
    主 Attention 只读取最多约 2051 个真实 K/V token
```

主要收益来自：

- Indexer K 只有一个 head、128 维；
- 每 4 token 压缩成一个检索 block；
- 主 K/V 只对 selected token 发生读取和计算；
- 显著减少长上下文下的主 K/V HBM 带宽；
- Split-K 提升 decode 小 batch 时的 GPU 并行度。

代价包括：

- 仍然需要完整主 K/V cache；
- 增加 raw/compressed index side caches；
- 增加 Indexer projection、scoring 和 Top-K；
- Attention 结果变为基于检索集合的稀疏近似。

---

## 24. NVIDIA 实现约束

PR 中的 NVIDIA QSA 路径目前要求：

- CUDA + Triton；
- FlashAttention 可用，用于 vLLM Attention backend/metadata 集成；
- 模型 dtype 为 BF16；
- 主 K/V cache 为 BF16；
- 不支持 KV cache quantization；
- 不支持 fused output quantization；
- 不支持 context parallelism；
- 不支持 decode context parallelism；
- 不支持 sliding-window attention；
- 不支持 ALiBi；
- 不支持 attention sinks；
- 不支持 dual-chunk RoPE；
- 必须是 causal decoder attention；
- `indexer_kv_heads == 1`；
- `indexer_budget` 必须能整除 `indexer_compress_ratio`；
- 当前配置校验要求 block Top-K 为 512 或 2048。

Qwen3.8-Flash-Next 使用：

```text
2048 / 4 = 512 blocks
```

符合 kernel 支持范围。

---

## 25. 完整 shape 流程

```text
HyperConnection residual state
[T,4,2560] = [T,10240]
        │
        ▼ HC mix
QSA input
[T,2560]
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼ 主 QKV fused projection                      ▼ Indexer projection
packed [T,13312]                                projected_qk [T,640]
        │                                              │
        ├── Q    [T,24,256]                            ├── iq [T,4,128]
        ├── Gate [T,24,256]                            └── ik [T,1,128]
        ├── K    [T,2,256]                                    │
        └── V    [T,2,256]                                    ▼
        │                                            每 4 个 raw ik 平均
        ▼                                                      │
Q/K Norm + partial RoPE                                        ▼
        │                                            compressed K [blocks,1,128]
        │                                                      │
        │                                            4-head ReLU dot-product
        │                                                      │
        │                                            Top-512 compressed blocks
        │                                                      │
        │                                            展开为 2048 token indices
        │                                            + 最多 3 个 causal tail
        │                                                      │
        └────────────────────────────┬─────────────────────────┘
                                     ▼
                         selected indices [T,2051]
                                     │
                                     ▼
                   主 exact paged K/V cache lookup
                                     │
                                     ▼
                         Sparse Paged GQA + Softmax
                                     │
                                     ▼
                         attn_output [T,24,256]
                                     │
                                     ▼ × sigmoid(Gate)
                         gated output [T,24,256]
                                     │
                                     ▼ flatten
                              [T,6144]
                                     │
                                     ▼ O projection
                             QSA output [T,2560]
                                     │
                                     ▼ HC combine_and_mix
                    residual state [T,10240]
                    MLP input       [T,2560]
```

---

## 26. 简化伪代码

### 26.1 Indexer

```python
def indexer_forward(hidden_states, positions):
    # [T,2560] -> [T,640]
    projected_qk = index_qk_proj(hidden_states)

    # [T,512], [T,128]
    projected_q, raw_k = split(projected_qk)

    # [T,4,128]
    index_q = projected_q.reshape(T, 4, 128)
    index_q = q_norm(index_q)
    index_q = partial_rope(index_q, positions, rotary_dim=64)

    # 保存每 token 的 raw index K，供跨 forward 组块
    store_raw_ring(raw_k)

    # 每 4 个 raw K 生成一个 compressed K
    pooled_k, first_position = average_complete_groups(
        raw_k,
        raw_ring,
        ratio=4,
    )
    compressed_k = k_norm(pooled_k)
    compressed_k = partial_rope(
        compressed_k,
        first_position,
        rotary_dim=64,
    )
    store_compressed_cache(compressed_k)

    # 对可见完整 blocks 打分
    scores = dot(index_q, compressed_cache)
    scores = relu(scores).sum(dim="index_heads") / sqrt(128)

    # 512 blocks -> 2048 token indices
    selected_blocks = topk(scores, k=512)
    selected_tokens = expand_each_block(selected_blocks, ratio=4)

    # 加入当前未完成 block 的最多三个 token
    return append_causal_tail(selected_tokens, max_tail=3)
```

### 26.2 主 QSA

```python
def qsa_forward(hidden_states, positions):
    # 唯一一次主 QKV Linear
    packed = qkv_proj(hidden_states)

    # split + Q/K norm + partial RoPE；不是第二次 Linear
    q, k, v, gate = project_qkv_gate(packed, positions)

    # 轻量检索分支
    selected_indices = indexer(hidden_states, positions)

    # 保存精确主 K/V
    update_main_paged_kv_cache(k, v)

    # 只在 selected indices 上做精确 GQA
    attn_output = sparse_paged_gqa(
        q,
        main_k_cache,
        main_v_cache,
        selected_indices,
    )

    # channel-wise attention output gate
    attn_output *= sigmoid(gate)

    # [T,24,256] -> [T,6144] -> [T,2560]
    return o_proj(attn_output.flatten(-2))
```

---

## 27. 常见理解误区

### 误区一：`_project_qkv_gate()` 又做了一次 QKV Linear

错误。

```text
qkv_proj：            真正的 fused Linear projection
_project_qkv_gate：   split + reshape + Q/K Norm + RoPE
```

### 误区二：QSA 直接选择 2048 个独立 token

不准确。

```text
先选 512 个 4-token blocks
再展开为 2048 个 token
最后追加最多 3 个 causal-tail token
```

### 误区三：Indexer K 就是主 Attention K

错误。两者来自不同投影，维度和用途都不同：

```text
Indexer K：1×128，只用于检索
主 K：    2×256，用于精确 Attention
```

### 误区四：compressed K 直接参与最终 value 聚合

错误。compressed K 只产生 selected indices；最终 Sparse Attention 从主 paged cache 读取每个 token 的精确 K/V。

### 误区五：QSA 不需要完整主 KV cache

错误。QSA 减少主 KV 的读取范围，但所有 token 的精确主 K/V 仍然需要缓存。

### 误区六：2051 表示模型训练时固定看 2051 个 token

不准确。2051 是 buffer 最大宽度：

```text
2048 token budget + 最多 3 个 causal tail
```

短上下文或可见 block 不足时，其余位置填 `-1`。

### 误区七：QSA output gate 等于 HC injection gate

错误：

```text
QSA output gate [T,24,256]：控制 Attention channel
HC injection     [T,4]：     控制 block output 写入四流
```

---

## 28. 调试检查清单

分析或适配 QSA 时，可以按以下顺序检查：

1. Decoder Layer 是否只在 `full_attention` 层实例化 QSA；
2. HC mix 后的 QSA input 是否为 `[T,2560]`；
3. 主 QKV packed projection 是否包含 Q、Gate、K、V；
4. `_project_qkv_gate()` 是否只做拆分、Norm 和 RoPE；
5. 主 Q/K 是否为 `[T,24,256]`、`[T,2,256]` 的逻辑 shape；
6. 主 RoPE 是否只覆盖前 64 维；
7. Indexer projection 是否为 `[T,2560] → [T,640]`；
8. Indexer Q/K 是否为 `[T,4,128]` 和 `[T,1,128]`；
9. raw K 是否在平均池化前保持未 Norm、未 RoPE 状态；
10. 每 4 个 raw K 是否平均成一个 compressed K；
11. compressed K 是否使用 group-first position 做 RoPE；
12. block score 是否为四个 ReLU dot-product 的和除以 `sqrt(128)`；
13. Top-K 是否选择 512 个 block，而不是 2048 个 block；
14. block 是否正确展开为 2048 个 token indices；
15. 是否为 causal tail 预留额外 3 个 index；
16. 无效 index 是否填 `-1`；
17. 主 K/V 是否在 sparse attention 前更新到 paged cache；
18. Sparse Attention 是否读取主 K/V，而不是 compressed K；
19. 主 GQA 是否保持 24 Q heads / 2 KV heads 的逻辑映射；
20. Sparse output 是否先乘 sigmoid gate 再进入 O projection；
21. O projection 后是否恢复 `[T,2560]`；
22. QSA output 是否由 HyperConnection injection 写回四流；
23. TP 下 Indexer 是否保持 replicated 语义；
24. MTP 后续 step 是否复用 Top-K 但仍更新 side cache；
25. 模型和三类 cache dtype 是否满足 BF16 限制。

---

## 29. 最终总结

Qwen3.8-Flash-Next QSA 的完整语义可以归纳为：

```text
HC 四流状态 [T,10240]
        ↓ mix
QSA input [T,2560]
        ↓
主分支：一次 fused projection 生成 Q/K/V/Gate
Indexer：独立生成 4-head Q 和 1-head raw K
        ↓
每 4 个 raw K 平均为一个 compressed micro-block key
        ↓
使用 4-head ReLU similarity 选择最多 512 个 block
        ↓
展开为 2048 个真实 token，并追加最多 3 个 causal tail
        ↓
主 Q 从精确 paged K/V cache 对 selected tokens 做 Sparse GQA
        ↓
Attention output × sigmoid(channel gate)
        ↓ O projection
QSA output [T,2560]
        ↓ HC injection
返回四流 residual 主干
```

其中最关键的四点是：

1. `qkv_proj` 只有一次线性投影，`_project_qkv_gate()` 不会再次执行 Linear；
2. Indexer 在 4-token micro-block 层面检索，实际选择 512 blocks，而不是独立选择 2048 tokens；
3. compressed index K 只用于选择位置，最终 Attention 使用主 paged K/V cache；
4. QSA 的模块输入输出始终是 `[T,2560]`，四流 `[T,10240]` 属于外围 HyperConnection。
