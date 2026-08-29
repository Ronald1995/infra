# Qwen3.8-Flash-Next N-gram Embedding 原理与 vLLM 实现分析

本文基于 Qwen3.8-Flash-Next 的公开配置，以及 vLLM PR [#53899](https://github.com/vllm-project/vllm/pull/53899) 中的 NVIDIA 实现，分析 PLE（Position Learning Enhancement）分支里的 N-gram Embedding。

本文聚焦以下问题：

- N-gram Embedding 在模型 forward 中处于什么位置；
- `ngram_size=3` 为什么会同时生成 2-gram 和 3-gram；
- N-gram 如何被哈希成 embedding ID；
- 多个 hash head 和不同大小的素数表如何缓解哈希碰撞；
- 约 51B 的 N-gram Embedding 参数量是如何形成的；
- prefill、decode、chunked prefill 和 CPU offload 下的数据如何流动。

除特别说明外，文中的 shape 均为逻辑全局 shape，不展开 Tensor Parallel 内部的参数分片。

---

## 1. 结论概览

Qwen3.8-Flash-Next 的 N-gram Embedding 可以理解为一个超大规模、可训练的“短语记忆表”：

1. 对当前 token 构造最近的 2-gram 和 3-gram；
2. 使用哈希将短语映射到多个 embedding 子表；
3. 从每个子表查询一个 160 维向量；
4. 将 16 个查询结果拼接成 2560 维 PLE embedding；
5. PLE 再根据当前四流 hidden state 判断这些短语信息是否有用；
6. 通过门控和 dilated short-conv 将结果注入模型主干。

主流程如下：

```text
input_ids + 前两个历史 token
              ↓
       构造 2-gram / 3-gram
              ↓
          多头哈希编码
              ↓
    16 次 embedding lookup
              ↓ concat
      PLE embedding [T, 2560]
              ↓
       key/value projection
              ↓
    与四流 hidden state 计算 gate
              ↓
     dilated depthwise short-conv
              ↓
    加回四流状态 [T, 4, 2560]
```

N-gram Embedding 本身只负责从离散 token 组合生成 `[T, 2560]` 的短语表示。后面的 key/value projection、门控和 short-conv 属于完整 PLE Layer，但不属于 embedding lookup 本身。

---

## 2. 真实模型配置

Qwen3.8-Flash-Next 的相关配置如下：

```text
hidden_size                         = 2560
hc_count                            = 4
ple_layer_ids                       = [2]    # 1-based
ple_embed_dim                       = 2560
ngram_size                          = 3
heads_per_ngram                     = 8
ngram_vocab_size_base               = 20,000,000
make_ngram_vocab_size_divisible_by  = 128
ple_conv_kernel_size                = 4
```

由此可得：

```text
N-gram 阶数数量 = ngram_size - 1
                 = 2

总 hash head 数 = (ngram_size - 1) × heads_per_ngram
                 = 2 × 8
                 = 16

每个 hash head 的 embedding 维度 = ple_embed_dim / 16
                                  = 2560 / 16
                                  = 160
```

16 个 hash head 的分配为：

```text
2-gram: 8 heads × 160 = 1280 维
3-gram: 8 heads × 160 = 1280 维
合计:                  = 2560 维
```

PLE 只插入在第 2 个 Decoder Layer，也就是 zero-based `layer_idx=1`。N-gram Embedding 并不是 48 个 Decoder Layer 各自持有一份。

---

## 3. 为什么同时存在 2-gram 和 3-gram

`ngram_size=3` 表示“最大使用到 3-gram”，而不是“只使用 3-gram”。vLLM 实现的循环范围是：

```python
for ngram in range(2, self.ngram_size + 1):
    ...
```

因此会生成：

```text
ngram=2 → 2-gram
ngram=3 → 3-gram
```

不额外生成 1-gram，是因为当前 token 已经由模型的普通 token embedding 表示。PLE 的目标是补充 token 组合信息。

以句子为例：

```text
I moved to New York City
                       ↑ 当前 token
```

处理 `City` 时，PLE 同时查询：

```text
2-gram: York City
3-gram: New York City
```

两者并不冗余：

- `York City` 是更稳定、更常见的局部搭配；
- `New York City` 是更精确的完整实体；
- 3-gram 是一次独立的哈希查表，模型无法从其 embedding 中无损拆出 2-gram embedding；
- 当完整 3-gram 较少见或发生碰撞时，2-gram 可以提供类似传统语言模型 backoff 的稳定信号。

另一个例子是：

```text
not very good
```

对应：

```text
2-gram: very good       # 局部上偏正面
3-gram: not very good   # 完整短语表达否定
```

同时保留多种粒度，使后续投影可以学习什么时候依赖通用局部搭配，什么时候依赖更具体的完整短语。

---

## 4. Forward 输入与 request 打包

N-gram Embedding 的核心入口是 `Qwen4ExpNGramEmbedding.forward_impl()`，输入包括：

```text
input_ids:       [T]
query_start_loc: [R + 1]
ngram_context:   [R, 2]
```

其中：

- `T` 是本次 forward 中所有 request 的调度 token 总数；
- `R` 是 request 数；
- `query_start_loc` 标记每个 request 在展平 token 数组中的区间；
- `ngram_context` 保存每个 request 在当前 chunk 之前的最近两个 token。

例如三个 request 本轮分别调度 3、1、2 个 token：

```text
query_start_loc = [0, 3, 4, 6]
```

表示：

```text
input_ids[0:3] → request 0
input_ids[3:4] → request 1
input_ids[4:6] → request 2
```

实现会先把展平 token 按 request 打包成二维 tensor，并在左侧拼接历史上下文：

```python
context = torch.cat([ngram_context, packed], dim=-1)
```

例如 decode 时：

```text
ngram_context = [New, York]
input_ids     = [City]

context       = [New, York, City]
```

这样，即使每次 decode 只输入一个 token，也可以正确构造跨 forward 边界的 2-gram 和 3-gram。

---

## 5. EOS 如何切断 N-gram

实现不会让 N-gram 跨过 EOS 边界。

它首先计算每个 token 距离上一个 EOS 的位置，然后在向前移动不足 1 或 2 个有效 token 时使用 EOS 填充。其效果类似：

```text
上一段文本 ... EOS | New York

处理 York 时：
2-gram = New York
3-gram = EOS New York
```

而不会生成：

```text
上一段最后一个 token + New + York
```

这个处理对于以下场景非常重要：

- 多轮对话中被 EOS 分隔的 segment；
- 多文档拼接；
- sequence 起始位置；
- chunked prefill 的边界恢复。

---

## 6. N-gram 哈希算法

对于当前位置 `t`，代码先准备：

```text
shifted[0] = x_t
shifted[1] = x_{t-1}
shifted[2] = x_{t-2}
```

其中无效历史位置已经被 EOS 替换。

每个位置有一个 layer-specific 的奇数乘子 `m_i`。对于 n 阶 N-gram，初始混合值为：

\[
h_n(t)
=
(x_t m_0)
\oplus
(x_{t-1}m_1)
\oplus
\cdots
\oplus
(x_{t-n+1}m_{n-1})
\]

其中 `⊕` 是 bitwise XOR。

对应代码逻辑：

```python
mixed = shifted[0] * layer_multipliers[0]
for index in range(1, ngram):
    mixed = torch.bitwise_xor(
        mixed,
        shifted[index] * layer_multipliers[index],
    )
```

乘子由以下信息派生：

```text
全局 seed
PLE layer ID
N-gram 内相对位置
SplitMix64
```

使用不同位置乘子的原因是保留 token 顺序：

```text
New York City
City York New
```

虽然包含相同 token，但不同 token 会乘以不同位置的乘子，最终哈希值通常不同。

---

## 7. 多个 Hash Head 的实际含义

“8 个 2-gram hash heads”不是 8 个 Attention heads，也不是重复查询同一行 embedding。

它表示同一个 2-gram 会进入 8 个不同大小、不同 offset 的哈希子表：

```text
2-gram mixed value
    ├─ mod P0 → head 0 embedding ID
    ├─ mod P1 → head 1 embedding ID
    ├─ mod P2 → head 2 embedding ID
    ├─ ...
    └─ mod P7 → head 7 embedding ID
```

3-gram 同样有另外 8 个子表。

对于第 `j` 个 head：

\[
id_{n,j}
=
h_n(t)\bmod P_{n,j}
+ offset_{n,j}
\]

其中：

- `P_{n,j}` 是该 head 的哈希表大小；
- `offset_{n,j}` 是该子表在物理大 embedding 表中的起始行。

最终：

```text
ngram_ids: [T, 16]
```

---

## 8. 为什么不同 Head 使用不同大小的素数表

哈希碰撞是指不同短语映射到同一个表位置。

例如两个短语的混合值为：

```text
短语 A → 38
短语 B → 73
```

在大小为 7 的表中：

```text
38 % 7 = 3
73 % 7 = 3
```

二者发生碰撞。

如果第二个 head 使用大小为 11 的表：

```text
38 % 11 = 5
73 % 11 = 7
```

则完整编码分别是：

```text
短语 A → [head0 row 3, head1 row 5]
短语 B → [head0 row 3, head1 row 7]
```

虽然在 head 0 中共享一行参数，但在 head 1 中仍然可以区分。

实现没有为每个 head 重新计算一套完全独立的 `mixed`，而是对同一个 `mixed` 使用不同的模数。因此更准确地说，这是一个由不同模数组成的哈希函数族。

使用大小相近但不同的素数有两个作用：

1. 避免多个表之间存在明显整除关系，使碰撞模式不容易同步；
2. 减少模数与 token ID、位置乘子共享因子造成的周期性聚集。

素数不会消除碰撞，但多个 head 可以显著降低“两个短语在所有子表中都无法区分”的概率。

---

## 9. 物理上为什么是一张大表

逻辑上有 16 个 embedding 子表，物理实现则把它们拼接成一个 `VocabParallelEmbedding`：

```text
[2-gram head 0]
[2-gram head 1]
...
[2-gram head 7]
[3-gram head 0]
...
[3-gram head 7]
```

假设前三个子表大小分别为 7、11、13，则 offset 是：

```text
head 0: offset = 0
head 1: offset = 7
head 2: offset = 18
```

即使三个 head 的余数都是 3，物理 ID 仍分别为：

```text
head 0: 3 + 0  = 3
head 1: 3 + 7  = 10
head 2: 3 + 18 = 21
```

因此不同 head 拥有独立参数空间。

所有子表总行数最后还会向 `make_ngram_vocab_size_divisible_by=128` 对齐，以便做 Tensor Parallel 分片。

---

## 10. 约 51B 参数是如何形成的

每个 hash head 的表大小约为：

```text
20,000,000 rows
```

共有：

```text
16 heads
```

所以总行数约为：

\[
16\times20,000,000
\approx320,000,000
\]

每行 embedding 维度为：

```text
160
```

总参数量约为：

\[
320,000,000\times160
\approx51.2\text{B parameters}
\]

各子表实际使用不同的相邻素数大小，并且总表会做对齐，因此真实值与 51.2B 会有少量差异，但量级就是约 51B。

仅计算 embedding 权重的理论存储量：

```text
BF16: 51.2B × 2 bytes ≈ 102.4 GB
FP8:  51.2B × 1 byte  ≈ 51.2 GB
```

这解释了为什么 PR #53899 专门实现 PLE CPU offload：即使语言模型主体每次只激活少量 MoE experts，这张 N-gram Embedding 表本身仍然非常大。

---

## 11. Embedding Lookup 的 Shape 变化

经过哈希后：

```text
ngram_ids: [T, 16]
```

每个 ID 查询一个 160 维向量：

```text
embedding lookup: [T, 16, 160]
```

随后 flatten 最后两维：

```text
[T, 16, 160]
    ↓ flatten
[T, 2560]
```

可以把 2560 维结果看成：

```text
[2-gram head 0 的 160 维
 | ...
 | 2-gram head 7 的 160 维
 | 3-gram head 0 的 160 维
 | ...
 | 3-gram head 7 的 160 维]
```

如果 checkpoint 使用 FP8 N-gram Embedding，lookup 输出使用全局 scale 反量化为模型计算 dtype：

```text
FP8 lookup result
    ↓ × weight_scale
BF16/目标 dtype embedding [T, 2560]
```

---

## 12. Embedding 如何进入 PLE

N-gram Embedding 输出：

```text
E: [T, 2560]
```

PLE 对其做两个投影：

```text
key_proj:
[T, 2560] → [T, 10240] → [T, 4, 2560]

value_proj:
[T, 2560] → [T, 2560]
```

当前模型四流 hidden state：

```text
query: [T, 4, 2560]
```

归一化后，对每条 stream 计算一个标量 gate：

\[
s_{t,c}
=
\frac{\langle Q_{t,c},K_{t,c}\rangle}{\sqrt{2560}}
\]

\[
g_{t,c}
=
\sigma\left(
\operatorname{sign}(s_{t,c})
\sqrt{\max(|s_{t,c}|,\epsilon)}
\right)
\]

得到：

```text
gate: [T, 4, 1]
```

再将同一个 value 按不同强度注入四条流：

```text
gated_value = gate × value.unsqueeze(stream_dim)
shape       = [T, 4, 2560]
```

最后执行 10240-channel、kernel size 4、dilation 3 的 depthwise causal short-conv，并返回：

```text
PLE output = gated_value + short_conv_output
shape      = [T, 4, 2560]
```

模型将其直接加到 HyperConnection 多流状态。

因此 N-gram Embedding 给出的是候选短语信息，而当前 hidden state 决定这些信息对四条 stream 分别有多大价值。

---

## 13. Prefill、Decode 与 Chunked Prefill

### 13.1 Prefill

对于完整 prompt：

```text
input_ids: prompt 中本轮调度的所有 token
ngram_context: 通常是 EOS padding，或此前 chunk 的最后两个 token
```

实现会批量构造所有 token 的 2-gram/3-gram ID，并一次查询 embedding。

### 13.2 Decode

普通 decode 每个 request 通常只有一个新 token：

```text
ngram_context = [x_{t-2}, x_{t-1}]
input_ids     = [x_t]
```

仍然可以构造：

```text
2-gram = [x_{t-1}, x_t]
3-gram = [x_{t-2}, x_{t-1}, x_t]
```

### 13.3 Chunked Prefill

假设 prompt 被分成两个 chunk：

```text
chunk 0: A B C D
chunk 1: E F G
```

处理 chunk 1 时：

```text
ngram_context = [C, D]
input_ids     = [E, F, G]
```

因此 `E` 的 2-gram/3-gram 是：

```text
D E
C D E
```

不会因为 chunk 切分而改变模型语义。

---

## 14. PR #53899 的 CPU Offload 路径

关闭 offload 时：

```text
GPU:
    计算 N-gram hash ID
    → GPU embedding lookup
    → key/value projection
    → PLE gate
    → short-conv
```

开启 `VLLM_PLE_CPU_OFFLOAD` 时：

```text
CPU offload worker:
    input_ids/query_start_loc/ngram_context
        ↓
    计算 N-gram hash ID
        ↓
    持有约 51B 参数的 embedding 权重并执行 lookup
        ↓
    将 [T, 2560] 结果异步拷贝到 GPU output buffer
        ↓
    semaphore = DONE

GPU model forward:
    先执行其他 GPU 工作
        ↓
    到达 PLE layer
        ↓
    等待 semaphore
        ↓
    读取 [T, 2560] embedding
        ↓
    继续执行 GPU key/value projection、gate 和 short-conv
```

也就是说，offload 边界位于：

```text
N-gram Embedding lookup 之后
key/value projection 之前
```

CPU worker 持有大 embedding 权重；GPU worker 中的 `PleOffloadLayer` 只是一个占位模块和预分配输出 buffer，不再分配完整 embedding 表。

同步机制包括：

- shared/pinned CPU input buffers；
- CUDA IPC output buffers；
- 独立 D2H/H2D copy stream；
- `cuStreamWaitValue32` / `cuStreamWriteValue32` 信号量；
- CUDA Graph 中的 opaque wait custom op。

CPU lookup 可以与 GPU 前面部分的计算重叠，但 GPU 到达 PLE layer 时必须等待对应 embedding 输出完成。

---

## 15. Tensor Parallel 与权重加载

物理大表使用 `VocabParallelEmbedding`，因此 embedding 行按照 TP rank 分片。

checkpoint 中 N-gram Embedding 又被拆成多个 shard 文件。加载时需要计算：

```text
checkpoint shard 覆盖的全局行区间
∩
当前 TP rank 持有的 embedding 行区间
```

只复制二者重叠部分。

因此需要区分三种分片概念：

1. 16 个逻辑 hash head 子表；
2. checkpoint 为避免单文件过大而使用的权重 shard；
3. runtime `VocabParallelEmbedding` 的 TP row shard。

它们是三个不同维度的问题，不能混为一谈。

CPU offload 模式下，CPU worker 持有可执行 lookup 的 embedding 参数；GPU rank 只接收 lookup 结果。各 TP rank 的后续 PLE projection 和主模型计算仍按正常并行配置执行。

---

## 16. 它与其他机制的区别

### 与普通 Token Embedding 的区别

```text
Token Embedding:
    当前 token ID → 向量

N-gram Embedding:
    当前 token + 最近历史 token → 短语向量
```

普通 token embedding 表示“当前是什么 token”；N-gram Embedding 表示“当前 token 处于什么局部组合中”。

### 与 RoPE 的区别

RoPE 编码 query/key 的几何相对位置；N-gram Embedding 直接记忆按特定顺序出现的离散 token 组合。两者并行存在，不互相替代。

### 与 Attention 的区别

Attention 动态读取任意历史 token；N-gram Embedding 只通过哈希查找局部短语记忆，不对整个上下文执行 softmax。

### 与 N-gram Speculative Decoding 的区别

N-gram speculative decoding 根据历史重复片段提出 draft token；这里的 N-gram Embedding 是模型参数的一部分，用于改变 hidden representation，不直接提出候选 token。

---

## 17. 简化伪代码

下面的伪代码保留了核心数据流：

```python
def ngram_embedding_forward(
    input_ids,          # [T]
    query_start_loc,    # [R + 1]
    ngram_context,      # [R, 2]
):
    # 1. 将展平 token 打包为 request-major 二维表示。
    packed = pack_by_request(input_ids, query_start_loc, pad=eos_token_id)

    # 2. 在左侧拼接此前两个 token，支持 decode/chunked prefill。
    context = concat([ngram_context, packed], dim=-1)

    # 3. 构造当前、前一个、前两个 token；不能跨 EOS。
    shifted = [
        context,
        shift_right(context, 1, fill=eos_token_id, stop_at_eos=True),
        shift_right(context, 2, fill=eos_token_id, stop_at_eos=True),
    ]

    id_blocks = []
    for n in [2, 3]:
        mixed = shifted[0] * multiplier[0]
        for i in range(1, n):
            mixed ^= shifted[i] * multiplier[i]

        # 每种 n-gram 有 8 个不同素数大小的子表。
        ids = mixed[..., None] % table_sizes[n]
        ids = ids + table_offsets[n]
        id_blocks.append(gather_current_tokens(ids))

    # [T, 16]
    ngram_ids = concat(id_blocks, dim=-1)

    # [T, 16, 160] → [T, 2560]
    return embedding_table(ngram_ids).flatten(-2)
```

---

## 18. 实现与调试检查清单

调试 N-gram Embedding 时，建议依次检查：

1. `query_start_loc` 是否与展平 token 排列一致；
2. `ngram_context` 是否是每个 request 已计算部分的最后两个真实 token；
3. speculative scheduling 的占位 token 是否在 lookup 前被处理；
4. EOS 是否正确切断跨 segment N-gram；
5. chunked prefill 边界是否得到与整段 prefill 相同的 N-gram ID；
6. `ngram_ids` shape 是否为 `[T, 16]`；
7. 每个 ID 是否落在对应 head 的 `[offset, offset + size)` 区间；
8. embedding lookup 输出是否为 `[T, 2560]`；
9. FP8 lookup 是否使用正确的全局 scale；
10. checkpoint shard、hash-head 子表和 TP shard 的行区间是否正确映射；
11. CPU offload 输出 buffer 是否在 GPU 消费完成后才被复用；
12. dummy run/CUDA Graph capture 是否正确满足 semaphore wait。

---

## 19. 源码索引

- NVIDIA PLE 与 N-gram Embedding：[`vllm/models/qwen4_exp/nvidia/ple_layer.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/nvidia/ple_layer.py)
- Qwen4Exp 主模型：[`vllm/models/qwen4_exp/nvidia/model.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/nvidia/model.py)
- PLE offload 抽象层：[`vllm/model_executor/layers/ple_offload_layer.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/model_executor/layers/ple_offload_layer.py)
- PLE offload connector：[`vllm/v1/ple_offload/connector.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/v1/ple_offload/connector.py)
- GPU Model Runner N-gram 输入准备：[`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/v1/worker/gpu_model_runner.py)
- Qwen3.8-Flash-Next 官方配置：[`config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
- vLLM PLE offload PR：[#53899](https://github.com/vllm-project/vllm/pull/53899)

---

## 20. 总结

Qwen3.8-Flash-Next 的 N-gram Embedding 本质上是一个多粒度、多哈希头的超大可学习短语表：

- 2-gram 提供稳定、通用的局部搭配；
- 3-gram 提供更具体的完整短语；
- 每种粒度使用 8 个不同素数大小的 hash heads 缓解碰撞；
- 16 个 160 维向量拼接为 2560 维 PLE embedding；
- 约 3.2 亿行、每行 160 维形成约 51.2B embedding 参数；
- `ngram_context` 保证 decode 和 chunked prefill 与整段计算语义一致；
- 后续 PLE gate 根据当前四流 hidden state选择性使用短语信息；
- PR #53899 通过 CPU offload 将巨大的 embedding 权重移出 GPU，同时保留 GPU 侧投影、门控和 short-conv。

它不是传统位置编码，也不是 speculative decoding，而是一个通过离散短语查表增强局部位置和组合模式表达的模型内存系统。
