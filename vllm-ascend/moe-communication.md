# vLLM-Ascend MoE AlltoAll 通信、计算与 Shape 变化

> **范围**：基于 vLLM-Ascend `main` 分支中 `AlltoAllCommImpl` / `TokenDispatcherWithAll2AllV` 的实现整理。重点说明 `Permute → Dispatch → Permute → GMM → Unpermute → AlltoAll → Unpermute`。MC2、Fused MC2 和 AllGather 是不同的实现路径，单独对比。
>
> **资料**：根据前述代码分析及提供的讨论材料整理；代码链接指向 `main`，后续提交可能改变实现。

## 1. 核心结论

普通 EP AlltoAll 的完整顺序是：

```text
输入 Token + Router TopK
    ↓
Permute 1           展开 TopK，并按目标专家 / EP Rank 排序
    ↓
AlltoAll Dispatch   发送 Token-Expert 副本至专家所在 Rank
    ↓
Permute 2           按接收 Rank 的本地 Expert 重新分组
    ↓
GMM1 → Activation → GMM2
    ↓
Unpermute 2         撤销本地 Expert 分组，恢复通信分段顺序
    ↓
AlltoAll Combine    专家结果返回 Token 的来源 Rank
    ↓
Unpermute 1         撤销首次排序并按 TopK 权重聚合
    ↓
输出 Token
```

**关键点**：

1. **第一次 Permute 为通信服务**：展开 TopK，按目标专家排序，便于打包各 EP Rank 的发送片段。
2. **第二次 Permute 为 GMM 服务**：跨 Rank 接收的数据按来源 Rank 分段，必须再按本地专家聚合成连续区间；当 `num_local_experts <= 1` 时可跳过。
3. **两次 Unpermute 对应撤销两次重排**：先恢复跨 Rank 通信布局，再恢复原始 Token 布局并执行 TopK 加权求和。
4. **AlltoAll 不改变 Hidden Size**，只改变 Token 副本的设备归属和数量；FFN 中的 GMM1 / 激活 / GMM2 才改变中间特征维度。

## 2. 代码入口与关键文件

| 源码 | 核心职责 |
| --- | --- |
| [`routed_experts.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/routed_experts.py) | `AscendRoutedExperts.forward_impl`、TopK 路由、Prepare / Finalize |
| [`moe_comm_method.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py) | `MoECommMethod.fused_experts`，组织 Dispatch → MLP → Combine |
| [`token_dispatcher.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/token_dispatcher.py) | `TokenDispatcherWithAll2AllV`，两次重排及两次 AlltoAll |
| [`moe_utils.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_utils.py) | `async_all_to_all`，封装 `dist.all_to_all_single` |
| [`moe_mlp.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_mlp.py) | `apply_moe_mlp`，执行 GMM1、激活/量化、GMM2 |
| [`prepare_finalize.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/prepare_finalize.py) | TP Token 切分、padding、后处理 Gather / unpad |

调用链：

```text
AscendRoutedExperts.forward_impl
 ├─ moe_comm_method.prepare
 │   └─ PrepareAndFinalizeWithAll2All.prepare
 │       ├─ padding（按条件）
 │       └─ TP token split（按条件）
 ├─ router._select_experts → topk_ids / topk_weights
 ├─ quant_method.apply
 │   └─ MoECommMethod.fused_experts
 │       ├─ token_dispatcher.token_dispatch
 │       │   ├─ _dispatch_preprocess
 │       │   │   ├─ _preprocess（统计 splits、group_list）
 │       │   │   └─ npu_moe_token_permute       [Permute 1]
 │       │   ├─ async_all_to_all                [Dispatch]
 │       │   └─ _dispatch_postprocess
 │       │       └─ npu_moe_token_permute       [Permute 2]
 │       ├─ apply_moe_mlp
 │       │   ├─ apply_gmm1
 │       │   ├─ activation / act_quant
 │       │   └─ apply_gmm2
 │       └─ token_dispatcher.token_combine
 │           ├─ _combine_preprocess
 │           │   └─ npu_moe_token_unpermute    [Unpermute 2]
 │           ├─ async_all_to_all               [Combine]
 │           └─ _combine_postprocess
 │               └─ npu_moe_token_unpermute    [Unpermute 1]
 └─ moe_comm_method.finalize
     ├─ TP all_gather（按条件）
     └─ unpad（按条件）
```

## 3. 符号和完整 Shape 流程

定义：

| 符号 | 含义 |
| --- | --- |
| $T_r$ | 源 EP Rank $r$ 在 `prepare` 后参与本层 MoE 的本地 Token 数 |
| $H$ | Hidden Size |
| $K$ | TopK：一个 Token 路由到的专家数 |
| $E$ | 全局专家数 |
| $P$ | EP Group 的 Rank 数 |
| $L$ | 一个目标 EP Rank 的本地专家数；常见均匀映射下 $E=P L$ |
| $I$ | **本地专家** FFN 的中间维度（可为分区后的维度） |
| $S_r$ | 源 Rank $r$ 发出的 Token-Expert 副本数：$S_r=T_rK$ |
| $C_{s\to r}$ | 源 Rank $s$ 发往目标 Rank $r$ 的副本数 |
| $R_r$ | 目标 Rank $r$ 接收的副本数：$R_r=\sum_{s=0}^{P-1}C_{s\to r}$ |

**一张表看完全部数据形状**：

| 阶段 | 输入 | 输出 | 变化原因 |
| --- | --- | --- | --- |
| Router TopK | `[T_r, H]` | `topk_ids/weights [T_r,K]` | 选择专家 |
| Permute 1 | `[T_r,H]` | `[S_r,H]` | 按 TopK 展开为 $K$ 份，按目标专家排序 |
| Dispatch AlltoAll | `[S_r,H]` | `[R_r,H]` | 跨 EP Rank 交换副本 |
| Permute 2 | `[R_r,H]` | `[R_r,H]` | 按本地专家排序（行数不变） |
| GMM1 | `[R_r,H]` | `[R_r,2I]` | Gate + Up Projection（常规 SwiGLU） |
| SwiGLU | `[R_r,2I]` | `[R_r,I]` | 激活并合并 Gate/Up |
| GMM2 | `[R_r,I]` | `[R_r,H]` | Down Projection |
| Unpermute 2 | `[R_r,H]` | `[R_r,H]` | 撤销专家分组 |
| Combine AlltoAll | `[R_r,H]` | `[S_r,H]` | 将计算结果返回源 Rank |
| Unpermute 1 | `[S_r,H]` | `[T_r,H]` | 恢复 Token 顺序并按权重合并 TopK |
| TP Finalize | `[T_r,H]` | 依具体 TP/SP 分布恢复 | Gather / unpad（按条件） |

> 上表描述**逻辑 Shape**；量化、SwiGLU 变体、融合算子可能改变实际 kernel 数量、dtype 及中间表示。

## 4. 前处理：TP 切分、padding 与 Router

`PrepareAndFinalizeWithAll2All.prepare` 在 `replace_allreduce=False` 时，会按配置对输入做 padding（若触发）并沿 Token 维进行 TP 切分：

```python
if not self.replace_allreduce:
    if self.tp_size > 1:
        split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
        split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
        hidden_states = split_hidden_states[self.tp_rank]
        router_logits = split_router_logits[self.tp_rank]
```

例如 `TP=2`，原始 `hidden_states=[8,4096]`、`router_logits=[8,8]`，切分后每个 TP Rank 分别处理 `[4,4096]` 和 `[4,8]`。各 Rank 随后独立执行 TopK 和 EP Dispatch。

当 `replace_allreduce=True`（在该调用链中由 `moe_config.is_sequence_parallel` 传入）时，此路径认为输入已完成适当的 Token 切分，不再重复切分。

**注意边界**：源码使用 `pad_size = self.tp_size - self.num_tokens`，这不是通用的“向上取整到 TP 整数倍”表达式；`torch.tensor_split` 也可能产生非等长分片。是否能安全执行 `finalize` 中的 `all_gather`，需结合配置与实测验证，不能仅根据逻辑示意断言任意非均匀 TP 都支持。

## 5. 七个阶段的具体实现

### 5.1 Permute 1：TopK 展开并按目标专家排序

`TokenDispatcherWithAll2AllV._dispatch_preprocess`：

```python
permutated_local_input_tokens, reversed_local_input_permutation_mapping = (
    torch_npu.npu_moe_token_permute(
        tokens=hidden_states,
        indices=topk_ids,
        num_out_tokens=topk_ids.numel(),
    )
)
```

输入 `hidden_states=[T_r,H]`，`topk_ids=[T_r,K]`；输出 `[T_r*K,H]`，并保存用于反向恢复的 `reversed_local_input_permutation_mapping`。

例如：

```text
A → E0,E2    B → E1,E3    C → E2,E3    D → E0,E1

原始 Token:      [A, B, C, D]                     [4,H]
TopK 展开:       [A0,A2,B1,B3,C2,C3,D0,D1]        [8,H]
按 Expert 排序:  [A0,D0,B1,D1,A2,C2,B3,C3]        [8,H]
```

这样按专家连续放置的 Token 副本可以按其目标 EP Rank 划分发送片段。

### 5.2 Dispatch：准备 splits 并执行 AlltoAllV

`_preprocess(topk_ids)` 先统计每个专家得到的副本数，然后求出每个目标 Rank 的发送/接收数量：

| 元数据 | Shape | 含义 |
| --- | --- | --- |
| `num_local_tokens_per_expert` | `[E]` | 当前源 Rank 对每个专家的路由计数 |
| `input_splits` | `[P]` | 当前 Rank 发往各 EP Rank 的副本数 |
| `num_global_tokens_per_expert` | `[P,E]` | 从全部 EP Rank 汇集的专家路由计数 |
| `num_global_tokens_per_local_expert` | `[P,L]` | 各源 Rank 发往本 Rank 的本地专家计数 |
| `output_splits` | `[P]` | 本 Rank 将从各源 Rank 接收的副本数 |
| `tokens_per_expert` | `[L]` | 本地每个专家的副本总数，供 GMM 使用 |

`tokens_per_expert` 作为 `group_list` 传给 GMM；普通 AlltoAll 路径使用 **Count 模式**，即 `group_list_type=1`，不是累计前缀和。

```python
_, global_input_tokens, handle = async_all_to_all(
    permutated_local_input_tokens,
    output_splits,  # 接收数量
    input_splits,   # 发送数量
    self.ep_group,
)
handle.wait()
```

封装内部调用：

```python
dist.all_to_all_single(
    a2a_out,
    input_.contiguous(),
    output_split_sizes=output_split_sizes,
    input_split_sizes=input_split_sizes,
    group=group,
    async_op=True,
)
```

由于每个目标 Rank 可以接收不同数量的 Token，实际为 **AlltoAllV（可变长度）**；关键不变量：

$$
\sum_{j=0}^{P-1}\mathrm{input\_splits}[j]=S_r,\qquad
\sum_{j=0}^{P-1}\mathrm{output\_splits}[j]=R_r.
$$

### 5.3 Permute 2：按本地专家聚合

Dispatch 后的数据通常首先按**源 Rank** 分段：

```text
来自源 Rank 0: [Expert 0 的数据][Expert 1 的数据]
来自源 Rank 1: [Expert 0 的数据][Expert 1 的数据]
```

而 GMM 希望看到按**专家**连续排列：

```text
Expert 0: [Rank 0 的数据][Rank 1 的数据]
Expert 1: [Rank 0 的数据][Rank 1 的数据]
```

因此 `_dispatch_postprocess` 再执行：

```python
global_input_tokens, reversed_global_input_permutation_mapping = (
    torch_npu.npu_moe_token_permute(
        global_input_tokens,
        global_input_tokens_local_experts_indices,
    )
)
```

此时 Token 已由 Permute 1 完成 TopK 展开；Permute 2 仅调整行顺序：`[R_r,H] → [R_r,H]`。

**特殊优化**：`num_local_experts <= 1` 时，无须再进行本地专家分组；Permute 2 与相应的 Unpermute 2 均跳过。

### 5.4 GMM：本地分组专家 FFN

`apply_moe_mlp` 依次调用 `apply_gmm1`、激活/量化、`apply_gmm2`，或走等价的融合实现。对常规无量化 SwiGLU：

```text
            输入 [R_r,H]
                 │
     GMM1, W13 [L,H,2I]（计算布局）
                 ▼
             [R_r,2I]
                 │
               SwiGLU
                 ▼
              [R_r,I]
                 │
      GMM2, W2 [L,I,H]（计算布局）
                 ▼
             [R_r,H]
```

无量化路径使用 `torch_npu.npu_grouped_matmul`，并将 `group_list_type` 和 `group_list` 传入。举例：

```python
# 本地两个专家分别处理 3、4 个 Token 副本
group_list = [3, 4]  # count 模式

# 概念上等价于
out0 = x[0:3] @ W0
out1 = x[3:7] @ W1
out = torch.cat((out0, out1), dim=0)
```

因此 `group_list` 与第二次 Permute 的结果必须完全一致，否则专家和输入数据就会错位。量化路径可能将 GMM1 + Activation + Quant 融合，不能把逻辑阶段数等同于实际 kernel 启动数。

### 5.5 Unpermute 2：撤销本地 Expert 排序

```python
rev_global = combine_metadata.reversed_global_input_permutation_mapping
if hidden_states.shape[0] > 0 and self.num_local_experts > 1 and rev_global is not None:
    hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, rev_global)
```

它将 `[R_r,H]` 从“按本地 Expert 分组”恢复成“按源 Rank 分段”，为返回源 Rank 做准备；**不进行 TopK 权重加和**。

### 5.6 Combine：反向 AlltoAll

```python
_, permutated_local_input_tokens, handle = async_all_to_all(
    hidden_states,
    combine_metadata.input_splits,   # 反向发送尺寸
    combine_metadata.output_splits,  # 反向接收尺寸
    self.ep_group,
)
handle.wait()
```

两次通信的 `splits` 必须交换：

```text
正向 Dispatch: async_all_to_all(x, output_splits, input_splits, EP)
反向 Combine:  async_all_to_all(y, input_splits, output_splits, EP)
```

目标 Rank 把计算结果送回原始 Token 所属 Rank，逻辑 Shape：`[R_r,H] → [S_r,H]`。

### 5.7 Unpermute 1：恢复原始 Token 顺序并加权聚合

```python
output = torch_npu.npu_moe_token_unpermute(
    permuted_tokens=permutated_local_input_tokens,
    sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(torch.int32),
    probs=combine_metadata.topk_weights.to(permutated_local_input_tokens.dtype),
    restore_shape=combine_metadata.hidden_shape_before_permute,
)
```

该操作完成首次 Permute 的反向映射及 TopK 专家加权求和：

$$
y_t=\sum_{j=1}^{K}p_{t,j}\operatorname{Expert}_{e_{t,j}}(x_t).
$$

因此 `[S_r,H] → [T_r,H]`，随后 `finalize` 根据 TP/SP 配置决定是否执行 Gather 或 unpad。

## 6. 实际算例：EP=2、TopK=2、四个专家

配置：

```text
EP = 2, TP = 1, E = 4, L = 2, K = 2
H = 4096, I = 14336
EP Rank 0 持有 Expert 0/1
EP Rank 1 持有 Expert 2/3
各源 Rank 有 4 个 Token
```

Token 路由：

| 源 Rank | Token | TopK |
| --- | --- | --- |
| 0 | A | 0, 2 |
| 0 | B | 1, 3 |
| 0 | C | 2, 3 |
| 0 | D | 0, 1 |
| 1 | E | 0, 2 |
| 1 | F | 1, 2 |
| 1 | G | 3, 1 |
| 1 | H | 2, 3 |

发送矩阵：

| 来源 / 目标 | EP Rank 0 | EP Rank 1 | 合计 |
| --- | ---: | ---: | ---: |
| 源 Rank 0 | 4 | 4 | 8 |
| 源 Rank 1 | 3 | 5 | 8 |
| **接收总量** | **7** | **9** | **16** |

第一次 Permute 两个源 Rank 都输出 `[8,4096]`：

```text
源 Rank 0: input_splits=[4,4]
源 Rank 1: input_splits=[3,5]
```

第一次 AlltoAll 后：

```text
EP Rank 0: output_splits=[4,3] → [7,4096]
EP Rank 1: output_splits=[4,5] → [9,4096]
```

第二次 Permute 后，两个 Rank 的本地专家分组为：

```text
EP Rank 0:
  Expert 0: A0,D0,E0        → 3
  Expert 1: B1,D1,F1,G1     → 4
  group_list=[3,4]

EP Rank 1:
  Expert 2: A2,C2,E2,F2,H2  → 5
  Expert 3: B3,C3,G3,H3     → 4
  group_list=[5,4]
```

GMM 具体 Shape：

| 阶段 | EP Rank 0 | EP Rank 1 |
| --- | --- | --- |
| GMM 输入 | `[7,4096]` | `[9,4096]` |
| GMM1 输出 | `[7,28672]` | `[9,28672]` |
| SwiGLU 输出 | `[7,14336]` | `[9,14336]` |
| GMM2 输出 | `[7,4096]` | `[9,4096]` |
| Unpermute 2 输出 | `[7,4096]` | `[9,4096]` |
| Combine AlltoAll 后 | `[8,4096]` | `[8,4096]` |
| Unpermute 1 后 | `[4,4096]` | `[4,4096]` |

例如 Rank 0 最终需要恢复：

```text
A = p(A,0)·A0 + p(A,2)·A2
B = p(B,1)·B1 + p(B,3)·B3
C = p(C,2)·C2 + p(C,3)·C3
D = p(D,0)·D0 + p(D,1)·D1
```

**结论**：EP Rank 0 收到 7 份、Rank 1 收到 9 份是正常的非均匀专家负载；只要 splits、专家分组及反向映射匹配，最终仍能各自恢复 4 个 Token。

## 7. TP / 量化 / MC2 的差异

### 7.1 TP + EP

在此普通 AlltoAll 路径下，未事先进行序列并行切分时，`prepare` 可能先沿 Token 维做 TP split；两次 AlltoAll 使用 `self.ep_group`，而不是 `self.tp_group`。`finalize` 再按条件进行 TP Gather。真正的 TP/EP/DP 混合拓扑仍取决于进程组映射，不能仅凭 Shape 示意推断所有组合均受支持。

### 7.2 量化链路

普通非量化路径：

```text
Permute 1 → AlltoAll(Activations) → Permute 2 → GMM
```

当 `with_quant=True`，可能出现：

```text
Permute 1 → Dynamic Quant
          ├─ AlltoAll(Scale)
          └─ AlltoAll(Quantized Activations)
          → Permute 2（包括 Scale 重排）→ Quantized GMM
```

量化 Scale 的具体 dtype / Shape 取决于 INT8、FP8、MXFP 等实现，不宜统一假定为 `[R,1]`。部分实现采用 `npu_moe_init_routing_v2` 而非普通 token permute 来完成第二次重排。

### 7.3 AlltoAll、MC2、Fused MC2 对照

| 逻辑阶段 | 普通 AlltoAll | MC2 | Fused MC2 |
| --- | --- | --- | --- |
| Dispatch 前的排序 | 显式 `npu_moe_token_permute` | 封装于 Dispatch 算子 | 融合 |
| 跨 EP Dispatch | `dist.all_to_all_single` | `npu_moe_distribute_dispatch_v2` | 融合 |
| 本地 Expert 排序 | 多专家时显式 Permute 2 | 算子输出已适合后续计算 | 融合 |
| 专家 FFN | GMM / 局部融合 | GMM / 局部融合 | 融合 |
| 反排序 + 返回 | Unpermute 2 + AlltoAll + Unpermute 1 | `npu_moe_distribute_combine_v2` | 融合 |
| TopK 加权聚合 | 最后一次 Unpermute | Combine 内部 | 融合 |

`FusedMC2CommImpl` 可通过 `torch.ops._C_ascend.dispatch_ffn_combine(...)` 或 CANN MegaMoe 完成融合。**融合意味着 Python 层看不到独立的排序/通信算子，不意味着底层无需完成相应的数据交换和索引恢复。**

`AllGatherCommImpl` 是另外一种路由通信策略，不能套用普通 AlltoAll 的两个显式 AlltoAll 阶段。

## 8. 正确性与性能验证重点

| 检查点 | 应满足的条件 |
| --- | --- |
| Permute 1 | 输出行数为 $T_rK$ |
| Dispatch send | `sum(input_splits) == T_r*K` |
| Dispatch receive | `sum(output_splits) == R_r` |
| Permute 2 | 行数仍为 $R_r$；每位本地专家对应连续 Token 区间 |
| GMM group_list | `sum(group_list) == R_r`，各专家区间与排序一致 |
| Unpermute 2 | 精确撤销第二次 Permute |
| Combine | `input_splits` / `output_splits` 在反向通信中互换 |
| Unpermute 1 | 恢复 `[T_r,H]`，并正确加权 TopK 输出 |
| TP Finalize | 输出数量、顺序、padding mask 与原始有效 Token 对齐 |

尤其值得关注：

- **Metadata 开销**：`_preprocess` 计算专家计数、做元数据 Gather，并把 splits 转到 CPU/NumPy；小 batch 下同步和启动开销可能较明显。
- **多本地专家的重排开销**：`num_local_experts > 1` 时需要 Permute 2 / Unpermute 2；当等于 1 时可绕过。
- **异步不等于重叠**：普通 AlltoAll 路径虽然传了 `async_op=True`，但紧接 `handle.wait()`，不能认为它天然实现了充分的通信计算重叠。
- **复杂并行边界**：对于 `replace_allreduce=True`、TP 非均匀 Token 分片、MC2 DP-wide padding，应结合真实进程组、mask 和输出 Gather 检查。

## 9. 一句话总结

普通 AlltoAll MoE 在三种数据布局之间往返：**原始 Token 顺序 → 按目标 EP Rank 分组 → 按本地 Expert 分组 → 按目标 EP Rank 分组 → 原始 Token 顺序**。两次 Permute/Unpermute 完成布局变换，双向 AlltoAll 完成跨设备搬运，GMM 完成专家计算。

---

**相关源码入口**：[`TokenDispatcherWithAll2AllV`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/token_dispatcher.py)、[`MoECommMethod.fused_experts`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py)、[`PrepareAndFinalizeWithAll2All`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/prepare_finalize.py)。
