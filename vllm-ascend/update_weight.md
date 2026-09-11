下面基于 **2026-09-10 的 vLLM-Ascend `main`、公开 issue/PR，以及我们前面讨论过的 upstream vLLM layerwise reload 机制**，重新整理一次权重更新风险。

这次我建议不要按“具体模型”分类，而按 **`process_weights_after_loading` / runtime weight finalization 做了什么操作**分类。因为从现在暴露出来的 #15666、#15463、#5915、#13905 等问题看，它们虽然落在不同模型/量化方法上，但根因其实高度重复。

先定义状态：

- **【已确认-未修】**：公开 issue/PR + 当前 `main` 代码仍能看到问题。
- **【已确认-已修】**：已有 merged PR 或 closed-completed issue。
- **【代码审计风险】**：当前 `main` 存在同类危险 pattern，但还没有找到直接对应的公开 bug report，因此不能当成已确认故障。
- **【架构缺口】**：不是单一 PWAL bug，而是 reload 生命周期本身没有覆盖该场景。

---

# 1. 当前权重更新中，后处理可能出问题的主要类型

从目前 vLLM-Ascend 的代码和已暴露问题看，可以归纳成下面 8 类：

| 类别 | 典型操作 | 主要失败模式 |
|---|---|---|
| 1. Graph-visible tensor 被重新分配 | `.to()`, `.contiguous()`, `.clone()`, `npu_format_cast()`, `self.x = new_tensor` | ACLGraph 继续引用旧地址，得到 stale weight / illegal memory |
| 2. Checkpoint/runtime layout 转换 | transpose、permute、NZ、reshape、flatten、repack | 第二次 `load_weights()` 对 runtime layout 按 checkpoint layout 写入，shape/shard 错误 |
| 3. Derived weight / derived state | `W_UK_T/W_UV`、`weight_fp32`、组合投影、scale bias | source 更新了，derived tensor 没更新，或者更新时换地址 |
| 4. Destructive source release | `dispose_layer()`、`del param`、删除 scale、Parameter→`list[Tensor]` | 下一轮 checkpoint reload 没有合法 destination |
| 5. PWAL 非幂等 / 生命周期错误 | 每次 PWAL 再 transpose；依赖 `wake_up` 做逆操作 | 多次 reload 后 layout 被重复转换 |
| 6. Quantization/repack 特有恢复 | MXFP8、block-FP8、WNA16、W8A8 等 | checkpoint format 与 kernel format 不同，需要 restore→load→repack |
| 7. Unmanaged/runtime-owned storage | plain tensor、workspace、offloader static buffer、kernel/list-owned tensor | generic layerwise 看不到，地址/alias/binding 失效 |
| 8. Reload transaction/backend 不一致 | HCCL 有 initialize/finalize，NPU IPC 没有 | 同一个模型在不同传输 backend 下 correctness 不一致 |

其中前 7 类主要是 **后处理操作自身的问题**；第 8 类决定了这些问题是否会被 native layerwise reload 屏蔽或者直接暴露。

从架构上，可以把所有风险归结为三个 invariant：

```text
I. load 前：
   destination 必须处于 weight_loader 理解的 checkpoint/model format

II. load 后：
   所有 derived/runtime state 必须反映新权重

III. graph replay 前：
   所有 graph-visible persistent storage 地址必须与 capture 时相同
```

也就是：

\[
\text{correct reload}
=
\text{representation correctness}
+
\text{value freshness}
+
\text{storage identity}
\]

下面逐类看当前 vLLM-Ascend 的具体问题。

---

# 2. 各类问题详细分析

## 2.1 Graph-visible tensor 重新分配：地址变化

这是目前证据最充分的一类。

典型危险代码包括：

```python
x = old.to(...)
x = old.transpose(...).contiguous()
x = old.clone()
x = torch_npu.npu_format_cast(...)
layer.foo = x
layer.weight.data = x
```

这些操作的共同点是：

```text
old storage A
      ↓ transform
new storage B
```

如果 `A` 已经进入 ACLGraph：

```text
ACLGraph → A

Python after reload → B
```

那么 graph replay 不会自动更新为 B。

---

### 2.1.1 DeepSeek-V4 `weight_fp32`：PR #15666

这是这一类最典型的新问题。

当前 `AscendUnquantizedLinearMethod.process_weights_after_loading()` 对：

```python
precast_fp32_weight=True
```

会：

```python
weight_fp32 = layer.weight.data.to(torch.float32)

layer.weight_fp32 = (
    weight_fp32
    if keep_nd_weight or skip_weight_nz_conversion
    else maybe_trans_nz(weight_fp32)
)
```

当前 `main` 仍然是直接赋值。

这里至少有一次新分配：

```python
layer.weight.data.to(torch.float32)
```

NZ 路径还可能再来一次：

```python
torch_npu.npu_format_cast(...)
```

因此第一次：

```text
weight_fp32 @ A
        ↓
ACLGraph capture A
```

第二次 RL reload + PWAL：

```text
new weight
    ↓ .to(float32)
weight_fp32_new @ B

layer.weight_fp32 = B
```

Graph 仍然：

```text
Graph → A
```

结果就是 **新 `layer.weight_fp32` 值正确，但 graph 继续使用旧值**。

PR #15666 正是为了解这个问题，方案是引入类似：

```python
update_tensor_inplace(layer, "weight_fp32", new_fp32)
```

已有兼容 old tensor 时：

```python
old.copy_(new)
```

而不是重新赋值。PR 截至当前仍是 open / unmerged。 PR 自己也明确把根因描述为 ACLGraph 捕获旧 tensor reference，而 reload 后替换 reference 导致继续使用 stale weight。

**当前结论：**

> 【已确认-未修】`main` 仍有。

**建议修复：**

短期采用 #15666 思路，但我更建议不要做一个只服务 `weight_fp32` 的 helper，而提供统一的 Ascend：

```python
update_tensor_inplace(
    module,
    name,
    new_tensor,
    *,
    graph_stable=True,
)
```

要求：

```text
old 不存在：
    cold load，允许 register/assign

old 存在且 shape/dtype/device 一致：
    old.copy_(new)

old 存在但不兼容：
    RL/graph 模式 fail closed
    不应该偷偷替换 storage
```

特别是最后一点。现在 #15666 patch 的 fallback `setattr()` 对 graph capture 后 representation 改变的场景仍然太宽松。

---

### 2.1.2 W8A8 MXFP8 transpose/contiguous：PR #13905

这是完全相同的 storage-identity 问题，只不过对象是正式的：

```text
weight
weight_scale
```

旧实现每次 PWAL 都：

```python
weight = weight.transpose(...).contiguous()
scale = scale.reshape(...).transpose(...).contiguous()
```

导致每轮 reload 生成新 storage。

PR #13905 明确记录了实际后果：

```text
ACLGraph captured pointer
       ↓
reload PWAL reallocates
       ↓
captured pointer points at stale/freed memory
       ↓
garbled output
```

修复方式是第一次分配：

```python
layer._mxfp8_weight_buf
layer._mxfp8_scale_buf
```

之后每轮：

```python
_mxfp8_weight_buf.copy_(new_transformed_weight)
_mxfp8_scale_buf.copy_(new_transformed_scale)
```

然后让：

```python
layer.weight.data = layer._mxfp8_weight_buf
layer.weight_scale.data = layer._mxfp8_scale_buf
```

一直指向稳定 storage。该 PR 已 merged 到对应 release，并且测试显式检查多轮 reload 的 `data_ptr()` 不变。

当前 main 也能看到 `_mxfp8_weight_buf/_mxfp8_scale_buf` 和 `restore_weights_for_rl_loading()` 这套机制。 

**当前结论：**

> 【已确认-已修其核心地址问题】，但它引出了后面 2.6 和 2.8 的另一个问题：**谁保证 reload 前调用正确的 restore 生命周期？**

---

### 2.1.3 SFA `W_UK_T` 的二次 NZ 转换

当前 SFA 已经显式处理了 `contiguous()` 地址问题：

```python
if not hasattr(self, "W_UV"):
    self.W_UV = ...
    self.W_UK_T = ...
else:
    self.W_UV.copy_(...)
    self.W_UK_T.copy_(...)
```

源码注释直接说明：graph + RL 下必须保持 weight address。

但是紧接着还有：

```python
if self.preprocess_type == PreprocessType.NATIVE:
    self.W_UK_T = maybe_trans_nz(self.W_UK_T)
```

而当前：

```python
maybe_trans_nz(weight)
```

在需要 NZ 时返回：

```python
torch_npu.npu_format_cast(
    weight,
    ACL_FORMAT_FRACTAL_NZ,
)
```

也就是可能得到新的 tensor/storage。 

因此流程可能变成：

```text
正确：
new ND derived
      ↓ copy_
old W_UK_T storage A

然后：
maybe_trans_nz(A)
      ↓
new NZ storage B

self.W_UK_T = B
```

如果 reload 时这个 branch 重跑，而且 graph 捕获的是此前 NZ tensor，那么前面的 `copy_` 保护会被最后一次 reassignment 抵消。

目前我没有找到一个公开 issue 明确报告这一点，因此：

> 【代码审计风险，高优先级】，不是已确认 bug。

**修复建议：**

不要：

```python
self.W_UK_T = maybe_trans_nz(self.W_UK_T)
```

而应该把最终 runtime representation 当作稳定 destination：

```text
cold:
derive ND
→ format cast NZ
→ allocate final runtime W_UK_T

reload:
derive ND temporary
→ format cast NZ temporary
→ old_runtime_W_UK_T.copy_(temporary_NZ)
```

也就是：

```text
最终 kernel-visible tensor
才是需要长期稳定的 storage
```

不是中间 ND tensor。

---

## 2.2 Checkpoint layout 与 runtime layout 不一致

这一类并不一定首先表现为 graph pointer 错，而通常直接表现为：

```text
shape mismatch
wrong TP shard
double transpose
silent wrong layout
```

核心问题是：

```text
weight_loader 的 contract = checkpoint/model format

PWAL 后 Parameter 的真实 format = runtime/kernel format
```

如果下一轮直接：

```python
model.load_weights(...)
```

而没有 restore：

```text
checkpoint tensor
       ↓
checkpoint-layout weight_loader
       ↓
runtime-layout destination
```

就是错误的。

---

### 2.2.1 MoE repeated transpose：Issue #5915

Issue #5915 当前仍 open。

issue 描述的场景是：

```text
load_weights
    ↓
w13/w2 transpose
    ↓
下一次 RL load_weights
    ↓
destination 已经是 transposed layout
```

历史代码尝试在：

```text
wake_up
```

阶段把 transpose 反过来。

但问题是：

> RL weight update 并不保证一定经历 `wake_up()`。

尤其 fully async training 可以一直保持 rollout engine awake。

Issue 原文明确指出这一点。

所以这是非常标准的：

```text
checkpoint/runtime layout restoration
错误绑定到 sleep/wakeup 生命周期
```

问题。

**当前结论：**

> 【已确认-未修/仍 open】。

**正确修复：**

不要依赖：

```text
sleep
→ wake_up
→ undo transpose
```

而应该依赖：

```text
start_weight_update
→ restore checkpoint representation
```

即：

```python
initialize_layerwise_reload()
```

或者未来：

```python
restore_weights_before_loading()
```

必须绑定到 **weight-update transaction**，而不是内存 offload transaction。

---

### 2.2.2 Issue #12226：反向证明 wake_up 里做 transpose 是错误抽象

#12226 是相反方向的真实 bug：

> sleep/wake 以后 MoE expert weight 又多 transpose 了一次。

它已经 closed/completed。 

它与 #5915 放在一起看很有价值：

```text
#5915:
希望 wake_up 帮 RL undo transpose
但 async RL 未必 wake_up

#12226:
普通 inference wake_up
却因为 transpose 生命周期绑定错误
把 weight 又 transpose 了一次
```

这说明根因不是某个 `if` 写错，而是：

> **weight representation restore 不应该属于 wake_up。**

应该属于：

```text
weight-update START phase
```

---

### 2.2.3 WNA16 / 310P / MoE 当前仍大量存在 destructive transpose/repack pattern

当前 main 搜索可以看到，例如 W8A16：

```python
layer.weight.data = (
    layer.weight.data
    .transpose(0, 1)
    .contiguous()
)

layer.weight.data = maybe_trans_nz(...)
```



310P FusedMoE 也有：

```text
transpose
→ contiguous
→ maybe_trans_nz
→ new Parameter
```



这些在 **native layerwise reload** 下可能通过：

```text
restore meta checkpoint layout
→ load
→ PWAL
→ copy runtime value back
```

被正确处理。

但是一旦入口是：

```python
model.load_weights()
```

直写，而没有 initialize：

> 它们就是同 #5915 一类的高风险点。

所以不能只按 quant method 判断“这个 PWAL 支不支持 RL”；还必须看 **调用它的 reload backend 是否执行 lifecycle**。

---

## 2.3 Derived weights / derived runtime state

这一类和 transpose 不同：

checkpoint 根本没有该 tensor。

例如：

```text
source checkpoint weight
        ↓
derive()
        ↓
runtime-only tensor
```

它有两个独立要求：

```text
1. value 要刷新
2. graph-visible storage 地址要稳定
```

---

### 2.3.1 DeepSeek V4 `weight_fp32`

#15666 实际同时属于两个类别。

它不仅是地址替换问题，还是 derived-state 问题：

```text
weight BF16
   ↓ .float()
weight_fp32
```

Trainer 更新：

```text
weight
```

不会发送：

```text
weight_fp32
```

因此 reload finish 必须显式重新：

```text
derive(new weight)
```

然后：

```text
old weight_fp32.copy_(new derived)
```

#15666 解决的是这两个动作结合起来的问题。

---

### 2.3.2 SFA `kv_b_proj → W_UK_T/W_UV`

当前 SFA：

```text
kv_b_proj.weight
        ↓ ND + transpose
        ↓ view
        ↓ split
   W_UK     W_UV
      ↓       ↓
 permute   transpose
      ↓       ↓
 W_UK_T    W_UV(runtime)
```

当前实现已经正确意识到这些是 graph-visible derived tensors，因此：

```python
first:
    allocate

reload:
    copy_
```



这是当前 vLLM-Ascend 里一个比较好的局部实现。

但它仍受两个其它问题影响：

```text
source kv_b_proj 被 dispose             → 2.4
W_UK_T 之后可能再 npu_format_cast       → 2.1
reload backend 未必执行 proper lifecycle → 2.8
```

所以“derived tensor 本身 copy_ 了”还不能证明整条 reload path 安全。

---

### 2.3.3 Kimi K3 `f_proj = f_b_proj @ f_a_proj`

PR #15168 是另一个很典型的 derived-weight 场景。

它明确需要：

```text
checkpoint:
f_a_proj
f_b_proj

         ↓ compose

runtime:
f_proj.weight =
    f_b_proj.weight @ f_a_proj.weight
```

并且 PR 描述明确说：

> composed weight 必须在 checkpoint load 后以及 **later source-weight reloads** 后重新生成。

该 PR 当前仍 open/unmerged。

这里揭示另一个重要风险：

```text
derived dependency graph
```

可能不止一级：

```text
source A/B
   ↓
derived F
   ↓
packed BFG
   ↓
kernel input
```

reload 不能只刷新第一层。

需要：

```text
dependency order:
source
→ level-1 derived
→ level-2 packed/derived
```

**建议修复：**

长期不应该靠“再跑整套 PWAL”隐式保证这些 derived state。

更好的接口是 upstream RFC 正在提出的：

```python
refresh_derived_state()
```

并要求：

```text
in-place
idempotent
dependency ordered
no arbitrary reallocation
```

---

### 2.3.4 W4A8 `scale_bias`：Issue #3152

Issue #3152 指出：

```text
scale_bias
```

在并行场景中是在 shard 后计算，因此结果与全量权重域上的计算不一致。

Issue 已因 stale 关闭为 `not_planned`，不是“确认已经修好”。

这一类不是 storage-address bug，而是：

> **derived state 在错误的数据域 / 错误顺序上计算。**

例如：

```text
错误：

global weight
    ↓ shard
local weight
    ↓ derive scale_bias

正确语义如果要求 global statistics：

global weight
    ↓ derive scale_bias
    ↓ shard corresponding state
```

所以 derived-state correctness 需要检查的不只是：

```text
有没有 refresh
```

还包括：

```text
refresh 在 TP/EP shard 前还是后？
输入属于 global domain 还是 local domain？
```

---

## 2.4 Destructive source release / Parameter registration 被破坏

这一类是 SFA #15463 的核心。

---

### 2.4.1 SFA `dispose_layer(kv_b_proj)`：Issue #15463

当前 main 在生成：

```text
W_UK_T
W_UV
```

之后仍明确执行：

```python
dispose_layer(self.kv_b_proj)
```



Issue #15463 当前仍 open。

`dispose_layer()` 的效果不是：

```text
remove registration
```

而是把 Parameter storage 变成空 tensor。

因此模型结构里：

```text
kv_b_proj.weight
```

名字仍存在，但 destination 已经类似：

```text
shape = [0]
```

下一轮直接收到 checkpoint：

```text
kv_b_proj.weight
shape ≈ [7168, 512]
```

然后：

```python
model.load_weights()
```

就会遇到：

```text
destination [0]
vs
incoming [7168,512]
```

这就是 #15463。

**当前结论：**

> 【已确认-未修】，main 仍存在 dispose。

**短期修复：**

RL/live reload 开启时：

```text
不要 dispose checkpoint source parameter
```

最简单：

```python
if not live_weight_reload_enabled:
    dispose_layer(self.kv_b_proj)
```

代价只是额外保留 source weight 显存。

Issue 中给出的一个 local BF16 示例 `[7168,512]` 约 7 MiB/层。

**更长期修复：**

允许 cold inference dispose，但 reload framework 必须同时具备：

```text
restore checkpoint metadata
+
materialize temporary source
+
load checkpoint
+
derive runtime state
+
copy derived state to old graph-visible storage
```

即 source 参数不一定必须永久保留，但其 **reload representation 必须可恢复**。

---

### 2.4.2 Block FP8：删除 `weight_scale_inv`

当前 block-FP8：

```python
resolved = resolve_block_scales(
    layer.weight.data,
    layer.weight_scale_inv.data,
    ...
)

del layer.weight_scale_inv
```

之后又根据路径重新创建：

```python
layer.weight = Parameter(...)
```

甚至：

```python
layer.weight = Parameter(quantized)
layer.weight_scale = Parameter(mx_scale)
```



这是典型的：

```text
checkpoint representation:
weight + weight_scale_inv

PWAL 后：
删除 weight_scale_inv
构造 runtime weight / weight_scale
```

对于 native layerwise：

```text
cold 时 restore_metadata 已捕获
```

理论上可以恢复 `weight_scale_inv` 的 meta representation。

但对于：

```text
NPU IPC direct model.load_weights
VERL direct model.load_weights
其它绕过 initialize 的 caller
```

它就是高风险路径。

所以这是：

> 【代码审计风险】，尤其对非-layerwise reload backend。

修复应该不是“永远不删 scale”，而是确保：

```text
reload START
    ↓
restore weight_scale_inv checkpoint destination

LOAD
    ↓
load incoming block scale

FINISH
    ↓
resolve/requantize
    ↓
copy into stable runtime storage
```

---

### 2.4.3 FusedMoE：Parameter → `list[Tensor]` 后删除 Parameter

当前 main 的 unquantized routed experts 有更激进的操作：

```python
layer.w13_weight_list = [
    weight.clone()
    for weight in layer.w13_weight.data.unbind(0)
]

layer.w2_weight_list = [
    weight.clone()
    for weight in layer.w2_weight.data.unbind(0)
]

del layer.w13_weight
del layer.w2_weight
```

发生在 MegaMoE / dynamic EPLB 等路径。

这里发生了三件危险事情：

```text
1. Parameter 被拆成 Python list
2. clone() 为每个 expert 新建独立 storage
3. 原 registered Parameter 被删除
```

upstream generic layerwise 最容易保护的是：

```text
_parameters
_buffers
```

而这里最终 graph/kernel 可能使用：

```text
list[Tensor]
```

它们并不属于 generic `kernel_tensors`。

所以这个场景比 #15463 还复杂。

**当前判断：**

> 【高优先级代码审计风险】。我目前没有找到公开 issue 明确报告“MegaMoE list weights + native reload”故障，因此不能标成已确认。

**建议修复有两条路线：**

第一种，优先推荐：

```text
不要把 graph-visible runtime experts 存成裸 list[Tensor]
```

可以把 storage 注册成明确的 Parameter/Buffer，然后在调用 kernel 时创建轻量 view/list：

```text
registered packed storage
     ↓ unbind views
kernel inputs
```

这样真实 owner 仍是 module。

第二种，如果 backend 必须持有 list：

```text
cold:
allocate expert_i runtime storage once

reload:
new transformed expert_i
        ↓
old_list[i].copy_(...)
```

并且禁止：

```python
w13_weight_list = [x.clone() ...]
```

在每轮 PWAL 中重新执行。

---

# 2.5 PWAL 非幂等 / 错误挂在 wake_up 等生命周期

这一类和 layout transform 经常一起发生，但根因值得单独看。

---

### 2.5.1 #5915：transpose 的逆操作依赖 `wake_up`

已经讨论过：

```text
reload lifecycle
≠
sleep/wakeup lifecycle
```

Issue #5915 仍 open，fully async RL 是明确的反例。

---

### 2.5.2 #12226：wake_up 重复 transpose

这是非幂等 PWAL 的经典表现：

```text
state S0
  ↓ transpose
S1

wake/reload
  ↓ transpose
S2
```

如果 transform：

\[
T(T(W)) \neq T(W)
\]

或者第二次输入已不是 transform 所期望的 representation，结果就错误。

#12226 已完成关闭。

---

### 2.5.3 当前 main 仍存在大量“每次调用都 mutate representation”的 PWAL

比如：

```python
layer.weight.data = maybe_trans_nz(...)
```

或者：

```python
layer.weight = Parameter(...)
```

或：

```python
del layer.weight_scale_inv
```

 

这些函数很多本质上是：

```text
cold-load finalizer
```

而不是：

```text
idempotent hot-reload refresher
```

因此不应该假设：

```python
process_weights_after_loading()
```

天然支持：

```text
call 1
call 2
call 3
...
```

RFC #2851 其实已经从架构层承认了这个问题：Ascend 每层散落 PWAL，导致 RL dynamic update 要额外 override 多个 loader，提出 NPUModelLoader 来集中管理。该 RFC 当前仍 open。

**建议修复：**

最终应该把：

```text
cold PWAL
```

和：

```text
reload refresh
```

拆开。

例如：

```python
process_weights_after_loading()   # cold build, can allocate/repack

restore_weights_before_loading()  # restore model/checkpoint representation

refresh_derived_state()           # reload finish, in-place/idempotent
```

而不是继续给每个 PWAL 增加：

```python
if is_rl:
if has_attr:
if wake_up:
```

---

# 2.6 Quantization/repack 的 restore-before-load 问题

这类问题的关键不是“PWAL 后地址有没有保持”，而是：

> 下一轮 checkpoint 来之前，Parameter 有没有恢复成 loader 能理解的 shape/layout？

---

### 2.6.1 MXFP8 已有 `restore_weights_for_rl_loading()`，但生产路径调用不统一

当前 MXFP8 main 中明确存在：

```python
restore_weights_for_rl_loading(self, layer)
```

其注释写得非常直接：

> Must be called BEFORE `model.load_weights()` in RL training. 

但是当前仓库搜索这个函数的调用，我找到的是：

```text
implementation
+
unit test
```

测试确实显式：

```text
process
→ restore
→ model.load_weights 模拟
→ process
```

 

没有找到一个通用 NPU reload engine 显式调用它。

这本身不代表 HCCL 有 bug，因为 HCCL 已采用 upstream：

```python
initialize_layerwise_reload()
```

它通过 restore metadata 完成等价职责。

但对 NPU IPC：

```text
没有 layerwise
也没有 generic restore hook
```

这就成为真实的 integration gap。

---

### 2.6.2 Block FP8 → MXFP8 有多级 representation

当前 block FP8：

```text
checkpoint:
FP8 weight
+
weight_scale_inv
       ↓
resolve_block_scales
       ↓
model dtype tensor
       ↓ A5:
dynamic MX quant
       ↓
MXFP8 weight + scale
       ↓
MXFP8 PWAL transpose/repack
```

当前源码完整展示了这条链，并在中途删除 `weight_scale_inv`、创建新 Parameter。

所以这不是简单的：

```text
checkpoint → runtime
```

而是：

```text
checkpoint
→ resolved
→ requantized
→ transformed runtime
```

每一个阶段都可能：

```text
改变 dtype
改变 shape
改变 storage
删除 source
产生 derived scale
```

这种 backend 如果绕过 layerwise，基本不应该假设裸 `model.load_weights()` 能工作。

---

### 2.6.3 WNA16 / W8A8 / 310P NZ 路径

当前源码搜索还有多处：

```text
transpose()
contiguous()
maybe_trans_nz()
flatten()
```

例如 W8A16。

310P dynamic W8A8 和 FusedMoE 也有类似 transform。 

这些都应该纳入同一测试矩阵：

```text
cold A
→ graph capture
→ reload B
→ reload C
→ compare cold-C
```

而不能只测：

```text
“第二次 model.load_weights 没报异常”
```

---

# 2.7 Unmanaged storage、alias/binding 与 offloader

这类 tensor 不一定是 Parameter/Buffer，但 runtime kernel 会长期引用它。

---

### 2.7.1 NZ static buffer binding：PR #15415

PR #15415 是一个很好的 Ascend 特有例子。

问题不是：

```text
weight value 算错
```

而是：

```python
torch_npu.npu_format_cast(...)
```

返回了一个 **新的 NZ tensor**。

StaticBufferPool 中已经换成：

```text
NZ buffer B
```

但是：

```text
param.data
_param_offloader._gpu_buffer
```

仍指向：

```text
旧 ND buffer A
```

最后 NZ-only kernel 使用的实际上还是 ND storage，导致稳定的 logprob drift。

PR 说明的根因和修复非常明确：重新 bind parameter 和 offloader `_gpu_buffer` 到新的 NZ buffer，并重新执行初始 prefetch。该 PR 当前仍 open/unmerged。

这说明后处理除了：

```text
value correctness
address correctness
```

还存在第三个维度：

```text
reference/binding topology correctness
```

即：

```text
BufferPool ──┐
Param.data ──┼── 必须指向同一个 runtime storage
Offloader  ──┘
```

只是“创建了正确的 NZ tensor”不够。

---

### 2.7.2 `npu_format_cast()` 是当前很值得全面审计的操作

当前 `maybe_trans_nz()`：

```python
return torch_npu.npu_format_cast(
    weight,
    ACL_FORMAT_FRACTAL_NZ,
)
```



而 #15415 已经给了非常强的实证：

> `npu_format_cast` 可以返回新的 tensor/storage。

因此当前 main 所有下面 pattern：

```python
layer.foo = maybe_trans_nz(layer.foo)

layer.foo.data = maybe_trans_nz(layer.foo.data)

self.foo = maybe_trans_nz(self.foo)
```

都应该重新审核：

```text
foo 是否 graph-visible？
foo 是否有其它 owner/alias？
reload 时会不会再次执行？
generic layerwise 能不能看到 foo？
```

当前搜索已经能看到：

```text
linear.weight
FusedMoE w13/w2
310P embedding weight_nz
WNA16
MLA/SFA W_UK_T
```

等多个位置。    

其中 registered Parameter 在 HCCL layerwise 下通常还能被 old-storage copy-back 保护；**plain tensor attribute** 则不能自动获得这种保护。

---

### 2.7.3 `weight_nz` 这类 plain derived attribute

310P embedding 当前有：

```python
layer.weight_nz = maybe_trans_nz(layer.weight)
```



这在代码形态上和 #15666：

```python
layer.weight_fp32 = new_tensor
```

非常接近。

如果：

```text
weight_nz
```

是 forward/ACLGraph 真正使用的 persistent tensor，并且 PWAL 会在 reload 后重跑，那么它就是：

```text
unregistered derived graph-visible tensor
+
direct reassignment
```

这是 #15666 同类风险。

我暂时没有找到对应公开 bug，所以：

> 【代码审计风险】。

优先确认：

```text
apply() 是否直接把 weight_nz 传给 graph-captured op
```

如果是，则应改为：

```text
cold allocate
reload copy_
```

或者把它注册成：

```python
register_buffer(..., persistent=False)
```

再配合正确 refresh。

---

# 2.8 Reload backend 生命周期不一致：这是当前最大的放大器

这不是单个 PWAL bug，但它决定前面所有 bug 是否会暴露。

---

### 2.8.1 HCCL：当前已经走正确 native layerwise transaction

当前 main：

```python
def start_weight_update(self):
    initialize_layerwise_reload(self.model)

def finish_weight_update(self):
    finalize_layerwise_reload(
        self.model,
        self.model_config,
    )
```



因此 HCCL 的语义是：

```text
START
save current runtime tensors
restore checkpoint representation

RECEIVE
model.load_weights(...)

FINISH
PWAL
copy processed result back
restore old graph-visible storage
```

这与 upstream vLLM 当前设计一致。

所以很多：

```text
registered Parameter transpose/repack
```

在 HCCL 下可以由 layerwise machinery 托底。

---

### 2.8.2 NPU IPC：当前完全没有 layerwise

当前 `main` 的 NPU IPC：

```python
def start_weight_update(self):
    """No-op for NPU IPC engine (no layerwise reloading)."""
    pass

def finish_weight_update(self):
    """No-op for NPU IPC engine (no layerwise reloading)."""
    pass
```



于是实际变成：

```text
checkpoint tensor
    ↓
direct model.load_weights
    ↓
current runtime model
```

没有：

```text
checkpoint-layout restore
graph-time tensor snapshot
PWAL deferred processing
runtime storage copy-back
```

这意味着：

- #5915 一类 transpose/layout 问题容易直接暴露；
- MXFP8 的 `restore_weights_for_rl_loading` 没有通用调用者；
- block-FP8 已删除的 source scale 无法自然恢复；
- SFA disposed `kv_b_proj` 直接成为 `[0]` destination；
- #15666 这种 unregistered derived tensor 更没有 generic storage protection。

所以：

> **当前 NPU IPC 本身就是一个架构级 correctness gap，而不是某个具体 quant backend 的局部 bug。**

---

### 2.8.3 NPU IPC packed path 还有一个更直接的问题

当前源码：

```python
if self.packed:
    weights = packed_npu_ipc_consumer(...)
else:
    ...
    weights.append(...)
    self.model.load_weights(weights)
```

注意：

```python
self.model.load_weights(weights)
```

目前仍缩进在 `else:` 中。

因此当前 main 的 `packed=True` 路径看起来是：

```text
successfully unpack weights
        ↓
weights variable exists
        ↓
return
```

但没有：

```text
model.load_weights(weights)
```

这是纯代码路径上可以直接确认的问题。

> 【当前 main 明确缺陷候选，置信度非常高】。

建议直接改成：

```python
if self.packed:
    weights = ...
else:
    weights = ...

self.model.load_weights(weights)
```

并同时补齐：

```python
start_weight_update
    → initialize_layerwise_reload

finish_weight_update
    → finalize_layerwise_reload
```

这样 NPU IPC 才与当前 HCCL/upstream CUDA IPC 的 semantic contract 一致。

---

# 3. 当前问题按严重性汇总

如果按“现在最应该修什么”排序，我会分成下面几档。

| 优先级 | 问题 | 当前状态 | 原因 |
|---|---|---|---|
| P0 | NPU IPC 没有 layerwise initialize/finalize | 当前 main 明确存在 | 会放大几乎所有 checkpoint/runtime PWAL 问题 |
| P0 | NPU IPC packed path 没看到 `model.load_weights()` | 当前 main 明确存在 | packed weight 可能根本未应用 |
| P0 | SFA `dispose_layer(kv_b_proj)` #15463 | open + main 仍有 | live reload destination 被破坏 |
| P0/P1 | DeepSeek-V4 `weight_fp32` #15666 | open + main 仍有 | ACLGraph 使用 stale FP32 derived weight |
| P1 | MoE transpose lifecycle #5915 | open | async RL 不经过 wake_up |
| P1 | FusedMoE Parameter→`list[Tensor]`+delete | main audit risk | generic layerwise 无法管理 graph-visible list storage |
| P1 | SFA `W_UK_T = maybe_trans_nz(...)` | main audit risk | 可能抵消前面的 in-place address protection |
| P1 | block-FP8 delete/recreate source/runtime params | main audit risk | 非-layerwise reload 无法安全恢复 |
| P1/P2 | 310P `weight_nz` 等 plain tensor reassignment | audit risk | 与 #15666 同 pattern |
| P2 | offloader NZ binding #15415 | open PR | storage layout正确但引用拓扑错误 |
| P2 | derived calculation domain #3152 | stale/not_planned | TP/EP 下 derived state 可能语义不同 |
| Historical | W8A8 MXFP8 #13905 | merged | 已证明 graph address stability 必须专门处理 |
| Historical | MoE wake double transpose #12226 | completed | 证明 lifecycle 绑定错误 |

---

# 4. 我认为 vLLM-Ascend 当前最核心的架构问题

这些 issue 看起来很分散：

```text
#13905 MXFP8
#15463 SFA
#15666 DSV4 FP32
#5915 MoE transpose
#15415 NZ offloader
```

但其实只是在重复暴露同一个设计问题：

```text
process_weights_after_loading()
```

当前同时承担了：

```text
cold-load initialization
layout transform
quantization
NZ conversion
derived state creation
runtime buffer construction
kernel preparation
memory optimization/disposal
```

而 RL 又试图：

```text
重新 load checkpoint
+
重新调用其中一部分 PWAL
```

这两个 contract 天然冲突。

vLLM-Ascend 自己的 RFC #2851 已经指出：每层分散实现 PWAL 导致 RL dynamic weight update 必须到处 override loader，应该集中化。

RFC #12073 也明确提出，需要用 Dense/MoE/FP8/VLM 和 HCCL/IPC 覆盖完整 weight-update 生命周期，而不是只测单次 inference。

因此我认为不能继续只按：

```python
if hasattr(self, "xxx"):
    self.xxx.copy_(...)
```

逐个打补丁。

这些 patch 是必要的，但不是最终架构。

---

# 5. 建议的统一修复方向

可以直接借鉴 upstream vLLM 已经形成的方向，把 Ascend PWAL 明确拆成三个职责：

```text
START
restore_weights_before_loading()

LOAD
model.load_weights()

FINISH
refresh_derived_state()
```

其中三类 state 分开处理。

### 第一类：checkpoint-backed registered Parameter/Buffer

交给 generic loader/layerwise：

```text
restore checkpoint metadata
→ load
→ runtime transform
→ copy_ old graph storage
```

### 第二类：derived graph-visible state

例如：

```text
weight_fp32
W_UK_T
W_UV
f_proj
_g1_alphas
weight_nz
```

统一要求：

```text
cold:
allocate final runtime storage

reload:
derive temporary result
→ old_runtime_tensor.copy_(new_result)
```

不要再：

```python
self.foo = new_tensor
```

### 第三类：backend-owned unmanaged storage

例如：

```text
MegaMoE expert lists
workspace
NZ static buffer
kernel strides/constants
```

要求 backend 显式声明：

```text
reuse_storage()
refresh_runtime_state()
```

或者重构成 registered non-persistent buffer。

---

# 6. 对当前代码最实用的一轮审计方法

如果现在要继续系统扫 vLLM-Ascend main，我会直接搜所有 PWAL 里的：

```text
= .contiguous()
= .clone()
= .to(...)
= maybe_trans_nz(...)
= torch_npu.npu_format_cast(...)
.data =
replace_parameter(...)
nn.Parameter(...)
del layer.xxx
dispose_layer(...)
resize_(...)
list(...clone...)
```

然后对每一处建立四列：

```text
1. Source representation
2. Final runtime representation
3. Is graph-visible?
4. Is storage managed by layerwise?
```

如果出现：

```text
graph-visible = YES
layerwise-managed = NO
reload reallocates = YES
```

就是 **#15666 / #13905 型**。

如果出现：

```text
checkpoint source deleted = YES
reload backend direct-load = YES
```

就是 **#15463 型**。

如果出现：

```text
runtime layout != checkpoint layout
reload before-load restore = NO
```

就是 **#5915 型**。

如果出现：

```text
derived from source weights = YES
reload refresh = NO
```

就是 **stale-derived-state 型**。

如果出现：

```text
kernel/offloader has another reference
new runtime storage allocated = YES
reference not rebound/reused
```

就是 **#15415 型**。

我认为这 5 个 predicate 基本可以覆盖现在 vLLM-Ascend 已暴露的绝大多数 weight-update/PWAL correctness bug。

---

# 7. 最终建议的修复优先顺序

最先应该修的是 **reload transaction 本身**，而不是逐模型修 PWAL：

```text
NPU IPC:

start_weight_update()
    ↓
initialize_layerwise_reload(model)

receive_weights()
    ↓
packed / unpacked
    ↓
统一 model.load_weights(weights)

finish_weight_update()
    ↓
finalize_layerwise_reload(model, model_config)
```

因为当前 HCCL 已经是这套语义。 NPU IPC 却明确是 no-op start/finish。

然后再做第二层审计：

```text
所有 layerwise 无法看到的 graph-visible state：
weight_fp32
W_UK_T/W_UV
weight_nz
expert tensor lists
workspace/static buffers
kernel-owned tensors
```

逐个改成：

```text
registered buffer/parameter
或
cold allocate + reload copy_
或
backend existing-storage reuse
```

第三层才处理：

```text
dispose/delete/source release
non-idempotent PWAL
derived-state dependency ordering
```

这样修完之后，整个 contract 会从现在的：

```text
“希望某个 PWAL 在第二次调用时恰好还能工作”
```

变成：

```text
checkpoint format
      ↓
explicit restore
      ↓
load
      ↓
explicit runtime refresh
      ↓
stable graph-visible storage
```

这才是 vLLM-Ascend 后续支持 VERL/slime/async RL、HCCL/IPC、Dense/MoE/FP8/SFA 时能够扩展的结构。