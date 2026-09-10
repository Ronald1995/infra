下面把当前 upstream vLLM 的权重加载 / 热更新流程统一整理一下，并把 VERL 当前实现放进同一个框架里看。这里重点讨论的是 **RL 场景下 checkpoint-format 权重反复更新，同时已经存在 CUDA Graph 的情况**。

一个贯穿全文的核心概念是：

```text
checkpoint/model representation
        ↓ load_weights
checkpoint-format Parameters
        ↓ process_weights_after_loading / PWAL
runtime/kernel representation
        ↓ CUDA Graph capture
graph-visible storage
```

热更新真正困难的地方不在 `model.load_weights()` 本身，而在于：

> 新 checkpoint 权重加载之后，如何重新得到正确的 runtime representation，同时保持 CUDA Graph 已捕获的 storage address 不变。

---

# 1. vLLM 权重加载整体流程

## 1.1 三阶段模型：Initialize → Load → Finalize

当前 vLLM layerwise reload 可以抽象成：

```text
┌─────────────────────────────────────────────┐
│ 1. Initialize                              │
│                                             │
│ initialize_layerwise_reload(model)          │
│                                             │
│ - 保存当前 runtime/kernel tensor            │
│ - 恢复 checkpoint/model-format metadata     │
│ - Parameter/Buffer 临时切回 meta            │
│ - 包装 weight_loader，延迟真正加载/处理       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ 2. Load                                    │
│                                             │
│ model.load_weights(weights)                 │
│                                             │
│ - name mapping                              │
│ - TP shard                                  │
│ - fused QKV / gate-up routing               │
│ - expert routing                            │
│ - weight_loader                             │
│ - layer ready 后执行 PWAL                    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ 3. Finalize                                │
│                                             │
│ finalize_layerwise_reload(model)            │
│                                             │
│ - 处理未完成 layer                          │
│ - deferred attention                        │
│ - attention scales                          │
│ - PWAL                                      │
│ - 新 runtime value copy 回旧 storage        │
│ - 恢复原 Parameter/Buffer object            │
└─────────────────────────────────────────────┘
```

vLLM 当前官方 RL reload 的核心就是这三个阶段。

---

## 1.2 Cold load 时先记录“模型格式”

在第一次 cold load 阶段，vLLM 会记录以后 reload 所需的 metadata：

```python
record_metadata_for_reloading(model)
```

内部遍历 module：

```python
info.restore_metadata = capture_layer_to_meta(layer)
```

保存的是：

```text
Parameter:
    name
    shape
    dtype
    stride
    Parameter subclass / attrs
    checkpoint/model representation

Buffer:
    name
    shape
    dtype
    persistent metadata
```

注意，这不是保存实际 GPU 权重副本，而是：

> 保存“如果以后重新接受 checkpoint-format 权重，这个 layer 应该恢复成什么结构”。

因此一个量化层可能有两个完全不同的世界。

Cold load 前：

```text
weight
shape = [N, K]
checkpoint layout
```

PWAL 后：

```text
weight
shape = packed / transposed / quantized runtime layout
```

以后 reload 前，必须有办法重新构造 `[N,K]` 那个 checkpoint-facing 参数结构。

---

## 1.3 Initialize：先保存 CUDA Graph 当前使用的 tensor

`initialize_layerwise_reload()` 最关键的一步：

```python
info.kernel_tensors = get_layer_params_buffers(layer)
```

当前 generic 机制保存的是：

```text
layer._parameters
layer._buffers
```

也就是现在已经被 PWAL 转换好的 runtime tensors。

假设：

```text
layer.weight
   ↓
Parameter P_old
   ↓
storage A
```

CUDA Graph capture 后：

```text
CUDA Graph ─────→ storage A
```

reload initialize 时：

```text
info.kernel_tensors ─────→ P_old ─────→ storage A
```

即使后面：

```python
delattr(layer, "weight")
```

old Parameter 仍然被 `info.kernel_tensors` 强引用。

因此 storage A 不会丢。

---

## 1.4 Initialize：恢复 checkpoint/model-format meta tensor

之后执行：

```python
restore_layer_on_meta(layer, info)
```

把当前 runtime Parameter/Buffer 拆掉，恢复 cold load 时保存的 metadata：

```text
runtime:

weight
shape = runtime_shape
ptr = A

             ↓ restore_layer_on_meta

checkpoint-facing:

weight
shape = checkpoint_shape
device = meta
```

这里的 meta 很关键：

```text
恢复 shape/dtype/layout
但不立即申请一整份 GPU storage
```

之后真正 layer ready 时再 materialize。

---

## 1.5 Load：`model.load_weights()` 到底负责什么

`model.load_weights()` 仍然是整个系统非常重要的入口。

它主要负责：

```text
checkpoint name
      ↓
model parameter mapping
      ↓
stacked mapping / expert mapping
      ↓
TP shard
      ↓
parameter.weight_loader(...)
```

比如 checkpoint：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

而模型 runtime registration 可能只有：

```text
qkv_proj.weight
```

那么 `AutoWeightsLoader / weight_loader` 会做：

```text
q_proj ─┐
k_proj ─┼─→ qkv_proj.weight 对应 shard
v_proj ─┘
```

MoE 同理：

```text
gate_proj
up_proj
down_proj
expert id
TP shard
```

都不能简单靠：

```python
dict(model.named_parameters())[name].copy_(tensor)
```

替代。

所以即使未来 selective reload，vLLM RFC 仍然坚持：

> 保留 `model.load_weights()` 和现有 `weight_loader`，因为 500+ 模型的名字映射、TP shard、fused mapping 都已经在那里。

---

## 1.6 Layerwise 的特殊之处：Load 阶段其实先缓存

Initialize 会把原 weight loader 包装成：

```text
online_process_loader
```

incoming 权重到来时，先记录：

```python
info.loaded_weights.append(...)
info.load_numel += ...
```

直到：

```python
info.load_numel >= info.load_numel_total
```

才调用：

```python
_layerwise_process(layer, info)
```

于是完整 layer 级流程：

```text
received checkpoint tensors
       ↓
buffer args
       ↓
layer complete
       ↓
materialize model-format tensors
       ↓
replay original weight_loader
       ↓
process_weights_after_loading
       ↓
copy runtime result back
```

这样做很重要，因为很多 transformation 必须等完整 layer 的参数到齐，例如：

```text
gate + up
    ↓
fused w13

q + k + v
    ↓
fused qkv

所有 expert weights
    ↓
MoE repack
```

不能一收到单个 shard 就立即 PWAL。

---

## 1.7 `_layerwise_process()`：真正完成一次 layer reload

核心代码逻辑是：

```python
materialize_layer(layer, info)

for name, args in info.loaded_weights:
    param.weight_loader(...)

quant_method.process_weights_after_loading(layer)

_copy_and_restore_kernel_tensors(layer, info)
```

也就是：

```text
new checkpoint tensor
       ↓
weight_loader
       ↓
temporary checkpoint parameter
       ↓
PWAL
       ↓
temporary runtime parameter
       ↓
copy_
       ↓
old graph-visible runtime storage
```



---

## 1.8 Finalize：为什么还需要一个 finalize

不是所有 layer 都能在收到最后一个 weight 的瞬间完成处理。

例如：

```text
Attention
padding parameter
checkpoint 中不存在的 runtime buffer
alias buffer
特殊 scale
```

因此：

```python
finalize_layerwise_reload(model, model_config)
```

会做补收尾：

```text
普通未完成 layer
       ↓
_layerwise_process

attention
       ↓
deferred processing

没有收到 checkpoint weight 的 layer
       ↓
恢复 old kernel tensors

attention scale
       ↓
重新创建/loading/process/copy-back
```

当前 attention 被特意延后，因为 attention PWAL 依赖其它模块已经完成处理。

---

# 2. 几种常见的权重更新方式

---

# 2.1 `reload_weights()`：直接从 iterator / 文件重新加载

这是最容易理解的一种。

当前 `GPUModelRunner.reload_weights()` 会区分：

```text
checkpoint format
vs
runtime/kernel format
```

对于 checkpoint-format：

```python
initialize_layerwise_reload(model)

model.load_weights(weights_iterator)

finalize_layerwise_reload(
    model,
    self.model_config,
)
```

当前源码就是这条路径。

因此：

```text
reload_weights
      │
      ├── checkpoint-format
      │       ↓
      │   native layerwise reload
      │
      └── processed/runtime-format
              ↓
          direct copy_
```

为什么 runtime-format 可以直接 copy？

因为：

```text
incoming representation
==
current graph-visible representation
```

例如：

```python
param.copy_(loaded_weight)
```

不需要：

```text
restore checkpoint layout
PWAL
```

---

# 2.2 NCCL 权重加载

当前 vLLM 的官方 NCCL weight-transfer engine 已经完整采用三阶段：

```text
Trainer
   │
   │ NCCL broadcast
   ▼
vLLM worker
```

### Start

```python
NCCLWeightTransferEngine.start_weight_update()
```

执行：

```python
initialize_layerwise_reload(self.model)
``` 


### Receive

每一批权重：

```text
NCCL broadcast
    ↓
received tensor
    ↓
model.load_weights(...)
```

因此 incoming 仍然按 checkpoint format 处理。

### Finish

```python
NCCLWeightTransferEngine.finish_weight_update()
```

执行：

```python
finalize_layerwise_reload(
    self.model,
    self.model_config,
)
``` 


完整流程：

```text
start_weight_update
    ↓
initialize_layerwise_reload
    ↓
receive bucket 0
    ↓
model.load_weights
    ↓
receive bucket 1
    ↓
model.load_weights
    ↓
...
    ↓
finish_weight_update
    ↓
finalize_layerwise_reload
```

非常重要的一点是：

> **initialize/finalize 包围的是整个 model update transaction，而不是每一个通信 bucket。**

因为一个 layer 很可能跨 bucket。

---

# 2.3 CUDA IPC 权重加载

当前 upstream CUDA IPC 和 NCCL 语义基本一致，只是通信方式不同。

### Start

当前：

```python
def start_weight_update(self):
    initialize_layerwise_reload(self.model)
```

### Receive

IPC 可以：

```text
unpacked IPC
```

每个 tensor 一个 IPC handle，也可以：

```text
packed IPC
```

多个 tensor 放进 packed staging buffer。

但两种路径最终都生成：

```python
weights = [(name, tensor), ...]
```

然后统一：

```python
with disable_mtp_completeness_check():
    self.model.load_weights(weights)
```



### Finish

```python
def finish_weight_update(self):
    finalize_layerwise_reload(
        self.model,
        self.model_config,
    )
    self._packed_importer.close()
```



因此 upstream 当前 IPC：

```text
start
  ↓
layerwise initialize
  ↓
IPC reconstruct tensor
  ↓
model.load_weights
  ↓
...
  ↓
layerwise finalize
  ↓
release IPC staging resources
```

设计已经比较完整。

---

## IPC 为什么特别强调 tensor lifetime

IPC tensor 可能只是 trainer-owned storage 的跨进程 view：

```text
trainer staging buffer
       │
       │ CUDA IPC handle
       ▼
rollout IPC tensor
```

所以 sender 必须保证：

```text
直到 receiver load 完
storage 不能被回收/覆盖
```

当前 upstream IPC sender 专门保存 strong refs：

```text
weight_refs
```

直到 `finish_weight_update()` 之后才释放。

这和 graph address preservation 是不同的问题：

```text
通信 buffer lifetime
≠
model runtime storage identity
```

两者都必须正确。

---

# 2.4 当前 VERL 的加载方式

这里是目前最值得关注的部分。

当前 VERL `vllm_rollout` 的主要 standard base-weight 更新仍然不是直接调用 upstream vLLM 的：

```text
start_weight_update()
receive_weights()
finish_weight_update()
```

而是有自己的：

```text
BucketedWeightSender
        ↓ ZMQ / IPC / SHM
BucketedWeightReceiver
```

然后每收到一个 bucket：

```python
self._update_weights(weights, ...)
```



对于普通非 FP8 base model：

```python
if param_updates:
    for model in self._iter_all_models():
        model.load_weights(param_updates)
```



所有 bucket 都收完之后，VERL 再：

```python
process_weights_after_loading(
    model,
    model_config,
    self.device,
)
```



因此当前 standard VERL 路径实际是：

```text
receive bucket 0
      ↓
model.load_weights(bucket0)

receive bucket 1
      ↓
model.load_weights(bucket1)

receive bucket 2
      ↓
model.load_weights(bucket2)

...
      ↓

all buckets finished
      ↓
process_weights_after_loading(model)
```

而不是 upstream 推荐的：

```text
initialize_layerwise_reload
      ↓
model.load_weights(bucket0)
      ↓
model.load_weights(bucket1)
      ↓
...
      ↓
finalize_layerwise_reload
```

这是两套本质不同的生命周期。

---

# 3. 权重后处理与图模式地址保护

---

# 3.1 `process_weights_after_loading` 中有哪些典型场景

PWAL 不能简单理解为：

```text
量化
```

实际上它承担了非常多类型的转换。

当前 vLLM RFC 对 88 个 PWAL 实现做过系统扫描，发现大致存在 no-op/delegation、format transform、derived state、online quantization 等多个类别。

可以进一步拆成下面这些常见场景。

---

## 3.1.1 Transpose / Permute

最常见。

例如：

```text
checkpoint:
[N, K]

runtime:
[K, N]
```

或者 MLA：

```text
W_UK:
[L,N,P]

       ↓ permute

W_UK_T:
[N,P,L]
```

原因通常是：

```text
kernel expected layout
memory coalescing
GEMM operand convention
backend-specific format
```

问题是：

```python
new_weight = old_weight.T.contiguous()
```

会产生新的 storage。

cold load 没问题。

graph capture 后重新执行则可能导致：

```text
old Graph → A
new Python tensor → B
```

---

## 3.1.2 Fuse

例如：

```text
gate_proj
up_proj
    ↓
w13_weight
```

或者：

```text
q_proj
k_proj
v_proj
    ↓
qkv_proj
```

这里通常既涉及：

```text
name mapping
sharding
concat
```

也可能随后再 repack。

因此必须等完整 fused parameter 的 source weights 到齐。

---

## 3.1.3 Quantization

例如：

```text
BF16
   ↓
FP8 / INT8 / INT4 / NVFP4 / MXFP4
```

可能生成：

```text
quantized weight
weight_scale
input_scale
zero point
block scale
```

这不仅修改 weight，还可能创建新的 derived metadata。

---

## 3.1.4 Repack / Shuffle

一些 kernel 不直接使用 checkpoint quantized layout。

例如：

```text
GPTQ layout
   ↓
Marlin packed layout
```

或者：

```text
generic FP8
   ↓
CUTLASS / DeepGemm runtime layout
```

操作可能包括：

```python
transpose()
permute()
contiguous()
reshape()
resize_()
bit packing
tile reorder
```

这种场景是 reload 最困难的一类。

---

## 3.1.5 Derived weights

典型就是 MLA：

```text
kv_b_proj.weight
      │
      ├── W_UK_T
      └── W_UV
```

这两个 tensor 不会由 trainer 单独发送。

必须：

```text
source checkpoint weight changed
       ↓
重新 derive
       ↓
更新 runtime derived tensor
```

---

## 3.1.6 Derived scales

例如：

```text
weight_scale
activation_scale
      ↓
fused alpha
```

可能得到：

```text
_g1_alphas
```

RFC 也把 `_g1_alphas` 列为典型 derived runtime state。

这种东西非常容易出现：

```text
weight 更新了
scale 更新了
derived alpha 没更新
```

即使地址没变，forward 仍然错。

---

## 3.1.7 Runtime layout metadata

例如：

```text
expert map
expert mask
routing table
```

它们可能不是 checkpoint tensor，而是根据：

```text
TP
EP
DP
local rank
expert placement
```

推导出来。

RFC 特别提到 RoutedExperts 一类存在多个：

```text
persistent=False buffers
```

它们不会由 checkpoint 恢复。

---

## 3.1.8 Runtime constant / FP32 copy

例如原始 checkpoint：

```text
BF16 scale
```

backend 为 kernel 生成：

```python
runtime_scale = scale.float()
```

这会产生独立 storage。

以后只更新 BF16 source：

```text
runtime_scale 不会自动更新
```

因此需要显式：

```python
runtime_scale.copy_(scale.float())
```

---

## 3.1.9 Workspace / scratch buffer

Marlin 是典型：

```text
workspace
```

它不是模型权重。

但是 forward：

```python
ops.marlin_gemm(
    ...,
    workspace,
)
```

会把它作为 kernel 参数。 

CUDA Graph capture 后：

```text
Graph remembers workspace.data_ptr()
```

所以 reload PWAL 不能：

```python
workspace = torch.empty(...)
```

随意换地址。

---

## 3.1.10 Kernel-owned tensor

比 workspace 更隐蔽：

```text
layer
 └── quant_method
      └── moe_kernel
           ├── strides
           ├── permutation
           ├── constants
           └── scratch
```

这些可能完全不在：

```text
layer._parameters
layer._buffers
```

但 custom op 会使用它们。

所以一样可能被 CUDA Graph capture。

---

## 3.1.11 Kernel reconstruction

有些 PWAL：

```python
self.moe_kernel = make_kernel(...)
```

cold load 没问题。

reload 时重新创建 kernel object，内部 tensor 可能全部换地址：

```text
old kernel
  strides A

new kernel
  strides B
```

Graph 却还引用 A。

这类问题 generic Parameter copy-back 完全解决不了。

---

# 3.2 vLLM 当前针对这些问题有哪些解决手段

可以分成六种。

---

## 3.2.1 `register_parameter`

适合：

```text
模型权重
kernel/runtime weight
需要 weight_loader 属性
需要 Parameter subclass metadata
```

优势是 generic layerwise 能自动保存它：

```python
info.kernel_tensors = get_layer_params_buffers(layer)
```

然后 reload 后：

```python
old_param.data.copy_(new_runtime_param)
```

恢复原 storage。

所以：

```text
registered Parameter
        ↓
generic storage protection
```

是目前 vLLM 最成熟的一类。

---

## 3.2.2 `register_buffer(..., persistent=True)`

适合：

```text
非训练 Parameter
但属于 serialized model state
```

例如某些：

```text
running state
scale
model state
```

`persistent=True`：

```text
属于 Module buffer
+
进入 state_dict
```

它也进入：

```text
layer._buffers
```

因此 generic layerwise 能保存/恢复 storage。

---

## 3.2.3 `register_buffer(..., persistent=False)`

适合：

```text
module-owned runtime state
但不应该进入 checkpoint
```

例如：

```text
expert mask
routing map
derived lookup table
runtime derived constant
```

特点：

```text
named_buffers ✅
module.to() ✅
layerwise 可见 ✅

state_dict ❌
checkpoint-owned ❌
```

reload 时 vLLM 还专门区分：

```python
if name in non_persistent \
        and name not in loaded_tensor_names:
    continue
```

意思是：

> checkpoint 没有真正更新这个 non-persistent buffer，就不要用临时 materialized state 去覆盖旧 runtime value。

---

## 3.2.4 `replace_parameter(..., prefer_copy=True)`

适合：

```text
PWAL-derived graph-visible Parameter
```

典型：

```text
MLA W_UK_T
MLA W_UV
unquantized MoE shuffled weight
```

cold load 可以：

```text
allocate new optimal runtime tensor
```

但 reload：

```text
new temporary runtime tensor
       ↓
copy_
       ↓
old Parameter storage
```

所以：

```python
replace_parameter(
    layer,
    name,
    new_tensor,
    prefer_copy=True,
)
```

核心语义是：

```text
compatible:
    old.copy_(new)

而不是：
    old = new
```

这特别适合：

```text
first PWAL:
    prefer_copy=False

later RL PWAL:
    prefer_copy=True
```

---

## 3.2.5 Generic layerwise copy-back

这是最重要的一层。

假设：

```text
cold runtime weight:

P_old → storage A
```

Graph capture：

```text
Graph → A
```

reload PWAL 临时生成：

```text
P_new → storage B
```

generic layerwise：

```python
P_old.data.copy_(P_new)
```

得到：

```text
P_old → storage A
        contains new values
```

然后：

```python
_place_kernel_tensors(...)
```

把 `P_old` 重新注册回 layer。

所以：

```text
Graph → A
Python → A
value = new model
```

---

## 3.2.6 Marlin：reuse unmanaged storage

Marlin workspace 是 generic layerwise 看不到的 plain/backend tensor。

因此当前 Marlin：

```python
self.workspace = marlin_make_workspace_new(
    device,
    existing=getattr(self, "workspace", None),
)
```

明确选择复用旧 storage。

也就是：

```text
cold:
allocate A

reload:
existing=A
    ↓
reuse A
```

而不是：

```text
reload:
allocate B
```

当前测试甚至要求如果 existing workspace 不兼容：

```text
fail
```

而不是偷偷换地址。

---

## 3.2.7 MoE：reload 时不要重建 kernel

另外一种方式更直接：

```text
kernel 已存在
    ↓
说明这是 weight update
    ↓
只 update weight
不要重新 init kernel
```

例如 unquantized FusedMoE 当前：

```python
is_weight_update = self.moe_kernel is not None

replace_parameter(
    ...,
    prefer_copy=is_weight_update,
)

if not is_weight_update:
    self._init_moe_kernel(layer)
```

这样：

```text
cold load:
build kernel

reload:
reuse kernel
reuse kernel-owned runtime state
```

避免整个 backend object 重新分配。

---

## 3.2.8 当前 generic 方案的边界

generic layerwise 只能自动保护：

```text
registered Parameter
registered Buffer
```

它不能自动保护：

```text
plain self.foo Tensor

quant_method.workspace

moe_kernel.strides

list[Tensor]

dict[str, Tensor]

nested kernel-owned tensor

detached runtime copy
```

因此可以把 vLLM 当前地址保护体系概括为：

```text
                    graph-visible state
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
    Parameter/Buffer   derived param    unmanaged state
          │                │                 │
          ▼                ▼                 ▼
   layerwise copy_   prefer_copy=True   backend-specific
                                         reuse / no rebuild
```

---

# 4. VERL 当前方式存在的问题以及修复方案

这是整件事情最重要的工程结论。

---

## 4.1 VERL 当前 standard path 绕过了 upstream reload transaction

当前 VERL 普通非 FP8：

```text
BucketedWeightReceiver
        ↓
bucket 0
        ↓
model.load_weights(bucket0)
        ↓
bucket 1
        ↓
model.load_weights(bucket1)
        ↓
...
        ↓
process_weights_after_loading(model)
```

 

它没有：

```python
initialize_layerwise_reload(model)
```

也没有：

```python
finalize_layerwise_reload(model)
```

因此直接失去了 upstream 为 RL reload 建立的核心 contract。

---

## 4.2 问题一：`weight_loader` 看到的是 runtime layout，而它理解的是 checkpoint layout

这是最直接的问题。

例如 cold PWAL 后：

```text
checkpoint:
weight [N,K]

runtime:
weight [K,N]
```

但 `weight_loader` 创建时仍然认为：

```text
output_dim = 0
input_dim = 1
```

这些信息是 checkpoint layout 的语义。

于是第二次裸：

```python
model.load_weights(new_checkpoint_weights)
```

就可能：

```text
incoming [N,K]
       ↓
weight_loader
       ↓
试图按 checkpoint dim 切
       ↓
destination 却是 runtime [K,N]
       ↓
shape mismatch / wrong shard / silent corruption
```

新 RFC #54477 专门把这个问题作为 selective reload 必须增加 `restore_weights_before_loading()` 的原因。

---

## 4.3 问题二：PWAL 前没有保存 capture-time storage

upstream：

```text
initialize
    ↓
save old runtime Parameter/Buffer
```

VERL 当前：

```text
直接 model.load_weights
```

所以没有：

```text
info.kernel_tensors = old graph-visible tensors
```

这意味着如果 PWAL：

```python
replace_parameter(...)
```

或者：

```python
self.foo = new_tensor
```

导致地址变化，没有 generic copy-back 能把新 value 写回 old storage。

---

## 4.4 问题三：最后直接跑一次完整 PWAL

VERL 当前所有 bucket 收完：

```python
process_weights_after_loading(model, ...)
```



问题是：

> PWAL 的 contract 原本是 cold-load finalization，不是严格的 hot-reload refresh API。

PWAL 允许：

```text
transpose
replace_parameter
allocate workspace
resize_
repack
kernel reconstruction
source parameter release
derived state creation
non-idempotent transformation
```

所以“所有权重收到以后再 PWAL 一次”虽然比每 bucket PWAL 好很多，但仍然没有解决：

```text
storage identity
idempotence
checkpoint/runtime representation
kernel-owned state
```

RFC #54477 当前也正是基于这一点提出：

```text
不要长期把完整 PWAL 当作 RL refresh API
```

而应该拆：

```text
restore_weights_before_loading()
load_weights()
refresh_derived_state()
```

其中 refresh contract 必须：

```text
in-place
idempotent
incremental
no arbitrary reallocation
```



---

## 4.5 问题四：bucket 会切开一个 layer

假设：

```text
bucket 0:
layer.10.q_proj

bucket 1:
layer.10.k_proj
layer.10.v_proj
```

如果只是：

```python
model.load_weights(bucket)
```

本身可以执行 loader。

但如果 transformation 需要：

```text
q+k+v 全部完成以后处理
```

必须有 transaction-level tracking。

upstream layerwise 的：

```text
load_numel
load_numel_total
```

就是解决这个问题的。

因此不能简单修成：

```text
for each bucket:
    reload_weights(bucket)
```

这是错误的。

应该：

```text
initialize once

for all buckets:
    model.load_weights(bucket)

finalize once
```

---

## 4.6 问题五：直接处理 Buffer 的路径可能绕过 vLLM reload contract

VERL 当前明确：

```python
param_updates, buffer_updates, named_buffers = split_buffer_updates(...)

...
apply_buffer_updates(...)
```



这意味着一部分 incoming state 不经过：

```text
vLLM layerwise weight loader
```

如果只是普通 checkpoint buffer，可能没问题。

但必须逐一确认：

```text
是否 graph-visible
是否 persistent
是否 derived
是否应该由 checkpoint 更新
是否应该保持 old storage
```

尤其：

```text
buffer = incoming_tensor
```

和：

```text
buffer.copy_(incoming_tensor)
```

语义完全不同。

图模式下后者通常才安全。

---

## 4.7 问题六：FP8/QAT/ModelOpt 已经开始出现大量 framework-specific workaround

当前 VERL 已经有：

```text
QAT:
prepare_qat_for_load_weights
manual_process_weights_after_loading

ModelOpt:
prepare_modelopt_for_weight_reload
modelopt_process_weights_after_loading

FP8:
prepare_quanted_weights_for_loading
load_quanted_weights
process_quanted_weights_after_loading

ROCm:
restore_moe_expert_maps
```



这其实已经说明了：

> generic `model.load_weights() + final PWAL` 并不能覆盖所有 representation lifecycle。

继续在 VERL 里增加：

```text
if FP8
if MoE
if QAT
if MLA
if Marlin
if Ascend
```

长期会越来越难维护。

---

# 4.8 推荐修复方案：VERL 使用 vLLM native layerwise transaction

短期最稳妥的 upstream vLLM 修复方案，不是重新发明一套。

而是：

```text
VERL receiver starts
       ↓
initialize_layerwise_reload(model)

for every communication bucket:
       ↓
model.load_weights(bucket)

all buckets received
       ↓
finalize_layerwise_reload(model, model_config)
```

也就是：

```python
initialize_layerwise_reload(model)

try:
    receiver.receive_weights(
        on_bucket_received=lambda weights, is_last:
            model.load_weights(weights)
    )

    finalize_layerwise_reload(
        model,
        model_config,
    )

except:
    # weight update is not transactional
    # model should be considered invalid
    raise
```

---

## 4.9 MTP / draft model 也必须是完整 transaction

VERL 当前会更新：

```text
main model
MTP drafter
```

所以应该：

```text
initialize(main)
initialize(draft)

receive all buckets:
    main.load_weights(...)
    draft.load_weights(...)

finalize(main)
finalize(draft)
```

不能：

```text
main native reload
draft old manual path
```

否则两个模型的 runtime representation 生命周期不同。

---

## 4.10 不建议“每 bucket 调 `reload_weights()`”

这是一个容易犯的错误。

不能：

```python
for bucket in buckets:
    model_runner.reload_weights(bucket)
```

因为 `reload_weights()` 自己相当于：

```text
initialize
load
finalize
```

如果 layer 跨 bucket：

```text
bucket0:
q

finalize
→ layer incomplete

bucket1:
k,v

new initialize
...
```

生命周期就破坏了。

正确粒度始终是：

```text
one logical model update
=
one initialize
+
N receive/load calls
+
one finalize
```

---

# 4.11 IPC buffer 还要额外注意 ownership

VERL 当前 bucket receiver 很可能复用通信 buffer。

因此 callback 里的：

```text
weights
```

不能默认永久持有。

如果 native layerwise 的 wrapped loader：

```text
缓存 tensor/reference
直到 layer complete
```

而 IPC/ZMQ bucket buffer 会在下一个 bucket 被覆盖，那么必须解决 tensor lifetime。

这是把 VERL 接入 native layerwise 时非常关键的一点。

需要保证：

```text
incoming tensor 的 storage 生命周期
至少覆盖
weight_loader 真正消费它的时间
```

有两类方案。

### 方案 A：receiver 给每个 weight 独立 owned storage

例如：

```python
owned_weight = incoming.clone()
```

再交给 layerwise。

正确但额外显存/带宽较高。

### 方案 B：修改传输层，使 buffer 只有在 layerwise 确认消费后才能 reuse

更高效，但 protocol 更复杂。

upstream packed IPC 已经通过 importer/refcount/strong references 明确处理 staging-buffer lifetime。

---

# 4.12 中期方案：直接接入 vLLM WeightTransferEngine

比手工：

```python
initialize_layerwise_reload
model.load_weights
finalize_layerwise_reload
```

更进一步的方案是：

> VERL 使用 upstream `WeightTransferEngine` 的 start / receive / finish contract。

即：

```text
VERL trainer
    ↓
vLLM TrainerWeightTransferEngine
    ↓
NCCL / IPC
    ↓
vLLM WeightTransferEngine
    ↓
start_weight_update
receive_weights
finish_weight_update
```

这样 VERL 不再知道：

```text
PWAL
layerwise
meta restoration
kernel tensor copy-back
packed IPC lifetime
```

这些全部由 vLLM ownership。

这是长期最合理的分层：

```text
VERL owns:
    trainer representation
    when to update
    what weights to send

vLLM owns:
    how checkpoint weights become runtime weights
    TP/expert routing
    PWAL
    graph-safe storage identity
    backend-specific refresh
```

---

# 4.13 长期方案：跟随 selective reload，而不是让 VERL 自己实现

当前 upstream RFC #54477 正在把：

```text
initialize layerwise
→ meta restore
→ full PWAL
→ copy back
```

进一步演进成：

```text
START
restore_weights_before_loading()

LOAD
model.load_weights()
weight_loader()

FINISH
refresh_derived_state()
```



它解决的是当前 layerwise 的几个固有问题：

```text
额外 temporary VRAM
完整 PWAL 太重
kernel 重建
non-idempotent PWAL
resize_/repack
derived state refresh 不够显式
```

因此 VERL 最好不要自己复制 selective reload 逻辑。

正确边界仍然应该是：

```text
VERL:
    start update
    send weights
    finish update

vLLM:
    决定内部使用
        layerwise reload
    或
        selective reload
```

这样未来 vLLM 从 layerwise 切 selective 时，VERL 不需要重写所有 backend workaround。

---

# 5. 把整个体系压缩成一张图

```text
                   COLD LOAD
                      │
                      ▼
              create Parameters/Buffers
                      │
                      ▼
               model.load_weights
                      │
                      ▼
          process_weights_after_loading
                      │
        ┌─────────────┼───────────────┐
        │             │               │
    transpose       derive          repack
        │             │               │
        ├──── scale / runtime state ───┤
        │                             │
        ▼                             ▼
 registered tensors           unmanaged tensors
 Parameter / Buffer           workspace/kernel state
        │                             │
        ▼                             ▼
 CUDA Graph capture          CUDA Graph capture


                    RL UPDATE
                       │
                       ▼
         initialize_layerwise_reload
                       │
              ┌────────┴────────┐
              │                 │
       save old storage     restore checkpoint
       A/B/C/...             metadata on meta
              │                 │
              └────────┬────────┘
                       ▼
                model.load_weights
                       │
                       ▼
                original loaders
                       │
                       ▼
                     PWAL
                       │
            temporary runtime state
                       │
          ┌────────────┼────────────┐
          │            │            │
 registered param   derived      unmanaged
          │            │            │
          ▼            ▼            ▼
generic copy_    prefer_copy    backend reuse
old storage      in-place       / no rebuild
          │            │            │
          └────────────┼────────────┘
                       ▼
          Graph-visible addresses stable
                       │
                       ▼
          finalize_layerwise_reload
```

---

# 6. 最终结论

整个 vLLM RL 权重更新可以用三个层级理解。

第一层是 **checkpoint loading**：

```text
model.load_weights
```

解决：

```text
name mapping
TP shard
expert routing
fused mapping
weight_loader
```

第二层是 **runtime representation transformation**：

```text
process_weights_after_loading
```

解决：

```text
transpose
fuse
quantize
repack
derive
scale generation
kernel preparation
runtime metadata
```

第三层是 **hot-reload correctness**：

```text
initialize_layerwise_reload
        +
finalize_layerwise_reload
        +
backend-specific storage reuse
```

解决：

```text
checkpoint/runtime layout mismatch
CUDA Graph address stability
derived-state refresh
Parameter/Buffer restoration
kernel/workspace storage identity
```

而 VERL 当前 standard 路径的问题，恰恰是：

```text
它使用了第一层：
model.load_weights

也手工调用了第二层：
process_weights_after_loading

但绕开了第三层：
native reload lifecycle
```

所以从架构上最推荐的修复不是继续给 VERL 增加更多：

```text
prepare_fp8
prepare_moe
restore_xxx
manual_pwal
```

而是把 VERL 的一次完整 weight-sync transaction 改成：

```text
START
vLLM initialize/start_weight_update

RECEIVE
N buckets → vLLM model.load_weights/receive_weights

FINISH
vLLM finalize/finish_weight_update
```

并最终最好让 VERL 直接依赖 **vLLM `WeightTransferEngine` 的 transaction contract**，让 graph-safe reload、PWAL、Marlin、MoE、MLA、未来 selective reload 都继续由 vLLM 自己负责。