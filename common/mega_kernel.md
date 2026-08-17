# Mega Kernel

本文将基于[Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)对mega kernel技术进行介绍。

这篇文章的核心并不是“又做了一个更快的 CUDA kernel”，而是在挑战当前 LLM 推理系统里一个很基础的执行范式：

> **为什么一定要把 Transformer 的一次 forward pass 拆成几十到上百个 GPU kernel？如果整个模型本身足够小，直接把整个 forward pass 变成一个巨大的 kernel，会发生什么？**

Hazy Research 给出的答案是：对于 **Llama-3.2-1B、batch size = 1、追求最低单请求延迟** 这种非常特定的 workload，传统的多-kernel 执行方式产生了大量“空泡（bubbles）”，而一个覆盖整个模型 forward pass 的 **megakernel** 可以显著减少这些空泡。他们报告 H100 上达到约 **78% 的显存带宽利用率**，相比 vLLM 快接近 2.5×、相比 SGLang 快 1.5× 以上，并声称实现了 1B+、BF16 模型单次 forward **低于 1 ms**。

---

## 一、先理解文章究竟在优化什么

这篇文章优化的场景非常窄：

- Llama-3.2-1B
- 单序列 decoding
- batch size = 1
- BF16
- H100 / B200
- 不使用 speculative decoding
- 目标是 **极致 token latency，而不是最大吞吐量**

这一点非常关键。

如果你熟悉 vLLM，通常会想到：

> continuous batching、PagedAttention、提高 GPU utilization、提高总 tokens/s。

但这篇文章在研究的是几乎相反的问题：

> **只有一个用户、一个序列，我怎样让这个 token 尽可能快出来？**

作者指出，在这个 workload 下，1B 模型太小了，矩阵运算本身很快，真正拖慢系统的反而变成了**每个算子之间的调度和同步开销**。

所以这里的优化目标不是：

\[
\text{maximize FLOPs}
\]

而更接近：

\[
\text{minimize idle time between memory transfers}
\]

---

# 二、最重要的洞察：Llama-1B decoding 是 memory-bound

对 batch=1 的自回归 decoding 来说，每生成一个 token，你基本都需要：

1. 读取模型权重；
2. 做 matrix-vector multiplication；
3. 得到下一个 hidden state。

因为 batch 只有 1，所以 GEMM 退化得更接近 GEMV，算术强度很低。

直觉上可以理解成：

假设模型权重约 1.24B 参数，BF16 每参数 2 bytes：

\[
1.24B \times 2 \approx 2.48GB
\]

H100 SXM 的峰值 HBM 带宽约：

\[
3.35TB/s
\]

NVIDIA 官方规格同样给出了 H100 SXM 约 3.35 TB/s 的 GPU memory bandwidth。

因此最理想情况下：

\[
\frac{3.35TB/s}{2.48GB}
\approx 1350
\]

也就是说理论上：

> H100 每秒最多大概能把整个模型权重“扫过”1350次。

所以理想 token latency 大约：

\[
1 / 1350 \approx 0.74ms
\]

这就是整篇文章的性能天花板。

**计算能力反而不是主要问题。**

这也是为什么作者一直讨论 memory bandwidth，而不是 TFLOPS。

---

# 三、传统 LLM runtime 的问题在哪里？

正常的 Transformer inference 大概长这样：

```text
RMSNorm
↓
QKV projection
↓
RoPE
↓
Attention
↓
O projection
↓
Residual
↓
RMSNorm
↓
Up/Gate projection
↓
SiLU
↓
Down projection
↓
Residual
```

一个 Llama block 可能被拆成若干 kernel。

16 层以后，就可能出现约 100 个 kernel launch。

作者指出三个主要问题。

---

## 1. Kernel launch overhead

每启动一个 CUDA kernel，都有固定开销。

他们测到 H100 上：

普通 CUDA stream：

\[
\approx 2.1\mu s
\]

CUDA Graph：

\[
\approx 1.3\mu s
\] 


1.3 μs 看起来很小。

但假设有：

\[
100\ kernels
\]

那么就是：

\[
100 \times 1.3\mu s = 130\mu s
\]

对一个目标只有：

\[
700\sim1000\mu s
\]

的 forward 来说，这已经是 **10%–20% 级别的 overhead**。

这就是所谓：

> 当计算本身越来越快时，framework overhead 开始变成主体。

这几年 GPU kernel 优化其实不断出现这种现象。

---

# 四、第二种 bubble 更有意思：tail effect

假设某个 kernel 有：

\[
512\ thread\ blocks
\]

GPU 有：

\[
148\ SMs
\]

理想调度大概：

```text
wave 1: 148 blocks
wave 2: 148
wave 3: 148
wave 4: 68
```

最后一波只剩：

\[
512 - 148\times3 = 68
\]

个 block。

于是：

\[
148-68 = 80
\]

个 SM 闲着。

问题在于：

> 下一个 kernel 不能直接使用这些空出来的 SM。

它必须等**上一个 kernel 整体结束**。

因此最后一小波 block 就产生了：

```text
████████████████
████████████████
████████████████
██████░░░░░░░░░░
                  ↑
               bubble
```

文章称这类现象为 **memory pipeline bubbles**。

---

# 五、第三个问题：kernel boundary 阻止 memory pipelining

这其实是整篇文章最核心的地方。

理想情况应该是：

```text
operation A compute
          ↓
          operation B weight load
                    ↓
                    operation B compute
```

也就是：

```text
A compute
██████████

      B load
      █████████

             B compute
             ███████
```

load 和前面的 compute/store 可以 overlap。

但如果 A 和 B 是两个 kernel：

```text
Kernel A
██████████

           launch / sync
           ░░░░░

                 B load
                 █████████

                          B compute
                          ███████
```

中间就有一块什么都没干的区域。

作者把它称为：

> bubble

于是文章标题：

> **Look Ma, No Bubbles!**

就是：

> 看，GPU pipeline 里几乎没有空泡了。

---

# 六、Megakernel 到底是什么？

普通执行：

```text
Kernel 1
↓
Kernel 2
↓
Kernel 3
↓
...
↓
Kernel 100
```

他们的方法：

```text
┌────────────────────────────┐
│        MEGAKERNEL          │
│                            │
│ RMSNorm                    │
│ QKV                        │
│ RoPE                       │
│ Attention                  │
│ O projection               │
│ MLP                        │
│ ...                        │
│ layer 16                   │
│ LM head                    │
│                            │
└────────────────────────────┘
```

**整个 Llama forward 是一次 kernel launch。** 


但这里有一个非常重要的误解要避免：

这并不是简单写一个：

```cpp
__global__ void llama() {
    for layer in layers:
        ...
}
```

真正困难的地方恰恰在：

> CUDA 原本帮你的 scheduler、dependency management 和 synchronization，现在都要自己做。

所以本质上，他们在 GPU kernel 内部又实现了一个：

> **小型 runtime / scheduler。**

这是整篇文章最漂亮的地方。

---

# 七、Megakernel 内部其实是一个“GPU interpreter”

他们定义了一组 instruction，例如：

```text
RMSNorm + QKV + RoPE

Attention

Attention reduction

O projection + residual

RMSNorm + up/gate + SiLU

Down projection + residual

Final RMSNorm + LM head
```

然后每个 SM 都会得到一个 instruction sequence。

概念上类似：

```text
SM0:
load W1
execute instruction A
load W2
execute instruction B
...

SM1:
load W1
execute instruction A
...
```

调度计划是在 Python 侧预先生成的，而且可以在很多 forward 中重复使用。

这就很像：

```text
CPU
  ↓ compile/schedule
GPU
  ↓
persistent interpreter
  ↓
execute whole neural network
```

因此我更愿意把这项工作理解成：

> **把模型 runtime 从 CPU/CUDA driver 下沉到了 GPU 内部。**

而不仅仅是“kernel fusion”。

---

# 八、Shared Memory Paging 是文章里最聪明的设计之一

为了持续加载权重：

```text
HBM
 ↓
Shared Memory
 ↓
Registers / Tensor cores / CUDA cores
```

问题来了：

shared memory 很小。

H100 一个 SM 的 shared memory 资源大约只有几百 KB 量级。

如果 instruction A 正占着 shared memory：

```text
[A A A A A A]
```

instruction B 即使想提前 load 权重：

```text
[B ? ? ?]
```

也没地方放。

于是又会产生 bubble。

作者的解决方案非常有意思：

> **给 shared memory 做分页。**

他们把 H100 上前约 213 KB shared memory 分成：

\[
13 \times 16KB
\]

的 page。

所以类似：

```text
Shared Memory

┌───────┐
│ page0 │
├───────┤
│ page1 │
├───────┤
│ page2 │
├───────┤
│ ...   │
├───────┤
│page12 │
└───────┘
```

每个 instruction：

```text
request pages
↓
load weight
↓
compute
↓
release pages
```

一旦 A 释放：

```text
page 3
```

B 马上就能开始：

```text
load next weights
```

于是 pipeline 变成：

```text
A compute
██████████

      B weight load
      ███████████

             B compute
             █████████
```

这实际上是在做一种：

> **software-managed GPU memory pipeline。**

---

# 九、Synchronization：他们把 kernel boundary 替换成 counter

传统 CUDA：

```text
Kernel A
───────── barrier ─────────
Kernel B
```

kernel boundary 本身就是全局 synchronization。

因此 dependency 很简单。

例如：

```text
QKV
↓
Attention
```

Attention 一启动，你知道 QKV 肯定完成。

但 megakernel 没有 kernel boundary。

于是必须自己管理：

```text
QKV completed?
        ↓
    counter++
```

下游：

```text
while(counter < expected)
    wait
```

作者就是在 global memory 维护 counter。

可以理解为：

```text
producer:
result = compute()
store(result)
counter++

consumer:
wait(counter)
load(result)
compute()
```

---

# 十、这反而允许比 CUDA kernel 更细的 dependency

这里是文章另一个很重要的洞察。

假设 MLP intermediate：

```text
[chunk0][chunk1][chunk2][chunk3]
```

传统 kernel：

```text
Up projection
████████████████████

                    Down projection
                    ████████████████
```

因为下一个 kernel 要等前一个全部完成。

他们的方法：

```text
chunk0 ready
↓
down projection chunk0

chunk1 ready
↓
down projection chunk1
```

形成：

```text
up:
████ ████ ████ ████
     ↓    ↓    ↓

down:
     ████ ████ ████ ████
```

也就是 producer-consumer pipeline。

文章说，他们给四个 chunk 各放一个 counter，让 down projection 不必等完整 hidden state。

这实际上比 kernel-level synchronization 更细粒度。

这也是 megakernel 真正的理论优势：

> 不只是少 launch 几次，而是**重新定义 dependency granularity**。

---

# 十一、结果到底有多快？

作者报告：

### H100

整个 1B+ BF16 forward：

\[
<1ms
\]

### B200

甚至：

\[
<680\mu s
\] 


相对于 baseline：

| GPU | vs vLLM | vs SGLang |
|---|---:|---:|
| H100 | ~2.5× | >1.5× |
| B200 | >3.5× | >1.5× |

这些是**作者自己的 benchmark 结果**，而不是独立第三方 benchmark，因此最好把它理解成这套实现展示出的上限和潜力，而不是“所有生产环境下都会快 2.5×”。

代码也确实开源了，仓库里包含 H100/B200 的 low-latency Llama demo 和 benchmark 脚本。

---

# 十二、一个很反直觉的数据：现在最大的瓶颈已经不是权重加载

B200 的一个 forward 约：

\[
600\mu s
\]

作者给的 breakdown 是：

| 部分 | 时间 |
|---|---:|
| activation store / sync / load | ~250 μs |
| RMSNorm + matrix-vector compute | ~200 μs |
| 等待 weight load | **~30 μs** |
| warp synchronization | ~40 μs |
| setup / bookkeeping | ~80 μs |

最值得注意的是：

> **weight load stall 只剩 30 μs。**

说明他们的 pipeline 真的基本把：

```text
HBM weight latency
```

藏起来了。

作者自己也写了一句很关键的话：

> “pipelining works!” 


---

# 十三、真正剩下的瓶颈是什么？

现在反而是：

### activation dependency latency

每个 instruction：

```text
store activation
↓
mark ready
↓
next op checks ready
↓
load activation
```

都有延迟。

所以即使 activation 数据量很小：

```text
bandwidth cost ≈ 小
```

但是：

```text
latency cost ≠ 小
```

这揭示了一个非常重要的系统规律：

> **Bandwidth-bound workload 优化到极致以后，很容易转化成 latency-bound workload。**

最初：

```text
瓶颈 = weight bandwidth
```

优化后：

```text
瓶颈 = dependency + synchronization + activation latency
```

这基本就是性能工程的“打地鼠”：

```text
优化 A
↓
B 成为 bottleneck
↓
优化 B
↓
C 成为 bottleneck
```

---

# 十四、为什么 CUDA Graph 没有解决问题？

很多人第一反应会是：

> CUDA Graph 不就是为了减少 kernel launch overhead 吗？

是，但它只能解决一部分。

作者测到：

```text
normal launch ≈ 2.1 μs
CUDA graph   ≈ 1.3 μs
``` 


更关键的问题在于 CUDA Graph 仍然保留：

```text
Kernel A
──────────
Kernel B
──────────
Kernel C
```

即：

> **execution abstraction 还是 kernel。**

而 megakernel：

```text
A
 ↘
  B
 ↘
  C
```

dependency 可以在内部任意细化。

因此两者差异不是简单的：

```text
1.3 μs → 0 μs
```

而是：

> **调度粒度发生变化。**

---

# 十五、为什么 NVIDIA 的 PDL 也不够？

NVIDIA 有 Programmatic Dependent Launch。

目的就是：

> 让后一个 kernel 在前一个 kernel 尚未完全结束时开始准备。

官方 CUDA 文档确实提供了相关的 kernel execution / synchronization 机制。

但 Hazy Research 认为其同步粒度仍然过粗。

文章给的例子是：

```text
Q
K
V
```

传统依赖可能相当于：

```text
Q complete
K complete
V complete
       ↓
Attention starts
```

megakernel 可以做到：

```text
head0 QKV ready
↓
head0 attention

head1 QKV ready
↓
head1 attention
``` 


也就是：

> **tensor / chunk / head level dependency**

而不是：

> kernel-level dependency。

---

# 十六、这篇文章真正挑战的是“GPU 编程抽象”

如果只看 headline：

> 1.5× faster Llama inference

其实会低估这篇文章。

它更深层的观点是：

今天主流 GPU programming stack 是：

```text
PyTorch
   ↓
torch.compile
   ↓
Triton / CUDA kernels
   ↓
CUDA runtime
   ↓
GPU scheduler
```

模型会变成：

```text
kernel
kernel
kernel
kernel
kernel
...
```

Hazy Research 在尝试：

```text
Python
↓
static schedule
↓
GPU-resident interpreter
↓
one persistent/mega kernel
```

也就是说：

> **把整个 neural network 本身当作一个 GPU program，而不是一串 GPU programs。**

这和传统 compiler 的区别很像：

传统：

```text
function call
function call
function call
```

megakernel：

```text
compile whole program
```

---

# 十七、它和 FlashAttention 的思想其实有亲缘关系

FlashAttention 的核心不是“attention 数学变了”。

而是：

> 重新设计 data movement。

从：

```text
HBM → compute → HBM
HBM → compute → HBM
```

变成：

```text
HBM
 ↓
SRAM
 ↓
尽可能多 compute
 ↓
HBM
```

Megakernel 更进一步：

FlashAttention：

```text
fuse operations inside attention
```

Megakernel：

```text
fuse operations across the whole model
```

所以可以把近年来 LLM systems 的演化粗略看成：

```text
operator optimization
        ↓
kernel fusion
        ↓
FlashAttention
        ↓
torch.compile
        ↓
persistent kernels
        ↓
megakernels
```

优化边界越来越大：

```text
operator
→ subgraph
→ layer
→ model
```

---

# 十八、但它有几个非常大的局限

这部分是读文章时最应该保持警惕的。

## ① 对 batch=1 最有优势

batch 增大以后：

```text
matrix-vector
```

逐渐变成：

```text
matrix-matrix
```

算术强度上升。

GPU Tensor Core utilization 上升。

此时：

```text
compute time >> kernel launch overhead
```

megakernel 的收益就可能快速下降。

文章自己也明确限定研究的是 **low-latency, batch-size-one inference**。

所以它不能直接证明：

> megakernel 会替代 vLLM。

两者目标其实不同。

vLLM：

```text
maximize throughput
```

megakernel：

```text
minimize single-request latency
```

---

## ② 对模型大小高度敏感

1B 模型：

```text
kernel runtime 很短
```

因此：

```text
launch overhead 占比很大
```

如果换成：

```text
70B
```

单层 GEMM 本身就很重。

这时：

```text
2 μs kernel launch
```

可能变得没那么重要。

所以标题里的：

> Llama-1B

不是偶然。

它几乎就是这套技术能展示最大收益的 sweet spot。

---

## ③ 工程复杂度明显更高

正常模型开发：

```python
x = rmsnorm(x)
qkv = linear(x)
attn = attention(qkv)
```

Megakernel：

```text
自己安排 SM
自己安排 shared memory
自己管理 dependency
自己管理 barrier
自己管理 instruction
自己 pipeline weight load
```

实际上已经开始变成：

> 手写一个 GPU execution runtime。

维护成本会显著提高。

---

## ④ portability 是问题

文章专门针对：

```text
H100
B200
```

甚至提到 Hopper 和 Blackwell 上最佳 compute 策略不同：

- Hopper：CUDA cores 更合适
- Blackwell：Tensor Cores 稍有优势 


意味着：

> hardware-specific optimization 很重。

这和 Triton / PyTorch 的理念正好形成张力：

```text
megakernel
↑ performance
↓ portability

framework
↓ performance ceiling
↑ portability
```

---

# 十九、我认为这篇文章最重要的三层意义

### 第一层：工程意义

证明：

> 对低 batch、小模型、超低延迟 inference，kernel launch / synchronization 已经是一级瓶颈。

这是非常实用的结论。

---

### 第二层：compiler 意义

未来 compiler 的 optimization scope 可能需要从：

```text
fuse nearby ops
```

变成：

```text
schedule whole network
```

也就是说 compiler 不只是生成 kernel：

```text
kernel generator
```

而可能变成：

```text
GPU program scheduler
```

---

### 第三层：hardware/software co-design 意义

文章事实上暴露出了 CUDA abstraction 的一个问题：

当前执行模型主要围绕：

```text
grid → kernel → grid → kernel
```

构建。

但 AI workload 越来越需要：

```text
fine-grained producer-consumer
```

例如：

```text
tensor tile ready
↓
下一阶段立刻消费
```

megakernel 本质是在软件层绕过这个限制。

从这个角度看，它很可能对未来 NVIDIA / AMD GPU 的：

- hardware scheduler
- async execution
- shared memory management
- cross-SM synchronization
- persistent kernels

设计产生启发。

---

# 二十、用一句话总结整篇文章

我会这样概括：

> **当 LLM 小到“算子本身已经不够慢”时，真正的瓶颈变成了算子之间的边界；Megakernel 的做法就是把这些边界全部删掉，在 GPU 内部重新实现一个针对整个 Transformer 的细粒度 scheduler。**

所以文章真正的关键词并不是：

**kernel fusion**

而是：

**execution scheduling + memory pipelining。**

它最有价值的地方，也不是单纯的“1.5×–2.5× faster”，而是提出了一个值得关注的方向：

\[
\boxed{\text{Model as a GPU program}}
\]

而不是：

\[
\boxed{\text{Model as a sequence of GPU kernels}}
\]

如果未来越来越多推理 workload 追求 **极低 latency、小模型、batch=1**——例如 voice agent、实时机器人、speculative decoding 中的小 draft model——这种设计思路会尤其有意义。作者自己也明确表示，他们认为 megakernel 的精细 GPU execution control 可以推广到更广泛的 AI workloads，但这在文章中仍然属于未来方向，而不是已经证明的结论。