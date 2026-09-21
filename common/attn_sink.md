可以把 vLLM 当前看到的 **learnable attention sink** 和 **gated attention** 理解成解决同一个根问题的两条路线：

> 标准 Softmax Attention 强制所有真实 token 的 attention 权重之和等于 1。即使当前 query 根本“不需要从历史里读东西”，它也必须把 100% 的概率质量分给某些 token。  
> Sink 的方案是增加一个“虚拟的空目标”吸收概率；Gated Attention 的方案是 attention 正常算完后，再显式决定“这次 attention 输出到底要放多少出来”。

而且二者在数学上其实有很强的对应关系。

### 1. 先看普通 Attention 为什么会产生 sink

普通单头 attention：

\[
l_j=\frac{q^\top k_j}{\sqrt d}
\]

\[
p_j=\frac{e^{l_j}}{\sum_k e^{l_k}}
\]

\[
o=\sum_j p_j v_j
\]

关键约束是：

\[
\sum_j p_j=1
\]

假设当前 token 其实不需要历史信息，理想情况可能是：

\[
o\approx 0
\]

但是 Softmax 本身没有“不要任何 token”的选项。

即使所有 \(qk\) 都很差，它还是必须找一个相对最大的 token：

```text
scores = [-10, -11, -12, -9]

softmax(scores)
      ↓
仍然总和 = 1
```

训练过程中模型会学出一些“专门接垃圾 attention mass 的位置”，经常是 BOS、首 token 等，这就是经典 attention sink 现象。

这里要区分两个概念：

**StreamingLLM 所说的 attention sink** 是“某些真实 token 变成 sink”。

而 GPT-OSS、DeepSeek V4 这类模型里的 `sinks/attn_sink`，是进一步把这个行为**显式参数化成一个没有 Value 的虚拟 sink**。DeepSeek V4 技术描述也是直接在 Softmax 分母中加入 learnable sink logit。:chatgpt-content-reference{index="0"}

---

## 2. Learnable Attention Sink 到底做了什么

对每个 attention head 增加一个可学习的 scalar：

\[
s_h
\]

原来的：

\[
p_j=\frac{e^{l_j}}{Z},
\qquad
Z=\sum_k e^{l_k}
\]

变成：

\[
p'_j
=
\frac{e^{l_j}}
{Z+e^{s_h}}
\]

注意：**sink 没有对应的 Value。**

也就是说，没有：

\[
p_\text{sink}v_\text{sink}
\]

它只是吃掉 denominator 中的一部分概率。

因此真实 token 的总 attention mass 变成：

\[
\sum_j p'_j
=
\frac{Z}{Z+e^{s_h}}
\leq 1
\]

所以现在模型可以表达：

```text
有有用历史信息：
real token mass ≈ 1

没有有用历史信息：
real token mass ≈ 0
```

DeepSeek V4 当前 vLLM 里就是一个 per-head `attn_sink` 参数；为了适配 kernel 的 padded heads，vLLM 初始化为 `-inf`，真实 head 的部分再从 checkpoint 加载。:chatgpt-content-reference{index="1"}

GPT-OSS 也是类似设计：

```python
self.sinks = torch.nn.Parameter(
    torch.empty(config.num_attention_heads // tp_size,
                requires_grad=False)
)
...
Attention(..., sinks=self.sinks)
```

也就是 TP 后每个 rank 保存自己的 local heads sink。:chatgpt-content-reference{index="2"}

---

# 3. Sink 其实就是一种“隐式 Gate”

这是理解两者关系最重要的一步。

普通 attention 输出：

\[
o_{\text{normal}}
=
\sum_j
\frac{e^{l_j}}{Z}
v_j
\]

加入 sink：

\[
o_{\text{sink}}
=
\sum_j
\frac{e^{l_j}}{Z+e^s}
v_j
\]

提取一个公共系数：

\[
o_{\text{sink}}
=
\frac{Z}{Z+e^s}
\sum_j
\frac{e^{l_j}}{Z}
v_j
\]

因此：

\[
\boxed{
o_{\text{sink}}
=
g_{\text{sink}}
\cdot o_{\text{normal}}
}
\]

其中：

\[
g_{\text{sink}}
=
\frac{Z}{Z+e^s}
\]

令：

\[
LSE=\log Z
\]

就得到：

\[
\boxed{
g_{\text{sink}}
=
\sigma(LSE-s)
}
\]

也就是说：

> **Learnable sink 本质上已经偷偷实现了一个 sigmoid gate。**

只是这个 gate 并不是额外用一个 Linear 算出来，而是：

```text
query
 ↓
QK logits
 ↓
LSE = logsumexp(logits)
 ↓
LSE - learned_sink
 ↓
sigmoid
 ↓
head output gate
```

已有研究也从这个角度把 Sink Attention 描述为一种 implicit gating。:chatgpt-content-reference{index="3"}

FlashMLA 的接口直接体现了这种实现方式：当提供 `attn_sink` 时，它把最终 output 乘以

\[
\frac{\exp(LSE)}
{\exp(LSE)+\exp(attn\_sink)}
\]

而无需真的插入一个 fake KV token。:chatgpt-content-reference{index="4"}

所以硬件 kernel 实际上可以这样做：

```text
正常 FlashAttention
       │
       ├── output
       └── LSE
             │
             ▼
      sigmoid(LSE - sink)
             │
             ▼
output *= gate
```

这和真的给 K/V 多插一个虚拟 token 数学等价，但便宜得多。

---

# 4. Gated Attention：直接把 Gate 做出来

以当前 vLLM 中 Qwen3-Next / Qwen3.5 full attention 为例。

它不是修改 softmax denominator，而是：

\[
o=Softmax(QK^T)V
\]

另外从 hidden state 计算：

\[
g=W_gx
\]

然后：

\[
\boxed{
o'=o\odot\sigma(g)
}
\]

vLLM 当前 Qwen3.5 复用 `Qwen3NextAttention`；Q projection 实际扩成了 `Q + gate`，即每个 head 里：

```text
[q_head | gate_head]
```

Q 会经过 QNorm / RoPE，而 gate 不经过这些操作，attention 算完后：

```python
attn_output = self.attn(q, k, v)
attn_output *= torch.sigmoid(gate)
```

vLLM 甚至有专门的 fused QK-Norm + RoPE + gate-copy Triton kernel，其中明确说明 `gate_out` 是 raw pre-sigmoid gate。:chatgpt-content-reference{index="5"}

这里特别注意：

**这和 Qwen3.5 里的 Gated DeltaNet 不是一回事。**

Qwen3.5 是 hybrid：

```text
linear_attention layers → Gated DeltaNet
full_attention layers   → Softmax Attention + output gate
```

我们这里比较的是后者的 **full-attention output gate**。

---

## 5. 两种方案最核心的差异

| 维度 | Learnable Sink | Gated Attention |
|---|---|---|
| 作用位置 | Softmax denominator / attention kernel | SDPA 输出之后 |
| 典型公式 | \(o'=\sigma(LSE-s_h)o\) | \(o'=\sigma(W_gx)\odot o\) |
| Gate 来源 | attention logits 的 LSE + learned threshold | hidden/query 的独立 projection |
| 参数规模 | 通常每 layer 每 head 一个 scalar | 通常每 token/head/head_dim 对应 projection |
| 动态性 | query-dependent，但只通过 LSE | 强 query/content-dependent |
| 粒度 | 通常一个 head 一个 scalar | 可以细到 head_dim 每个 channel |
| 是否改变真实 token 相对 attention | **不改变** | **不改变** |
| 是否改变真实 token 权重总和 | 是，变成 ≤1 | Softmax 内仍为 1，输出再缩放 |
| 额外 GEMM | 无 | 通常有 gate projection |
| Attention kernel | 必须支持 sink | 普通 SDPA kernel 即可 |
| 表达能力 | 较弱、结构化 | 更强 |
| 参数/计算成本 | 极低 | 更高 |

这张表里有一个很重要但容易忽略的共同点：

两者都**不会改变真实 token 之间的相对 attention 比例**。

Sink 中：

\[
\frac{p'_i}{p'_j}
=
\frac{e^{l_i}}{e^{l_j}}
\]

和没有 sink 完全一样。

Gated Attention 更明显：

```text
Softmax(QK) → 已经决定 token 间比例
             ↓
       整个输出再乘 gate
```

所以这两种设计主要解决的不是：

> “应该关注历史中的哪个 token？”

而是：

> **“这个 attention head / channel 这一次到底需不需要输出信息？”**

---

# 6. 为什么 Gated Attention 比 sink 更强

Sink 的 gate：

\[
g_h=\sigma(LSE_h-s_h)
\]

它的信息来源非常受限。

模型只能根据：

```text
这个 head 当前所有 QK logits 总体有多强？
```

来决定是否开启 head。

比如两个 query：

```text
query A:
logits = [8, 1, 0]

query B:
logits = [8, 7, 7]
```

它们的 LSE 不同，所以 gate 会不同。

但 sink 看不到 attention output 的具体语义，也没有一个自由的 \(W_gx\)。

而 explicit gated attention：

\[
g=\sigma(W_gx)
\]

理论上可以从当前 token 的完整 hidden representation 判断：

```text
这个 token 是标点？
这个 head 当前有没有必要工作？
某些 feature channel 应该打开？
某些 channel 应该关闭？
```

尤其 Qwen3.5 这种 gate shape 与 Q head 输出相同，相当于：

\[
g_{t,h,d}
\]

而 sink 通常只有：

\[
g_{t,h}
\]

所以：

```text
sink
一个 head 一起开/关

gated attention
一个 head 内不同 feature dimension 也可以分别开/关
```

这也是 gated attention 表达能力更强的重要原因。相关 gated-attention 工作发现，post-SDPA sigmoid gate 不仅可以缓解 attention sink，还引入额外非线性和 query-dependent sparsity。:chatgpt-content-reference{index="6"}

---

# 7. 但 Sink 有一个很漂亮的归纳偏置

虽然 Gated Attention 更强，sink 有一个非常漂亮的解释：

\[
s_h
\]

相当于这个 head 的一个 **attention confidence threshold**。

因为：

\[
g_h=\sigma(LSE_h-s_h)
\]

当：

\[
LSE_h \gg s_h
\]

表示当前 query 对某些 KV 有很强匹配：

\[
g_h\approx1
\]

attention 正常工作。

反之：

\[
LSE_h \ll s_h
\]

表示当前 attention 没找到值得看的内容：

\[
g_h\approx0
\]

输出自动关闭。

因此可以把 sink 理解成：

```text
            QK Attention
                 │
                 ▼
       “我找到有用东西了吗？”
                 │
         ┌───────┴────────┐
         │                │
        yes               no
         │                │
    输出 attention       输出≈0
```

不需要额外训练一个复杂 gate projection。

---

# 8. 为什么 DeepSeek V4 选择 sink 很合理

DeepSeek V4 的 attention 已经很复杂：

```text
Sliding-window KV
        +
Compressed KV
        +
Indexer / Top-K
        ↓
Sparse core attention
```

对于 sparse attention，一个 query 得到的 candidate set 有时候可能整体质量不高。

如果仍然：

\[
Softmax(topK)=1
\]

那么即使 top-k 都不太相关，模型还是必须从这些 candidate 中读满 100% 的 value。

增加 sink 以后：

\[
\sum_j p_j \le 1
\]

模型就可以表达：

> “Indexer 给我的这些候选都不值得读。”

这对 sparse/compressed attention 尤其自然。

DeepSeek V4 当前 vLLM 的 `attn_sink` 正是 per-head parameter，FlashMLA sparse kernel直接消费该参数。:chatgpt-content-reference{index="7"}

---

# 9. 从 vLLM 实现角度，两者区别更明显

vLLM 的 generic `Attention` 在初始化时直接检查：

```python
self.has_sink = extra_impl_args.get("sinks") is not None
```

然后把 `has_sink` 交给 attention backend selector。

所以 **sink 是 attention backend capability**：

```text
Model
  ↓
Attention(sinks=...)
  ↓
has_sink=True
  ↓
backend selection
  ↓
FlashAttention / FlashInfer / TRTLLM...
       必须支持 sink
```

vLLM 的 backend interface 也专门存在 `supports_sink()`，不支持的 backend 会被排除。:chatgpt-content-reference{index="8"}

而 Gated Attention：

```text
ordinary Attention backend
       ↓
   attn_output
       ↓
 sigmoid(gate)
       ↓
 elementwise mul
       ↓
     o_proj
```

对于 attention backend 本身，它甚至可以完全不知道 gate 的存在。

因此从工程性看：

```text
Sink
优点：几乎零参数、零额外 GEMM
缺点：侵入 attention kernel/backend

Gated Attention
优点：attention kernel 不需要特殊语义
缺点：多一个 projection / 更大的 Q projection
```

---

# 10. 这也解释了你之前看到的 sink 权重更新特殊问题

这两类参数在 vLLM weight loading / online update 中的性质其实非常不同。

DeepSeek V4 当前：

```python
self.attn_sink = nn.Parameter(...)
```

而且它不是普通 Linear 的 `.weight`。

加载时需要专门处理 local head slice：

```text
global attn_sink
      ↓ TP narrow
local head sink
      ↓
self.attn_sink
```

DeepSeek V4 代码还需要保留 padded heads，未使用部分为 `-inf`。:chatgpt-content-reference{index="9"}

GPT-OSS 同样对 `"sinks"` 有专门的 TP narrow + copy 逻辑。:chatgpt-content-reference{index="10"}

这意味着：

```text
q_proj.weight
k_proj.weight
v_proj.weight
...
```

通常进入统一的 `LinearBase.weight_loader` 生命周期。

但：

```text
attn_sink
```

很容易成为特殊 standalone parameter：

```text
checkpoint
   ↓
model-specific load_weights special case
   ↓
param.data.copy_()
   ↓
attention kernel directly reads this tensor
```

这就是为什么它在 **layerwise reload / RL live weight update / cudagraph 地址保持** 场景中特别容易漏掉。

而 Qwen3.5 gated attention 的 gate weight 本身通常属于：

```text
q_proj / qkv_proj
```

的一部分；Kimi K3 则可以有 `g_proj`，但依然是标准 Linear。当前 vLLM 的 Kimi K3 实现也明确有 optional sigmoid output gate。:chatgpt-content-reference{index="11"}

因此 gated attention 参数通常天然走：

```text
Linear weight
→ normal weight_loader
→ quantization/TP loader
→ reload
```

相比 standalone `sink`，weight-update 生命周期更统一。

---

## 最后用一个公式把两者统一起来

可以把两者都写成：

\[
\boxed{
O=G(Q,X,K)\odot
\left[
Softmax(QK^T)V
\right]
}
\]

区别只是 \(G\) 怎么来。

Sink Attention：

\[
\boxed{
G_\text{sink}
=
\sigma(
\operatorname{LSE}(QK^T)-s
)
}
\]

它是 **attention 自身置信度驱动的隐式 scalar gate**。

Gated Attention：

\[
\boxed{
G_\text{gate}
=
\sigma(W_gX)
}
\]

它是 **hidden state 驱动的显式、通常更细粒度 gate**。

所以从架构思想上，我更倾向于这样记：

```text
Vanilla Attention
    强制 read something
           │
     ┌─────┴─────┐
     │           │
 Sink Attention  Gated Attention
     │           │
 增加 no-op      增加 output gate
 destination
     │           │
 implicit gate   explicit gate
     │           │
 σ(LSE-s)        σ(Wx)
```

**Sink 并不是在“消灭 attention sink”，而是把原来落在 BOS/首 token 上的非语义 sink，变成一个显式、无 Value 的虚拟 sink；Gated Attention 则进一步连这个虚拟位置都不需要，直接控制 attention 输出强度。** 这是两种方案最本质的差异。 :chatgpt-content-reference{index="12"}