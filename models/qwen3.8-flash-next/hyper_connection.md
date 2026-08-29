# Qwen3.8-Flash-Next HyperConnection 原理与 Forward 流程

本文基于 Qwen3.8-Flash-Next 的公开配置，以及 vLLM PR [#53899](https://github.com/vllm-project/vllm/pull/53899) 中的 NVIDIA 实现，分析模型中的 HyperConnection（HC，也称 Gated Residual）机制。

本文重点回答：

- 为什么模型层间状态是 `4 × 2560 = 10240`，Attention 和 MLP 输入却仍是 2560；
- 四流状态如何混合成标准 Transformer block 输入；
- Attention/MLP 的单流输出如何重新写回四条 residual stream；
- mixing gate 和 injection gate 分别控制什么；
- NVIDIA 实现为什么延迟执行 combine；
- PLE、Pipeline Parallel 和 MTP 如何改变 HyperConnection 的执行边界。

除特别说明外，文中的 shape 均为逻辑全局 shape，不展开 Tensor Parallel 内部的参数分片。

---

## 1. 核心结论

Qwen3.8-Flash-Next 在层与层之间保存 4 条并行 residual stream：

```text
层间状态：       [T, 4, 2560] = [T, 10240]
                         ↓ HyperConnection mix
Attention 输入： [T, 2560]
Attention 输出： [T, 2560]
                         ↓ HyperConnection combine + mix
MLP 输入：       [T, 2560]
MLP 输出：       [T, 2560]
                         ↓ HyperConnection combine
下一层状态：     [T, 4, 2560] = [T, 10240]
```

因此：

> `10240` 是 HyperConnection residual state 的宽度；`2560` 是 Attention、GDN、QSA、MoE、MLP 和 LM Head 的 block hidden size。

HyperConnection 的作用可以概括为两个方向：

```text
读取：4 条 stream → gated mix → 1 条 2560 维 block input

写回：1 条 2560 维 block output → gated injection → 4 条 stream
```

这样可以扩大层间状态容量，但不需要把 Attention 和 MoE 的主要计算宽度扩大 4 倍。

---

## 2. 真实模型配置与符号

Qwen3.8-Flash-Next 的相关配置：

```text
hidden_size = H = 2560
hc_count    = C = 4
hc_lowrank  = R = 320
```

因此：

```text
hyper_hidden_size = C × H
                  = 4 × 2560
                  = 10240
```

本文使用以下符号：

| 符号 | Shape | 含义 |
|---|---:|---|
| `T` | 标量 | 本次 forward 中展平后的 token 数 |
| `X` | `[T, C, H]` | 四流 HyperConnection 状态 |
| `X_flat` | `[T, C×H]` | 四流状态的物理存储形式 |
| `B` | `[T, H]` | Attention 或 MLP 的 block output |
| `I` | `[T, C]` | block 输出写回四流的 injection logits |
| `G` | `[T, C, H]` | 四流读取阶段的 mixing gate logits |
| `R` | `[T, 320]` | low-rank mixing 中间状态 |

源码中的 hidden state 通常以 `[T, 10240]` 存储，需要时再视为 `[T, 4, 2560]`。

---

## 3. 为什么需要四条 Residual Stream

标准 Transformer 每层只有一条 residual state：

```text
X
├─ Norm → Attention → Add
└─────────────────────┘
```

其状态宽度为：

```text
[T, H]
```

HyperConnection 把 residual state 扩展成多条并行 stream：

```text
X0 [T,H] ─┐
X1 [T,H] ─┤
X2 [T,H] ─┤→ 学习如何读取和写回
X3 [T,H] ─┘
```

每个 block 不直接消费全部 `[T,4H]`，而是动态混合出一个 `[T,H]` 的视图。block 计算完成后，同一个 `[T,H]` 输出又以不同强度写入四条 stream。

可以把它理解为：

- 四条 stream 是模型在层间维护的四份长期工作状态；
- Attention/MLP 每次只读取一份动态组合后的工作视图；
- block 输出可以选择性更新每条状态，而不是强制四条状态完全相同。

这与简单把 hidden size 改成 10240 不同。后者会让 QKV、MoE 和输出投影的矩阵规模显著膨胀；HyperConnection 主要增加较小的 HC mixing/injection 计算。

---

## 4. 初始四流状态如何产生

在第一个 Pipeline Parallel rank，模型先得到普通 token 或多模态 embedding：

```text
embedding: [T, 2560]
```

然后执行：

```python
hidden_states = hidden_states.repeat(1, hc_count)
```

得到：

```text
hidden_states: [T, 10240]
```

逻辑上：

```text
X = [embedding | embedding | embedding | embedding]
```

即初始时：

\[
X_{t,0}=X_{t,1}=X_{t,2}=X_{t,3}
\]

四条 stream 会在后续层通过以下机制逐渐分化：

- 每条 stream 独立的 RMSNorm affine 参数；
- 每 token、每 stream、每 channel 的 mixing gate；
- 每 token、每 stream 的 injection gate；
- Attention、GDN、QSA、MoE 和 PLE 输出的持续写入。

---

## 5. HyperConnection 模块组成

每个 Decoder Layer 有两个独立的 `GatedResidual`：

```python
self.attn_hyper_connection = GatedResidual(...)
self.mlp_hyper_connection = GatedResidual(...)
```

它们分别负责：

```text
attn_hyper_connection:
    四流状态 → Attention/GDN/QSA 输入
    生成 Attention 输出的 injection logits

mlp_hyper_connection:
    将 Attention 输出写回四流
    四流状态 → MoE/MLP 输入
    生成 MoE/MLP 输出的 injection logits
```

模型最后还有一个：

```python
self.hyper_connection_mixer = GatedResidual(use_combine=False)
```

它负责在 48 层结束后，将最终四流状态压缩为 LM Head 使用的单流 `[T,2560]`。

---

## 6. Grouped Gemma RMSNorm

HyperConnection 不使用普通的单流 pre-norm，而是对四条 stream 分别执行 Gemma 风格 RMSNorm。

输入：

```text
X_flat: [T, 10240]
```

逻辑 reshape：

```text
X: [T, 4, 2560]
```

对每个 token、每条 stream 独立计算：

\[
RMS(X_{t,c})
=
\sqrt{
\frac{1}{H}
\sum_{d=1}^{H}X_{t,c,d}^{2}
+\epsilon
}
\]

Gemma 风格 affine 为：

\[
X^n_{t,c,d}
=
\frac{X_{t,c,d}}{RMS(X_{t,c})}
(1+w_{c,d})
\]

配置中：

```python
hc_per_branch_norm=True
```

因此 norm weight shape 是：

```text
[4 × 2560] = [10240]
```

四条 stream 各自拥有不同 affine 参数，而不是共享同一组 2560 维权重。

输出 shape 不变：

```text
[T, 10240]
```

---

## 7. Mix：从四流读取一个 2560 维 Block Input

`GatedResidual.mix()` 的目标是：

```text
[T, 4, 2560]
    ↓ learnable gated mix
[T, 2560]
```

完整流程：

```text
X [T,4,2560]
    ↓ grouped Gemma RMSNorm
Xn [T,4,2560]
    ↓ flatten
Xn [T,10240]
    ├─ down projection → low-rank [T,320]
    └─ injection projection → [T,4]
             ↓
low-rank [T,320]
    ↓ SiLU(x / 4)
    ↓ up projection
mixing logits [T,10240]
    ↓ reshape + sigmoid
mixing gate [T,4,2560]
    ↓ Xn × gate，沿 stream 维求平均
block_input [T,2560]
```

### 7.1 合并的 Down + Injection Projection

NVIDIA 实现把 low-rank down projection 和 injection projection 合并成一个 GEMM：

```python
input_mix_weight_down_block_inject
```

逻辑输出大小：

```text
hc_lowrank + hc_count
= 320 + 4
= 324
```

为使 skinny GEMM 输出维度按 16 对齐，额外加入 12 个 padding channel：

```text
320 + 4 + 12 = 336
```

物理 GEMM：

```text
[T,10240] × [336,10240]ᵀ
    ↓
[T,336]
```

拆分为：

```text
low-rank state:   [T,320]
injection logits: [T,4]
padding:          [T,12]  # 不参与数学计算
```

逻辑公式：

\[
L=X^nW_{down}\in\mathbb{R}^{T\times320}
\]

\[
I=X^nW_{inject}\in\mathbb{R}^{T\times4}
\]

`L` 用于计算如何从四流读取 block input；`I` 被保存，等 block 输出返回后控制如何写回四流。

### 7.2 Low-rank Mixing Gate

首先执行：

\[
L'=\operatorname{SiLU}\left(\frac{L}{C}\right)
\]

即：

```text
[T,320] → [T,320]
```

然后 up projection：

\[
G=L'W_{up}
\]

shape：

```text
[T,320] → [T,10240] → [T,4,2560]
```

`hc_gate_mix` 内部执行 sigmoid：

\[
\hat G_{t,c,d}=\sigma(G_{t,c,d})
\]

这个 mixing gate 的粒度是：

```text
每个 token × 每条 stream × 每个 hidden channel
```

因此同一条 stream 内不同 hidden channel 可以使用不同读取权重。

### 7.3 Gated Mean

最终 block input 为：

\[
BlockInput_{t,d}
=
\frac{1}{C}
\sum_{c=1}^{C}
\hat G_{t,c,d}X^n_{t,c,d}
\]

代入 `C=4`：

\[
BlockInput_{t,d}
=
\frac{1}{4}
\sum_{c=1}^{4}
\sigma(G_{t,c,d})X^n_{t,c,d}
\]

shape 变化：

```text
normalized state: [T,4,2560]
mixing gate:      [T,4,2560]
逐元素乘法:       [T,4,2560]
对 4 条流求平均:  [T,2560]
```

Attention、GDN、QSA 或 MLP 收到的就是这个 `[T,2560]`。

---

## 8. Combine：将 2560 维 Block Output 写回四流

Attention 或 MLP 的输出 shape 为：

```text
BlockOutput: [T,2560]
```

HyperConnection 同时保存了 mix 阶段产生的 injection logits：

```text
InjectionLogits: [T,4]
```

先转成注入系数：

\[
J_{t,c}
=
2\sigma\left(\frac{I_{t,c}}{C}\right)
\]

当 `C=4`：

\[
J_{t,c}
=
2\sigma\left(\frac{I_{t,c}}{4}\right)
\]

然后把同一个 block output 按不同强度写入四条 stream：

\[
X'_{t,c,d}
=
X_{t,c,d}
+J_{t,c}BlockOutput_{t,d}
\]

shape 变化：

```text
原四流状态:        [T,4,2560]
block output:       [T,2560]
                    ↓ unsqueeze
                    [T,1,2560]
injection coefficient:
                    [T,4]
                    ↓ unsqueeze
                    [T,4,1]
broadcast 乘法:     [T,4,2560]
residual add:       [T,4,2560]
flatten:            [T,10240]
```

### 为什么使用 `2 × sigmoid`

\[
2\sigma(x)\in(0,2)
\]

当 injection logit 为 0：

\[
2\sigma(0)=1
\]

此时 combine 退化为标准 residual 强度：

\[
X'_c=X_c+BlockOutput
\]

模型还可以学习：

```text
接近 0：几乎不更新该 stream
接近 1：标准 residual add
接近 2：增强该 stream 的更新
```

如果不乘 2，零初始化对应 0.5，会天然把 residual 输出减半。

---

## 9. Mixing Gate 与 Injection Gate 的区别

这两个 gate 容易混淆。

| Gate | Shape | 使用时机 | 控制对象 |
|---|---:|---|---|
| Mixing gate | `[T,4,2560]` | block 执行前 | 四条流如何合并成 block input |
| Injection gate | `[T,4]` | block 执行后 | block output 以多大强度写回每条流 |

Mixing gate 是 channel-wise：

```text
stream 0 的 hidden channel 0 和 channel 1 可以有不同读取权重
```

Injection gate 是 stream-wise：

```text
同一条 stream 的 2560 个 channel 共享一个 block 输出注入系数
```

因此一次 HC 操作同时学习：

```text
Read：从哪里读取、读取哪些 channel
Write：把 block 结果写入哪些 stream、写入多强
```

---

## 10. 单个 Token 的直观例子

令 `T=1`，四条流分别为：

```text
x0 [2560]
x1 [2560]
x2 [2560]
x3 [2560]
```

### 10.1 读取阶段

HyperConnection 生成 channel-wise gate：

```text
g0 [2560]
g1 [2560]
g2 [2560]
g3 [2560]
```

Attention 输入为：

\[
a=
\frac{
g_0\odot Norm(x_0)
+g_1\odot Norm(x_1)
+g_2\odot Norm(x_2)
+g_3\odot Norm(x_3)
}{4}
\]

shape：

```text
a: [2560]
```

### 10.2 Block 计算

```text
y = Attention(a)
y: [2560]
```

### 10.3 写回阶段

假设 injection 系数为：

```text
j0 = 0.2
j1 = 0.8
j2 = 1.1
j3 = 1.6
```

更新后：

```text
x0' = x0 + 0.2 × y
x1' = x1 + 0.8 × y
x2' = x2 + 1.1 × y
x3' = x3 + 1.6 × y
```

新四流状态：

```text
X' = [x0' | x1' | x2' | x3']
shape = [10240]
```

所以四流状态不是由 Attention 输出复制四份得到，而是：

```text
Attention 前保留下来的四流 residual state
+
按四个 injection gate 广播写回的 Attention output
```

---

## 11. Attention 前后的完整流程

进入当前 Attention 边界时：

```text
hidden_states: [T,10240]
```

执行：

```python
hidden_states, block_input, injection = attn_hc.mix(hidden_states)
```

返回：

```text
hidden_states: [T,10240]  # 原四流 residual 主干，继续保留
block_input:   [T,2560]   # Attention/GDN/QSA 输入
injection:     [T,4]      # Attention 输出写回时使用
```

Attention 只处理 `block_input`：

```text
[T,2560]
    ↓ GDN 或 QSA
[T,2560]
```

此时同时存在：

```text
原四流 residual state: [T,10240]
Attention output:       [T,2560]
Attention injection:    [T,4]
```

原四流状态没有被 Attention 覆盖或丢弃。

---

## 12. MLP 前的四流状态从哪里来

Attention 完成后调用：

```python
hidden_states, mlp_input, mlp_injection = (
    mlp_hc.combine_and_mix(
        hidden_states,
        attn_out,
        attn_injection,
    )
)
```

第一步 combine：

```text
Attention 前保留的四流状态 [T,4,2560]
+
Attention 输出 [T,2560] × injection [T,4]
    ↓
Attention 后的新四流状态 [T,4,2560]
```

第二步 mix：

```text
新四流状态 [T,4,2560]
    ↓ grouped RMSNorm
    ↓ low-rank mixing gate
    ↓ gated mean over streams
MLP input [T,2560]
```

同时生成：

```text
MLP injection logits: [T,4]
```

因此 MoE/MLP 的输入仍然是：

```text
[T,2560]
```

MoE/MLP 输出也是：

```text
[T,2560]
```

---

## 13. 一个 Decoder Layer 的完整 Shape 流程

```text
进入当前层：
hidden_states       [T,10240]
prev_mlp_output     [T,2560] 或 None
prev_mlp_injection  [T,4]    或 None
```

### 13.1 Attention 前

如果有上一层 pending MLP output：

```text
四流状态 [T,10240]
+ prev MLP output [T,2560] × injection [T,4]
    ↓ combine
materialized state [T,10240]
```

然后 mix：

```text
materialized state [T,10240]
    ↓ grouped norm + gated mix
attention input [T,2560]
```

### 13.2 Attention/GDN/QSA

```text
attention input  [T,2560]
    ↓
GDN 或 QSA
    ↓
attention output [T,2560]
```

### 13.3 MLP 前

```text
四流状态 [T,10240]
+ attention output [T,2560] × injection [T,4]
    ↓ combine
attention 后四流状态 [T,10240]
    ↓ grouped norm + gated mix
MLP input [T,2560]
```

### 13.4 MoE/MLP

```text
MLP input  [T,2560]
    ↓ MoE/MLP
MLP output [T,2560]
```

### 13.5 当前层返回

```text
hidden_states: [T,10240]  # Attention 已写回
block_output:  [T,2560]   # MLP 输出，暂未写回
injection:     [T,4]      # MLP 写回 gate
```

MLP 输出会在下一层 Attention HC 边界被写回。

---

## 14. NVIDIA Delayed Combine

### 14.1 普通实现

最直接的实现是每个 block 结束后立即 combine：

```text
X
    ↓ mix
BlockInput
    ↓ Block
BlockOutput
    ↓ combine
X'
    ↓ 下一个 HC norm + mix
```

这会单独产生：

```text
combine kernel
grouped RMSNorm kernel
mixing projection kernels
```

### 14.2 NVIDIA 实现

NVIDIA 版本把 block output、injection 暂存到下一个 HC 边界：

```text
当前 block：
    计算 BlockOutput
    暂不 combine

下一个 HC：
    combine 上一个 BlockOutput
    + grouped RMSNorm
    + 当前 HC mix
```

接口为：

```python
combine_and_mix(
    hidden_states,
    prev_block_output,
    prev_injection,
)
```

其中：

```text
prev_block_output: 上一个 Attention/MLP 的 [T,2560] 输出
prev_injection:    上一个 HC mix 产生的 [T,4] injection logits
当前 HC module:    提供 combine 后所需的 norm 和下一次 mix 参数
```

`hc_combine_norm` 将：

```text
combine
+
下一边界的 grouped RMSNorm
```

融合为一个操作，减少一次 `[T,10240]` 中间结果的显存读写。

---

## 15. Layer 间的实际执行顺序

对普通、没有特殊加法的层，执行时序是：

```text
Layer N Attention HC:
    消费 Layer N-1 pending MLP output
    → combine_and_mix
    → Layer N attention input

Layer N Attention/GDN/QSA:
    [T,2560] → [T,2560]

Layer N MLP HC:
    消费 Layer N attention output
    → combine_and_mix
    → Layer N MLP input

Layer N MoE/MLP:
    [T,2560] → [T,2560]
    → 作为 pending output 返回
```

所以一个 Decoder Layer 结束时，逻辑上的最终 MLP residual add 可能尚未真正执行。

这也是 `Qwen4ExpDecoderLayer.forward()` 返回三元组而不是单一 hidden state 的原因：

```python
return hidden_states, mlp_out, injection
```

---

## 16. PLE 为什么会打断 Delayed Combine

PLE 直接对完整四流状态做加法：

```python
hidden_states = hidden_states + self.ple(...)
```

如果进入 PLE 层时，上一个 MLP output 仍处于 pending 状态，则当前 `hidden_states` 还不是完整 materialized state。

因此必须先执行：

```python
hidden_states = attn_hc.combine(
    hidden_states,
    prev_block_output,
    prev_injection,
)
```

得到完整：

```text
[T,10240]
```

然后：

```text
materialized HC state
    + PLE output [T,10240]
    ↓
PLE 增强后的四流状态
    ↓ attn_hc.mix
Attention input [T,2560]
```

官方模型中 `ple_layer_ids=[2]`，所以 zero-based Layer 1 会发生这次 materialization。

任何直接修改完整多流状态的外部加法都需要类似处理。代码中的 multimodal deepstack 注入也会终止 delayed combine，不过官方 Qwen3.8-Flash-Next 配置没有启用 deepstack。

---

## 17. Pipeline Parallel 边界

一个 PP stage 内可以把 pending 状态保存为 Python 层面的三元组：

```text
hidden_states
block_output
injection
```

但 PP stage 间只传输统一的 `IntermediateTensors`，不能把 delayed tuple 留给远端 stage。

因此非最后 PP rank 返回前必须执行：

```python
hidden_states = last_layer.mlp_hyper_connection.combine(
    hidden_states,
    block_output,
    injection,
)
```

得到完整四流状态：

```text
[T,10240]
```

然后传递：

```python
IntermediateTensors({"hidden_states": hidden_states})
```

下一个 PP rank 直接从 `[T,10240]` 继续执行，不再需要上一 stage 的 pending output 和 injection。

---

## 18. Final HyperConnection Mixer

最后一层结束后通常仍有 pending MLP output：

```text
hidden_states: [T,10240]
block_output:  [T,2560]
injection:     [T,4]
```

最后执行：

```python
multi_hidden, sample_hidden_states, _ = (
    final_mixer.combine_and_mix(
        hidden_states,
        block_output,
        injection,
    )
)
```

`final_mixer` 使用：

```python
use_combine=False
```

这里的含义不是“不 combine 最后一个 MLP output”，而是：

- `combine_and_mix` 仍会消费传入的最后一个 MLP output；
- final mixer 自己后面没有新的 block，因此不需要再生成新的 injection logits。

返回：

```text
multi_hidden:         [T,10240]
sample_hidden_states: [T,2560]
```

其中：

- `multi_hidden` 是最后一个 MLP 已写回的完整四流状态；
- `sample_hidden_states` 是 final gated mix 得到的单流状态；
- LM Head 只接收 `[T,2560]`。

---

## 19. MTP 中的四流状态

开启 MTP 时，目标模型不仅需要给 LM Head 返回单流状态，还要保留：

```text
multi_hidden: [T,10240]
```

主模型将它写入：

```python
self._mtp_hidden_buffer
```

MTP draft 模型第一步接收：

```text
目标模型 multi hidden: [T,4,2560]
新 token embedding:    [T,2560]
```

分别投影后，把 token embedding 广播到四条 stream：

```text
fc_hidden(multi_hidden): [T,4,2560]
fc_embedding(token):     [T,2560]
                         ↓ unsqueeze/broadcast
                         [T,1,2560]

相加：                   [T,4,2560]
flatten：                [T,10240]
```

随后 MTP 的 Decoder Layer 继续使用同样的 HyperConnection 读取/写回机制。

每个 draft step 最终同时输出：

```text
sample_hidden [T,2560]   → MTP LM Head
multi_hidden  [T,10240]  → 下一个 draft step
```

如果只保留 final mixer 后的 `[T,2560]`，下一 MTP step 就失去了目标模型真实的四流状态。

---

## 20. Attention、MLP 与 PLE 的工作宽度

不同模块的工作宽度如下：

| 模块 | 输入/输出宽度 |
|---|---:|
| 层间 HyperConnection residual state | 10240 |
| Grouped Gemma RMSNorm | 10240 |
| HC down projection 输入 | 10240 |
| HC low-rank state | 320 |
| HC up projection输出 | 10240 |
| HC mixing gate | `4 × 2560` |
| HC injection gate | 4 |
| GDN 输入/输出 | 2560 |
| QSA 输入/输出 | 2560 |
| MoE/MLP 输入/输出 | 2560 |
| LM Head 输入 | 2560 |
| PLE key/query/gated value | `4 × 2560` |
| PLE short-conv channels | 10240 |
| MTP multi hidden | 10240 |

所以“模型 hidden size 是 2560 还是 10240”必须结合语境：

```text
配置和 block hidden size：2560
HyperConnection state width：10240
```

---

## 21. Tensor Parallel 行为

本文前面的 shape 是模块接口的逻辑 shape。

Attention、GDN、QSA 和 MoE 内部会按 TP/EP 配置分片，但对 Decoder Layer 暴露的接口仍是：

```text
[T,2560] → block → [T,2560]
```

NVIDIA HyperConnection 中：

- merged down+injection projection 使用 `disable_tp=True`；
- down/up mixing projection采用 replicated linear 语义；
- HC 需要在各 TP rank 上获得一致的多流 gate 和 injection；
- 主 block 的 TP 通信仍封装在 Attention/GDN/QSA/MoE 内部。

因此不能把 QKV 的 per-rank head shape 与 HyperConnection 对外 `[T,2560]` 的 block shape 混为一谈。

---

## 22. 计算与存储开销直觉

如果直接把 Transformer hidden size 从 2560 扩大到 10240：

- QKV、output projection 的权重规模会大幅增加；
- MoE router 和每个 expert 的输入输出矩阵会大幅增加；
- 主干 FLOPs 和通信量会非常高。

HyperConnection 则保持主 block 宽度 2560，只增加：

- `[10240 → 320]` 的 low-rank down projection；
- `[320 → 10240]` 的 up projection；
- `[10240 → 4]` 的 injection projection；
- grouped norm、逐元素 gate、stream mean 和 broadcast residual add。

因此它用相对低秩的控制网络管理更宽的层间状态，而不是让所有主干算子直接运行在 10240 维。

NVIDIA 实现又通过以下方式降低额外成本：

- 合并 down projection 和 injection projection；
- 输出维度 padding 到 16 对齐，便于 skinny GEMM；
- 融合 combine + grouped RMSNorm；
- 延迟 combine，减少中间 `[T,10240]` 的显存往返；
- 在适用的 NVIDIA GPU 上使用 low-latency skinny GEMM。

---

## 23. 与标准 Residual 的关系

标准 residual：

\[
X'=X+Block(X)
\]

HyperConnection 可以理解为其多流、门控推广：

### 读取

\[
BlockInput
=
Mix(X_0,X_1,X_2,X_3)
\]

### Block

\[
B=Block(BlockInput)
\]

### 写回

\[
X'_c=X_c+J_cB
\]

当：

- 四条 stream 相同；
- mixing gate 产生等价平均；
- injection coefficient 为 1；

它的行为接近标准 residual。训练则可以学习更丰富的读写策略。

---

## 24. 简化伪代码

### 24.1 HyperConnection Mix

```python
def hc_mix(x):                    # x: [T, 4, 2560]
    xn = grouped_gemma_rmsnorm(x) # [T, 4, 2560]

    packed = merged_down_inject(xn.flatten(-2))  # [T, 336]
    lowrank = packed[:, :320]                    # [T, 320]
    injection = packed[:, 320:324]               # [T, 4]

    lowrank = silu(lowrank / 4)
    gate_logits = up_proj(lowrank)                # [T, 10240]
    gate = sigmoid(gate_logits.view(T, 4, 2560))

    block_input = (gate * xn).mean(dim=1)         # [T, 2560]
    return x, block_input, injection
```

### 24.2 HyperConnection Combine

```python
def hc_combine(x, block_output, injection):
    # x:            [T, 4, 2560]
    # block_output: [T, 2560]
    # injection:    [T, 4]

    inject = 2 * sigmoid(injection / 4)           # [T, 4]
    x = x + block_output[:, None, :] * inject[:, :, None]
    return x                                      # [T, 4, 2560]
```

### 24.3 Decoder Layer

```python
def decoder_layer(x, prev_output, prev_injection):
    # x: [T, 4, 2560]

    if has_ple:
        if prev_output is not None:
            x = attn_hc.combine(x, prev_output, prev_injection)
            prev_output = prev_injection = None
        x = x + ple(x)

    if prev_output is not None:
        x, attn_input, attn_injection = attn_hc.combine_and_mix(
            x, prev_output, prev_injection
        )
    else:
        x, attn_input, attn_injection = attn_hc.mix(x)

    attn_output = attention_or_gdn(attn_input)    # [T, 2560]

    x, mlp_input, mlp_injection = mlp_hc.combine_and_mix(
        x, attn_output, attn_injection
    )

    mlp_output = moe_or_mlp(mlp_input)            # [T, 2560]

    # MLP output 延迟到下一 HC 边界 combine。
    return x, mlp_output, mlp_injection
```

---

## 25. 调试检查清单

调试 HyperConnection 时，建议依次检查：

1. 初始 embedding 是否从 `[T,2560]` 正确扩展为 `[T,10240]`；
2. flatten 布局是否为 `HC outer, hidden inner`，即 `[T,4,2560]`；
3. grouped norm 是否按每个 2560 维 stream 独立计算 RMS；
4. norm affine weight 是否为 `[10240]`；
5. merged down+injection 输出是否为 `[T,336]`；
6. split 是否严格为 `[320,4,12]`；
7. mixing gate 是否为 `[T,4,2560]`，并在 stream 维做 mean；
8. block input 是否为 `[T,2560]`；
9. block output 是否仍为 `[T,2560]`；
10. injection logits 是否为 `[T,4]`，而不是 `[T,4,2560]`；
11. combine 时 block output 是否沿 stream 维 broadcast；
12. Attention output 是否由 `mlp_hc.combine_and_mix` 消费；
13. MLP output 是否由下一层 `attn_hc.combine_and_mix` 消费；
14. PLE 前是否先 materialize pending output；
15. PP stage 返回前是否清空 pending tuple 并只传 `[T,10240]`；
16. final mixer 是否同时得到 `[T,10240]` 和 `[T,2560]`；
17. MTP 是否使用 final mixer 前的真实 multi-stream state；
18. CUDA Graph padding token 是否不会污染真实四流状态。

---

## 26. 常见理解误区

### 误区一：Attention 输入是 10240

错误。Attention/GDN/QSA 输入都是 HC mix 后的 `[T,2560]`。

### 误区二：Attention 输出会复制四份形成新状态

不准确。原四流 residual state 一直保留；Attention 输出按四个 injection coefficient 增量写回。

### 误区三：四条 stream 每条只有一个 mixing 权重

错误。mixing gate 是 `[T,4,2560]`，每个 hidden channel 都有独立读取 gate。

### 误区四：mixing gate 和 injection gate 是同一个张量

错误。前者控制读，shape 为 `[T,4,2560]`；后者控制写，shape 为 `[T,4]`。

### 误区五：一层结束时 `hidden_states` 已包含本层 MLP 输出

在 NVIDIA delayed-combine 路径中通常不是。MLP output 会作为 pending tuple 返回，在下一 HC 边界才写回。

### 误区六：`final_mixer(use_combine=False)` 不会写回最后一个 MLP

错误。它仍通过 `combine_and_mix` 消费传入的 pending MLP；`use_combine=False` 只表示 final mixer 不为后续不存在的 block 生成新 injection。

---

## 27. 源码索引

- NVIDIA HyperConnection：[`vllm/models/qwen4_exp/nvidia/hyperconnection.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/nvidia/hyperconnection.py)
- HyperConnection 通用配置与 Grouped Norm：[`vllm/models/qwen4_exp/common/hyperconnection.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/common/hyperconnection.py)
- NVIDIA HC kernels：[`vllm/models/qwen4_exp/nvidia/ops/hc.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/nvidia/ops/hc.py)
- Qwen4Exp Decoder 与 Model forward：[`vllm/models/qwen4_exp/nvidia/model.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/nvidia/model.py)
- NVIDIA MTP：[`vllm/models/qwen4_exp/nvidia/mtp.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/nvidia/mtp.py)
- NVIDIA PLE：[`vllm/models/qwen4_exp/nvidia/ple_layer.py`](https://github.com/vllm-project/vllm/blob/95dc96d1d012a25ff5c3823a1e77197c8dae4654/vllm/models/qwen4_exp/nvidia/ple_layer.py)
- Qwen3.8-Flash-Next 官方配置：[`config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
- vLLM PR：[#53899](https://github.com/vllm-project/vllm/pull/53899)

---

## 28. 总结

Qwen3.8-Flash-Next HyperConnection 的核心是：

```text
四流保存状态，单流执行主 block，门控完成读写。
```

具体来说：

- 层间 residual state 是 `[T,4,2560]`，物理 flatten 为 `[T,10240]`；
- Attention、GDN、QSA、MoE 和 MLP 的输入输出仍是 `[T,2560]`；
- grouped Gemma RMSNorm 对四条 stream 分别归一化；
- `10240 → 320 → 10240` 的 low-rank 网络产生 channel-wise mixing gate；
- gated mean 将四流压缩为一个 2560 维 block input；
- `[T,4]` injection gate 将 block output 分别写回四条流；
- NVIDIA delayed combine 将写回推迟到下一个 HC 边界，并融合 combine + norm；
- PLE、PP 边界和其他直接多流加法会强制 materialize pending output；
- final mixer 同时保留 MTP 使用的 `[T,10240]` 多流状态，并生成 LM Head 使用的 `[T,2560]` 单流状态。

HyperConnection 并不是把整个 Transformer hidden size 扩大到 10240，而是用低秩门控网络管理一个 10240 维的多流 residual memory，同时让昂贵的 Attention 和 MoE 主计算保持在 2560 维。
