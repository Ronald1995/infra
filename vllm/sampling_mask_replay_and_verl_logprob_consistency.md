# vLLM Sampling Mask Replay 与 verl Logprob 一致性

> 本文基于 vLLM PR [#49577](https://github.com/vllm-project/vllm/pull/49577)、DeepSeek-V3.2 技术报告的 Keep Sampling Mask 方法，以及 verl `3c62a15d` 代码快照整理。本文讨论 vLLM rollout 的 `processed_logprobs`、verl 训练侧重新计算的 logprob、top-p/top-k 截断引入的 action-space mismatch，以及 sampling mask replay 如何消除这类系统偏差。

## 1. 核心结论

1. verl 使用 vLLM rollout 时，默认把 `logprobs_mode` 设置为 `processed_logprobs`。该 logprob 对应 vLLM 实际采样分布，而不是 processor 执行前的原始模型分布。
2. `processed_logprobs` 本身是正确选择。问题出现在 rollout 使用 top-p/top-k 截断，而训练侧仍在完整词表上归一化时。
3. 仅在训练侧重新执行相同的 `top_p`、`top_k` 参数不能保证精确一致，因为训练侧重新得到的 support 可能不同于 rollout 当时的 support。
4. Sampling mask replay 保存 rollout 每一步实际保留的 token ID 集合，并在训练侧对当前策略复用该集合，使两侧在完全相同的 action subspace 上计算 logprob。
5. 当前 verl 默认配置为 `top_p=1`、`top_k=-1`、`repetition_penalty=1.0`、`temperature=1.0`，默认没有 top-p/top-k 截断，因此不能笼统地说 verl 一直存在截断导致的系统误差。
6. 当用户设置 `top_p<1`、`top_k>0`、`min_p>0` 或其他会删除词表 support 的 processor，且训练侧没有精确 replay rollout mask 时，确实存在结构性的 action-space mismatch。

完整链路可以概括为：

```text
vLLM rollout
  raw logits
    → penalty / temperature / min-p / top-k / top-p
    → processed logits
    → sampling support S_t
    → sampled token a_t
    → processed rollout logp + S_t

verl training with mask replay
  training logits
    → temperature 等可重放变换
    → 固定应用 rollout support S_t
    → masked log_softmax
    → current/old training logp
    → importance ratio / KL
```

## 2. Raw Logprob 与 Processed Logprob

设模型原始 logits 为 $z$，完整词表为 $V$。

### 2.1 Raw logprob

`raw_logprobs` 在 temperature、penalty、top-k、top-p 等采样处理之前计算：

$$
\log p_{\mathrm{raw}}(a\mid s)
=z_a-\log\sum_{j\in V}\exp(z_j).
$$

它描述原始模型 softmax 分布，但在启用采样 processor 后，不一定等于真正生成 token 的行为策略概率。

### 2.2 Processed logprob

设经过 penalty、temperature、min-p、top-k 和 top-p 后的 logits 为 $z'$，最终保留的 token 集合为 $S\subseteq V$。vLLM 的 `processed_logprobs` 为：

$$
\log p_{\mathrm{processed}}(a\mid s)
=z'_a-\log\sum_{j\in S}\exp(z'_j).
$$

被过滤 token 的 logits 被设置为 $-\infty$，因此它们的概率为 0。`processed_logprobs` 对应实际采样分布，适合作为 importance sampling 中的行为策略分母。

如果为了与训练侧 full-vocabulary logp 表面一致而切换到 `raw_logprobs`，但 token 实际仍由 top-p/top-k 分布生成，那么记录下来的 logprob 反而不再是行为策略的真实概率。

## 3. verl 当前的 vLLM Logprob 数据流

verl rollout 配置默认包含：

```yaml
logprobs_mode: processed_logprobs
calculate_log_probs: true
```

vLLM AsyncLLM 初始化时，verl 将 `self.config.logprobs_mode` 传入 engine。生成请求中，布尔配置 `calculate_log_probs` 最终被转换为：

```python
sampling_params["logprobs"] = 0
```

这里的 `0` 不是“不返回 logprob”，而是只返回实际采样 token 的 logprob。生成完成后，verl 按每个生成 token ID 从 vLLM 输出中取出对应值：

```python
log_probs = [
    step_logprobs[token_ids[i]].logprob
    for i, step_logprobs in enumerate(output.logprobs)
]
```

数据随后经过 agent loop 对齐：

```text
vLLM CompletionOutput.logprobs
  → TokenOutput.log_probs
  → AgentLoopOutput.response_logprobs
  → batch["rollout_log_probs"]
```

multi-turn 场景中，模型生成 token、tool observation 和 padding 由 `response_mask` 区分；只有模型实际生成的 action token 参与策略损失。

## 4. verl 中的三种策略概率

正常 decoupled 模式下，verl 区分：

| 概率 | 典型字段 | 来源 | 用途 |
|---|---|---|---|
| $\pi_{\mathrm{rollout}}$ | `rollout_log_probs` | vLLM processed logp | 行为策略、rollout correction |
| $\pi_{\mathrm{old}}$ | `old_log_probs` | 训练后端重新 forward | PPO 固定近端锚点 |
| $\pi_\theta$ | `log_prob` | 当前 minibatch actor forward | 当前待优化策略 |

PPO 内部 ratio 为：

$$
r_t^{\mathrm{PPO}}
=\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{old}}(a_t\mid s_t)
\right).
$$

通常 $\pi_\theta$ 与 $\pi_{\mathrm{old}}$ 都由训练后端在完整词表上计算，因此 PPO ratio 内部口径一致。但 rollout 数据实际来自 $\pi_{\mathrm{rollout}}$；如果 rollout 做了截断，数据生成策略与训练定义的 old policy 并不完全相同。

rollout correction 使用：

$$
r_t^{\mathrm{corr}}
=\frac{\pi_{\mathrm{old}}(a_t\mid s_t)}
       {\pi_{\mathrm{rollout}}(a_t\mid s_t)}
=\exp\left(
\log\pi_{\mathrm{old}}
-\log\pi_{\mathrm{rollout}}
\right).
$$

bypass mode 则直接设置：

```python
old_log_probs = rollout_log_probs
```

此时 PPO ratio 直接比较训练侧 current full-vocabulary logp 与 vLLM processed rollout logp。如果两侧 support 不同，偏差会直接进入 PPO ratio。

## 5. Top-p/Top-k 为什么产生系统偏差

假设 rollout 与训练模型权重完全相同，只考虑 top-p/top-k 截断。令 full-vocabulary 分布为 $p$，rollout 保留集合为 $S$，集合内总概率质量为：

$$
Z_S=\sum_{j\in S}p(j).
$$

vLLM 在 $S$ 内重新归一化后，行为策略为：

$$
q(a)=\frac{p(a)}{Z_S},\quad a\in S.
$$

如果训练侧仍计算 full-vocabulary probability，则：

$$
\frac{p(a)}{q(a)}=Z_S<1.
$$

因此，即使：

- rollout 与训练权重完全同步；
- 没有 BF16/FP32 误差；
- temperature 完全一致；
- 输入 token 完全相同；

rollout correction ratio 仍会系统性小于 1。若保留概率质量约为 0.9，则 ratio 通常也接近 0.9，而不是 1。

进一步有：

$$
\mathbb{E}_{a\sim q}\left[\frac{p(a)}{q(a)}\right]
=p(S)=Z_S<1.
$$

目标策略在 $S$ 外仍有概率质量，但行为策略永远不会采到这些动作；这违反了 importance sampling 所需的 support/coverage 条件。增加采样数量不能消除该结构性问题。

## 6. 为什么只同步 Top-p/Top-k 参数仍不精确

训练侧当然可以执行相同的：

```yaml
top_p: 0.9
top_k: 50
```

但采样 support 是 logits 的函数，而不是配置参数本身：

$$
S_t^{\mathrm{rollout}}
=\operatorname{TopKTopP}(z_t^{\mathrm{vLLM}}),
$$

$$
S_t^{\mathrm{train}}
=\operatorname{TopKTopP}(z_t^{\mathrm{train}}).
$$

以下因素都可能使二者不同：

- actor 已经过若干 minibatch 更新；
- vLLM 与训练后端精度不同；
- softmax、排序和 top-p kernel 不同；
- tensor-parallel reduction 顺序不同；
- 量化、LoRA 合并或权重同步时序不同；
- repetition penalty、bad words、grammar、logit bias 没有完整重放；
- top-p 边界 token 对微小数值变化非常敏感。

若 rollout 采到 token $a\in S_t^{\mathrm{rollout}}$，但训练重新生成的 mask 中 $a\notin S_t^{\mathrm{train}}$，训练 logprob 会成为 $-\infty$，importance ratio 变成 0。相反，训练新 support 中但 rollout support 外的动作又不可能出现在该 batch 中。

因此正确做法不是重新决定 mask，而是固定重放 rollout 当时的 $S_t^{\mathrm{rollout}}$。

## 7. Sampling Mask 的对外数据格式

vLLM PR #49577 对外定义的结构为：

```python
@dataclass
class SamplingMask:
    token_ids: list[list[int]]
```

每个 completion token 对应一个 support 列表：

```python
generated_token_ids = [42, 17, 99]

sampling_mask.token_ids = [
    [3, 42, 55],       # 生成 token 42 时的 support
    [1, 17],           # 生成 token 17 时的 support
    [8, 9, 23, 99],    # 生成 token 99 时的 support
]
```

必须满足：

```python
generated_token_ids[i] in sampling_mask.token_ids[i]
```

support 内 token 的顺序不应作为训练语义使用；训练侧只依赖集合成员关系。

Python API 示例：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="model-path",
    return_sampling_mask=True,
    logprobs_mode="processed_logprobs",
)

outputs = llm.generate(
    "Hello",
    SamplingParams(
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        logprobs=1,
    ),
)

completion = outputs[0].outputs[0]
print(completion.token_ids)
print(completion.logprobs)
print(completion.sampling_mask.token_ids)
```

HTTP `/inference/v1/generate` 返回形式为：

```json
{
  "choices": [
    {
      "token_ids": [42, 17, 99],
      "sampling_mask": [
        [3, 42, 55],
        [1, 17],
        [8, 9, 23, 99]
      ],
      "finish_reason": "stop"
    }
  ]
}
```

## 8. vLLM 内部 CSR 表示

二维 Python list 会带来大量对象和序列化开销，因此 vLLM 内部使用 CSR 风格的压缩结构：

```python
class SamplingMaskLists(NamedTuple):
    token_ids: np.ndarray
    offsets: np.ndarray
    cu_num_generated_tokens: list[int] | None
```

对于：

```python
supports = [
    [3, 42, 55],
    [1, 17],
    [8, 9, 23, 99],
]
```

内部表示为：

```python
token_ids = [3, 42, 55, 1, 17, 8, 9, 23, 99]
offsets = [0, 3, 5, 9]
```

第 $i$ 个生成 token 的 support 为：

```python
support_i = token_ids[offsets[i] : offsets[i + 1]]
```

字段含义如下：

| 字段 | 长度 | 含义 |
|---|---:|---|
| `token_ids` | 所有保留 token 总数 | 所有 support 拼接后的 token ID |
| `offsets` | 生成 token 数 + 1 | 每个 support 在 `token_ids` 中的起止位置 |
| `cu_num_generated_tokens` | request 数 + 1 | batch 中每个 request 对应的生成位置范围 |

例如一个 batch 中两个请求分别生成 2 和 3 个 token：

```python
cu_num_generated_tokens = [0, 2, 5]
```

表示 request 0 使用 sampling position `[0, 2)`，request 1 使用 `[2, 5)`。

## 9. vLLM 如何构造 Sampling Mask

每一步采样大致执行：

```text
raw logits
  → allowed token / bad word / logit bias
  → repetition / frequency / presence penalty
  → temperature
  → min-p
  → top-k / top-p
  → 被过滤 token 设为 -inf
  → softmax 与随机采样
```

例如 processed logits 为：

```python
[2.0, -inf, 1.0, -inf, 0.0]
```

则：

```python
torch.isfinite(processed_logits)
# [True, False, True, False, True]
```

sampling support 为：

```python
[0, 2, 4]
```

该 mask 不是 attention mask，也不是 response mask，而是该生成位置上仍有非零采样概率的 vocabulary token ID 集合。

PR 中的实现将 mask 与 sampled token 一起异步执行 GPU 到 CPU 的传输，请求结束后合并各 decode chunk，并在最终响应中转换成 `list[list[int]]`。

## 10. 训练侧如何 Replay Mask

假设：

```python
training_logits = [2.0, 0.5, 1.0, -1.0, 0.0]
sampling_mask = [0, 2, 4]
sampled_token = 2
```

训练侧应用：

```python
keep = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
keep[sampling_mask] = True
masked_logits = training_logits.masked_fill(~keep, float("-inf"))
log_prob = torch.log_softmax(masked_logits, dim=-1)[sampled_token]
```

mask 后：

```python
masked_logits = [2.0, -inf, 1.0, -inf, 0.0]
```

采样 token 2 的概率为：

$$
p_{\mathrm{mask}}(2)
=\frac{e^{1.0}}{e^{2.0}+e^{1.0}+e^{0.0}}
\approx0.2447,
$$

$$
\log p_{\mathrm{mask}}(2)\approx-1.408.
$$

不用 mask 时：

$$
p_{\mathrm{full}}(2)
=\frac{e^{1.0}}
{e^{2.0}+e^{0.5}+e^{1.0}+e^{-1.0}+e^{0.0}}
\approx0.2071,
$$

$$
\log p_{\mathrm{full}}(2)\approx-1.574.
$$

即使两侧 logits 完全相同，logp diff 仍约为：

$$
-1.574-(-1.408)=-0.166,
$$

ratio 约为：

$$
\exp(-0.166)\approx0.847.
$$

重放同一 mask 后，两侧使用相同分母；权重相同时 ratio 应接近 1，剩余差异才主要反映数值精度、kernel 或权重同步误差。

## 11. Temperature 与随机采样为什么不是同类问题

temperature 定义：

$$
\pi_\tau(a\mid s)
=\frac{\exp(z_a/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

temperature 改变概率，但不删除有限 logits 对应的 token，因此：

$$
S_\tau=V.
$$

只要 vLLM 和训练侧使用相同 temperature，两侧都对 logits 执行 `z / temperature`，它不会产生 top-p/top-k 类型的 support mismatch。`temperature=1` 时更有：

$$
z/\tau=z.
$$

随机采样只决定本次选中了哪个 token，不修改该 token 的概率。训练侧不会重新采样，而是通过 teacher forcing 对 vLLM 已生成的同一个 token 重新计算 logprob。因此，如果两侧分布相同：

$$
\log\pi_{\mathrm{train}}(a_t)
-\log\pi_{\mathrm{rollout}}(a_t)=0
$$

对每一个可能采到的 $a_t$ 都成立。

temperature 仍可能放大或缩小已有的数值差异。若 vLLM logits 为 $z+\epsilon$、训练 logits 为 $z$，则缩放后的局部差异为：

$$
\frac{z+\epsilon}{\tau}-\frac{z}{\tau}
=\frac{\epsilon}{\tau}.
$$

因此 $\tau<1$ 可能放大后端数值差异，但这不是随机采样本身造成的结构性误差。

## 12. KL 指标如何受影响

### 12.1 PPO approximate KL

PPO 中 current 与 old 都由训练后端按相同口径计算时：

$$
\Delta_t=\log\pi_\theta-\log\pi_{\mathrm{old}}
$$

不会直接混入 top-p normalization 差异。但数据实际由 truncated rollout policy 生成，因此 on-policy 假设仍不完全成立。

### 12.2 Rollout-vs-training KL

若比较：

$$
\mathbb{E}_{a\sim q}
[\log q(a)-\log p(a)],
$$

即使 rollout 与训练权重相同，也有：

$$
\log q(a)-\log p(a)=-\log Z_S>0.
$$

该值可以被解释为 truncated rollout 分布与 full-vocabulary 分布的真实 KL；但如果用它诊断 vLLM 与训练后端的数值一致性，它会把截断差异误认为后端或策略漂移。

### 12.3 Reference-policy KL

如果 actor 与 reference logp 都在训练后端完整词表上计算，两者口径一致。但 token 来自 truncated rollout 分布，因此 token-sampled KL 的期望测度仍是 rollout policy，而不一定是训练侧 old policy。

## 13. 该方法是否是最新理论

DeepSeek-V3.2 技术报告于 2025 年 12 月公开，在 “Keep Sampling Mask” 小节明确提出：保存 $\pi_{\mathrm{old}}$ 采样时的截断 mask，并在训练 $\pi_\theta$ 时重放，以保证相同 action subspace。vLLM PR #49577 是该策略的近期工程实现。

但其背后的理论并不新。Importance sampling 一直要求行为策略对目标策略需要估计的动作具有足够 support；PPO 也一直使用 $\pi_\theta/\pi_{\mathrm{old}}$ 形式的 ratio。近期工作的贡献主要是把现代 LLM RL 中的推理 sampler 明确定义为 policy 的一部分，并把每 token 的动态 support 作为需要跨 inference/training 系统重放的状态。

不能据现有公开资料断言 DeepSeek 是历史上第一个想到类似方法的团队。更准确的表述是：DeepSeek-V3.2 是近期公开、清晰命名并报告大规模 RL 稳定性收益的代表性来源。

## 14. 为什么该问题过去没有被广泛暴露

主要原因包括：

1. 很多训练使用 `top_p=1`、`top_k=-1`，本身不存在截断 support mismatch。
2. 早期 rollout 与训练模型耦合较紧；现代系统则常用 vLLM/SGLang 生成、FSDP/Megatron 训练，后端差异更明显。
3. top-p/top-k 长期被当作生成质量技巧，而不是 RL policy 定义的一部分。
4. PPO clipping、truncated importance sampling、rejection sampling 和 reward normalization 会部分遮蔽症状。
5. 现代 reasoning rollout 长度显著增加，微小的 per-token ratio 偏差会沿序列累积。
6. 保存和训练 replay 变长 mask 需要 CSR 编码、异步传输、multi-turn 对齐和高效 sparse masked logsumexp，工程成本较高。

如果每个 token 的 ratio 都存在固定偏差 $c$，序列级 ratio 会按：

$$
r_{\mathrm{seq}}=c^T
$$

累积。例如 $c=0.95$、$T=100$ 时：

$$
0.95^{100}\approx0.0059.
$$

长 CoT 会显著放大原本不易察觉的 mismatch。

## 15. verl 是否一直存在该系统误差

答案是否定的，需要按配置判断。

### 15.1 当前默认配置

verl 默认：

```yaml
temperature: 1.0
top_k: -1
top_p: 1
repetition_penalty: 1.0
```

在没有其他删除 support 的 processor 时：

$$
S=V.
$$

因此不存在 sampling mask replay 所针对的 top-p/top-k 截断偏差。仍可能存在的 mismatch 包括：

- vLLM 与训练后端精度和 kernel 不同；
- TP reduction、量化和 LoRA 行为不同；
- 权重同步延迟或 async policy staleness；
- MoE expert routing 不一致；
- speculative decoding 或其他推理优化差异。

### 15.2 启用截断采样

如果设置：

```yaml
top_p: 0.9
top_k: 50
```

且没有精确 mask replay，则：

```text
vLLM rollout：在 S_t 上归一化
verl training：在完整词表 V 上归一化
```

此时确实存在不可通过增加采样数量消除的结构性偏差。Rollout correction、ratio clipping、TIS 和 rejection sampling可以降低风险，但不能恢复相同 action support。

## 16. 在 verl 中接入 Mask Replay 的建议链路

完整接入至少需要：

1. vLLM engine 开启 `return_sampling_mask`，并保持 `logprobs_mode=processed_logprobs`。
2. vLLM rollout server 从 `CompletionOutput.sampling_mask` 取出 per-token support。
3. `TokenOutput`、`AgentLoopOutput` 与 `DataProto` 增加 sampling mask 字段。
4. multi-turn 场景将 mask 与模型生成 token 对齐；tool observation 和 padding 不应具有策略 mask。
5. 使用 CSR/packed 表示传输，避免构造 `[batch, response_length, vocab_size]` dense bool tensor。
6. FSDP、Megatron、vocab parallel 等训练后端实现 masked logsumexp 和 sampled-token logprob。
7. 分别明确 PPO ratio、rollout correction、reference KL 是否使用 replay mask，避免无差别截断所有概率项。
8. 增加一致性测试：权重相同且使用 replay mask 时，`logp_diff` 应接近数值精度允许的 0，importance ratio 应接近 1。

对 current/old 与 rollout 的 importance ratio，建议使用 rollout 固定 support：

$$
\log\pi_\theta^{\mathrm{replay}}(a_t)
=\log\operatorname{softmax}_{S_t^{\mathrm{rollout}}}
(z_{\theta,t})_{a_t}.
$$

对于 reference KL 是否应用相同 mask，应由目标函数定义决定：若目标是完整模型相对 reference 的 KL，可能仍需 full-vocabulary 语义；不能因为接入 mask replay 就自动截断所有 KL 项。

## 17. 配置建议

在 verl 尚未接通 mask replay 时：

- 若优先保证概率口径严格一致，使用 `top_p=1`、`top_k=-1`，并避免其他删除 support 的 processor。
- 保持 vLLM `logprobs_mode=processed_logprobs`；不要为了贴近训练 full-vocabulary logp 而改用与实际行为策略不一致的 `raw_logprobs`。
- 保证 rollout 与训练使用相同 temperature，并监控 `rollout_log_probs` 与重新计算 `old_log_probs` 的差值分布。
- 若必须使用截断采样，优先使用 decoupled old-policy recomputation 配合 rollout correction，并明确接受这只是缓解方案，不等价于 sampling mask replay。
- 对 bypass mode 更加谨慎，因为 full-vocabulary current logp 与 truncated rollout logp 的差异会直接进入 PPO ratio。

## 18. 参考资料

- [vLLM PR #49577: Mask Replay](https://github.com/vllm-project/vllm/pull/49577)
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/html/2512.02556)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- verl rollout 默认配置：`verl/trainer/config/rollout/rollout.yaml`
- verl vLLM 接口：`verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- verl agent-loop logprob 对齐：`verl/experimental/agent_loop/agent_loop.py`
- verl rollout correction：`verl/trainer/ppo/rollout_corr_helper.py`

