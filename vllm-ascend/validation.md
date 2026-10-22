# MoE 专家数 256 → 16（topk 不变）对 vLLM-Ascend 推理特性与 RL「以小验大」的影响

> 结论基于 **远程 Ascend A3（910C）实机测试** + 代码约束核对，不是纯代码分析。
> 测试环境：`liurong_vllm` 容器，8 die（0,1,2,4,5,6,7,8），CANN 9.1.0，torch_npu 2.10.0，
> vLLM 0.26.0+empty，vllm-ascend `origin/main` @ `5591facb3`，工作树 `/home/liurong/ws/moe-ep8`。

---

## 0. TL;DR（先看这 6 条）

1. **同 topk、同 EP 度下，16 专家并不比 256 专家快。** 实测 decode 1550–1556 tok/s（16E）vs 1619–1629 tok/s（256E），prefill+decode 3150–3334 vs 3165–3252 tok/s，差异 <5%。原因很直接：topk 不变 ⇒ 每 token 激活的专家数、每 rank 收到的 token 数都不变，计算和通信量不变；裁剪只省权重显存（30B 模型 61 GB → 9.7 GB）。
   **因此「以小验大」不能用来验证吞吐/时延，只能验证算法与显存可行性。**
2. **16 专家时 EP 上限就是 16**（`num_experts % ep_size == 0` 且每 rank ≥1 专家）。256 专家常见的 32/64 卡大 EP、层次化通信、跨机 PD 分离，在 16 专家上**根本无法构造**，这些特性无法被「以小验大」覆盖。
3. **EPLB 仍能跑，但冗余粒度 = EP size**：16E/8 卡最少要加 8 个冗余专家（+50% 专家权重）；16E/16 卡要 +16（+100%）。256E/8 卡只需 +8（+3.1%）。实测 16E+8 冗余 EPLB warm-up 成功，但收益空间被「topk/num_experts=50% 天然均衡 + 每 rank 仅 2–3 个专家」大幅压缩。
4. **MC2 / FUSED_MC2（MegaMoe）在 16E/8 卡（2 专家/rank）可用**，且 FUSED_MC2 与 MC2 输出逐 token 完全一致（hash 相同）。但 256E/8 卡（32 专家/rank）在 `gpu_memory_utilization=0.6` 下开 FUSED_MC2 直接 **OOM（要再分配 4.35 GB 对称缓冲）**——MegaMoe 的显存开销随本地专家数增长，16E 反而更容易开。
5. **精度断崖**：Qwen3-30B-A3B（128 专家，topk 8）实测 PPL：完整 128E = **8.18**；裁到 64E = **20.13**；裁到 16E（不蒸馏）= **2416.66**（295×）。这说明「128→16、topk 不变」原始敏感度极高，蒸馏要补的缺口非常大；蒸馏后的 16E 是否还能代表 256E 的行为，必须单独验证。
6. **RL 生命周期有实打实的坑**：`sleep(level=2)` 需要 `enable_sleep_mode=True`（否则 expandable_segments 断言直接失败）；开启后 sleep 11.4 s / wake 3.5 s、释放 54.65 GiB，但**同一进程内 sleep 前两次 greedy 输出完全一致，wake 后 greedy 输出变了**（`same_before=true, same_after=false`）。对 on-policy RL 来说这是策略漂移/可复现性风险。

---

## 1. 测试方法

### 1.1 合成 MoE（控制变量，测特性与吞吐）
用 `Qwen3MoeForCausalLM` 生成 4 层、hidden 1024、moe_intermediate 512、topk=8 的随机权重模型（`--load-format dummy`），只改 `num_experts`：

- `qwen3moe-16e` / `qwen3moe-64e` / `qwen3moe-256e`
- 用例：TP=1/2/4/8 + `--enable-expert-parallel`、`enable_fused_mc2=1`、`--enable-eplb`+冗余、ACL graph、`quantization=ascend`。
- 吞吐数字全部带 **warmup**（先跑 8 条预热，再计时），否则会把首次 TBE/kernel 编译的 ~5 s 算进去，得到“16E 比 256E 慢 10 倍”的假象（我们第一轮就踩了这个坑，见 §3.1 注）。

### 1.2 真实模型（测精度与 RL 生命周期）
- 模型：`Qwen/Qwen3-30B-A3B`（48 层、128 路由专家、topk 8、moe_intermediate 768）。
- 裁剪：按每层 router 行 L2 范数取 top-K（K=64、16），同步切 `mlp.gate.weight` 行、专家权重按原顺序重编号；**不做蒸馏**。产物 30 GB / 9.7 GB。
- PPL：8 184 token 的仓库文档语料（英中混合），`prompt_logprobs` 逐 token NLL。
- sleep/wake：`llm.sleep(level=2)` → `wake_up()`，比较前后 greedy 输出 hash，并在 sleep 前跑两次做确定性对照。

### 1.3 关键约束（代码核对）
| 约束 | 位置 | 对 16 专家的含义 |
|---|---|---|
| `num_experts % ep_size == 0` | vLLM `determine_expert_map` / 实测日志 `Local/global number of experts: 2/16` | EP ≤ 16，且只能整除 |
| EPLB 冗余 `(n_expert+n_redundant) % ep_size == 0` | `vllm_ascend/eplb/core/eplb_utils.py:46-56` | 16E/8 卡最少 +8；16E/16 卡 +16 |
| MC2 密度策略（A2）`ep≥16 且 experts/device≤24` | `ascend_forward_context.py:304-315` | A2 上 16E 想用 MC2 必须 EP=16（1 专家/rank） |
| 1 专家/rank 特殊路径（`torch.npu.synchronize()`、跳过 permute） | `ops/fused_moe/token_dispatcher.py:644-651, 670, 705` | EP=16 时走低效/特殊分支（本次 8 die 无法实测） |
| MegaMoe `num_experts % ep == 0`、hidden/moe_inter 白名单、topk≤32 | `ascend_config.py:980-1000` | 16E/8 卡满足；EP=16 时 1 专家/rank 未验证 |
| MXFP/FP4 仅 A5 | `device/device_op.py:218` 等 | V4 的 `expert_dtype=fp4` 在 A3 上不能原生跑，只能 w8a8/w4a8 |
| 层次化通信 ≤512 专家 | `ascend_config.py:845-853` | 16E 不受上限约束，但 EP 规模先被 16 卡死 |

---

## 2. 实测结果

### 2.1 吞吐：16E vs 256E（同 topk=8、同 TP=8/EP=8）

| 用例 | 专家/rank | 输出 tok/s（warm） | 备注 |
|---|---|---|---|
| 16E decode（64 req × 32 tok） | 2 | **1550.6 / 1556.5** | 两次重复 |
| 256E decode（同上） | 32 | **1619.7 / 1629.0** | 两次重复 |
| 16E prefill+decode（512 req × 16+8 tok） | 2 | **3334.3 / 3150.5** | 两次重复 |
| 256E prefill+decode（同上） | 32 | **3252.0 / 3165.6** | 两次重复 |
| 16E TP=1（无 EP，参考） | 16 | 373（16 req × 8 tok，冷） | 单卡 |

> 注：未 warmup 时曾测到 16E 25.8 tok/s、256E 242.1 tok/s（9×），以及 16E a2a 525 vs 256E 3178 tok/s（6×）。加 warmup 后差异消失——**首次执行的算子编译/预热约 5 s**，在 RL 首个 rollout、sleep 后首个 batch 上都会出现，不能当成稳态性能。

**解释**：topk 不变时，每 token 激活的专家数相同；每 rank 收到的 token 数 = N·topk/ep，也与专家总数无关。裁剪减少的是“总权重/显存”，不是“每 token 计算量/通信量”。所以 16E 的收益是：
- 权重 61 GB → 9.7 GB（6.3×），同卡可放更大 KV / 更大 batch；
- 可以塞进更少卡、更小集群；
- 但**不会让单 token 更快**，也不能用它的 tok/s 去外推 256E。

### 2.2 特性可用性矩阵（NPU 实测）

| 特性 | 16E（2 专家/rank） | 256E（32 专家/rank） | 结论 |
|---|---|---|---|
| TP=1 无 EP | ✅ 373 tok/s | ✅（未单卡跑，权重 3.6 GB 可） | 基线 |
| TP=8 + EP（MC2 路径） | ✅ 日志 `method=MoECommType.MC2` | ✅ | 16E 可用 |
| `enable_fused_mc2=1`（MegaMoe/dispatch_ffn_combine） | ✅ 输出 hash 与 MC2 完全一致 | ❌ OOM @ util 0.6（+4.35 GB 对称缓冲） | 16E 反而更容易开；256E 需更高 util/更多卡 |
| EPLB（`num_redundant_experts=8`） | ✅ warm-up 完成，3 专家/rank | 未跑（理论 +8 = 3.1%） | 16E 需 +50% 冗余 |
| ACL graph（`enforce_eager=False`） | ✅ 173 tok/s（冷），capture 使 load +46 s | 未跑 | 16E 可用 |
| w8a8 量化（`quantization=ascend`） | ❌ 需要 ModelSlim 的 `quant_model_description.json`；dummy 权重无法验证 | 同 | **本次未验证**，不能下结论 |
| EP=16（1 专家/rank） | 未测（芯片不足 16 die） | — | 代码有特殊低效分支，风险高 |
| 大 EP（32/64 卡）/ PD 分离 | ❌ 构造不出来（EP≤16） | ✅ | 无法用小模型覆盖 |

### 2.3 真实模型裁剪精度（Qwen3-30B-A3B，topk 8，不蒸馏）

| 模型 | 专家数 | 体积 | PPL（8 184 tok） | 相对 128E |
|---|---|---|---|---|
| 完整 | 128 | 61 GB | **8.18** | 1.00× |
| 裁剪 | 64 | 30 GB | **20.13** | 2.46× |
| 裁剪 | 16 | 9.7 GB | **2416.66** | 295× |

- 64E 时输出还算正常（PPL 20），16E 时 greedy 输出已经退化成 `" a a a a..."`、`"ab it it it..."` 这类重复串。
- 这是**最坏基线**（按 router 范数选专家、零蒸馏、零校准）；你们的流程有蒸馏，绝对值会好很多。但它说明：**128→16 的原始信息损失极大，蒸馏不是“微调一下”，而是要重建大部分能力**；16E 蒸馏模型与 256E 的差距/行为差异必须实测，不能假设“基本精度”自动成立。

### 2.4 RL 生命周期：sleep / wake（16E 裁剪模型，TP=1，`rl_config.enabled=true`）

| 指标 | 实测 |
|---|---|
| sleep 前空余 HBM | 6.11 GiB |
| sleep 后空余 HBM | 60.76 GiB |
| 释放 | **54.65 GiB** |
| sleep 耗时 | **11.44 s** |
| wake 耗时 | **3.46–3.76 s** |
| sleep 前两次 greedy | hash 相同（`same_before=true`） |
| wake 后 greedy | **hash 变化（`same_after=false`）** |

- 没开 `enable_sleep_mode=True` 时：`AssertionError: Expandable segments are not compatible with memory pool`（`platform.py:1323` 只在非 sleep 模式设 expandable_segments）。
- wake 后输出变化说明**状态没有完全恢复**（权重/缓冲/随机性至少有一项漂移）。对 RL 来说，这意味着「sleep→wake 后继续 rollout」与「不 sleep 连续 rollout」不是同一个策略，on-policy 数据会混入这种漂移；需要进一步定位（我们只做了单次复现，但 pre-sleep 确定性对照已排除“本来就不确定”）。

---

## 3. 对 vLLM-Ascend：哪些推理特性“用不了/退化”

### 3.1 直接不可用（被专家数卡死）
1. **大 EP / 多机 EP / 层次化通信 / 大规模 PD 分离**
   16 专家 ⇒ EP ≤ 16（单机 8 卡最多 2 专家/rank，16 卡 1 专家/rank）。32/64 卡的 EP、跨机 all-to-all、`mc2_comm_alg=hierarchy`（为多机设计）、1P:3D 之类的大规模 PD 拓扑，在 16E 上**无法构造**，因此这些路径的 bug/性能特性不会被「以小验大」发现。
2. **EPLB 的“细粒度”负载均衡**
   冗余专家数必须是 EP 的整数倍：16E/8 卡 +8（+50%），16E/16 卡 +16（+100%）。同时 topk/num_experts=8/16=50%，每 rank 只有 2–3 个专家，单个热专家就占 rank 负载的 1/2–1/3，EPLB 的“复制热专家”粒度极粗；而 256E/8 卡是 32 专家/rank、+3.1% 冗余，EPLB 才有发挥空间。**结论：16E 上 EPLB 只能验证“能启动”，验证不了收益与大部分调度逻辑。**
3. **1 专家/rank（EP=16）路径**
   `token_dispatcher.py` 在 `num_local_experts<=1` 时走特殊分支：`_preprocess` 里 `torch.npu.synchronize()`（device 同步）、`_dispatch_postprocess` 返回 `None` 映射、`_combine_preprocess` 跳过 unpermute。也就是说 EP=16 时 MoE 通信/重排走的是**未优化路径**，且 `expert_ids_per_ep_rank` 不构建。本次 8 die 无法实测，但代码上这是明确的退化点。
4. **MTP / 投机推理**
   V4 的 MTP 层本身也是 MoE（`num_nextn_predict_layers=1`，`mtp.*` 权重）。裁 256→16 必须把 MTP 层一起裁并保持 expert id 一致，否则权重形状/路由对不上；即使裁了，MTP 的接受率依赖专家容量，16E 上测到的接受率/加速比不能外推到 256E。
5. **量化组合**
   - V4 的 `expert_dtype=fp4` 在 A3 上不原生支持（`MXFP MoE quantization is only supported on Ascend A5`），只能走 w8a8/w4a8 反量化/转换路径；
   - EPLB 明确拒绝 `W4A8MXFP`，且 `enable_fused_mc2=1` 与部分量化布局互斥（`eplb/adaptor/vllm_adaptor.py:141-144`）；
   - 我们这次**没能验证** w8a8/w4a8 在 16E 上的数值/性能（需要 ModelSlim 量化 checkpoint），这是报告里的已知空白。

### 3.2 能跑但退化/需要额外条件
1. **MC2**：16E/8 卡（2 专家/rank）在 A3 上实测可用，日志确认 `method=MC2`。但 A2（910B）的策略是 `ep≥16 且 experts/device≤24` 才选 MC2，16E 在 A2 上想开 MC2 就必须 EP=16 → 落到 1 专家/rank 特殊路径，等于“为了开 MC2 而进入低效分支”；A3（本次实测，910C）按 token 数选择，EP=8、2 专家/rank 即可，二者要分开看。
2. **FUSED_MC2 / MegaMoe**：16E 可开且数值与 MC2 一致；256E 在 0.6 util 下 OOM（对称缓冲随本地专家数增长）。MegaMoe 的设计收益在“多专家/大 EP”，16E 上开它主要是验证“能跑”，不是验证收益。
3. **ACL graph**：16E/8 卡可 capture（load +46 s，稳态 173 tok/s），但 16E 的图捕获尺寸/复用率和 256E 不同，图相关 bug 不能互推。
4. **Sleep mode**：能跑，但 wake 后输出漂移（§2.4）；且必须 `enable_sleep_mode=True`，否则 expandable_segments 断言失败。RL 若依赖 level-2 sleep 做显存腾挪，需要把“wake 后一致性”纳入验收。
5. **共享专家 / shared-expert DP / multistream overlap**：V4 有 1 个 shared expert，路由专家从 256 降到 16 后 shared 部分占比显著上升，`enable_shared_expert_dp`、`multistream_overlap_shared_expert` 的收益结构会变；这些是“比例变化”，不是“不可用”。
6. **Group-limited routing**：V4 配置 `n_group=None`，不适用；但如果拿 V3/GLM 系（`n_group=8, topk_group=4`）做 16E 裁剪，16/8=2 专家/组、topk=8 会选满 4 组×2=8 个候选，**组路由退化成“全选”**，等于该特性失效。
7. **Hash routing（V4 前 3 层）**：V4 配置 `num_hash_layers=3`，前 3 层按 token id 查表（`gate.tid2eid`）直接选专家，不走可学习 router（`vllm/models/deepseek_v4/*/model.py` 中 `is_hash_moe = layer_index < num_hash_layers`）。专家从 256 裁到 16 后，这张 token→expert 表必须整体重建并重新蒸馏这 3 层，否则前 3 层输出直接错；这类“结构耦合”是纯 16E 实验最容易漏掉的部分。

### 3.3 仍然完好的部分
- TP/EP 的基础 all-to-all/allgather/MC2 路径、ACL graph、EPLB 启动、FUSED_MC2（16E）、dummy/真实权重加载、PPL/生成接口都正常。
- 16E 的显存收益是实打实的：30B 模型 61 GB → 9.7 GB，同卡 KV 容量/并发可以大幅提升。

---

## 4. RL「以小验大」的功能风险

1. **权重同步拓扑不一致**
   rollout 是 16 专家，trainer 是 256 专家：`weight_transfer`（HCCL/NPU IPC，`vllm_ascend/distributed/weight_transfer/`）按参数名/shape 打包传输，专家数、每 rank 本地专家数、EPLB 物理专家布局都不同。**不能把 256E trainer 的权重直接推给 16E rollout**，必须让 trainer 也加载同一份裁剪+蒸馏 checkpoint（含 `mtp.*`），并保持 expert id 映射一致。否则「以小验大」验证的 weight-sync 路径和线上不是同一条。
2. **EPLB 在 16E 上验证不到**
   16E 即使开 EPLB 也只是“能启动”（我们实测 warm-up 成功），冗余专家放置、D2D 热迁移、`log2phy` 重映射、32+ rank 的调度阈值（`policy_flashlb` 对 `world_size<32` 有特殊阈值）都不会被触发。线上 256E 的 EPLB bug 会在小模型实验里“隐身”。
3. **Sleep/Wake 生命周期**
   - 必须 `enable_sleep_mode=True`，否则直接失败；
   - level-2 sleep 11.4 s（16E/9.7 GB），256E/61 GB 会更久，RL 的 step time 结构不同；
   - **wake 后 greedy 输出漂移**（实测），on-policy 数据一致性存疑。
4. **Training consistency / batch invariance**
   `rl_config.enable_training_consistency` 决定是否启用 FA3；`enable_batch_invariant` 设 `VLLM_BATCH_INVARIANT=1`。16E 与 256E 选择的 MoE 通信/算子（MC2 vs ALLTOALL vs FUSED_MC2）不同，**bitwise 对齐训练器的难度和结论都不同**；小模型上“对上了”不代表 256E 对得上。
5. **MTP / 投机解码**
   16E 的 MTP 接受率、draft 长度收益不能外推；如果裁剪时漏掉 MTP 层还会直接 shape mismatch。
6. **PD 分离 / 大规模部署**
   EP≤16 决定了很多 PD 拓扑根本搭不出来；相关功能（KV 传输、跨机调度）无法用小模型验证。
7. **性能外推无效（最重要）**
   同 topk 下 16E 与 256E 的稳态吞吐基本相同（实测 <5%），但**显存/并发能力差 6×**。所以：
   - 用 16E 测“RL rollout 吞吐/时延”会得到与 256E 几乎一样的数（好消息：算法实验的 step time 可参考）；
   - 但用 16E 测“最大并发/batch/KV 容量/显存水位”会严重高估 256E 的能力；
   - 首次 rollout/sleep 后首个 batch 的 ~5 s 编译/预热在两种规模都存在，别把它当成规模差异。
8. **冷启动与缓存**
   我们实测到未 warmup 时 16E 25.8 tok/s、256E 242 tok/s 的假差异，全部来自首次算子编译。RL 的 benchmark 脚本必须先 warmup，否则“以小验大”的结论会被冷启动噪声支配。

---

## 5. RL「以小验大」的精度风险

1. **裁剪本身的信息损失是断崖式的**
   实测 128E→64E：PPL 8.18→20.13（2.5×）；128E→16E（不蒸馏）：PPL 8.18→2416.66（295×），生成已退化。蒸馏要补的不是“几个点”，而是绝大部分能力。**“蒸馏后基本精度”必须用同一套 eval 在 16E 和 256E 上分别测，不能只看 16E 的绝对分数。**
2. **专家密度从 3.1% 变成 50%**
   256E/topk8：每 token 用 3.1% 的专家；16E/topk8：用 50%。后果：
   - 路由熵、负载分布、专家 specialization 完全不同；
   - 16E 上每个专家承担 N/2 token，热专家粒度 = rank 的 1/2，负载不均更尖锐；
   - RL 的 entropy bonus / KL / capacity 相关超参在 50% 密度下调好，搬到 3.1% 密度大概率不最优。
3. **蒸馏目标的偏差**
   蒸馏通常对齐 teacher 的 logits/top-k 输出，但**路由分布**（哪些 token 去哪些专家）不一定对齐；16E 的 router 可能学成“近似 uniform”，而 256E 是高度稀疏的。RL 会放大这种差异（reward 对 routing 的敏感度不同）。
4. **MTP 接受率下降**
   MTP 层容量随专家数下降，draft 接受率降低，投机解码的加速比在 16E 上被高估；如果 RL 用 MTP 加速 rollout，实测 step time 不能外推。
5. **量化链路不一致**
   256E 线上可能是 fp4/w4a8，16E 实验为了跑通常用 bf16/w8a8；数值误差、溢出行为不同，精度结论不可直接搬。
6. **Sleep/Wake 漂移**
   实测 wake 后输出变化；如果 RL 在 sleep/wake 之间继续用旧 KV/旧策略，reward 与 logprob 会不可复现。
7. **奖励黑客与容量上限**
   16E 容量小，更容易被 reward hack（输出退化但 reward 高）；在 16E 上“有效”的 reward shaping/数据配比，到 256E 可能无效或反效。
8. **评测饱和/污染**
   16E 蒸馏模型可能在常用 benchmark 上接近 teacher（因为 benchmark 简单/被蒸馏数据覆盖），但在长尾/长上下文/多语言上差距大；RL 训练分布一旦偏移，差距会暴露。

---

## 6. 建议（怎么把「以小验大」做对）

1. **明确“验什么”**：
   - 用 16E 验：算法、reward、数据配比、训练稳定性、显存可行性；
   - 不用 16E 验：吞吐/时延上限、最大并发、EPLB 收益、大 EP/PD、MTP 加速比、量化数值。
2. **16E 的部署约束写死**：EP ≤ 8（保持 ≥2 专家/rank），不要用 EP=16 的 1 专家/rank 路径；EPLB 若开，冗余按 `k*ep` 配（8 卡 → +8）。
3. **权重同步同源**：trainer 与 rollout 用同一份裁剪+蒸馏 checkpoint（含 `mtp.*`），expert id 映射固定；不要在 256E trainer 与 16E rollout 之间直接同步。
4. **精度验收双跑**：同一 eval suite 同时跑 16E 蒸馏模型与 256E，记录 PPL/任务分数、路由分布（每专家负载 max/mean、Gini）、MTP 接受率；差异超过阈值就不要用 16E 的 RL 结论外推。
5. **sleep/wake 一致性纳入 CI**：sleep 前后 greedy/logprob 对比；当前实测不一致，建议先定位再用于生产 RL。
6. **benchmark 必 warmup**：至少 1 个完整 batch 预热后再计时；RL 的首个 rollout 单独统计。
7. **量化单独验证**：用 ModelSlim 量化后的 16E checkpoint 跑 w8a8/w4a8，确认 16 专家下 kernel/精度都 OK（本次因 dummy 权重无法验证）。

---

## 7. 复现与产物

- 工作树：`/home/liurong/ws/moe-ep8`（`origin/main` @ `5591facb3`）
- 合成模型：`/home/liurong/ws/moe-scale/models/qwen3moe-{16e,64e,256e}`
- 真实模型：`.../Qwen3-30B-A3B`、`...-pruned64`、`...-pruned16`
- 结果：
  - `results/summary.txt`（特性矩阵）
  - `results/perf_summary.txt`（warmup 后的 16E/256E 吞吐）
  - `results/ppl_full.log` / `ppl_pruned64.log` / `ppl_pruned_dev9.log`（PPL 8.18 / 20.13 / 2416.66）
  - `results/sleep_rl2.log`（sleep/wake 与 hash 对比）
  - `results/e16_tp8_dbg.log`（`MoE comm method selected: ... MC2`）
  - `results/e256_tp8_fmc2.log`（MegaMoe OOM）
- 关键命令示例：
  ```bash
  source /home/liurong/ws/moe-ep8.env
  export PYTHONPATH=/vllm-workspace/vllm:$NPU_SLOT_WORKTREE:$PYTHONPATH
  cd "$NPU_SLOT_WORKTREE"
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,4,5,6,7,8 CASE_WARM=1 \
    CASE_MODEL_DIR=.../qwen3moe-16e CASE_TP=8 CASE_EP=1 CASE_N=64 CASE_MAXTOK=32 \
    python run_case.py
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,4,5,6,7,8 MODEL=.../Qwen3-30B-A3B TP=8 MODE=ppl \
    python real_eval.py
  ```

### 未覆盖/待补
- EP=16（1 专家/rank）实测（需 16 die）
- w8a8/w4a8 量化后的 16E 数值与性能
- 真蒸馏模型（本次只做了无蒸馏最坏基线）
- MTP/投机解码接受率（需 V4 fp4 + A5，或可跑的 MTP checkpoint）
- 真实 trainer 的 weight-transfer 全链路