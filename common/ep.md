这次昇腾 EP 调研涉及的仓库和分支如下。表里保留了源码入口、主要实现、硬件范围和公开验证情况，方便按 **EP64/128 纯通信、MoonEP 负载均衡、MegaMoE 通算融合、训练侧 dispatcher 组合** 四个方向查找；文末另有一张跨章节去重的 EP 算子汇总表。

## EP 通信库与 MoonEP 移植

| 仓库地址 | 硬件与定位 | 基本实现内容 | 关键分支、PR 或公开验证 |
|---|---|---|---|
| [deepseek-ai/DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) | **Ascend 950DT**，DeepSeek 官方通信库 | `EPBuffer` 提供 MoE Dispatch/Combine，支持 FP8 Dispatch、BF16 Combine 和延迟 epilogue。Ascend C 内核通过 DeepJIT 编译，通信使用 HCCL/HCOMM、UBMEM、URMA。 | README 公布 950DT PoC 的 **EP8/16/32/64/128** 实测。[#12](https://github.com/deepseek-ai/DeepEP-Ascend/pull/12) 提议在 EP128、每 rank BS ≤ 32 时走 MTE 直写，仍开放。 [GitHub](https://github.com/deepseek-ai/DeepEP-Ascend/blob/main/README.md?utm_source=chatgpt.com) |
| [Ascend/DeepEP](https://gitcode.com/Ascend/DeepEP) | **Ascend 950**，昇腾侧 DeepEP V2 推理实现 | `ElasticBuffer` 统一高吞吐 Prefill 与低延迟 Decode 的 Dispatch/Combine，使用 CANN SHMEM，并包含实验性 W4A8 MegaMoE。 | 重点 MR 为 `!18` ElasticBuffer、`!25` FP8 Peer Table Dispatch、`!33` MegaMoE 推理。主仓本次访问受限，实现说明已通过[公开分支副本](https://gitcode.com/lishaoxun/DeepEP_9283)交叉核对。 [AtomGit](https://gitcode.com/lishaoxun/DeepEP_9283?utm_source=chatgpt.com) |
| [lueelu/DeepEP](https://github.com/lueelu/DeepEP) | **Ascend 950**，DeepEP V2 与 MegaMoE 实验 | 基于 Ascend C、CANN SHMEM 提供高吞吐和低延迟 Dispatch/Combine，`experiments/megamoe` 使用 CATLASS，将 Dispatch、两次 GMM、SwiGLU 和 Combine 融合。 | README 列出 W4A8 Prefill EP32/64/128、Decode EP64/128 的实验配置。[#10](https://github.com/lueelu/DeepEP/pull/10) 增加 BF16 融合推理和训练，PR 报告了 EP8/16/32 前反向验证，仍开放。 [lueelu/DeepEP · GitHub](https://github.com/lueelu/DeepEP?utm_source=chatgpt.com) |
| [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) | **A2 / A3 / A5**，SGLang NPU 算子库 | [`python/deep_ep`](https://github.com/sgl-project/sgl-kernel-npu/tree/main/python/deep_ep) 提供 Normal、Low-Latency 两套 EP 接口，按芯片和模式使用自定义 Ascend C、`torch.distributed` All-to-All 或 CANN 算子。FuseEP 暴露 FusedDeepMoe、DispatchFFNCombine 和 CANN MegaMoE。 | [#873](https://github.com/sgl-project/sgl-kernel-npu/pull/873) 的 A5 MegaMoE 接入与 [#986](https://github.com/sgl-project/sgl-kernel-npu/pull/986) 的 A2 INT8 Dispatch 已合入。[#963](https://github.com/sgl-project/sgl-kernel-npu/pull/963) ZeRCMoE 仍为实验 PR。 [GitHub](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/doc/A2_DEEPEP.md?utm_source=chatgpt.com) |
| [tianxinghui/ascend_deepep](https://gitcode.com/tianxinghui/ascend_deepep) | 昇腾节点内，DeepEP V1 兼容实现 | 兼容 V1 节点内的布局计算、Dispatch、Combine 和通信 Buffer，另有面向专家 GMM 的布局转换接口。 | `deepep_v1` 是已核对的公开分支。另一个公开入口为 [yan_yw/ascend_deepep](https://gitcode.com/yan_yw/ascend_deepep)。 [AtomGit](https://gitcode.com/tianxinghui/ascend_deepep?utm_source=chatgpt.com) |
| [Ascend/MoonEP](https://gitcode.com/Ascend/MoonEP) | **Ascend 950 / A5**，MoonEP NPU 后端 | `npu_patch` 基于 Ascend C 和 CANN SHMEM 对称内存，提供动态冗余专家规划、Dispatch/Combine、权重预取和梯度归并。数据搬运支持 MTE 和 URMA/UDMA。 | [`dev_npu`](https://gitcode.com/Ascend/MoonEP/tree/dev_npu) 是 NPU 适配入口。[`dev_server_dedup`](https://gitcode.com/Ascend/MoonEP/tree/dev_server_dedup) 提供 Rank 和 Server 两种去重模式，Server 模式以连续 8 个 rank 为一个去重域，使用专用 kernel 和 workspace。 |
| [chenjingsheng/MoonEP-Ascend](https://gitcode.com/chenjingsheng/MoonEP-Ascend) | **A2 / A3**，MoonEP 训练移植 | Ascend C 实现路由规划、Dispatch/Combine、权重预取和梯度归并。对称堆上的跨 rank 数据搬运走 SHMEM MTE GM2GM，HCCL 用于同步。 | 依赖该项目配套的 [chenjingsheng/shmem](https://gitcode.com/chenjingsheng/shmem)。 [AtomGit](https://gitcode.com/chenjingsheng/MoonEP-Ascend?utm_source=chatgpt.com) |
| [jianzhnie/MoonEP-Ascend-910C](https://github.com/jianzhnie/MoonEP-Ascend-910C) | **910C / A3**，MoonEP 社区移植 | 在 Host 侧做负载均衡规划，通过对称内存和 MTE 执行 Dispatch/Combine，并提供权重预取和梯度归并。 | 公开验证主要覆盖单机多卡，仓库提供测试和 benchmark。 [GitHub](https://github.com/jianzhnie/MoonEP-Ascend-910C?utm_source=chatgpt.com) |
| [gcw_rYW0Lkr2/Ascend_moonep_v3](https://gitcode.com/gcw_rYW0Lkr2/Ascend_moonep_v3) | **Ascend 950 / A5**，MoonEP V3 | 支持二级规划、紧凑比例路由、各 rank 不等 token 数和源端去重。数据面支持 MTE 与 URMA/UDMA 两条路径。 | 仓库提供单机和多机运行说明。现有融合加权 Combine 与部分源端去重配置互斥。 [AtomGit](https://gitcode.com/gcw_rYW0Lkr2/Ascend_moonep_v3?utm_source=chatgpt.com) |
| [lonelylone/MoonEP-Ascend](https://gitcode.com/lonelylone/MoonEP-Ascend) | **A2 / A3 移植方向**，MoonEP RoCE 版本 | 使用 cann-shmem 的 RoCE 引擎、Ascend C 和 HCCL，实现贪心规划、Dispatch/Combine 和 rank 同步。 | `prefetch_weight` 和 `reduce_grad` 在 README 中标为演示接口。 [AtomGit](https://gitcode.com/lonelylone/MoonEP-Ascend?utm_source=chatgpt.com) |

## CANN 原生算子、通算融合和框架集成

| 仓库地址 | 定位 | 基本实现内容 | 源码或集成入口 |
|---|---|---|---|
| [CANN/ops-transformer](https://gitcode.com/cann/ops-transformer) | CANN 原生 EP 算子源码 | **MoeDistributeDispatch/Combine** 使用 MC2/HCCL 实现 EP All-to-AllV，可配合 TP 通信。**MoeEpDispatch/Combine** 面向 950DT，通过 HCOMM/URMA 操作对称通信窗口，支持缓存路由。**MegaMoE、DispatchFFNCombine、AlltoAllvGroupedMatMul** 等覆盖整层或单侧通算融合。 | [`mc2/`](https://gitcode.com/cann/ops-transformer/tree/master/mc2) 和 [MegaMoE 源码文档](https://gitcode.com/cann/ops-transformer/blob/master/mc2/mega_moe/README.md)。硬件覆盖按具体算子和版本确认。 [AtomGit](https://gitcode.com/cann/ops-transformer/tree/master/mc2?utm_source=chatgpt.com) |
| [Ascend/op-plugin](https://github.com/Ascend/op-plugin) | `torch_npu` 算子绑定 | 对 PyTorch 暴露 `npu_moe_distribute_dispatch_v2`、`npu_moe_distribute_combine_v2`、`npu_moe_distribute_combine_add_rms_norm` 等接口。 | 适合核对参数、量化模式、通信组和版本约束。 [Ascend/op-plugin · GitHub](https://github.com/Ascend/op-plugin?utm_source=chatgpt.com) |
| [deepseek-ai/DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) | **Ascend 950**，DeepSeek GEMM 库与融合 MoE | MegaMoE 将 EP Dispatch、两次 Grouped GEMM、SwiGLU 和 Combine 融合执行。 | README 有 EP8 示例和性能数据。vLLM-Ascend [#20299](https://github.com/vllm-project/vllm-ascend/pull/20299) 尝试接入 DeepSeek V4.1 TP1/EP8 MXFP4，仍开放。 [GitHub](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/main/README.md?utm_source=chatgpt.com) |
| [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | 推理框架 EP 调度 | `ALLTOALL`、`ALLGATHER`、`MC2`、`FUSED_MC2` 四条主要路径，按平台、token 数和配置选择 HCCL、CANN Dispatch/Combine、DispatchFFNCombine 或 MegaMoE。 | [通信策略源码](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py)。DeepEP HT [#21178](https://github.com/vllm-project/vllm-ascend/pull/21178) 和 A5 Hierarchical MC2 [#21841](https://github.com/vllm-project/vllm-ascend/pull/21841) 仍开放。 [GitHub](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py?utm_source=chatgpt.com) |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 推理框架 EP 后端集成 | `deepep` 对接 sgl-kernel-npu 的 Normal/Low-Latency 路径，`ascend_ops` 对接昇腾原生通信，`ascend_fuseep` 走 Dispatch＋专家 FFN＋Combine 融合路径。 | 参考 [Ascend 参数文档](https://sgl-project.github.io/platforms/ascend/ascend_npu_support_features.html) 和 [MiniMax-M3 Prefill FuseEP RFC](https://github.com/sgl-project/sglang/issues/39521)。 [GitHub](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/overrides.py?utm_source=chatgpt.com) |
| [Ascend/FSDPTurbo](https://gitcode.com/Ascend/FSDPTurbo) | 昇腾 FSDP2 训练库的 EP 调度 | `ep_plan.dispatcher` 提供 `all2all`、`mc2`、`allgather`、`chunkmoe`、`domino`、`moonep` 六条内置路径和 `custom_ep_forward_eager/fused` 自定义路径，`use_fused_ops` 控制计算侧是否走 Ascend 融合算子；EP 侧不含自研 Ascend C 内核，靠算子组合与重叠调度 | [EP 文档](https://gitcode.com/Ascend/FSDPTurbo/blob/main/docs/zh/features/expert_parallel.md)、[dispatcher 源码](https://gitcode.com/Ascend/FSDPTurbo/tree/main/fsdp_turbo/distributed/expert_parallel)；`chunkmoe`、`allgather` 与 `ep_balance` 由 [`1fa40ad3`](https://gitcode.com/Ascend/FSDPTurbo/commit/1fa40ad350eb681ed413ca64bffedc166c986e0f) 一次引入，通信形态与算子清单见后两节 |
| [Ascend/MindSpeed](https://gitcode.com/Ascend/MindSpeed) | Megatron 训练加速 | 包含 MoE All-to-All Dispatcher、Token Permute/Unpermute 融合、前后向通信重叠，以及通过热点专家复制做负载均衡。 | [特性目录](https://gitcode.com/Ascend/MindSpeed/tree/dev/docs/features)。 [AtomGit](https://gitcode.com/Ascend/MindSpeed?utm_source=chatgpt.com) |
| [Ascend/MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM) | LLM 训练框架集成 | 在模型训练配置中接入 EP Dispatcher、GroupedMatMul 和 MC2，覆盖 Megatron 和 FSDP2 训练路线。 | [MC2 集成文档](https://github.com/Ascend/MindSpeed-LLM/blob/master/docs/en/pytorch/features/mcore/mc2.md)。 [Ascend/MindSpeed-LLM · GitHub](https://github.com/Ascend/MindSpeed-LLM?utm_source=chatgpt.com) |

## FSDPTurbo 的 EP dispatcher 与通信形态

`Ascend/FSDPTurbo` 是面向昇腾的 FSDP2 训练库（官方支持 950PR/950DT 与 Atlas A3/A2 训练产品），EP 侧的设计重点是**同一套 permute / GroupedMatMul / 激活算子配不同通信形态**：`distributed.ep_plan.dispatcher` 选通信与重叠方式，`use_fused_ops` 决定计算侧是否走 Ascend 融合算子（关闭后回落到 CPU eager 实现）。仓库里没有自研的 Ascend C EP 内核，`fsdp_turbo/ops/npu/` 全是 `torch_npu`、`cann_ops_nn`、`npu_patch` 的适配层，所以下面这张表只看 dispatcher 的通信形态，算子细节统一到下一节的「EP 算子汇总」里只写一次。公开 EP 示例覆盖 Qwen3.5-35B-A3B（EP+EFSDP）、Qwen3 MoE、DeepSeek V4 和 Kimi K3。

| dispatcher | 通信与重叠方式 | 约束与配套 |
|---|---|---|
| `all2all`（默认） | permute → AllToAllV → permute → GMM → 激活 → GMM → unpermute → AllToAllV → unpermute；反向按 autograd 反序 | 通用性最好；支持量化、`ep_balance`、a2a split 缓存；量化路径通过 `inject_quant_ep_callbacks` 拿 `dispatch_preprocess`/`alltoall_dispatch`/`alltoall_combine` 三个回调复用这套 A2A |
| `mc2` | 单侧通算融合，没有独立 A2A：dispatch+第一个 GEMM 一个算子，第二个 GEMM+combine 一个算子，中间只留激活 | 只支持未量化 BF16/FP16 专家，量化 MoE + `mc2` 在接线阶段直接报错；任一源 rank 无 token 时整组回退 `all2all`；950 上对应 ACLNN 算子公布的 `epWorldSize` 为 2/4/8/16/32/64；split 缓存与 device 侧 `group_list` 可省掉 host 求和与 H2D |
| `allgather` | 三份 AllGather（index / weights / hidden）→ 本地专家 mask（非本地 token 映射到 dummy 专家）→ permute → GMM → 激活 → GMM → unpermute → ReduceScatter | 每卡持有全量 token，显存随 EP 线性增长，适合 EP 较小或 A2A 不划算的场景；支持 `ep_balance` 与量化 |
| `chunkmoe` | token 按 `moe_num_chunks`/`moe_chunk_size` 切块、两块一组：组内先把两块的 A2A 都发出去，再依次等数据、算专家、发 combine，用 issue/wait 两个 autograd 边界把 A2A 与专家计算错开 | chunk 数需在 EP 组内一致（`all_reduce(MAX)`），尾部空 chunk 补零以保证各 rank 集合通信顺序一致；支持量化 |
| `domino` | 固定切 2 片：A2A 放独立 comm stream，用 event 与计算流对接，dispatch / 专家 GEMM / combine 三段重叠 | 切片重叠思路同 Domino（[arXiv:2409.15241](https://arxiv.org/abs/2409.15241)）；量化专家走模块自己的 `experts_forward`，MXFP8 随 GLM5.3 接入 |
| `moonep` | 对称内存路径：每卡固定收 `S*K` 行，规划、dispatch/combine 与冗余专家权重预取都在 CANN SHMEM 符号堆上，权重梯度 FP32 暂存后按 home rank 三段式归约 | 只支持 BF16 稠密专家、静态形状、`expert_parallel_size > 1` 且 `expert_fully_shard_parallel_size: 1`；与 MoE 量化互斥；真机验证目前只有单节点 4/8 卡 |
| `custom_ep_forward_eager` / `custom_ep_forward_fused` | 模型自定义 | 专家计算语义不是标准 GLU/SwiGLU 时使用；其余 `custom_<method>` 与 callable dispatcher 保持旧调用约定 |

三条与选型直接相关的开关：`use_fused_ops: false` 只影响计算侧，`permute`/`unpermute`/`grouped_matmul`/`swiglu` 回落 CPU eager 实现（通信仍走 HCCL）；`all2all` 与 `mc2` 的 a2a split 缓存由 `distributed.ep_plan.reuse_dispatch_splits` 控制，重计算重放和反向 wgrad 复用前向缓存；`enable_ep_balance` 只支持 `all2all` 与 `allgather`，冗余专家的参数和梯度靠点对点 send/recv 同步，不占集合通信。

## EP 算子汇总（跨三章去重）

前三章的表格按仓库/方案组织，同一个算子会在多个仓库里重复出现；这一节按算子去重，每行只在「出现位置」列标注它被哪些仓库或方案用到，`类型` 用来区分 EP 专用融合算子、被复用的 CANN 基础/量化算子、通信库接口、框架调度和底层通信设施。

| 算子 / 接口 | 类型 | 出现位置 | 说明 |
|---|---|---|---|
| `MoeDistributeDispatch` / `MoeDistributeCombine`（torch_npu：`npu_moe_distribute_dispatch_v2` / `npu_moe_distribute_combine_v2`） | EP 融合算子 | ops-transformer、op-plugin、vllm-ascend | MC2/HCCL 实现 EP All-to-AllV，可叠加 TP 通信；昇腾侧最通用的 dispatch/combine |
| `npu_moe_distribute_combine_add_rms_norm` | EP 融合算子 | op-plugin | combine 之后直接接残差加法与 RMSNorm，省一次激活读写 |
| `MoeEpDispatch` / `MoeEpCombine` | EP 融合算子 | ops-transformer（950DT） | 通过 HCOMM/URMA 操作对称通信窗口，支持缓存路由 |
| `MegaMoE` | EP 融合算子（整层） | ops-transformer、DeepGEMM-Ascend、Ascend/DeepEP（W4A8）、lueelu/DeepEP、sgl-kernel-npu、vllm-ascend | 把 Dispatch、两次 GroupedMatmul、SwiGLU 与 Combine 融成一层 |
| `DispatchFFNCombine` | EP 融合算子（整层） | ops-transformer、sgl-kernel-npu FuseEP、vllm-ascend `FUSED_MC2` | dispatch＋专家 FFN＋combine 的中间档融合 |
| `FusedDeepMoe` | EP 融合算子（整层） | sgl-kernel-npu FuseEP、sglang `ascend_fuseep` | FuseEP 暴露的融合 MoE 前向入口 |
| `AlltoAllvGroupedMatMul` | EP 融合算子（单侧，MC2） | ops-transformer、FSDPTurbo `mc2`（`npu_alltoallv_gmm`） | AllToAllV→GMM；可选共享专家 MM 与 expert-first `permuteOut`；950 另有量化版 `AlltoAllvQuantGroupedMatMul` |
| `GroupedMatMulAlltoAllv` | EP 融合算子（单侧，MC2） | ops-transformer、FSDPTurbo `mc2`（`npu_gmm_alltoallv`） | GMM→rank-first→AllToAllV，与上一个方向配对成 dispatch/GMM、GMM/combine 两段融合 |
| `GroupedMatmul`（`npu_grouped_matmul`） | CANN 基础算子 | ops-transformer、op-plugin、MindSpeed-LLM、FSDPTurbo（含 mc2 反向 wgrad）、MoonEP GMM | 专家 GEMM 与反向；FSDPTurbo 前向 `split_item=2`、wgrad `split_item=3, group_type=2`，MoonEP 用 `group_list_type=0` 累积布局 |
| `MoeTokenPermute` / `MoeTokenUnpermute`（`npu_moe_token_permute` / `npu_moe_token_unpermute`） | CANN 基础算子 | MindSpeed（Token Permute/Unpermute 融合）、FSDPTurbo 五条路径 | A2A 前后按专家重排与还原，`unpermute` 可带 `probs` 加权 |
| `SwiGLU`（`npu_swiglu`） | CANN 基础算子 | MegaMoE、FSDPTurbo | 默认专家激活；FSDPTurbo 的 `domino` 用自带 `gate/up` 规则，不调它 |
| `ClippedSwiglu` / `SituGlu`（`cann_ops_nn.clipped_swiglu(_grad)` / `situ_glu(_grad)`） | CANN 基础算子 | FSDPTurbo（DeepSeek-V4.1 clipped、Kimi Situ） | 带 `limit`/`alpha` 的 clipped SwiGLU，以及吃整块 `gate_up` 的 Situ |
| 量化专家 GMM 与量化激活 | 量化算子 | FSDPTurbo（MXFP8 / W8A16 / W4A8 / W4A16，入口 `fsdp_turbo/ops/quantize.py`） | 量化 `experts_forward` 复用 EP 的 A2A；`mc2` 与 `moonep` 不支持量化专家 |
| `EPBuffer` / `ElasticBuffer` / `deep_ep`（Normal、Low-Latency） | 通信库接口 | DeepEP-Ascend、Ascend/DeepEP、lueelu/DeepEP、tianxinghui（V1）、sgl-kernel-npu、sglang | 库级 dispatch/combine 入口而非单个 kernel；含 FP8 dispatch、BF16 combine、延迟 epilogue，ElasticBuffer 统一 Prefill/Decode |
| MoonEP 对称内存 EP 后端（`npu_patch` 与各移植的 dispatch/combine kernel） | 通信库接口 | Ascend/MoonEP、A2/A3/910C 各移植、FSDPTurbo `moonep` | 在对称内存上做规划、dispatch/combine、权重预取与梯度归并；Ascend C 实现，数据面 MTE/URMA/UDMA 或 cann-shmem RoCE；`npu_patch` 的 dispatch/combine/plan kernel 内含全 rank 栅栏 |
| `FuseEP` / `ascend_fuseep` | 通信库接口 | sgl-kernel-npu、sglang | 把 FusedDeepMoe、DispatchFFNCombine、CANN MegaMoE 暴露给框架 |
| 推理框架通信策略：`ALLTOALL` / `ALLGATHER` / `MC2` / `FUSED_MC2`（vllm-ascend）、`deepep` / `ascend_ops` / `ascend_fuseep`（sglang） | 框架调度 | vllm-ascend、sglang | 按平台、token 数在 HCCL、CANN Dispatch/Combine、DispatchFFNCombine、MegaMoE 之间选路 |
| 训练框架调度：MindSpeed EP Dispatcher（含 Token Permute 融合、热点专家复制）、MindSpeed-LLM EP Dispatcher/MC2、FSDPTurbo 六条 dispatcher + `ep_balance` | 框架调度 | MindSpeed、MindSpeed-LLM、FSDPTurbo | 决定通信形态、前后向重叠与冗余专家负载均衡；FSDPTurbo 的具体形态见上一节 |
| HCCL AllToAllV / AllGather / ReduceScatter | 底层通信 | 几乎所有方案 | `all_to_all_single`、`all_gather_into_tensor`、`reduce_scatter_tensor`；FSDPTurbo 的 A2A 都由它承载 |
| CANN SHMEM | 底层通信 | DeepEP V2、多个 MoonEP 移植、FSDPTurbo `moonep` | 对称内存与跨 rank 访问，数据面可选 MTE、RoCE 或 URMA/UDMA |

去重后全文共涉及 **20 个条目**：8 个 EP 专用融合算子（`MoeDistributeDispatch/Combine`、`combine_add_rms_norm`、`MoeEpDispatch/Combine`、`MegaMoE`、`DispatchFFNCombine`、`FusedDeepMoe`、`AlltoAllvGroupedMatMul`、`GroupedMatMulAlltoAllv`）、5 个被复用的 CANN 基础/量化算子、3 个通信库接口、2 个框架调度、2 个底层通信设施。真正在 kernel 层面新增的只有第一类；`all2all`/`allgather`/`chunkmoe`/`domino` 这些方案本身不引入新算子，差别在通信形态与重叠方式。

## 上游参考与底层依赖

| 仓库地址 | 与昇腾 EP 的关系 |
|---|---|
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | DeepEP 上游 API 和 NVIDIA 通信实现，提供 Dispatch/Combine、FP8 Dispatch 等。昇腾内核入口见 DeepEP-Ascend。 [GitHub](https://github.com/DeepSeek-ai/DeepEP?utm_source=chatgpt.com) |
| [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP) | MoonEP 上游算法和 GPU 实现，定义在线冗余专家规划、静态通信形状、权重预取和梯度归并。昇腾入口见 `Ascend/MoonEP` 的 NPU 分支。 [GitHub](https://github.com/MoonshotAI/MoonEP/blob/master/README.md?utm_source=chatgpt.com) |
| [CANN/shmem](https://gitcode.com/cann/shmem) | 提供对称内存、跨 rank 访问和底层搬运接口，是多个 MoonEP 和 DeepEP 移植的通信基础。具体数据面由各实现选择 MTE、RoCE 或 URMA/UDMA。 [AtomGit](https://gitcode.com/cann/shmem?utm_source=chatgpt.com) |
| [CANN/catlass](https://gitcode.com/cann/catlass) | 提供专家 GEMM 等设备侧计算组件，供 MegaMoE 等融合实现使用。 [lueelu/DeepEP · GitHub](https://github.com/lueelu/DeepEP?utm_source=chatgpt.com) |
| [gitcode_lijd/doc](https://gitcode.com/gitcode_lijd/doc) | 存放 MoonEP 接口、历史适配方案，以及 MC2 和 ElasticBuffer 的对比资料。 [AtomGit](https://gitcode.com/gitcode_lijd/doc?utm_source=chatgpt.com) |

**EP64/128 已公开的同口径数据**可直接参考 DeepSeek DeepEP-Ascend 的 950DT PoC。EP64 的 Dispatch / Combine 分别为 **323–327 / 294–298 GB/s**，EP128 为 **313–320 / 272–278 GB/s**。`lueelu/DeepEP` 也列出了 EP64/128 的 W4A8 MegaMoE 实验配置，衡量的是包含专家 FFN 的融合路径，适合按整层耗时比较。 [GitHub](https://github.com/deepseek-ai/DeepEP-Ascend?utm_source=chatgpt.com)