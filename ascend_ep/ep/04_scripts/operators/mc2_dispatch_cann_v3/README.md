# mc2_dispatch_cann_v3

此目录测量 CANN 上下文版 `MoeDistributeDispatchV3` / `MoeDistributeCombineV3`。Torch 入口为 `MoeDistributeBuffer.low_latency_dispatch` / `low_latency_combine`，带 `context` 时调用 ACLNN **V5**。带 `groupEp` 的 ACLNN **V3** 入口属于另一套公开接口。 [Dispatch 绑定](https://gitcode.com/cann/ops-transformer/blob/68a67e130417593f37ea3cece1ebfb4be366b95f/mc2/moe_distribute_dispatch/torch_extension/csrc/moe_distribute_dispatch.cpp) · [Combine 绑定](https://gitcode.com/cann/ops-transformer/blob/68a67e130417593f37ea3cece1ebfb4be366b95f/mc2/moe_distribute_combine/torch_extension/csrc/moe_distribute_combine.cpp)

- 参考提交为 `68a67e130417593f37ea3cece1ebfb4be366b95f`。950DT 文档支持上下文 Dispatch，Combine 仅列 Atlas A3。950DT 支持 L1 BF16/FP16 Dispatch 和 INT8 量化校验。A3 另有正向 Combine、L2 串行对照和 L3 专家前向。
- 输入 `[T,H]`、INT32 `[T,K]` 与本地连续专家分片。单个调用每 rank 最多 512 tokens。上下文和 HCCL 窗口在计时外分配，正式采样包括路由转换与实际调用。
- 950DT 通过原生 route label 核对逆映射。A3 通过专家分段、token 多重集和 Combine 回环校验。设备显式区分状态。有效计算和通信模型见 [测试方法](../../../03_benchmark/ep_benchmark_methodology.md)。
- 需要与固定源码兼容的 CANN、torch_npu 和 CANN Torch 扩展。当前没有设备执行记录。 [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_dispatch_cann_v3.benchmark --config operators/mc2_dispatch_cann_v3/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): none.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-001, MOE-002, MOE-003, MOE-004, MOE-005, MOE-006, MOE-007, MOE-008, MOE-009, LAYER-007; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: L3 replay needs a real_trace_replay workload; this configuration does not select one; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true; this case measures rank dedup; the plan selects dedup.mode=none; this case measures server dedup; the plan selects dedup.mode=none; tokens per rank exceeds adapter capacity 512.

- **Correctness contract**: dispatch is checked through native route labels (950DT) or expert segments and token multisets (A3); combine is checked as a round trip. Runtime rejections are preserved as status plus raw log.
- **Timing contract**: context and HCCL window are allocated outside the timed region; one low-latency dispatch/combine call is timed and `local_complete` recorded. A single call is limited to 512 tokens per rank.
- **Registry constraints**: `ep_sizes` unconstrained, `dtypes=[bfloat16,float16]`, `hidden_size_range=[1024,8192]`, `top_k_range=[1,16]`, `num_experts_range=[2,2048]`, `max_tokens=512`, `hardware=[Ascend 950DT dispatch, Atlas A3 dispatch/combine]`.
- **Source**: CANN ops-transformer at commit `68a67e130417593f37ea3cece1ebfb4be366b95f` (`operators/mc2_dispatch_cann_v3/backend.py`); not ascendep-vendored, and no device execution record exists.
