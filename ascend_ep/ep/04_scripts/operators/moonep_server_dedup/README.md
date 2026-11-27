# moonep_server_dedup

Native API: `MoonEP dev_server_dedup dispatch / combine`. Source commit `d8495c4745ba6310b40eddbdb9e3e9ef7b2c2b5d`. [Source](https://gitcode.com/Ascend/MoonEP/tree/dev_server_dedup)

- **Adapter** · `integrated`. COMM-010 checks a native dispatch/combine roundtrip. LAYER-005/006 measure router, expert forward and backward, native rank or server dedup, replica weight prefetch, and gradient reduction. Use `configs/dedup.yaml` for both modes with a two-node, 16-rank placement. Configure a physical board map for server mode.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L4. `rank_dedup`, `server_dedup`, `dedup_training`. [Capabilities](../registry.yaml)
- **Controls** · The pinned Buffer accepts only `rank` and `server` dedup modes. The two-node suite includes a fixed-owner HCCL path for the uncompressed L4 control. Payload estimates use the original home-expert routing and report one Dispatch.
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.moonep_server_dedup.benchmark --config operators/moonep_server_dedup/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): none.
- **Conditional cases** (runnable only under a matching configuration): COMM-010 (L1 独立通信): only when dedup={'mode': 'server'}; LAYER-005 (L4 路由器/规划/去重): only when dedup={'mode': 'rank'}; LAYER-006 (L4 路由器/规划/去重): only when dedup={'mode': 'server'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-001, MOE-002, MOE-003, MOE-004, MOE-005, MOE-006, MOE-007, MOE-008, MOE-009, LAYER-007; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-008, LAYER-009.
- **Why not**: L3 replay needs a real_trace_replay workload; this configuration does not select one; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline; operation requires combine, dispatch ….

- **Correctness contract**: LAYER-005/006 validate the full training step with `validate_moe(use_router=True)` (input gradient, expert weight gradients) and add the planner/dedup fields; COMM-010 checks the native dispatch/combine round trip.
- **Timing contract**: `fresh_plan` re-plans every step, `cached_plan` reuses a plan built outside the timer; replica weight prefetch and gradient reduction are reported as `stage_ns` (`benchmarks/cases/l4_router_planner.py::_planned`).
- **Registry constraints**: `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `min_tokens=1`; `dedup.mode` must be `rank` or `server` (the case rule `layer_dedup_mode` reports the other mode `not_applicable`), and server mode needs an explicit rank-to-server map.
- **Source**: MoonEP `dev_server_dedup` at commit `d8495c4745ba6310b40eddbdb9e3e9ef7b2c2b5d`; not ascendep-vendored.
