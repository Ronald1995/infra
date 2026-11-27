# mc2_combine_add_rmsnorm_cann

Native API: `npu_moe_distribute_combine_add_rms_norm`. Source commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d`. [Source](https://github.com/Ascend/op-plugin)

- **Adapter** · `integrated`. COMM-011 measures the fused combine, residual add and RMSNorm. DispatchV2 prepares the native handle outside the timed region. The reference checks normalized output, inverse RMS and residual sum.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1. `combine_add_rmsnorm`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按源码安装目标依赖。计划保留接入缺口。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_combine_add_rmsnorm_cann.benchmark --config operators/mc2_combine_add_rmsnorm_cann/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): none.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-001, MOE-002, MOE-003, MOE-004, MOE-005, MOE-006, MOE-007, MOE-008, MOE-009, LAYER-007; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: L3 replay needs a real_trace_replay workload; this configuration does not select one; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true; this case measures rank dedup; the plan selects dedup.mode=none; this case measures server dedup; the plan selects dedup.mode=none; tokens per rank exceeds adapter capacity 512.

- **Known gap — the case that matters cannot execute.** `COMM-011` is planned as runnable, but the backend inherits `operators.ascendep.MC2V2Backend`, whose `initialize` imports `ep.mc2_combine_add_rmsnorm_cann.demo`. That adapter does not exist in the pinned ascendep tree (21 adapters are vendored; this is not one of them), so the run ends `source_unavailable` from `operators/ascendep.py::_import_demo`.
- **Correctness contract**: `benchmarks/cases/l1_communication.py::_combine_add_rmsnorm` checks normalised output, inverse RMS and residual sum against an FP32 reference with the dtype tolerance. It cannot be reached while the import above fails.
- **Timing contract**: the DispatchV2 handle that prepares the layout is built outside the timed region; one `rmsnorm_combine` call is timed and `local_complete` is recorded.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `hidden_size=7168`, `max_tokens=512`, `equal_tokens`, `contiguous_placement`.
- **Source**: native `torch_npu.npu_moe_distribute_combine_add_rms_norm` from op-plugin `c83e36b808c4e053c9245f95c7bc45cfaae0000d`; the adapter *base class* is vendored, the adapter module is not.
