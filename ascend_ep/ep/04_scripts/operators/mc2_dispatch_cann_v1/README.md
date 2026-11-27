# mc2_dispatch_cann_v1

Native API: `npu_moe_distribute_dispatch / npu_moe_distribute_combine`. Source commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d`. [Source](https://github.com/Ascend/op-plugin)

- **Adapter** · `integrated`. V1 public bindings. The opaque reverse route stays in the native handle; correctness checks per-expert token multisets and weighted combine.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `dispatch`, `combine`, `moe_forward`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_dispatch_cann_v1.benchmark --config operators/mc2_dispatch_cann_v1/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): none.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-001, MOE-002, MOE-003, MOE-004, MOE-005, MOE-006, MOE-007, MOE-008, MOE-009, LAYER-007; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: L3 replay needs a real_trace_replay workload; this configuration does not select one; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true; this case measures rank dedup; the plan selects dedup.mode=none; this case measures server dedup; the plan selects dedup.mode=none; tokens per rank exceeds adapter capacity 512.

- **Correctness contract**: `dispatch` returns an opaque reverse-route handle, so `validate_dispatch` checks per-expert token multisets and expert boundaries rather than row order; combine is checked with the weighted reference.
- **Timing contract**: L1 handlers submit and wait inside the timed block and record `local_complete`; the V1 group is created in `initialize` (`MC2V1Backend`), outside the timer.
- **Registry constraints**: `ep_sizes=[2,4,8,16,32,64,128,144,256,288]`, `dtypes=[bfloat16,float16]`, `hidden_size=7168`, `max_tokens=512`, `equal_tokens`, `contiguous_placement`.
- **Source**: op-plugin commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d` (`operators/cann_special.py::MC2V1Backend`); not ascendep-vendored.
