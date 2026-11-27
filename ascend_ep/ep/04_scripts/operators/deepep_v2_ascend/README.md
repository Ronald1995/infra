# deepep_v2_ascend

Native API: `deep_ep.ElasticBuffer`. Source commit `69090a120cfee06be431a79873692677bc07b042`. [Source](https://gitcode.com/lishaoxun/DeepEP_9283)

- **Adapter** · `integrated`. The public source mirror provides a distinct ElasticBuffer. This adapter exercises the high-throughput expanded BF16 path only. It records the actual EP32/64/128 selection.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `dispatch`, `combine`, `cached_dispatch`, `async_dispatch`, `async_combine`, `moe_forward`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the native wheel from the pinned DeepEP V2 public mirror in its own Python environment.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.deepep_v2_ascend.benchmark --config operators/deepep_v2_ascend/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): none.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-001, MOE-002, MOE-003, MOE-004, MOE-005, MOE-006, MOE-007, MOE-008, MOE-009, LAYER-007; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: EP8 outside adapter's supported EP sizes [32, 64, 128]; L3 replay needs a real_trace_replay workload; this configuration does not select one; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true; this case measures rank dedup; the plan selects dedup.mode=none; this case measures server dedup; the plan selects dedup.mode=none.

- **Correctness contract**: dispatch layout via `validate_dispatch`; the L3 forward via `validate_moe`. The registry has no `moe_backward`, so no gradient case is claimed.
- **Timing contract**: L1 dispatch/combine handlers submit and wait inside the timed block and record `local_complete`; the async variants wait on the adapter's deferred callback.
- **Registry constraints**: `ep_sizes=[32,64,128]` (the only adapter that lists EP64/EP128), `dtypes=[bfloat16]`, `hidden_size=7168`, `min_tokens=256`, `max_tokens=10240`, `equal_tokens`, `contiguous_placement`.
- **Source**: upstream `lishaoxun/DeepEP_9283` at commit `69090a120cfee06be431a79873692677bc07b042`; not ascendep-vendored, so the native ElasticBuffer build comes from `dependencies/manifest.yaml`.
