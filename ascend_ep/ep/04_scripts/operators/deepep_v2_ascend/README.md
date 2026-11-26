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

- **Cases it can run** (capability-level): L1 COMM-001..COMM-008, COMM-010; L2 FUSE-001, FUSE-003; L3 LAYER-007, MOE-001, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-009/011/012 (`quantized_dispatch`/`combine_add_rmsnorm`/`planned_dispatch`); FUSE-002/004/005/006/007/008/009; LAYER-001..LAYER-006, LAYER-008, LAYER-009 (`router_layer`/`dynamic_planner`/dedup); MOE-002/003/004 (`moe_backward`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). They stay in the manifest as `unsupported`.
- **Correctness contract**: dispatch layout via `validate_dispatch`; the L3 forward via `validate_moe`. The registry has no `moe_backward`, so no gradient case is claimed.
- **Timing contract**: L1 dispatch/combine handlers submit and wait inside the timed block and record `local_complete`; the async variants wait on the adapter's deferred callback.
- **Registry constraints**: `ep_sizes=[32,64,128]` (the only adapter that lists EP64/EP128), `dtypes=[bfloat16]`, `hidden_size=7168`, `min_tokens=256`, `max_tokens=10240`, `equal_tokens`, `contiguous_placement`.
- **Source**: upstream `lishaoxun/DeepEP_9283` at commit `69090a120cfee06be431a79873692677bc07b042`; not ascendep-vendored, so the native ElasticBuffer build comes from `dependencies/manifest.yaml`.