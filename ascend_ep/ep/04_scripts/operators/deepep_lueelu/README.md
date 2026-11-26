# deepep_lueelu

Native API: `deep_ep.ElasticBuffer (Mesh+Clos)`. Source commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`. [Source](https://github.com/penn513/ascendep)

- **Adapter** · `historical_verified`. Mesh+Clos ElasticBuffer HT. L1 uses a native per-route scalar label to validate cached and asynchronous dispatch and the inverse mapping.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `moe_forward`, `moe_backward`, `router_weight_grad`, `expert_weight_grad`, `dispatch`, `combine`, `cached_dispatch`, `async_dispatch`, `async_combine`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.deepep_lueelu.benchmark --config operators/deepep_lueelu/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-001..COMM-008, COMM-010; L2 FUSE-001, FUSE-003; L3 LAYER-007, MOE-001, MOE-002, MOE-003, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-009 (`quantized_dispatch`), COMM-011 (`combine_add_rmsnorm`), COMM-012 (`planned_dispatch`); FUSE-002/004/005/006/007/008/009; LAYER-001..006, LAYER-008, LAYER-009 (`router_layer`/`dynamic_planner`/dedup); MOE-004 (`recompute`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). They stay in the manifest as `unsupported` with the missing token as the reason.
- **Correctness contract**: dispatch layout via `validate_dispatch`; combine and L3 via relative L2 / `validate_moe`.
- **Timing contract**: L1 handlers submit and wait inside the timed block, recording `local_complete`; the `cached_dispatch` variant reuses a routing handle produced outside the timer.
- **Registry constraints**: `ep_sizes=[2,4,8,16,32,64,128,256,512]`, `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `max_tokens=16384`, `swiglu_limit=10` (no `top_k`/`num_experts` pin).
- **Source**: vendored `operators/vendor/ascendep/ep/deepep_lueelu/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; `deep_ep` is a native build, `EP_ASCENDEP_ROOT` is only an optional override.
