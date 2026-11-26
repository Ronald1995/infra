# deepep_ascend

Native API: `deep_ep.EPBuffer`. Source commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`. [Source](https://github.com/penn513/ascendep)

- **Adapter** · `historical_verified`. Expanded EPBuffer. L1 tests full and cached dispatch and uses the native expanded route scalar channel to validate source mapping.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `moe_forward`, `moe_backward`, `router_weight_grad`, `expert_weight_grad`, `dispatch`, `combine`, `cached_dispatch`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.deepep_ascend.benchmark --config operators/deepep_ascend/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-001, COMM-002, COMM-003, COMM-004, COMM-007, COMM-008, COMM-010; L2 FUSE-001, FUSE-003; L3 LAYER-007, MOE-001, MOE-002, MOE-003, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-005/006 (`async_dispatch`/`async_combine`), COMM-009 (`quantized_dispatch`), COMM-011 (`combine_add_rmsnorm`), COMM-012 (`planned_dispatch`); FUSE-002/004/005/006/007/008/009 (`dispatch_gmm`/`gmm_combine`/`chunk_pipeline`/`tile_schedule`/`comm_resources`/`quantized_fusion`/`fused_backward`); LAYER-001..006, LAYER-008, LAYER-009; MOE-004 (`recompute`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). They stay in the manifest as `unsupported`.
- **Correctness contract**: COMM-001/002 validate the dispatch layout against the reference; COMM-003/004/FUSE-001/FUSE-003 add a relative-L2 combine comparison; L3 uses `validate_moe`.
- **Timing contract**: L1 dispatch/combine handlers submit and wait inside the timed block and record `local_complete`; the `cached_dispatch` variant reuses the routing handle produced outside the timer.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`; EP64/EP128 points are `unsupported`.
- **Source**: vendored `operators/vendor/ascendep/ep/deepep_ascend/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; `deep_ep` itself is a native build from `dependencies/manifest.yaml`.
