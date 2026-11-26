# deepep_direct_lueelu

Native API: `deep_ep.backends.direct.api.ElasticBuffer`. Source commit `83252a54d2363734a706a7dd8368bebc3094af5a`. [Source](https://github.com/lueelu/DeepEP)

- **Adapter** · `integrated`. Direct ElasticBuffer is selected explicitly. Set benchmark.deepep_transport to mte or udma for the topology. L1 uses a native per-route scalar label to check inverse mapping.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `moe_forward`, `moe_backward`, `router_weight_grad`, `expert_weight_grad`, `dispatch`, `combine`, `cached_dispatch`, `async_dispatch`, `async_combine`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.deepep_direct_lueelu.benchmark --config operators/deepep_direct_lueelu/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-001..COMM-008, COMM-010; L2 FUSE-001, FUSE-003; L3 LAYER-007, MOE-001, MOE-002, MOE-003, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-009 (`quantized_dispatch`), COMM-011 (`combine_add_rmsnorm`), COMM-012 (`planned_dispatch`); FUSE-002/004/005/006/007/008/009 (`dispatch_gmm`/`gmm_combine`/`chunk_pipeline`/`tile_schedule`/`comm_resources`/`quantized_fusion`/`fused_backward`); LAYER-001..006, LAYER-008, LAYER-009; MOE-004 (`recompute`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). They stay in the manifest as `unsupported`.
- **Correctness contract**: dispatch layout via `validate_dispatch` against the reference collective; combine and L3 via relative L2 / `validate_moe`.
- **Timing contract**: L1 handlers submit and wait inside the timed block and record `local_complete`; `async_dispatch`/`async_combine` wait on the adapter's deferred callback.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`.
- **Source**: adapter module from the vendored tree (`upstream: deepep_ascend`, so `operators/vendor/ascendep/ep/deepep_ascend/demo.py` at `24ff84d…`, no external checkout); the native library is `lueelu/DeepEP` at `83252a54d2363734a706a7dd8368bebc3094af5a`.
