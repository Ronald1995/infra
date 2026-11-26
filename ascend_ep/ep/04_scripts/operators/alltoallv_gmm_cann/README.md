# alltoallv_gmm_cann

原生入口为 `npu_alltoallv_gmm / npu_gmm_alltoallv`。参考提交为 `24ff84d3ab8c01e1e2071802465173e624e1f89d`。 [源码](https://github.com/penn513/ascendep/tree/24ff84d3ab8c01e1e2071802465173e624e1f89d)

- **接入与验证** · `historical_verified`。参考报告覆盖 950DT EP8。当前框架运行时重新验证正确性和性能。
- **布局和范围** · 参考适配配置为 BF16、EP2/EP8、24 个专家、Top-6、各 rank 等长。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L2/L3。`moe_forward`、`moe_backward`、`router_weight_grad`、`expert_weight_grad`、`dispatch_gmm`、`gmm_combine`、`fused_backward`。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.alltoallv_gmm_cann.benchmark --config operators/alltoallv_gmm_cann/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-010; L2 FUSE-002, FUSE-004, FUSE-009; L3 LAYER-007, MOE-001, MOE-002, MOE-003, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-001..009, COMM-011, COMM-012 (no `dispatch`/`combine`/`cached_dispatch`/`async_*`/`quantized_dispatch`/`combine_add_rmsnorm`/`planned_dispatch`); FUSE-001/003/005/006/007/008 (no `dispatch`/`chunk_pipeline`/`tile_schedule`/`comm_resources`/`quantized_fusion`); LAYER-001..006, LAYER-008, LAYER-009 (no `router_layer`/`dynamic_planner`/`rank_dedup`/`server_dedup`/`dedup_training`); MOE-004 (`recompute`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). They stay in the manifest as `unsupported` with the missing token as the reason.
- **Correctness contract**: FUSE-002/004 compare the native dispatch+GMM / GMM+combine pair against the reference collective + reference GMM (relative L2); FUSE-009 uses the fused backward path; L3 uses `validate_moe`.
- **Timing contract**: `dispatch_gmm`/`gmm_combine` are one adapter entry point inside the timed call (`benchmarks/cases/l2_fusion.py::_dispatch_gmm`, `_gmm_combine`); FUSE-009 inherits MOE-003's boundary (forward and backward in one call).
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`.
- **Source**: vendored `operators/vendor/ascendep/ep/alltoallv_gmm_cann/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; no external checkout, `EP_ASCENDEP_ROOT` is an optional override.
