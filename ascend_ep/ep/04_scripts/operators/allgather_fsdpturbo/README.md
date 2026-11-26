# allgather_fsdpturbo

原生入口为 `all_gather_experts_forward_fn`。参考提交为 `24ff84d3ab8c01e1e2071802465173e624e1f89d`。 [源码](https://github.com/penn513/ascendep/tree/24ff84d3ab8c01e1e2071802465173e624e1f89d)

- **接入与验证** · `historical_verified`。参考报告覆盖 950DT EP8。当前框架运行时重新验证正确性和性能。
- **布局和范围** · 参考适配配置为 BF16、EP2/EP8、24 个专家、Top-6、各 rank 等长。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L3。`moe_forward`、`moe_backward`、`router_weight_grad`、`expert_weight_grad`。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.allgather_fsdpturbo.benchmark --config operators/allgather_fsdpturbo/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level, `CaseSpec.requires` <= registry `capabilities`): L1 COMM-010; L3 LAYER-007, MOE-001, MOE-002, MOE-003, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-001..009, COMM-011, COMM-012 (no `dispatch`/`combine`/`cached_dispatch`/`async_*`/`quantized_dispatch`/`combine_add_rmsnorm`/`planned_dispatch`); FUSE-001..009 (no `dispatch_gmm`/`gmm_combine`/`chunk_pipeline`/`tile_schedule`/`comm_resources`/`quantized_fusion`/`fused_backward`); LAYER-001..006, LAYER-008, LAYER-009 (no `router_layer`/`dynamic_planner`/`rank_dedup`/`server_dedup`/`dedup_training`); MOE-004 (`recompute`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). Those tasks stay in the manifest as `unsupported` with the missing token as the reason (`common/scheduler.py::status_for`).
- **Correctness contract**: MoE output and gradients checked by `common/validation.py::validate_moe` (CPU FP32 oracle below 2M elements, collective reference above): `moe_forward_output`, `input_gradient`, `expert_weight_gradient`.
- **Timing contract**: `benchmarks/cases/common.py::moe_case`; the timed call contains the forward (and the backward for a training case), `wait` is the identity, and the sample ends after `ctx.synchronize()` (`common/timer.py::measure`).
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`; a mismatch (EP64/EP128 included) is `unsupported` from `common/registry.py::check_constraints`.
- **Source**: vendored `operators/vendor/ascendep/ep/allgather_fsdpturbo/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; no external checkout is needed, `EP_ASCENDEP_ROOT` is an optional override (`operators/ascendep.py`).
