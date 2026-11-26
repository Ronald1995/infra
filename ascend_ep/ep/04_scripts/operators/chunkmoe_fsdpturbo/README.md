# chunkmoe_fsdpturbo

原生入口为 `chunkmoe_experts_forward_fn`。参考提交为 `24ff84d3ab8c01e1e2071802465173e624e1f89d`。 [源码](https://github.com/penn513/ascendep/tree/24ff84d3ab8c01e1e2071802465173e624e1f89d)

- **接入与验证** · `historical_verified`。参考报告覆盖 950DT EP8。当前框架运行时重新验证正确性和性能。
- **布局和范围** · 参考适配配置为 BF16、EP2/EP8、24 个专家、Top-6、各 rank 等长。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L3/L2 chunk。`moe_forward`、`moe_backward`、`router_weight_grad`、`expert_weight_grad`、`chunk_pipeline`。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.chunkmoe_fsdpturbo.benchmark --config operators/chunkmoe_fsdpturbo/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-010; L2 FUSE-005 (`chunk_pipeline`, variants `chunks_2`/`chunks_4`/`chunks_8`); L3 LAYER-007, MOE-001, MOE-002, MOE-003, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-001..009, COMM-011, COMM-012 (no `dispatch`/`combine`/`cached_dispatch`/`async_*`/`quantized_dispatch`/`combine_add_rmsnorm`/`planned_dispatch`); FUSE-001/002/003/004/006/007/008/009 (`dispatch`/`dispatch_gmm`/`gmm_combine`/`tile_schedule`/`comm_resources`/`quantized_fusion`/`fused_backward`); LAYER-001..006, LAYER-008, LAYER-009; MOE-004 (`recompute`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). They stay in the manifest as `unsupported`.
- **Correctness contract**: FUSE-005 checks the chunked forward with `validate_moe`; the chunk count is the variant, so the three variants are three different measurements, not repeats.
- **Timing contract**: `benchmarks/cases/l2_fusion.py::_chunk_pipeline`; `run` calls `backend.chunk_forward(work, chunks)` once inside the timed block.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`.
- **Source**: vendored `operators/vendor/ascendep/ep/chunkmoe_fsdpturbo/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; no external checkout, `EP_ASCENDEP_ROOT` is an optional override.
