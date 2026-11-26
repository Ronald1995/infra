# dispatch_ffn_combine_vllm_ascend

Native API: `torch.ops._C_ascend.dispatch_ffn_combine`. Source commit `0f63cfa1a52225f301ffd4a335b8daa3028c283f`. [Source](https://github.com/vllm-project/vllm-ascend)

- **Adapter** · `integrated`. The pinned BF16 custom op is built for Atlas A3. It consumes NZ weights. Validation checks that the selected workload stays within the shared SwiGLU clipping limit.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L3. `moe_forward`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.dispatch_ffn_combine_vllm_ascend.benchmark --config operators/dispatch_ffn_combine_vllm_ascend/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-010; L3 LAYER-007, MOE-001, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-001..009, COMM-011, COMM-012 (`dispatch`/`combine`/`cached_dispatch`/`async_*`/`quantized_dispatch`/`combine_add_rmsnorm`/`planned_dispatch`); FUSE-001..009; LAYER-001..006, LAYER-008, LAYER-009; MOE-002/003/004/007/009 (`moe_backward`, `recompute`, `concurrent`, `dynamic_shapes`). They stay in the manifest as `unsupported`.
- **Correctness contract**: `validate_moe` over the forward output; the registry constrains `hardware=[Atlas A3 (ascend910_93)]`, so on any other SoC the point is `unsupported` and no correctness claim is made.
- **Timing contract**: `moe_case`; the fused custom op runs inside one timed call and the sample closes after `ctx.synchronize()`.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `min_tokens=1`.
- **Source**: upstream vllm-ascend at commit `0f63cfa1a52225f301ffd4a335b8daa3028c283f`; not ascendep-vendored, so the torch extension must come from `dependencies/manifest.yaml`.
