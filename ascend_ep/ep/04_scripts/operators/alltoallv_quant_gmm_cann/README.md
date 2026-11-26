# alltoallv_quant_gmm_cann

Native API: `npu_alltoallv_quant_gmm`. Source commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d`. [Source](https://github.com/Ascend/op-plugin)

- **Adapter** · `integrated`. FUSE-008 measures dynamic MXFP8 activation quantization, packing and the native AllToAllV-GMM call. Weight layout and static quantization are prepared before timing.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L2. `quantized_fusion`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.alltoallv_quant_gmm_cann.benchmark --config operators/alltoallv_quant_gmm_cann/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-010; L2 FUSE-008 (`quantized_fusion`).
- **Cases it cannot run**: every other L1 case (no `dispatch`/`combine`/`cached_dispatch`/`async_*`/`quantized_dispatch`/`planned_dispatch`/`combine_add_rmsnorm`); FUSE-001..007 and FUSE-009 (`dispatch`/`dispatch_gmm`/`gmm_combine`/`chunk_pipeline`/`tile_schedule`/`comm_resources`/`fused_backward`); all L3 `moe_forward`/`moe_backward` cases; all L4 `router_layer`/`dynamic_planner`/dedup cases. They stay in the manifest as `unsupported` with the missing token as the reason.
- **Correctness contract**: `validate_quantized_fusion` compares the dequantised result with the reference FP32 GMM path; the registry records `numeric_contract: mxfp8_activations_and_weights`, so this result is not comparable with a BF16 fused case.
- **Timing contract**: `benchmarks/cases/l2_fusion.py::_quantized_fusion`; one native call inside the timed block, weight packing/static quantisation outside it.
- **Registry constraints**: `ep_sizes=[2,4,8,16,32,64,128,256]`, `dtypes=[bfloat16,float16]`, `contiguous_placement`, `min_tokens=1`; no `top_k`/`num_experts` pin.
- **Source**: native op-plugin binding at commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d`; the operator is not ascendep-vendored and needs `torch_npu` plus the matching `cann_ops_transformer` build.
