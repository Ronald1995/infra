# megamoe_deepgemm_ascend

Native API: `deep_gemm.fp8_fp4_mega_moe`. Source commit `3a35326271b069b8b5bf1a754435acd6cc91575b`. [Source](https://github.com/deepseek-ai/DeepGEMM-Ascend)

- **Adapter** · `integrated`. FP8 activations and FP4 weights. The reference dequantizes the operands and applies routing probability before activation quantization. The adapter uses no shared experts.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L3. `moe_forward`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.megamoe_deepgemm_ascend.benchmark --config operators/megamoe_deepgemm_ascend/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L3 完整 MoE MOE-001, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-002, MOE-003, MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline; operation requires combine, dispatch ….

- **Correctness contract**: `validate_moe` over the forward output; the registry records `numeric_contract: mxfp8_activations_mxfp4_weights_no_shared_expert`, so the reference dequantises both operands and no shared expert is modelled.
- **Timing contract**: `moe_case`; one `deep_gemm.fp8_fp4_mega_moe` call inside the timed block, quantisation of the static weights outside it.
- **Registry constraints**: `ep_sizes=[8]` (EP8 only), `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `min_tokens=1`.
- **Source**: DeepGEMM-Ascend at commit `3a35326271b069b8b5bf1a754435acd6cc91575b`; not ascendep-vendored.
