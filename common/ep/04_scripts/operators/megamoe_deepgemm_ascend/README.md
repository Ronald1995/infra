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
