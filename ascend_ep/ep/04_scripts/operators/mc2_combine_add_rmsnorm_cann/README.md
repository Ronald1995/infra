# mc2_combine_add_rmsnorm_cann

Native API: `npu_moe_distribute_combine_add_rms_norm`. Source commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d`. [Source](https://github.com/Ascend/op-plugin)

- **Adapter** · `integrated`. COMM-011 measures the fused combine, residual add and RMSNorm. DispatchV2 prepares the native handle outside the timed region. The reference checks normalized output, inverse RMS and residual sum.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1. `combine_add_rmsnorm`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按源码安装目标依赖。计划保留接入缺口。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_combine_add_rmsnorm_cann.benchmark --config operators/mc2_combine_add_rmsnorm_cann/config.yaml
```
