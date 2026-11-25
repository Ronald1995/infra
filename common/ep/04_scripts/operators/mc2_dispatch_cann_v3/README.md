# mc2_dispatch_cann_v3

原生入口为 `aclnnMoeDistributeDispatchV3 / aclnnMoeDistributeCombineV3`。参考提交为 `68a67e130417593f37ea3cece1ebfb4be366b95f`。 [源码](https://gitcode.com/cann/ops-transformer)

- **接入与验证** · `not_implemented`。The published V3 ACLNN API takes a communication context. The pinned op-plugin does not expose V3 dispatch/combine Python bindings.
- **布局和范围** · 平台、容量及布局限制见源码。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · 待接入。独立能力待接入。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按源码安装目标依赖。计划保留接入缺口。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_dispatch_cann_v3.benchmark --config operators/mc2_dispatch_cann_v3/config.yaml
```
