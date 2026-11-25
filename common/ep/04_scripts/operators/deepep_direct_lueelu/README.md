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
