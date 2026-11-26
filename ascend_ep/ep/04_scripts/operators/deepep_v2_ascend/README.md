# deepep_v2_ascend

Native API: `deep_ep.ElasticBuffer`. Source commit `69090a120cfee06be431a79873692677bc07b042`. [Source](https://gitcode.com/lishaoxun/DeepEP_9283)

- **Adapter** · `integrated`. The public source mirror provides a distinct ElasticBuffer. This adapter exercises the high-throughput expanded BF16 path only. It records the actual EP32/64/128 selection.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `dispatch`, `combine`, `cached_dispatch`, `async_dispatch`, `async_combine`, `moe_forward`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the native wheel from the pinned DeepEP V2 public mirror in its own Python environment.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.deepep_v2_ascend.benchmark --config operators/deepep_v2_ascend/config.yaml
```
