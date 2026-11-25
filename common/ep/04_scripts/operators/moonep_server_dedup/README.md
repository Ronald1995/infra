# moonep_server_dedup

Native API: `MoonEP dev_server_dedup dispatch / combine`. Source commit `d8495c4745ba6310b40eddbdb9e3e9ef7b2c2b5d`. [Source](https://gitcode.com/Ascend/MoonEP/tree/dev_server_dedup)

- **Adapter** · `integrated`. COMM-010 exercises native rank or eight-rank server dedup. Configure a physical board map for server mode. The codec checks a native dispatch/combine roundtrip. L4 dedup is pending.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1. `rank_dedup`, `server_dedup`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.moonep_server_dedup.benchmark --config operators/moonep_server_dedup/config.yaml
```
