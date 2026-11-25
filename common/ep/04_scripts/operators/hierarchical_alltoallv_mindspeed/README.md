# hierarchical_alltoallv_mindspeed

原生入口为 `--moe-hierarchical-alltoallv / --moe-tp-extend-ep`。参考提交为 `c68e201a5733065b3aa59a3d44dc5e2ec700d7c9`。 [源码](https://gitcode.com/Ascend/MindSpeed)

- **接入与验证** · `not_implemented`。Pinned ascendep demo is a launcher for a separate Megatron training driver. The TP-extend-EP model and workload integration is pending.
- **布局和范围** · 平台、容量及布局限制见源码。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · 待接入。独立能力待接入。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按源码安装目标依赖。计划保留接入缺口。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.hierarchical_alltoallv_mindspeed.benchmark --config operators/hierarchical_alltoallv_mindspeed/config.yaml
```
