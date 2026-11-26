# ep_gmm_fused_lueelu

原生入口为 `deep_ep_experimental.ep_gmm_fused`。参考提交为 `24ff84d3ab8c01e1e2071802465173e624e1f89d`。 [源码](https://github.com/penn513/ascendep/tree/24ff84d3ab8c01e1e2071802465173e624e1f89d)

- **接入与验证** · `historical_verified`。参考报告覆盖 950DT EP8。当前框架运行时重新验证正确性和性能。
- **布局和范围** · 参考适配配置为 BF16、EP2/EP8、24 个专家、Top-6、各 rank 等长。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L2/L3。FUSE-006 通过原生 `tiling` 参数变更 `m0/n0/k0`，FUSE-007 变更 `self_copy_cores`。扫描使用 [Lueelu 固定源码](https://github.com/lueelu/DeepEP/tree/83252a54d2363734a706a7dd8368bebc3094af5a/experiments/ep_gmm_fused)。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.ep_gmm_fused_lueelu.benchmark --config operators/ep_gmm_fused_lueelu/config.yaml
```
