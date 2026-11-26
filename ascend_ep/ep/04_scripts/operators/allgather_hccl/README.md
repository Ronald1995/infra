# allgather_hccl

原生入口为 `torch.distributed.all_gather / reduce_scatter_tensor`。使用环境中实际安装的源码或二进制版本。 [源码](https://github.com/Ascend/pytorch)

- **接入与验证** · `integrated`。支持 NPU/HCCL。CPU/Gloo 路径用于验证公共逻辑。
- **布局和范围** · 支持非等长源 token 和指定专家归属。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L1/L2 serial/L3/L4。`moe_forward`、`moe_backward`、`router_weight_grad`、`expert_weight_grad`、`dispatch`、`combine`、`router_layer`、`recompute`。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · PyTorch。NPU 另需 torch_npu 和 HCCL。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.allgather_hccl.benchmark --config operators/allgather_hccl/config.yaml
```
