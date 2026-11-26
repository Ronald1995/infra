# mc2_dispatch_cann_v3

此目录测量 CANN 上下文版 `MoeDistributeDispatchV3` / `MoeDistributeCombineV3`。Torch 入口为 `MoeDistributeBuffer.low_latency_dispatch` / `low_latency_combine`，带 `context` 时调用 ACLNN **V5**。带 `groupEp` 的 ACLNN **V3** 入口属于另一套公开接口。 [Dispatch 绑定](https://gitcode.com/cann/ops-transformer/blob/68a67e130417593f37ea3cece1ebfb4be366b95f/mc2/moe_distribute_dispatch/torch_extension/csrc/moe_distribute_dispatch.cpp) · [Combine 绑定](https://gitcode.com/cann/ops-transformer/blob/68a67e130417593f37ea3cece1ebfb4be366b95f/mc2/moe_distribute_combine/torch_extension/csrc/moe_distribute_combine.cpp)

- 参考提交为 `68a67e130417593f37ea3cece1ebfb4be366b95f`。950DT 文档支持上下文 Dispatch，Combine 仅列 Atlas A3。950DT 支持 L1 BF16/FP16 Dispatch 和 INT8 量化校验。A3 另有正向 Combine、L2 串行对照和 L3 专家前向。
- 输入 `[T,H]`、INT32 `[T,K]` 与本地连续专家分片。单个调用每 rank 最多 512 tokens。上下文和 HCCL 窗口在计时外分配，正式采样包括路由转换与实际调用。
- 950DT 通过原生 route label 核对逆映射。A3 通过专家分段、token 多重集和 Combine 回环校验。设备显式区分状态。有效计算和通信模型见 [测试方法](../../../03_benchmark/ep_benchmark_methodology.md)。
- 需要与固定源码兼容的 CANN、torch_npu 和 CANN Torch 扩展。当前没有设备执行记录。 [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_dispatch_cann_v3.benchmark --config operators/mc2_dispatch_cann_v3/config.yaml
```
