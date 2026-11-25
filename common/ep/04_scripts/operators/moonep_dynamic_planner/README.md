# moonep_dynamic_planner

原生入口为 Ascend MoonEP `dev_npu` 的 `npu_patch.Buffer.dispatch`。固定参考提交 `b4127baa1bec0bcccd49b48c2dc2a3be70da9a0f`。 [源码](https://gitcode.com/Ascend/MoonEP/tree/dev_npu)

- L1 `COMM-012` 测量从本 rank 专家计数开始，到原生规划和分发结果可用的整体时间。每轮均调用新 plan。
- 使用 BF16 `[T,H]`、FP32 路由权重和每 rank 等长 `[T,K]`。原生分片的 capacity 为 S × K，规划需要本地专家数等于冗余槽位数。校验检查实际规划的元数据、路由权重和 Combine 回环。
- 安装该分支的原生 npu_patch 和匹配的 SHMEM。默认使用 UDMA 数据面和 MTE 规划。多节点设置 `benchmark.shmem_rendezvous`。
- 独立规划耗时、权重刷新、冗余专家梯度归并和整层训练周期尚未接通。`LAYER-003/004/008` 保留待接入状态。
- [注册表](../registry.yaml) · [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.moonep_dynamic_planner.benchmark --config configs/load_balance.yaml --cases COMM-012
```
