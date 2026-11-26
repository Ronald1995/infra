# EP 后端选型

| 场景 | 待测候选 | 主要评价指标 |
|---|---|---|
| 大 Batch Prefill | DeepEP、MC2、MegaMoE | 吞吐、带宽 |
| 小 Batch Decode | DeepEP LL、MC2 | 低时延 |
| EP64 / EP128 训练 | MegaMoE、DeepEP、MoonEP | 前反向吞吐、扩展性 |
| 专家负载不均 | MoonEP 动态规划 | 负载均衡收益、额外显存 |
| 跨 Server 通信受限 | Server Dedup | 跨节点流量与归并开销 |
| BF16 通算融合训练 | MegaMoE、FSDPTurbo MC2 | 整体前反向耗时 |
| 量化推理 | DeepGEMM、MegaMoE、FuseEP | 量化吞吐、数值误差 |

## 选型结论

当前记录候选方案。基于同口径性能、正确性结果与可用拓扑填写最终选择。
