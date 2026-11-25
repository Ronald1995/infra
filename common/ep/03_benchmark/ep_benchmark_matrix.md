# EP Benchmark 矩阵

以下是待验证的候选维度。每个测点须满足目标平台、拓扑和后端能力。

| 维度 | 候选配置 |
|---|---|
| 硬件与拓扑 | A3、950DT。单机、超节点内、跨超节点 |
| EP size | 8、16、32、64、128 |
| Tokens per rank | 1、16、128、512、1024、4096 |
| Hidden size | 4096、7168、8192 |
| Top-K | 4、6、8 |
| 专家数 | 24、128、256、384 |
| 数据类型 | BF16、FP8、MXFP |
| 路由分布 | 均衡、随机、热点专家、真实模型路由 |
| 负载均衡 | Planner Off / On |
| 去重 | None / Rank / Server |
| 执行模式 | Forward / Backward / Forward + Backward |

## 专项测试集合

- **基础性能**：EP8 BF16 均衡路由，分层建立通信与专家子层基线。
- **规模扩展**：分别固定每 rank 的 token 数和全局 token 数，保持同一组实验的模型参数一致。
- **负载均衡与去重**：同一份路由输入下对比 Planner 和去重策略。
- **量化与推理**：为量化、Prefill 和 Decode 分别选择适合的 dtype、token 数和调用路径。

参数模板见 [脚本目录](../04_scripts/README.md)。配置是否可执行由具体后端检验。
