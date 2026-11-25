# EP 负载均衡与去重分析

## 对比方式

复用相同的路由输入，分别比较 Planner Off / On 和 None / Rank / Server 去重。

## 记录指标

- 专家负载的 MaxVio、Expert Max / Mean 与 Rank Max / Mean。
- 冗余专家数量、权重显存、Planner 耗时与收益。
- Rank / Server 去重前后的 token 流量、权重预取与反向梯度归并开销。

## 证据与结论

待补充测点、输入分布与 Profiling 索引。
