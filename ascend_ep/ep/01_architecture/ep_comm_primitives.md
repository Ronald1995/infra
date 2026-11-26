# EP 通信原语

这里记录各通信原语的输入分布、输出布局、反向通信和通信量口径。

| 原语 | 分析边界 |
|---|---|
| AllGather | 各 rank 输入与接收缓冲区，区分本地数据和远端传输 |
| ReduceScatter | 归约操作与返回分片，记录输入规模和数据类型 |
| AllToAllV | rank 间可变 split、路由布局和收发字节数 |
| Dispatch | token 从源 rank 到专家 rank 的映射，区分逻辑路由与去重后的传输 |
| Combine | 专家输出返回源 rank 的归并、路由权重和累加精度 |

统计通信量时分别记录 payload、路由元数据、量化 scale 与补齐开销，并标明统计的是发送、接收还是链路流量。测试口径见 [Benchmark 方法](../03_benchmark/ep_benchmark_methodology.md)。
