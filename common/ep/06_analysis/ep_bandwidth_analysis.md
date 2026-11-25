# EP 通信带宽分析

## 统计口径

区分单向、双向和聚合理论带宽。记录每个 rank 的发送与接收字节数、有效 payload、计时窗口和实际走过的链路。

## 对比项

- 原始流量与 Rank / Server 去重后的流量。
- MTE、UBMEM、URMA、UDMA 路径和设备利用率。
- 同拓扑、同传输方向、同消息规模下的带宽瓶颈。

拓扑和互联见 [EP 网络拓扑](../01_architecture/ep_network_topology.md)，待补充实测结果。
