# EP 网络拓扑

本文记录设备间的物理互联、通信域和 EP rank 映射。目标平台、组网和版本需要随具体实验记录。

## 拓扑与通信域

- 记录昇腾 950 超节点的 Mesh、Clos、UB、UBoE 和跨超节点互联。
- 对每个测点登记物理节点与设备、EP 组、rank 放置以及流量是否经过跨节点链路。
- 分别记录单链路单向带宽、双向带宽与组内聚合能力。

## 现有资料

- [HCCL 与 HCOMM](../../../UB/HCCL_HCOMM.md)
- [950 超节点组网图](../../../UB/assets/hccl_hcomm/f9_supernode_fabric.png)
- [EP 通信硬件路径](ep_comm_hardware.md)
