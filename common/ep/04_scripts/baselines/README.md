# 公共基线

- [L0 设备与通信](l0/README.md)。
- L1 由 `reference/collectives.py` 实现 AllToAllV 和 AllGather/ReduceScatter。NPU 使用 HCCL，CPU 使用 Gloo 验证通信与梯度。
- L2 测量 Dispatch → GMM1 和 GMM2 → Combine。L3 的 AllToAllV 参考路径测量串行完整专家 FFN。融合分析按同语义实际计时配对。
