# L0 公共设备基线

`configs/hardware.yaml` 从 4 KiB 到 1 GiB 扫描通信与设备 Copy。`HW-006` 分别测量 Gather 和 ReduceScatter。`HW-010` 使用 trace 的专家计数，分别测量两层 GMM、Dgrad 和 Wgrad。`HW-011` 测量 SwiGLU 前反向。`HW-012` 使用独立 NPU stream 执行 AllToAll 与 GMM1。

`HW-007` 通过 CANN SHMEM Python buffer API 执行原生 MTE Put。`HW-008` 的 UDMA 路径直接调用 Ascend C Put 和 Quiet。将 `benchmark.shmem_rendezvous` 设为共享 SHMEM TCP rendezvous，端口须和 torchrun 分开。当前源码公开 UDMA 引擎，URMA 别名保留为独立 API 不可用的状态。 [原生 SHMEM 源码](https://gitcode.com/cann/shmem/tree/f0bcfc0fe51bce1d866ba6935d2adc6e5e5800e5)

```sh
bash baselines/l0/native/build_udma.sh /path/to/pinned/shmem /path/to/new/build
```

在配置中将 `benchmark.shmem_udma_library` 设为构建输出的 `libep_udma_kernel.so`。Python SHMEM 运行时须与构建组件匹配。

设备 Copy 报告逻辑读写量。工作集大小和实际 HBM 计数器决定能否用于评估物理带宽。
