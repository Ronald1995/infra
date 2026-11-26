---
name: ep-benchmark
description: Run, validate, profile and analyze distributed Ascend expert-parallel benchmarks.
---

# EP Benchmark

1. 读取 [算子调研表](../../ep.md)、[测试方法](../03_benchmark/ep_benchmark_methodology.md)、[用例矩阵](../03_benchmark/ep_benchmark_matrix.md) 和 [算子注册表](operators/registry.yaml)。按用户给出的硬件、EP、目标和算子选 suite 和 case。
2. 执行 `python run_benchmark.py doctor`。核对型号、CANN、torch_npu、HCCL、SHMEM、原生依赖、拓扑和可用 HBM。按各算子 README 核对源码。确认多机共享结果目录和 rank 配置。
3. 执行 `plan --config CONFIG --operators IDS --levels LEVELS`，保存 manifest、算子能力、资源和缺口。按目标选代表形状。
4. 用 `configs/smoke_npu.yaml --ep-size 1` 检验 GMM/SwiGLU，用 `--ep-size 2` 验证通信及梯度。失败时保留 rank 日志。
5. 执行 `baseline --config CONFIG --levels L0,L1,L2`。相同环境和配置下复用归档，保留基线来源。
6. 执行 `run --config CONFIG --operators IDS` 或单算子入口。校验后采样，记录通过、跳过、依赖错误和接入缺口。中断后执行 `resume --run-id RUN_ID`。
7. 按目标选训练尾部、高吞吐、强弱扩展、拓扑、负载、去重、融合或量化用例。多机扩展每次选择一个 EP Size 并保存各点 run ID。
8. 执行 `profile --run-id RUN_ID` 独立采集已通过测点。保留 rank Trace 和校验值，较大产物上传到项目已授权的位置。
9. 执行 `analyze --run-id RUN_ID`，核对可比性、可见通信和设备空闲范围。异常复测保留新的 attempt，报告依据及待验证假设。
10. 执行 `report --run-id RUN_ID` 生成 Markdown、JSON、CSV 和可用图表。单次一键执行可用 `pipeline`。
11. 交付覆盖情况、真实通过和缺口、可比组结果、证据与原始路径。历史导入附来源提交。NPU 性能依据目标设备真实采样。
