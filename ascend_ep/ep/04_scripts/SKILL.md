---
name: ep-benchmark
description: Run, validate, profile and analyze distributed Ascend expert-parallel benchmarks.
---

# EP Benchmark

1. 读取 [算子调研表](../../ep.md)、[测试方法](../03_benchmark/ep_benchmark_methodology.md)、[测试矩阵设计](../04_scripts/docs/benchmark_matrix.md) 和 [算子注册表](operators/registry.yaml)。
2. 执行 `python run_benchmark.py doctor`。核对型号、CANN、torch_npu、HCCL、SHMEM、原生依赖、拓扑和可用 HBM。按各算子 README 核对源码。确认多机共享结果目录和 rank 配置。
3. 默认用统一计划生成完整矩阵：`python run_benchmark.py plan --plan configs/benchmark_plan.yaml`（`mode: full_matrix`，`all` + 全部专项 Suite）。核对 EP8/EP64/EP128 的 Case 集合是否一致、每个 Suite 是否真的贡献了 workload、`unsupported`/`not_applicable`/`dependency_missing` 的原因是否合理；需要缩小范围时用 `--ep-size`、`--suite`、`--levels`、`--cases`、`--operators`。
4. 用 `configs/smoke_npu.yaml --ep-size 1` 检验 GMM/SwiGLU，用 `--ep-size 2` 验证通信及梯度。L0 单独用 `configs/hardware.yaml`、`configs/memory.yaml`、`configs/transport.yaml`、`configs/collectives.yaml`、`configs/overlap.yaml` 验证。失败时保留 rank 日志。`same_server`/`cross_server` 需要显式 Rank→Server 映射；没有可用 Peer 时必须是 `not_applicable`，不能退化成同机通信。
5. 执行 `run --plan configs/benchmark_plan.yaml`（或单配置 `run --config CONFIG`）。EP Size 顺序执行，单 Case 失败不影响后续 Case，单个 EP Size 失败不影响其他 EP Size。中断后执行 `resume --run-id RUN_ID`。
6. 执行 `capabilities --plan configs/benchmark_plan.yaml` 查看每个 EP Size × Operator 的支持/未支持统计；`unsupported` 不计入 `passed`。
7. 按目标选训练尾部、高吞吐、强弱扩展、拓扑、负载、去重、融合或量化 Suite。强/弱扩展只改 workload 生成规则，Case 集合保持一致。
8. 执行 `profile --run-id RUN_ID` 独立采集已通过测点。保留 rank Trace 和校验值，较大产物上传到项目已授权的位置。
9. 执行 `analyze --run-id RUN_ID`，核对可比性、可见通信和设备空闲范围。异常复测保留新的 attempt，报告依据及待验证假设。
10. 执行 `report --run-id RUN_ID` 生成 Markdown、JSON、CSV 和可用图表；`run` 命令也会写覆盖率报告。单次一键执行可用 `pipeline`。L0 报告保留 requested/observed/verification，只有 `verified` 路径可做机制横向比较；`l0_scaleout.md` 按完整参数变体分行，`N/A` 保留原因。
11. 交付覆盖情况（`reports/coverage_matrix.md`、`variant_coverage.md`、`operator_coverage.md`、`ep_size_comparison.md`、`l0_scaleout.md`、`ep_summary.json` 的 `completion_status` 与状态桶）、真实通过和缺口、可比组结果、证据与原始路径。`not_run` 不是通过，也不是失败：完整的 Full Matrix 必须每个预期 task 都有明确终态。历史导入附来源提交。NPU 性能依据目标设备真实采样。
