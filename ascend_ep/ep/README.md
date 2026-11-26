# 昇腾 EP 专题

[EP 总览及算子索引](../ep.md) 按具体后端列出实现入口和支持情况。专题按硬件原理、实现、测试、结果和分析的阅读顺序组织。

| 目录 | 作用 |
| --- | --- |
| `01_architecture/` | 说明卡内和跨机链路、通信原语与物理拓扑，为后端实现提供硬件背景。 |
| `02_implementations/` | 对照各后端的通信、计算融合、路由与前反向实现。 |
| `03_benchmark/` | 定义 L0–L4 的计时边界、输入、正确性要求、指标和用例矩阵。 |
| `04_scripts/` | 通过统一 CLI 将配置、公共基线和原生算子适配编排成可执行测点。 |
| `05_results/` | 归档计划、配置与环境快照、原始样本、正确性、Profiling 及报告。 |
| `06_analysis/` | 基于结果分析带宽、通算重叠、负载与去重、扩展效率和后端适用范围。 |

执行时，`04_scripts/configs/` 选择用例和算子，`common/` 统一构造输入、路由和分布式任务。`benchmarks/` 调用 `baselines/` 或 `operators/` 执行测点，写入 `05_results/raw/<run-id>/`。`metrics/`、`profiling/`、`analysis/` 和 `reporting/` 将样本和 Trace 转为指标与报告。关于可执行目录和入口见 [Benchmark 方法与入口](03_benchmark/ep_benchmark_methodology.md#九测试脚本目录与入口)。

## 01 通信架构

- [通信硬件与数据路径](01_architecture/ep_comm_hardware.md)
- [网络拓扑](01_architecture/ep_network_topology.md)
- [通信原语](01_architecture/ep_comm_primitives.md)

## 02 算子实现

- [DeepEP-Ascend 与 ElasticBuffer](02_implementations/deepep_ascend.md)
- [MoonEP](02_implementations/moonep.md)
- [MegaMoE](02_implementations/megamoe.md)
- [Lueelu MegaMoE 通信与架构](02_implementations/lueelu_megamoe.md)
- [CANN MC2](02_implementations/cann_mc2.md)
- [HCCL EP](02_implementations/hccl_ep.md)
- [框架 EP](02_implementations/framework_ep.md)

## 03 Benchmark 规范

- [测试方法与指标](03_benchmark/ep_benchmark_methodology.md)
- [测试矩阵](03_benchmark/ep_benchmark_matrix.md)

## 04 测试脚本

- [脚本目录、配置和入口](04_scripts/README.md)
- [36 个算子接入矩阵](04_scripts/operators/README.md)
- [Agent Benchmark Skill](04_scripts/SKILL.md)

## 05 测试结果

- [EP8 结果模板](05_results/ep_benchmark_ep8.md)
- [EP64 / EP128 结果模板](05_results/ep_benchmark_ep64_ep128.md)
- [原始结果及 Profiling 归档约定](05_results/raw/README.md)

## 06 专项分析

- [通算重叠](06_analysis/ep_overlap_analysis.md)
- [负载均衡与去重](06_analysis/ep_load_balance.md)
- [通信带宽](06_analysis/ep_bandwidth_analysis.md)
- [规模扩展](06_analysis/ep_scalability_analysis.md)
- [后端选型](06_analysis/ep_backend_selection.md)

旧入口 [ep_comm_hardware.md](ep_comm_hardware.md) 和 [lueelu_megamoe.md](lueelu_megamoe.md) 指向对应的分类文档。
