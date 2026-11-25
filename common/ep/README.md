# 昇腾 EP 专题

[EP 总览及算子索引](../ep.md) 按具体后端列出实现入口和支持情况。专题按硬件原理、实现、测试、结果和分析的阅读顺序组织。

## 01 通信架构

- [通信硬件与数据路径](01_architecture/ep_comm_hardware.md)
- [网络拓扑](01_architecture/ep_network_topology.md)
- [通信原语](01_architecture/ep_comm_primitives.md)

## 02 算子实现

- [DeepEP-Ascend 与 ElasticBuffer](02_implementations/deepep_ascend.md)
- [MoonEP](02_implementations/moonep.md)
- [MegaMoE](02_implementations/megamoe.md)
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
