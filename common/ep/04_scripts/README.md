# EP 测试脚本

当前版本建立配置、适配接口和测试入口。测试用例、分布式执行、Profiling 采集与结果写入待接入。

## 目录

| 路径 | 职责 |
|---|---|
| `configs/` | EP8 / EP64 / EP128、Prefill、Decode、负载均衡与去重配置 |
| `backends/` | 后端能力声明、原生接口适配和注册表 |
| `benchmarks/` | 通信、GMM、单段融合、专家子层和完整 MoE 测试入口 |
| `common/` | 进程组、路由、输入、计时、指标、正确性和显存 |
| `profiling/` | 采集、解析及重叠、空闲和带宽分析 |
| `analysis/` | 跨后端、规模、负载均衡对比及报告生成 |
| `launchers/` | EP8 / EP64 / EP128 配置入口 |

## 入口

在本目录运行以下命令可查看参数和计划接入的后端。入口仅使用 Python 标准库。

```bash
python run_benchmark.py --help
python run_benchmark.py --list-backends
```

`--config` 接受已有配置路径。当前入口会明确提示执行尚未接入，适配模块就绪后再启用测试。三个 launcher 只是转发对应配置。

## 配置与能力

- 基线配置为 [EP8 BF16](configs/ep8_bf16.yaml)，另有 [EP64](configs/ep64_bf16.yaml) 和 [EP128](configs/ep128_bf16.yaml)。同组规模实验须匹配模型参数与扩展策略。
- 专项配置包括 [Decode](configs/decode.yaml)、[Prefill](configs/prefill.yaml)、[负载均衡](configs/load_balance.yaml) 和 [去重](configs/dedup.yaml)。
- 后端声明原生支持的 L1–L4 层次。公共测试流程负责共享路由、正确性与度量。分层口径见 [Benchmark 方法](../03_benchmark/ep_benchmark_methodology.md)。
- 测试产物按 [原始结果归档](../05_results/raw/README.md) 记录。
