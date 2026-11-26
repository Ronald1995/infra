# 950DT EP8 BF16 归档模板

此目录展示归档格式，当前状态为 `not_run`。

| 文件 | 运行后记录的内容 |
|---|---|
| [config.yaml](config.yaml) | 测点实际生效的配置快照 |
| [environment.json](environment.json) | 芯片、拓扑、CANN 版本和源码提交 |
| [routing_trace.json](routing_trace.json) | 种子、路由摘要和 trace 校验值 |
| [benchmark.json](benchmark.json) / [benchmark.csv](benchmark.csv) | 分层性能结果，CSV 模板只有表头 |
| [correctness.json](correctness.json) | 输出、输入梯度和专家权重梯度的误差 |
| `logs/`、`profiling/` | 本地产物，外部归档地址和校验值写入正式报告 |
