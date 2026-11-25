# 原始结果归档

每次运行独立归档，保存配置快照、运行环境、路由摘要、正确性校验、性能数据和产物索引。目录名应能关联报告中的测点或运行 ID。

[950DT EP8 BF16](950dt_ep8_bf16/README.md) 提供一个文件结构模板。大型 Profiling 文件和日志存放在外部归档，记录存储地址、SHA-256 与对应测点。

每个 run 保存配置、环境、拓扑、测点样本和报告。失败及重试按 attempt 归档。历史结果可通过 `import-ascendep` 导入并保留来源提交。 [运行命令](../../04_scripts/README.md) · [算子结果索引](../../04_scripts/operators/README.md)
