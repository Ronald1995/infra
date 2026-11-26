# EP Benchmark 矩阵

测试范围由一份 Benchmark Plan 展开，见 [测试矩阵设计](../04_scripts/docs/benchmark_matrix.md) 与
[`configs/benchmark_plan.yaml`](../04_scripts/configs/benchmark_plan.yaml)。

```
EP Size × Suite × Case × Operator × Parameter Variant
```

每个测点须满足目标平台、拓扑和后端能力；不满足的组合保留 `unsupported / dependency_missing /
not_implemented / not_applicable` 状态与原因，不会从 Manifest 中删除。

| 维度 | 当前执行定义 |
|---|---|
| 硬件与拓扑 | A3、950DT；单机、超节点内、跨超节点（`benchmark_plan.yaml` 的 `topology` 生成 rank→server/supernode 映射） |
| EP size | 8、64、128（同一份 Case Catalog，EP 配置只描述环境） |
| 层级 | L0、L1、L2、L3、L4 全部纳入完整计划 |
| L0 | MEM-001～004、LINK-001～013、EXEC-001～006、HW-001～006、HW-010～011、OVERLAP-001～009 |
| L1 | COMM-001～012 |
| L2 | FUSE-001～009 |
| L3 | MOE-001～009 + LAYER-007 |
| L4 | LAYER-001～006、008、009 |
| 算子 | 注册表中的全部算子；按 Case 级能力判定，不按 README 的层级声明判定 |
| Tokens per rank | Decode 16、Prefill 4096、Training tail 16/64/128/512、高吞吐 4096/8192/16384 |
| 路由分布 | 均衡、随机、热点专家、rank/server 热点、高局部性、跨服务器、去重受控、真实 trace replay |
| 负载均衡 | Planner Off（固定归属 HCCL 对照）/ On；inference 与 training 两种 phase |
| 去重 | None（COMM-001/MOE-001 对照）/ Rank / Server |
| 执行模式 | Forward / Backward / Forward + Backward |
| 参数变体 | Case-specific：消息大小窗口、peer fan-out、fixed/workload 计算 Shape、chunk/tile/plan 变体 |

## 完整 Case Catalog

- 总量由 `common/cases.py` 动态计算（当前 83 个 ID：79 canonical + 4 legacy alias）。
- `HW-007/HW-008/HW-009/HW-012` 分别 alias 到 `LINK-012/LINK-013/MEM-001/OVERLAP-001`，
  默认不重复生成 task，但在 manifest 与覆盖率报告中显式记录。
- 报告中的 Case 总数、覆盖率与通过数都从实际 Catalog 和实际结果计算，不硬编码。

## 专项测试集合

- **L0 公共基线**：硬件与通信基线在每个 EP Size 下各执行一次，不与算子做笛卡尔积。
- **基础性能**：EP8 BF16 均衡路由，分层建立通信与专家子层基线。
- **规模扩展**：强扩展固定全局 token，弱扩展固定每 rank token/专家数；两者 Case 集合一致，只改工作负载生成规则。
- **负载均衡与去重**：同一份路由输入下对比 Planner、固定归属 HCCL 和 None/Rank/Server 去重。
- **量化与推理**：为量化、Prefill 和 Decode 分别选择适合的 dtype、token 数和调用路径。

参数模板见 [脚本目录](../04_scripts/README.md)。配置是否可执行由具体后端与 Case 级 capability 检验。
