# EP Benchmark 测试矩阵设计

本目录把 EP Benchmark 从「若干独立配置」升级为「一份 Benchmark Plan 生成完整测试矩阵」。核心目标是：

```
EP Size × Suite × Case × Operator × Parameter Variant
```

并且 **不能因为 EP Size、Suite、YAML 继承或 Operator Capability 差异而静默丢失用例**。

## 1. 分层职责

| 层 | 位置 | 职责 |
|---|---|---|
| Case Catalog | `common/cases.py` | 唯一的 Case 定义来源（83 个 ID / 79 个 canonical + 4 个 legacy alias） |
| Suite | `common/suites.py` | 按 level / category / case id 选择 Case，不定义 Case |
| Parameter Variant | `common/variants.py` | Case-specific 参数展开（消息大小窗口、peer sweep、shape mode） |
| EP 环境配置 | `configs/ep{8,64,128}_bf16.yaml` | 只描述 EP 环境：EP/TP、模型、专家数、拓扑来源、rank mapping |
| Suite 配置 | `configs/{suite}.yaml` | 只描述场景：workload、benchmark 维度、operators |
| Benchmark Plan | `configs/benchmark_plan.yaml` | 组合 base → EP → Suite → Case variant，生成三个 EP Size 的完整 Manifest |
| Capability | `common/registry.py` + `common/scheduler.py` | 按 Case 级能力与参数约束判定状态 |

`common/config.py` 的 `ENVIRONMENT_SECTIONS` 明确列出「环境层」字段：plan 组合时 Suite 配置不能覆盖
`schema_version / device / distributed / model / topology / launch`。因此 `decode.yaml` 不会把 EP8 的
`num_experts` 带进 EP64。Suite 的 `matrix` 中标记了 `distributed.ep_size` 的条目只应用于对应 EP Size
（强/弱扩展配置就是这种写法）。

## 2. 展开规则

### L0

```
EP Size × L0 Case × Parameter Variant
```

L0 是公共硬件基线，**不做 operator 笛卡尔积**，`backend_id` 固定为 `hardware`。六类 Case：

| 类别 | Case | Suite |
|---|---|---|
| Local Memory | MEM-001～004 | `memory` |
| Native Transport | LINK-001～013 | `transport` |
| Communication Execution | EXEC-001～006 | `collectives` |
| Framework Collective | HW-001～006 | `hardware` |
| Compute | HW-010～011 | `hardware` |
| Overlap | OVERLAP-001～009 | `overlap` |

L0 在每个 EP Size 下都完整执行一次；共享单卡基线不会被算作多 Rank 并发测试的替代品。

### L1～L4

```
EP Size × Case × Operator × Parameter Variant
```

36 个已注册 operator 全部参与能力检查。是否可运行由 Case 的 `requires`、operator 的
`capabilities/constraints` 和参数约束共同决定。不支持的组合仍然生成 task，并写入
`status` 与 `reason`。

### Legacy Alias

`HW-007 → LINK-012`、`HW-008 → LINK-013`、`HW-009 → MEM-001`、`HW-012 → OVERLAP-001`。
默认 `aliases: dedupe`：alias 不生成重复 task，但在 `manifest.matrix.case_index` 中以
`{"status": "alias", "alias_of": ...}` 记录，覆盖率报告同样显示 `alias->CANONICAL`。
`--cases HW-007` 显式指定时仍会执行该 alias（用户明确要求）。

## 3. Suite

| Suite | 选择方式 |
|---|---|
| `all` / `full` | 全 Catalog（默认完整模式） |
| `hardware` | L0 category ∈ {collective, compute} |
| `memory` | L0 category = memory |
| `transport` | L0 category = transport |
| `collectives` | L0 category = execution |
| `overlap` | L0 category = overlap + FUSE-001～009 + MOE-007 |
| `decode` | COMM-001、COMM-003、MOE-001、LAYER-001（小 token） |
| `prefill` | 与 decode 相同 Case，超大 token |
| `training_tail` | COMM-005/006/007、MOE-002、MOE-003 |
| `high_throughput` | COMM-001/003/004/008、MOE-003、MOE-009 |
| `strong_scaling` | 固定全局 token，Case 集合与 weak 相同 |
| `weak_scaling` | 固定每 rank token/专家数，Case 集合与 strong 相同 |
| `topology` | COMM-001/002/003、MOE-001 |
| `load_balance` | COMM-012、MOE-005、LAYER-003/004/008 |
| `dedup` | COMM-010、LAYER-005/006 + `none` 对照组（COMM-001、MOE-001） |
| `quantized` | COMM-009、FUSE-008、MOE-006 |

Suite 不是唯一入口：所有 Case 都能通过 `all` 执行，同一个 Case 可以属于多个 Suite，
不同 Suite 下同一个 Case 可以使用不同参数（例如 decode 与 prefill 的 token 数）。

## 4. 参数矩阵

- 消息大小候选池由 `benchmark.message_bytes` 给出（4 KiB～1 GiB），每个 L0 Case 按
  `SIZE_WINDOWS` 选择窗口：延迟类 ≤1 MiB，吞吐类 ≥1 MiB，memory/overlap 各自有窗口。
  不做全维度笛卡尔积。
- Transport：`peer_counts` 包含固定 fan-out（1/2/4）与 `0`（全部 peer）；`peer_policies`
  支持 `ring / same_server / cross_server`，`ring` 按 rank 旋转，避免固定 peer 数在
  EP8/EP128 选到不同链路类型。SHMEM 只记录 `group` 策略。
- Compute：`shape_mode=fixed` 使用 `benchmark.compute.fixed_shape`（M/N/K/groups 跨 EP
  一致），`shape_mode=workload` 使用真实 dispatch 后的 expert rows。两种模式不会互相比较。
- Overlap：通信字节数取自 `message_bytes`，计算量取自 dispatch 后的 GMM/SwiGLU rows；
  结果记录通信/计算/并发耗时与设备事件交集，无法测量真实重叠时
  `overlap_verified=false` 并写明原因。
- Scaling：`strong` 用 `scaling.global_tokens / ep_size`（必须整除），
  `weak` 用固定 `tokens_per_rank` 和 `experts_per_rank × ep_size`。
- AllToAll：`message_bytes` 不能被 EP Size 整除时直接生成 `unsupported` task 并说明原因。

## 5. Capability 状态

| 状态 | 含义 |
|---|---|
| `not_run` | 可运行，但尚未执行 |
| `passed` | 实测通过 |
| `unsupported` | 能力/形状/参数不支持 |
| `dependency_missing` | 依赖缺失（原生桥、SHMEM、hccl_test 等） |
| `environment_unavailable` | 设备或环境不满足 |
| `not_implemented` | 适配器能力待接入 |
| `not_applicable` | 该配置与 Case 定义不符（如 dedup 关闭时的去重 Case、forward-only 下的训练 Case） |

`unsupported` / `dependency_missing` / `not_applicable` 永远不会计入 `passed`。

## 6. Manifest 与完整性校验

`common/plan.py::expand` 生成 task 后调用 `common/scheduler.py::check_completeness`：

1. 每个 EP Size 必须包含该 plan 选中的全部 canonical Case；
2. 三个 EP Size 的 Case 集合必须完全一致；
3. 不支持的组合保留 task，不删除。

每个 task 记录 `ep_size / suite_ids / case_category / case_family / alias_of /
config_fingerprint / config_source / plan_point`。`manifest.json` 附带 `matrix` 元数据
（catalog、alias、config points、case_index、coverage），`resolved_config.yaml` 对 plan run
按配置指纹去重存储。多 EP Size 运行时额外写 `manifests/ep{size}.json` 索引。

## 7. 报告

`reporting/coverage.py` 生成：

- `coverage_matrix.{md,csv,json}`：Case × EP Size 最差状态（`missing` / `out_of_scope` / `alias` 显式区分）；
- `case_operator_coverage.csv`：EP Size × Case × Operator 三维覆盖（状态、任务数、原因）；
- `operator_coverage.md`：EP Size × Operator 的 passed/failed/unsupported/dependency_missing/not_run 统计；
- `ep_size_comparison.md`：同 Case、同 Operator、同参数变体、同模型、同每 rank workload、
  同计时边界的跨 EP Size 对比；L0 仅比较 `verified` 路径；
- `l0_scaleout.md`：L0 机制/消息大小/peer 数的 EP Size 扩展视图。

`reporting/l0_report.py` 额外输出 `requested / observed / verification` 与每个 EP Size 的
L0 覆盖情况。

## 8. CLI

```sh
python run_benchmark.py plan   --plan configs/benchmark_plan.yaml
python run_benchmark.py run    --plan configs/benchmark_plan.yaml
python run_benchmark.py run    --plan configs/benchmark_plan.yaml --ep-size 64
python run_benchmark.py run    --plan configs/benchmark_plan.yaml --suite overlap
python run_benchmark.py run    --plan configs/benchmark_plan.yaml --cases COMM-001,COMM-003
python run_benchmark.py run    --plan configs/benchmark_plan.yaml --operators deepep_ascend,moonep
python run_benchmark.py capabilities --plan configs/benchmark_plan.yaml
python run_benchmark.py run    --config configs/ep8_bf16.yaml --cases COMM-001   # 兼容单配置
```

`--config` 与 `--plan` 互斥；`--cases` 是用户显式缩小范围，不再受配置默认 level 限制。

## 9. 扩展方式

- 新增 Case：只在 `common/cases.py` 添加定义，并按类别加入 Suite（如属于现有 Suite 的
  category 则无需改 Suite 表）。
- 新增 Suite：在 `common/suites.py` 增加声明式 Suite，并在 `configs/benchmark_plan.yaml`
  的 `suite_configs` 中登记可选场景文件。
- 新增参数维度：在 `common/variants.py` 增加 Case-specific 展开函数，不要在
  `scheduler.py` 里硬编码。
- 新增 EP Size：新增 `configs/ep{N}_bf16.yaml`（只写环境），并在 plan 的 `ep_configs`
  登记；`check_completeness` 会验证它与其它 EP Size 的 Case 集合一致。

## 10. 测试

`tests/test_matrix.py` 覆盖：EP Size 覆盖一致性、L0 展开与 alias、L1～L4 展开与 capability、
Suite 映射与去重、Strong/Weak Scaling、拓扑/peer 约束、覆盖率与性能对比报告、CLI 组合。
所有测试均为离线规划/聚合测试，不代表真实 NPU 性能。
