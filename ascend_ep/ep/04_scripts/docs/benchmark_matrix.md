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
| Parameter Variant | `common/variants.py` | Case-specific 参数展开（消息大小窗口、peer sweep、shape mode、机制可用性） |
| EP 环境配置 | `configs/ep{8,64,128}_bf16.yaml` | 只描述 EP 环境：EP/TP、模型、专家数、拓扑来源、rank mapping |
| Suite 配置 | `configs/{suite}.yaml` | 只描述场景：workload、benchmark 维度、operators |
| Benchmark Plan | `configs/benchmark_plan.yaml` | 组合 base → EP → Suite → Case variant，生成三个 EP Size 的完整 Manifest |
| Capability | `common/registry.py` + `common/scheduler.py` | 按 Case 级能力与参数约束判定状态 |

`common/config.py` 的 `ENVIRONMENT_SECTIONS` 明确列出「环境层」字段：plan 组合时 Suite 配置不能覆盖
`schema_version / device / distributed / model / topology / launch`。因此 `decode.yaml` 不会把 EP8 的
`num_experts` 带进 EP64。`matrix` 条目同样会被清洗（`matrix_overlay`）：`distributed.ep_size` 只作为
EP Size 选择器，`device / topology / launch` 直接丢弃，`model` 仅保留 scaling 定义所必需的
`num_experts`。这就是「Suite 只能改场景，不能改环境」的硬约束。

### 1.1 两种测试语义（plan.mode）

| 模式 | 行为 |
|---|---|
| `catalog` | 每个列出的 Suite 各自展开；`all` 只用 Base 配置。旧 Plan 的语义，也是未声明 `mode` 时的默认值 |
| `full_matrix` | `all` 展开完整 Catalog，其余每个专项 Suite 都用**自己的场景配置**展开 |

默认 `configs/benchmark_plan.yaml` 使用 `full_matrix`，并显式列出
`all + hardware + memory + transport + collectives + overlap + decode + prefill + training_tail +
high_throughput + strong_scaling + weak_scaling + topology + load_balance + dedup + quantized`。
因此 decode 用小 token、prefill 用大 token、training tail 展开 16/64/128/512、high throughput
展开大 token、strong/weak scaling 形成独立矩阵、topology 展开 high locality、load balance 启用
planner、dedup 展开 None/Rank/Server、quantized 使用自己的算子与形状约束。

完全相同的任务去重为一个 task（保留全部 `suite_ids` 与 `plan_points`）；工作负载、参数或测试边界
不同的任务保持独立。`full_matrix` 必须包含 `all`，否则 `load_plan` 直接报错，避免"看起来是完整
矩阵、实际只有几个 Suite"。

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
| `all` / `full` | 全 Catalog（Full Matrix 中承担"统一默认配置"的基准点） |
| `hardware` | L0 category ∈ {collective, compute} |
| `memory` | L0 category = memory |
| `transport` | L0 category = transport |
| `collectives` | L0 category = execution |
| `overlap` | L0 category = overlap + FUSE-001～009 + MOE-007 |
| `decode` | COMM-001、COMM-003、MOE-001、LAYER-001（小 token） |
| `prefill` | 与 decode 相同 Case，超大 token |
| `training_tail` | COMM-005/006/007、MOE-002、MOE-003（16/64/128/512 token） |
| `high_throughput` | COMM-001/003/004/008、MOE-003、MOE-009（4096/8192/16384 token） |
| `strong_scaling` | 固定全局 token，Case 集合与 weak 相同 |
| `weak_scaling` | 固定每 rank token/专家数，Case 集合与 strong 相同 |
| `topology` | COMM-001/002/003、MOE-001（balanced / high_locality） |
| `load_balance` | COMM-012、MOE-005、LAYER-003/004/008，planner 开启，train/forward 两个阶段 |
| `dedup` | COMM-010、LAYER-005/006 + `none` 对照组（COMM-001、MOE-001），None/Rank/Server |
| `quantized` | COMM-009、FUSE-008、MOE-006，小 token + forward |

Suite 不是唯一入口：所有 Case 都能通过 `all` 执行，同一个 Case 可以属于多个 Suite，
不同 Suite 下同一个 Case 可以使用不同参数（例如 decode 与 prefill 的 token 数）。

## 4. 参数矩阵

- 消息大小候选池由 `benchmark.message_bytes` 给出（4 KiB～1 GiB），每个 L0 Case 按
  `SIZE_WINDOWS` 选择窗口：延迟类 ≤1 MiB，吞吐类 ≥1 MiB，memory/overlap 各自有窗口。
  不做全维度笛卡尔积。
- Transport 机制过滤：`benchmark.transport.enabled` 与 `L0_PARAMETERS[case]["mechanism"]` 对照，
  未启用的机制不产生可执行 Variant，Case 仍留在 Manifest 中并标记 `not_applicable`（原因写明机制名）。
  显式 `[]` 表示不启用任何机制；未配置（`null`）保留旧的"全部允许"语义。
- Transport peer：`peer_counts` 包含固定 fan-out（1/2/4）与 `0`（当前策略下全部 peer）；
  `peer_policies` 支持 `ring / same_server / cross_server`。`ring` 按 rank 旋转；`same_server` /
  `cross_server` 必须由显式 Rank→Server 映射支撑，缺少映射、没有候选 Peer、或请求的
  `peer_count` 超过该策略实际可选数时，一律 `not_applicable` 并给出原因，**不会**回退到普通 Peer，
  也不会悄悄减少 Peer 数量（`baselines/l0/transport/peers.py` 与 planner 使用同一条规则）。
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
| `not_run` | 可运行，但还没有执行结果（终态缺失，不是通过） |
| `passed` | 实测通过 |
| `unsupported` | 能力/形状/参数不支持 |
| `dependency_missing` | 依赖缺失（原生桥、SHMEM、hccl_test 等） |
| `environment_unavailable` | 设备或环境不满足 |
| `not_implemented` | 适配器能力待接入 |
| `not_applicable` | 该配置与 Case 定义不符（dedup 关闭时的去重 Case、forward-only 下的训练 Case、disabled 机制、无可用跨机 Peer 等） |

`unsupported` / `dependency_missing` / `not_applicable` / `not_run` 永远不会计入 `passed`。

配置决定的状态与被跳过要区分清楚：当前 Plan 组合没有选中的测试条件（例如 planner 关闭时的
`LAYER-003`、没有 trace 的 `LAYER-007`）记为 `not_applicable` 并写明原因；`not_run` 只表示
"应该跑但还没有结果"。因此一次完整执行之后仍然存在的 `not_run` 就是真实的执行缺口。

执行结束后 `ep_summary.json` 用**互斥且穷尽**的状态桶统计每个 EP：

```json
{
  "completion_status": "incomplete",
  "require_complete_execution": true,
  "planned_tasks": 100, "completed_tasks": 90,
  "passed": 85, "failed": 2, "unsupported": 1, "not_applicable": 1,
  "dependency_missing": 1, "environment_unavailable": 0, "not_implemented": 0,
  "not_run": 10,
  "executed": 87, "blocked": 1, "unexecuted": 10,
  "consistency": {"planned_tasks": 100, "sum_of_buckets": 100, "consistent": true},
  "ep_sizes": {"8": {"completion_status": "...", "...": "..."}}
}
```

所有状态桶之和等于 `planned_tasks`；`executed = passed + failed`；`blocked` 包含
`dependency_missing / environment_unavailable / source_unavailable`。`completion_status: complete`
只要求"每个预期 task 都拿到明确终态"，**不等于全部通过**：存在 `unsupported` 或
`not_applicable` 的 Plan 可以是 complete，但这些状态不会进入性能通过数。Full Matrix 默认
`execution.require_complete_execution: true`：出现未解决的 `not_run` 时运行以非零码结束。

## 6. Manifest 与完整性校验

`common/plan.py::expand` 生成 task 后调用 `common/scheduler.py::check_completeness`：

1. 每个 EP Size 必须包含该 plan 选中的全部 canonical Case；
2. 三个 EP Size 的 Case 集合必须完全一致；
3. 不支持的组合保留 task，不删除。

每个 task 记录 `ep_size / suite_ids / plan_points / case_category / case_family / alias_of /
config_fingerprint / config_source / plan_point`。`manifest.json` 附带 `matrix` 元数据
（`mode`、catalog、alias、config points、每个 config point 的 `task_count`、`topology`、
case_index、coverage），`resolved_config.yaml` 对 plan run 按配置指纹去重存储。多 EP Size 运行时
额外写 `manifests/ep{size}.json` 索引。

### 6.1 拓扑来源与验证状态

Topology 按优先级组合，并总是记录来源：

1. EP 环境配置里的 `topology.server_ids`（部署自己的映射，plan 不会覆盖）→ `source: environment`
2. Plan 里显式的 `topology.rank_to_server` / `server_to_supernode`（支持 `[每 rank 一个域]`、
   `{rank: domain}`、`{domain: [ranks]}` 三种写法）→ `source: explicit`
3. Plan 的模式推导（`server_size` / `supernode_size`）→ `source: plan`，`verified: false`

推导出来的映射只是布局假设，**不会被当成已验证的物理拓扑**：每个 config point 与
`manifest.matrix.topology` 都记录 `source` / `verified` / `servers` / `supernodes`，
`l0_report.md` 的 EP 覆盖表也会展示。只有 `same_server` / `cross_server` 需要的映射完整
（每个 rank 恰好一个域）时才允许执行；不完整时直接 `not_applicable`。要声明"这是实测拓扑"，
在显式映射上写 `verified: true`。

### 6.2 Case × Operator × Variant 覆盖

`reporting/coverage.py::task_matrix` 以 **Manifest 的 task 列表为唯一预期来源**，逐条与 Result
核对并输出 `variant_coverage.{md,csv}` 与 `coverage_matrix.json:case_operator_variant`：

- 每个 `(EP, Case, Operator, Variant)` 的 `planned / measured / missing / failed / unsupported /
  not_applicable / blocked / not_run` 计数满足 `accounted == planned`；
- 缺失 Variant 精确到 `task_id`，不会被折叠成"整个 Case 缺失"；
- Manifest 中没有的 Result 记录报告为 `orphans`，预期但完全没有 task 的 Case 报告为
  `cases_without_tasks`。

## 7. 报告

`reporting/coverage.py` 生成：

- `coverage_matrix.{md,csv,json}`：Case × EP Size 最差状态（`missing` / `out_of_scope` / `alias` 显式区分）；
- `case_operator_coverage.csv`：EP Size × Case × Operator 三维覆盖（状态、任务数、原因）；
- `variant_coverage.{md,csv}`：EP Size × Case × Operator × Variant，预期 task 与 Result 一一核对，
  缺失 Variant 精确到 `task_id`；
- `operator_coverage.md`：EP Size × Operator 的 passed/failed/unsupported/dependency_missing/not_run 统计；
- `ep_size_comparison.md`：同 Case、同 Operator、同参数变体、同模型、同每 rank workload、
  同计时边界的跨 EP Size 对比；L0 仅比较 `verified` 路径；
- `l0_scaleout.md`：L0 的 EP Size 扩展视图。分组键是
  `(case, category, observed-or-requested mechanism, execution resource, 全部 case 参数的 fingerprint)`，
  因此 `ring / same_server / cross_server`、`device / host`、`gmm / dgrad / wgrad`、
  `fixed / workload` 都是独立行；同一 EP Size 出现多个不同 task 时报告 `ambiguous` 而不是互相覆盖；
  `N/A` 保留具体原因（例如 EP8 没有跨机 Peer），`-` 表示该 EP Size 没有计划这个 Variant。

`reporting/l0_report.py` 额外输出 `requested / observed / verification`、每个 EP Size 的
L0 覆盖情况以及拓扑来源与验证状态。

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
Suite 映射与去重、Strong/Weak Scaling、Full Matrix 组合与去重、拓扑来源、执行完整性统计、
Case×Operator×Variant 覆盖、L0 scale-out 分组、覆盖率与性能对比报告、CLI 组合。
`tests/test_l0.py` 覆盖：transport 机制过滤（enabled / 空列表 / 未配置）、peer 策略（无跨机 Peer、
无拓扑、peer_count 超限、确定性轮转、不回退）、capability 矩阵与路径验证。
所有测试均为离线规划/聚合测试，不代表真实 NPU 性能。

常用命令：

```sh
python -m pytest -q tests/test_matrix.py tests/test_l0.py tests/test_framework.py
python -m pytest -q tests/test_integration.py      # CPU 双 rank Gloo，需要 PyTorch
```
