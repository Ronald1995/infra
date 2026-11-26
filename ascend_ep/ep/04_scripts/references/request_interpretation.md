# Request interpretation

The skill turns one sentence into `scripts/workflow/request.schema.json`. Two things are
fixed: **interpreting the intent** and **resolving the defaults**. Everything else is
overridable by the caller, and the resolved request is shown before anything runs.

## Intent

| Intent | Trigger words (examples) | What the workflow does |
|---|---|---|
| `full_benchmark` | 完整 / 全部 / full / 整个 / 全量 | every phase, every suite the plan declares |
| `partial_benchmark` | 只跑 / 只测 / only run, or an explicit EP/level list | every phase, narrowed scope |
| `smoke_test` | smoke / 冒烟 / 小规模 / 快速验证 | every phase, small scope (CPU runs use `configs/smoke_cpu.yaml`) |
| `plan_only` | 只生成计划 / 不要执行 / plan only / dry run | up to `resource_estimation`, nothing executed |
| `estimate` | 估算 / 预估 / estimate / 需要多少时间 | up to `resource_estimation` |
| `resume` | 继续 / 恢复 / 上次 / resume / unfinished | resume the latest unfinished workflow |
| `profile` | profiling / 性能剖析 / 采集 | profiling + analysis + report of an existing run |
| `analyze` | 分析已有 / 只分析 / re-analyze | analysis + report of an existing run |
| `report` | 只生成报告 / 重新生成报告 | report + final validation of an existing run |
| `environment_check` | 环境检查 / 检查环境 / doctor only | request + environment + dependency + capability |

Order matters when a prompt contains several words: `resume` > `environment_check` >
`estimate` > `plan_only` > `profile`/`analyze` > `report` > `smoke` > `partial` > `full`.
"估算…不要执行测试" is an estimate, not a plan; "只跑 EP64" is partial, not full.

## Scope

* **EP sizes** come from `EP<n>` tokens anywhere in the prompt. An explicit list replaces
  the plan's list; it never narrows anything else.
* **Levels** come from `L0`..`L4`. A range (`L0-L4`, `L1 到 L3`) expands.
* **Suites, operators and cases** are *not* parsed from prose. They come from the plan.
  Switching to a narrower suite/operator set is a structured edit of the request, not a
  guess from a sentence — a wrong guess there changes the test scope silently.
* An explicit dimension overrides only itself. "EP64, L1-L2" keeps every suite and
  operator the plan declares.

## Safety policy

| Field | Default | Interpretation |
|---|---|---|
| `environment.dependency_policy` | `check_only` | `install_missing` only when the prompt says 安装 / install / 缺少依赖就 |
| `environment.allow_network` | `false` | only 允许网络 / 联网 / download / 下载 |
| `environment.allow_sudo` | `false` | only 允许 sudo |
| `execution.profile_policy` | `representative` | `full` for 全部/所有 profiling, `none` for a plan/estimate/report request |
| `execution.time_budget_minutes` | unset | "30 分钟" / "2 小时" |
| `execution.nnodes` | `1` | multi-node is never inferred; it must be stated |

`full_benchmark` **never** implies sudo, network access or a driver change. Installation
is a separate, explicit permission.

## Defaults and their sources

Every resolved field records where it came from (`sources` in the request):

* `plan:configs/benchmark_plan.yaml` — read from the plan file,
* `prompt` — parsed from the instruction,
* `default:<why>` — a documented default (for example `default:representative`,
  `default:check_only (no install authorization)`),
* `intent:<why>` — implied by the intent (for example a CPU run cannot install a driver).

The orchestrator prints the resolved request and writes `resolved_request.json` before the
first phase; a wrong interpretation is visible there, not 20 000 tasks later.

## When to ask instead of guess

Only when a missing value cannot be derived and would change the measurement:

* multi-node placement (nodes, `master_addr`, `master_port`, `session`, devices per node),
* the physical topology when the plan's derived layout is not the deployment's,
* the target SoC when it cannot be read from `npu-smi`,
* installing something that needs sudo or the network when the prompt did not say so.

Anything else has a default from the plan; asking the user to repeat it is not autonomy.
