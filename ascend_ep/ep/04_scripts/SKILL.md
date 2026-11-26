---
name: ep-benchmark
description: Use this skill whenever the user asks to run, execute, benchmark, test, profile, compare, resume, or analyze Ascend EP/MoE communication operators. Interpret natural-language benchmark requirements, automatically prepare the environment, resolve installable dependencies, inspect hardware capabilities, build available kernels, generate and validate the test matrix, execute benchmarks, recover interrupted runs, analyze performance, and deliver verified reports.
---

# EP Benchmark: autonomous benchmark agent

This skill turns one instruction into a complete, auditable benchmark workflow:

> 在 Ascend 950 上运行完整 EP Benchmark，EP8、EP16、EP32、EP64、EP128，覆盖 L0-L4 和所有算子。
> 如果缺少依赖，尝试安装；如果需要编译 Kernel，尝试编译。完成后进行性能分析和
> Profiling，生成完整报告。

The user never edits a YAML file and never needs to know which phase runs which command.
You interpret the request, hand it to the workflow orchestrator, and report the result.

**Hard rule: this file does not implement a benchmark.** Every phase calls an existing
command (`run_benchmark.py`, `scripts/doctor.py`, `scripts/install_deps.sh`,
`kernels/asc_comm/build.sh`). Never re-implement the scheduler, the runner, the timer or the
case catalog to make the workflow simpler.

## 1. When this skill applies

Trigger on any request to run, benchmark, test, measure, profile, compare, resume, analyze
or report an Ascend EP / MoE / dispatch / combine / GMM / AllToAll benchmark. Also trigger
on requests that only ask about feasibility ("can this machine run EP64?"), coverage,
dependencies or timing estimates. Do **not** trigger for a code change to the benchmark
itself.

## 2. Turn the instruction into a request

Write a `request.json` (schema: `scripts/workflow/request.schema.json`) or let the
interpreter do it:

```sh
python scripts/benchmark_workflow.py interpret "<the user's sentence>" --out request.json
```

| The user says | intent | what changes |
|---|---|---|
| 完整 / 全部 / full Benchmark | `full_benchmark` | nothing narrowed; the plan's scope |
| 只跑 EP64 的 L1、L2 | `partial_benchmark` | only `scope.ep_sizes` / `scope.levels` |
| 冒烟 / smoke / 快速验证 | `smoke_test` | small scope; CPU hosts use `configs/smoke_cpu.yaml` |
| 估算…不要执行 | `estimate` | stops after `resource_estimation` |
| 只生成计划 | `plan_only` | stops after `resource_estimation` |
| 继续上次没跑完的 | `resume` | resumes the latest unfinished workflow |
| 执行 Profiling | `profile` | profiling + analysis + report of an existing run |
| 分析已有结果 / 只生成报告 | `analyze` / `report` | only the downstream phases |
| 检查环境 | `environment_check` | doctor + dependencies + capability only |

Rules for scope:

* **Defaults come from the plan**, not from this document. EP sizes, levels and suites are
  read from `configs/benchmark_plan.yaml` (EP8/16/32/64/128, L0-L4, all suites, `profile: full`) unless the user
  names others.
* An explicit value overrides **only its own dimension**. "EP64, L1-L2" narrows the EP size
  and the level; it does not drop suites, operators or cases.
* Profiling defaults to `representative` (one passing task per EP size × level × operator),
  never to the whole matrix.
* Ask the user **only** for what cannot be derived: multi-node placement, the real physical
  topology, an unreadable SoC, or a permission you do not have (sudo, network).
* Never ask the user to repeat what the plan already declares.

Details, including the precedence between trigger words: `references/request_interpretation.md`.

## 3. Run the workflow

```sh
python scripts/benchmark_workflow.py execute --request request.json
python scripts/benchmark_workflow.py plan    --request request.json   # plan + estimate only
python scripts/benchmark_workflow.py resume  --workflow-id <id>
python scripts/benchmark_workflow.py status  --workflow-id <id> --json
```

Phases, in order, each with a gate (`references/workflow_gates.md`):

```
request_resolution -> environment_discovery -> dependency_check -> dependency_install
-> capability_probe -> kernel_build -> plan_generation -> resource_estimation
-> benchmark_execution -> result_validation -> profiling -> analysis
-> report_generation -> final_validation
```

The workflow decides the phases; a phase that is not part of the intent is recorded as
`skipped` with that reason. Never call the phases yourself in a different order: the
orchestrator is what makes the run resumable and auditable.

State lives in `05_results/workflows/<workflow_id>/`:

```
state.json  events.jsonl  request.json  environment.json  dependency_report.json
capability_report.json  kernel_report.json  plan_manifest.json  resource_estimate.json
execution_summary.json  result_validation.json  profiling_summary.json
analysis_summary.json  report_summary.json  final_report.json  logs/
```

## 4. Automation rules

1. **One request, one run.** Execution is always `run` → `profile` → `analyze` → `report`.
   Never `run` and then `pipeline`; never start a second run for the same request. An
   incomplete run continues with `resume --run-id <same id>`.
2. **An exit code is not a task status.** `run` exits non-zero when a task failed or is
   blocked. If every task reached a terminal status, continue to analysis and report, and
   say what failed. Only a run with no manifest or an inconsistent summary is a failed
   phase.
3. **A blocked item blocks its own scope.** A missing operator dependency, a missing AIV/CCU
   kernel, or an EP size larger than the host blocks exactly the tasks that need it. The rest
   still runs. Never measure AIV/CCU cases through the host NBI path.
4. **`not_run` is not a pass**, and `completion_status: complete` only means every task
   reached a terminal status. Report the difference.
5. **Nothing is invented.** No backend ranking from one measurement, no scaling trend from
   one EP size, no overlap from a latency ratio, no physical bandwidth from a byte model, no
   CPU result presented as Ascend performance.
6. **Every claim is traceable.** Each phase writes its JSON artifact and every command keeps
   its exit code and log path.

## 5. Safety boundaries

Allowed without asking (when the request implies them):

* read-only checks: `doctor`, `capabilities`, `--dry-run`, `--check-only`;
* plan generation, resource estimation;
* the benchmark the user asked for;
* installing a dependency the catalog marks installable **only** with
  `dependency_policy: install_missing` **and** network or a wheelhouse: the installer is
  manifest-driven and pinned, so it fetches/builds exactly `dependencies/manifest.yaml`
  (never an unpinned checkout) into that dependency's environment;
* building a kernel whose source and toolchain are both present;
* analysis and report generation.

Requires explicit permission in the request: network downloads, `sudo`, system packages,
multi-node launches, profiling runs that would exceed the user's budget.

Never, under any permission: install or upgrade the driver / firmware / CANN / device nodes,
change system network configuration, delete an existing build artifact or a pinned source
cache, delete historical results, kill another user's process, edit the plan file to hide a
failure, or switch CANN/PyTorch versions. A full-benchmark request is not a
system-administration permission.

The vendored adapter tree needs no installation and no external checkout: verify it with
`python scripts/vendor_ascendep.py --check`. Whether a dependency is installable, host-only
or absent is decided by the catalog's `ownership` + `pin` + `install_spec`, and per-operator
readiness is in `python scripts/dependency_report.py`; deployment from a bare host is
`docs/installation.md`.

Details: `references/dependency_policy.md`, `references/execution_policy.md`.

## 5.5 Coverage profiles and what a run must prove

A run narrows the matrix only through a declared execution profile; it never silently
drops a requirement:

| Profile | Scope | Use |
|---|---|---|
| `smoke` | one case per level (L0, L1), one value per swept dimension | environment and correctness check |
| `core` | every EP size, every standard case, every applicable operator, all dimension values, standard suites only | baseline comparison |
| `full` | the plan's whole suite list, including routing targets, stragglers, combined long tail, topology, scaling, dedup, quantized | the default benchmark |
| `exhaustive` | full plus the legacy alias identifiers as their own tasks | deep parameter study |

`--execution-profile` selects it for one invocation; the plan's own `profile:` is the
default. The manifest records the resolved profile, so a `smoke` run can never be
reported as benchmark coverage.

Before delivering, verify the four coverage numbers from `analysis/results.json`
(`coverage_metrics`): planning, execution, correctness and verified performance. They
are separate on purpose: planning can be 100% while a mechanism without an
implementation keeps verified coverage below it, and the report must say which one is
low and why.

Routing targets and slow-rank injections are *paired* measurements: every stressed
point needs a baseline under the same condition, and the long-tail section reports the
slowdown, retention, amplification and calibration error -- or lists the point as
`not_applicable` with the reason when no baseline exists. Never quote a slowdown ratio
without its baseline.

## 6. Failure and recovery

Resume the workflow, do not restart the run:

```sh
python scripts/benchmark_workflow.py resume --workflow-id <id>
```

The workflow resumes at its first phase that is not `passed`/`skipped`, keeps the run id,
and invalidates the plan when the request hash or the repository commit changed. A
retryable task status (`runtime_failed`, `timeout`, `oom`, ...) is retried by `resume`;
`correctness_failed` is not retried blindly. Details: `references/recovery_policy.md`.

## 7. Deliver

Report, in this order:

1. **what ran** — workflow id, run id, device, EP sizes, levels, commit;
2. **what passed** — counts, sample counts, which groups are actually comparable;
3. **what is blocked and why** — dependency, environment, unsupported, kernel, capability,
   with the recorded reason for each;
4. **what was not verified** — no profile, no L0 verification, single measurement;
5. **where the evidence is** — `reports/benchmark_report.md`, `reports/*`, raw `metrics.json`,
   `05_results/workflows/<id>/`.

Acceptance gates for the delivery (`references/report_acceptance.md`): the status buckets sum
to the planned tasks, coverage accounts for every variant, and every link in the main report
resolves. Final `partial` is a real delivery; final `failed`/`blocked` means explain what
stopped it.
