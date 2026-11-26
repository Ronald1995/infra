# Autonomous benchmark workflow

The skill (`SKILL.md`) interprets an instruction; `scripts/benchmark_workflow.py` executes
it. This document records the architecture, the change it made to the previous procedure and
two recorded runs.

## 1. Before and after

### Before

```
human
  |
  +-- reads SKILL.md (ten steps, ten commands)
  +-- decides which profile, which plan, which EP size
  +-- runs doctor -> install_deps -> capabilities -> build -> plan -> run
  |          -> validate -> profile -> analyze -> report, by hand
  +-- on failure: decides alone whether to re-run, resume or stop
  `-- copies numbers out of the report into the answer
```

The procedure was accurate but manual: nothing recorded which step ran, nothing could be
resumed, and a prompt like "run the full benchmark and install what is missing" had to be
translated by the human into six commands with the right flags.

### After

```
prompt
  |
  v
scripts/workflow/request.py     intent + scope + safety policy, defaults read from the plan
  |
  v
scripts/benchmark_workflow.py   phase loop, gates, state, events, resume
  |
  +-> environment.py   doctor --profile <p> --json        -> environment.json + rank capacity
  +-> dependency.py    doctor checks + install_deps.sh    -> dependency_report.json
  +-> capability.py    capabilities --plan --runtime      -> capability_report.json
  +-> kernel.py        kernels/asc_comm/build.sh          -> kernel_report.json
  +-> planning.py      run_benchmark.py plan [+ n tasks]  -> plan_manifest.json
  |                    resource estimate from history     -> resource_estimate.json
  +-> execution.py     run | resume -> ep_summary.json    -> execution_summary.json
  |                    result_validation                  -> result_validation.json
  +-> profiling.py     profile --task-ids/--cases/...     -> profiling_summary.json
  +-> analysis.py      analyze                            -> analysis_summary.json
  +-> reporting.py     report + accounting + links        -> report_summary.json
                                                          -> final_report.json
```

Every arrow is an existing command. The new code is about 3 000 lines of orchestration, state
and gates; it contains no scheduler, runner, timer, case catalog or metric definition.

## 2. Module map

| Module | Responsibility |
|---|---|
| `scripts/workflow/request.py` | intent/scope/safety interpretation, plan-derived defaults, `request.json` schema, `request_resolution` handler |
| `scripts/workflow/state.py` | phase table, statuses, `state.json`, `events.jsonl`, resume points, invalidation |
| `scripts/workflow/phases.py` | phase specifications, gates, `CommandRunner`, `PhaseOutcome`, the execution loop |
| `scripts/workflow/environment.py` | doctor + rank capacity: which EP sizes this host can run |
| `scripts/workflow/dependency.py` | classify doctor findings, permitted install, doctor re-check |
| `scripts/workflow/capability.py` | static per-EP support and the staged runtime probe |
| `scripts/workflow/kernel.py` | kernel source/toolchain/artifact check, build without `--clean` |
| `scripts/workflow/planning.py` | temporary narrowed plan, manifest counts, resource estimate |
| `scripts/workflow/execution.py` | run/resume, manifest-hash guard, correctness accounting |
| `scripts/workflow/profiling.py` | profiling policy, selection, coverage |
| `scripts/workflow/analysis.py` | analyze + "is this analysis of this run?" |
| `scripts/workflow/reporting.py` | report + delivery gate + link check |
| `scripts/benchmark_workflow.py` | CLI: `interpret`, `execute`, `plan`, `estimate`, `resume`, `status` |

## 3. What changed in the existing code

| File | Change | Why |
|---|---|---|
| `run_benchmark.py` | `capabilities --plan ... --runtime --json` now actually probes (it returned before the probe); `--ep-size` accepts a list; `profile` accepts `--ep-size`, `--operators`, `--task-ids` | the workflow needs a runtime verdict, a multi-EP plan and a precise profiling selection |
| `common/controller.py` | `profile_selected(task, args)` | profiling a 20k-task matrix is never the goal |
| `SKILL.md` | rewritten as an agent contract (trigger, interpretation, defaults, phases, safety, recovery, delivery) | it was a manual operating procedure |
| `references/*.md` | seven policy documents | keep SKILL.md short while stating the rules precisely |

Nothing in `common/plan.py`, `common/scheduler.py`, `common/runner.py`, `common/timer.py`,
`analysis/` or `reporting/` changed for this feature.

## 4. State and evidence

```
05_results/workflows/<workflow_id>/
  request.json  resolved_request.json  state.json  events.jsonl
  environment.json  dependency_report.json  capability_report.json  kernel_report.json
  plan_manifest.json  resource_estimate.json  execution_summary.json
  result_validation.json  profiling_summary.json  analysis_summary.json
  report_summary.json  final_report.json
  logs/                     # one log per command, with its exit code
  preview/<id>-plan/        # the validated manifest, never the measured one
  plan/request_plan.yaml    # the temporary narrowed plan (repository config untouched)
```

`state.json` carries `completed_phases`, `blocked_phases`, `run_id`, `repo_commit`,
`request_hash`, `plan_hash` and every command with its exit code and log path. A workflow is
resumable from that file alone.

## 5. Recorded runs

### 5.1 CPU smoke, complete flow

Command:

```sh
python scripts/benchmark_workflow.py execute \
  --prompt "在当前 CPU 环境测试整个框架" --workflow-id wf-cpu-smoke
```

Result: `partial` (a real delivery: the unsupported points are listed, nothing is faked).

```
request_resolution     passed
environment_discovery  passed
dependency_check       partial   3 installable dependencies are unsatisfied (torch, numpy, pyyaml)
dependency_install     skipped   policy check_only: nothing installed
capability_probe       passed
kernel_build           skipped   a CPU run needs no Ascend device kernel
plan_generation        passed    93 tasks planned, 39 executable
resource_estimation    partial   no historical timing matches these tasks
benchmark_execution    passed    run completed, every task terminal
result_validation      partial   39 passed, 37 unsupported, 17 not applicable
profiling              skipped   the request set profile_policy to none
analysis               passed
report_generation      passed
final_validation       partial   39 passed, 37 unsupported + 17 not applicable, all explained
```

Accounting from `final_report.json`: `planned 93`, buckets `passed 39 + unsupported 37 +
not_applicable 17 = 93`, coverage `planned 93 / matched 93 / missing 0`, no passed task
without a correctness record.

The run directory is `05_results/raw/wf-cpu-smoke/`; its `reports/benchmark_report.md` is the
delivery, and `05_results/workflows/wf-cpu-smoke/` holds the workflow evidence.

### 5.2 NPU request on a host without a device

Command:

```sh
python scripts/benchmark_workflow.py plan \
  --prompt "在 Ascend 950 上运行完整 EP Benchmark，EP8、EP64、EP128，覆盖 L0-L4 和所有算子。" \
  --workflow-id wf-npu-plan
```

Result: `blocked` at `environment_discovery`, before any plan or measurement:

```
environment_discovery  blocked  no requested EP size can run on this host:
                                EP8: no Ascend device was detected (source: unavailable) ...
```

This is the intended behaviour: the workflow does not run the NPU matrix on a host without an
NPU, and it does not silently degrade EP128 to a smaller group.

## 6. Tests

```
python -m pytest tests/test_workflow_state.py tests/test_workflow_resume.py \
                 tests/test_workflow_gates.py tests/test_workflow_execution.py -q
53 passed
```

The harness (`tests/conftest.py::FakeBenchmark`) models a benchmark host: doctor output,
capability JSON, plan/run/profile/analyze/report artifacts, dependency checks and failure
modes. A whole workflow therefore runs offline, writes real files in the real shapes, and
asserts the decisions (one run, no pipeline, gates, resume, blocked scope).
