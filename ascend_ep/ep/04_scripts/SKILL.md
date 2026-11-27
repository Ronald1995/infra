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
command (`run_benchmark.py`, `scripts/npu_smoke.py`, `scripts/doctor.py`,
`scripts/install_deps.sh`, `kernels/asc_comm/build.sh`). Never re-implement the scheduler,
the runner, the timer or the case catalog to make the workflow simpler.

## 1.1 Formal tests run on NPU only (no CPU fallback)

Two execution modes exist and they are **not** interchangeable backends
(`common/execution_mode.py`):

| mode | who may use it | counts as operator evidence |
|---|---|---|
| `npu_only` (default) | every formal Smoke and Benchmark run | only a real `device: npu` pass |
| `framework` | framework self-checks: import, config parsing, planner expansion, constraint filtering, reporting | never |

The mode is `npu_only` unless `--execution-mode framework` (or `EP_EXECUTION_MODE`) is
passed explicitly. In `npu_only` a CPU configuration is refused *before* the scheduler
expands it, so no CPU result can be reported as operator coverage.

Never, while a formal run is in progress:

* substitute CPU for NPU, or Gloo for HCCL, in a formal communication test;
* skip a case because an NPU kernel is missing — record `dependency_missing` instead;
* mark a mock or CPU number as an operator pass;
* decide "the operator works" from a Python import alone.

If the NPU gate fails, the affected tasks are blocked with a recorded reason and everything
else still runs. A blocked task is never a pass.

## 1.2 Which command owns which stage

| stage | command |
|---|---|
| environment gates, dependencies, kernel build | `python scripts/doctor.py`, `scripts/install_deps.sh`, `kernels/asc_comm/build.sh` |
| third-party sources (offline) | `python scripts/fetch_third_party.py --check` / `--fetch` |
| isolated environments | `python scripts/env_manager.py --base-check` / `--check` / `--create` |
| native build | `python scripts/native_build.py --list` / `--run` / `--verify` |
| **NPU Smoke (L0 gate + one real run per operator)** | `python scripts/npu_smoke.py` |
| planning / full benchmark / profiling / analysis / report | `python run_benchmark.py …` |
| end-to-end orchestration of the above | `python scripts/benchmark_workflow.py` |

## 1.4 Platform support: a release note is a hint, a run is a verdict

`dependencies/manifest.yaml` defines `supported_soc` as what a *release is documented
for*. A dependency whose notes omit Ascend950 is therefore **undocumented**, not
unsupported — an earlier revision turned that gap into `BLOCKED_PLATFORM` and wrongly
dropped eleven adapters, several of which carry 950DT reference reports.

The four states live in `dependencies/platform_support.yaml` and are enforced by
`common/platform_support.py`:

| status | meaning | may block the platform? |
|---|---|---|
| `documented` | the release declares this SoC | no |
| `undocumented` | notes omit it → prepare it and let a real run decide | **no** |
| `validated` | a real run on this SoC passed (cite the run) | no |
| `rejected` | a real run or the vendor proved it cannot work (cite the evidence) | yes |

Only two things may produce `BLOCKED_PLATFORM`: an operator's own
`constraints.hardware` excluding the host SoC, or a `rejected` verdict carrying
evidence. `platform_support.promote()` refuses a rejection without evidence.

## 1.5 Dependency preparation pipeline

```
detect host SoC + CANN -> load 36 operators -> drop the operators whose own
constraints.hardware excludes the SoC (2) -> resolve the closure of the remaining 34
-> verify third_party checkouts -> CANN/HCCL/HCOMM probe -> TUNA python plan ->
isolated environments -> source installs -> A5 native build -> doctor -> Stage 1 L0
-> Stage 2 operator smoke -> benchmark -> report
```

| step | command | states it may report |
|---|---|---|
| sources | `fetch_third_party.py --check` | `FOUND` / `MISSING` / `COMMIT_MISMATCH` / `UNPINNED` |
| environments | `env_manager.py --create` | `PRESENT` / `ABSENT` / `BROKEN`; refuses to build on an unusable NPU stack |
| native build | `native_build.py --run` | `BUILT` / `BUILT_HOST_ONLY` / `BUILD_FAILED` / `BLOCKED_DEPENDENCY` |
| dependencies | `npu_smoke.py deps` | `READY` / `MISSING` / `VERSION_MISMATCH` / `BUILD_FAILED` / `SOURCE_UNAVAILABLE` / `UNDOCUMENTED_ON_SOC` / `UNSUPPORTED_PLATFORM` |

Rules that keep the pipeline honest:

* a checkout is verified by full commit id, never by "the clone succeeded";
* `BUILT_HOST_ONLY` exists because `kernels/asc_comm` links a host library even when
  `aiv/write_nbi/kernel.cpp` / `ccu/dsl/kernel.cpp` are absent — a host link is not a
  device kernel;
* a similarly named PyPI project is never a substitute; `dependencies/distributions.yaml`
  records the four traps (megatron/megatron-core, mindspeed, deep-ep, deep-gemm) and
  `common/distributions.py::check_module_origin` verifies the imported module actually
  lives in the pinned checkout;
* torch and torch_npu come from the container and are never installed or downgraded;
* HCCL and HCOMM come from CANN: not third-party sources, never reinstalled here.

## 1.3 Per-operator smoke profiles

A smoke run does **not** use "the first case the planner marked runnable" — that can pick a
case that is not an adapter's main path. Every operator declares its own profile in
`operators/smoke_profiles.yaml`:

```yaml
alltoall_hccl:
  family: hccl
  level: L1
  case: COMM-001
```

The table is validated against `operators/registry.yaml` and the case catalog by
`common/smoke_profile.py` before anything runs, and the run is refused when:

* an operator has no profile;
* the case's `requires` capabilities are not declared by the adapter;
* the level does not match the case;
* the planner cannot turn the profile into an executable task (a profile is a promise that
  the point runs, so a declaration the planner rejects is a defect in the table).

Whether a backward pass is required is derived from the adapter's own capabilities: an
adapter that declares `moe_backward` must run `MOE-002` as well, and a profile cannot opt
out silently.

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
| 冒烟 / smoke / 快速验证 | `smoke_test` | NPU_ONLY: Stage 1 L0 gate + one declared case per operator; never a CPU configuration |
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

## 3.5 NPU Smoke: the command and what it must prove

```sh
# 1. expand the profiles; touches no device, safe on any host
python scripts/npu_smoke.py plan

# 2. the real smoke: Stage 1 L0 gate + Stage 2 one declared case per operator
python scripts/npu_smoke.py run --output 05_results/smoke --run-id smoke-ep8

# narrow without changing the policy
python scripts/npu_smoke.py run --output 05_results/smoke --run-id smoke-comm \
    --stage operators --operators alltoall_hccl,moonep --direction forward

# gate only: build the plan, apply the NPU/hardware preflight, create no run
python scripts/npu_smoke.py run --output 05_results/smoke --run-id gate --dry-run

# continue an interrupted run: results already on disk are reused, only tasks that
# are not terminal are retried, and a "too few cards" verdict is re-evaluated
python scripts/npu_smoke.py run --output 05_results/smoke --run-id smoke-ep8 --resume

# the Full Benchmark gate: only operators the smoke actually passed
python run_benchmark.py run --plan configs/benchmark_plan.yaml --ep-size 8 \
    --operators "$(python scripts/npu_smoke.py gate --report 05_results/smoke/smoke-ep8/smoke_report.json)"
```

A run directory is created once. A second `run` into the same `--run-id` is refused
unless `--resume` is passed, and `--resume` refuses a stored plan that differs from the
current request instead of silently mixing two plans. `gate` is operator-level by design: a
run where some adapters are blocked by a missing package still lets the passed ones proceed
to measurement; `--strict` additionally requires the smoke run's own conclusion to be `PASS`.

Exit codes: `0` complete coverage, `1` ran but coverage incomplete or something failed,
`2` invalid request/profile table (nothing executed).

Artifacts, under `<output>/<run-id>/`:

```
smoke_report.json   machine-readable report (environment, stage1, operators, coverage, gaps)
smoke_report.md     the same, for a human
manifest.json       standard manifest: every task the run planned
metrics.json        per task, including correctness and latency of the points that ran
logs/<task>/…       per-attempt worker logs
```

Stage 1 **environment smoke** exercises the L0 families a host must be able to run — P2P,
collectives, GMM, SwiGLU, communication/compute overlap, plus memory and native transport as
reported-but-optional families. It is cheap by construction
(`configs/smoke_environment.yaml`: one value per swept dimension) and it is what gives L0
real coverage; no operator profile repeats it.

Stage 2 **operator smoke** runs each profile's forward case, plus the backward case when the
adapter declares `moe_backward`. A profile passes only when **every** task it expanded to
passed on the NPU.

The report must state three coverages separately, because they can disagree:

| coverage | means |
|---|---|
| operator | operators whose main path really executed and passed on the NPU |
| level | L0-L4, each backed by a real passed result (L0 from Stage 1) |
| execution path | every path in `required_paths` (backend, compute, direction, routing, dedup, precision) backed by a passed profile |

A required path nobody covers, a level with no passed result, and an operator with no profile
are all reported as `COVERAGE_GAP`. Overall `PASS` requires all three coverages complete and
no gaps — "every operator passed" and "every required path was exercised" are different
claims and the report keeps them apart.

Delivery buckets (`common/status_map.py`) keep blockers distinguishable:

`PASSED` · `FAILED` · `BLOCKED_DEPENDENCY` · `BLOCKED_HARDWARE` · `BLOCKED_PLATFORM` ·
`BLOCKED_ENVIRONMENT` · `BUILD_FAILED` · `SKIPPED_BY_USER`

In particular an EP size larger than the host's card count is `BLOCKED_HARDWARE`, never a
CPU run and never a plain failure; a build that fails is `BUILD_FAILED`, not a missing
dependency.

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
default. A request may set `scope.execution_profile` explicitly (the workflow passes it
to the plan), and the interpreter only recognises the words `smoke`, `core` and
`exhaustive` (plus `冒烟`): a vague phrase never narrows the matrix silently. The manifest records the resolved profile, so a `smoke` run can never be
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
3. **what is blocked and why** — dependency, hardware, platform, environment, build, with
   the recorded reason for each. Use the `BLOCKED_*` bucket names, never a single "failed"
   count. A blocked task is not a pass and does not contribute coverage;
4. **the three coverages, separately** — for a smoke run: operator, level and execution
   path from `<output>/<run-id>/smoke_report.json`, plus every `COVERAGE_GAP` it lists. Never
   claim complete coverage because "all operators passed";
5. **what was not verified** — no profile, no L0 verification, single measurement;
6. **where the evidence is** — `reports/benchmark_report.md`, `reports/*`, raw
   `metrics.json`, `05_results/workflows/<id>/`, and for smoke
   `<output>/<run-id>/smoke_report.{json,md}`.

Acceptance gates for the delivery (`references/report_acceptance.md`): the status buckets sum
to the planned tasks, coverage accounts for every variant, and every link in the main report
resolves. Final `partial` is a real delivery; final `failed`/`blocked` means explain what
stopped it.
