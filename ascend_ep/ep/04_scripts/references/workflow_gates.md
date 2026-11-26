# Workflow gates

Every phase has one gate. A gate is what turns "the command exited" into "we know
something". This file lists them, what evidence satisfies each, and what happens when it
does not — the same table the orchestrator uses (`scripts/workflow/phases.py`).

Statuses: `pending`, `running`, `passed`, `partial`, `blocked`, `failed`, `skipped`.
`partial` is a real outcome: the phase produced usable evidence and a limitation, and the
workflow continues. `blocked`/`failed` stop the workflow unless the phase is skippable.

| Phase | Gate | Evidence | If it fails |
|---|---|---|---|
| `request_resolution` | the request is complete enough to run | `resolved_request.json`: intent, scope, safety policy, every default and its source | `blocked` with the missing field named; nothing else runs |
| `environment_discovery` | a doctor report exists and names python/CANN/SoC/devices | `environment.json` + `rank_capacity` | `failed` when `doctor --json` produced no JSON; `blocked` when no requested EP size has enough ranks |
| `dependency_check` | every dependency of the scope is classified | `dependency_report.json` (present / installable / buildable / host-provided / unknown) | `blocked` only when a host dependency the workflow may not install is missing |
| `dependency_install` | nothing installable is missing, or the installer ran and doctor was re-run | `dependency_after_install.json` | `blocked` when network/sudo/wheelhouse was not authorised; `failed` when the installer failed |
| `capability_probe` | per EP size the static support and the runtime channel status are recorded | `capability_report.json` | `partial` when some EP sizes are blocked; `blocked` when none is runnable |
| `kernel_build` | every engine is built or explicitly blocked | `kernel_report.json` (source, toolchain, ABI, artifacts) | `partial`: the cases that need the engine stay blocked, nothing falls back to the host path |
| `plan_generation` | the manifest covers every requested EP size / level / operator and every non-executable task has a reason | `plan_manifest.json` + `plan_hash` | `failed` on a coverage gap; `blocked` when no EP size has an executable task |
| `resource_estimation` | counts, iteration counts, a runtime estimate with its basis and a storage estimate exist | `resource_estimate.json` | `partial` when no historical timing exists; `blocked` when the estimate exceeds the requested budget |
| `benchmark_execution` | one run, every planned task terminal, bucket sum == planned | `execution_summary.json` + `ep_summary.json` + manifest hash | `failed` when no manifest or the executed manifest differs from the validated plan; `partial` when tasks are still `not_run` |
| `result_validation` | every passed task has a correctness record; no blind retry | `result_validation.json` | `failed` when a task is `passed` without correctness; `partial` for missing records or retryable failures |
| `profiling` | the selected tasks were profiled, or profiling was disabled | `profiling_summary.json` (selection, collected, failed, per-EP coverage) | `skipped` when the policy is `none`; `partial` when no artifact was parsable |
| `analysis` | `analysis/results.json` exists and belongs to this run's task ids | `analysis_summary.json` | `failed` when the analysis references unknown task ids; `partial` when no comparable group exists |
| `report_generation` | the required report files exist and every main-report link resolves | `report_summary.json` | `failed` on a missing file or a dangling link |
| `final_validation` | buckets sum to planned, coverage accounts for every variant, failures are named | `final_report.json` | `failed` on an accounting error; `partial` when failures/blocked points exist but are explained |

## Rules that override a single gate

1. **A non-zero exit code is not a failed phase.** `run` exits non-zero when a task failed
   or was blocked. If `ep_summary.json` is consistent and every task is terminal, the
   workflow continues and the report states what failed.
2. **A blocked item blocks its own scope.** A missing operator dependency, a missing
   device kernel or an EP size larger than the host blocks exactly the tasks that need it.
3. **`not_run` is not a pass.** A workflow with `not_run` tasks is `partial` and says so.
4. **Skipped always has a reason.** A phase that is not part of the intent is recorded as
   `skipped` with the intent as the reason, never left `pending`.
