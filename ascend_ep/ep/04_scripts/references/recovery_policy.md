# Recovery policy

A workflow that dies must be resumable, and a run that partially failed must still be
deliverable. Both are the difference between automation and a fragile script.

## What is persisted

| File | Content |
|---|---|
| `state.json` | workflow status, phase statuses/reasons, run id, request hash, plan hash, repo commit, per-phase commands with exit codes and log paths |
| `events.jsonl` | append-only history (created / phase_started / command / phase_finished / revalidated / workflow_finished) |
| `request.json` | the resolved request |
| `*.json` per phase | `environment.json`, `dependency_report.json`, `capability_report.json`, `kernel_report.json`, `plan_manifest.json`, `resource_estimate.json`, `execution_summary.json`, `result_validation.json`, `profiling_summary.json`, `analysis_summary.json`, `report_summary.json`, `final_report.json` |
| `logs/` | one log per command, with its exit code |

A phase is written the moment it finishes, so a crash loses at most the running phase.

## Resume

```sh
python scripts/benchmark_workflow.py resume --workflow-id <id>
python scripts/benchmark_workflow.py resume            # the latest unfinished workflow
```

* the workflow continues at its first phase that is not `passed`/`skipped`;
* a completed phase is never re-run;
* an interrupted `benchmark_execution` continues with
  `run_benchmark.py resume --run-id <the same id>`, so finished tasks are not re-measured;
* a changed request hash or repository commit invalidates the plan and everything after
  it (`needs_revalidation`), and the invalidation appears in `state.json` as
  `invalidated: <reason>`.

## Failure → behaviour

| Scenario | Behaviour |
|---|---|
| Python base dependency missing | install only if authorised, otherwise stop at `dependency_check` |
| no NPU present | the NPU performance phases stop; a CPU smoke request is unaffected |
| one operator's dependency missing | that operator's tasks stay blocked; the rest runs |
| ASC-COMM device kernel missing | the cases that need it stay blocked; nothing falls back to the host NBI path |
| EP64 has too few devices | EP64 is blocked, EP8 continues |
| a case fails correctness | recorded, not retried blindly |
| runtime timeout | retried by `resume` per `schemas.RETRYABLE` |
| a task OOMs | recorded; independent tasks continue |
| the workflow is interrupted | resume at the last unfinished phase |
| analysis fails | raw results are kept; only `analysis` (and later) re-run |
| report fails | results and analysis are kept; only `report_generation` (and later) re-run |

The workflow never discards existing results because a command exited non-zero. It
distinguishes the command exit code, the task status and the workflow status, and only the
last one decides delivery.
