# Execution policy

One request produces **one** official run. Everything about execution follows from that.

## One run, one path

The workflow always uses path B:

```
run -> profile -> analyze -> report
```

`run` -> `pipeline` is forbidden: `pipeline` re-executes the measurement, and two runs
with different ids for one request make the results unattributable. `pipeline` stays
available for a human who wants one command; the workflow never uses it.

| Situation | Behaviour |
|---|---|
| the run directory already exists and `ep_summary.json` says `complete` | the phase is `skipped` ("already complete; results reused") |
| the run directory exists but is incomplete | `run_benchmark.py resume --run-id <id>` — never a new run id |
| the run directory does not exist | `run_benchmark.py run ...` |
| the command timed out | `partial`; resume the workflow to continue the same run |
| the executed manifest differs from the validated preview | `failed`: a task list that was never checked is never measured |

## Scope arguments

The run re-expands the same plan the preview validated. When the request narrows the
declared scope, the workflow writes a temporary plan under its own output directory
(`<workflow>/plan/request_plan.yaml`) with absolute layer paths and passes that. The
repository's configuration files are never edited.

## Exit codes, task statuses, workflow statuses

These are three different things and the workflow keeps them apart:

| Layer | Meaning |
|---|---|
| command exit code | 0 / non-zero of one CLI invocation; recorded with its log |
| task status | `passed`, `runtime_failed`, `unsupported`, `dependency_missing`, ... (one per task) |
| workflow status | `passed` (all phases done and delivered), `partial` (delivered with failures/blocked), `blocked`, `failed` |

`run` exits non-zero when any task failed or never reached a terminal status. That does
**not** discard the results: if `ep_summary.json` is consistent and every task is
terminal, the workflow continues, and the report lists the failures. A run that produced
no manifest, or an inconsistent summary, is a failed phase.

## Retry and resume

* A retryable task (`schemas.RETRYABLE`: `not_run`, `runtime_failed`, `timeout`, `oom`,
  `environment_unavailable`, `dependency_missing`, `source_unavailable`) is retried by
  `resume`, which keeps the run id and the attempt history.
* `correctness_failed` is **not** retried blindly: the cause must be fixed and the report
  must show it.
* A workflow resumes at its first phase that is not `passed`/`skipped`. Completed phases
  are not re-run. A changed request hash or repository commit invalidates the plan and the
  measurement (`needs_revalidation`), and the invalidation is recorded in `state.json`.

## Multi-node

Multi-node is never inferred. It requires `execution.nnodes > 1` plus `nproc_per_node`,
`master_port` and `session`; without them the request is blocked at
`request_resolution`. EP sizes larger than the local device count are blocked with the
counted reason, never run on a smaller group.
