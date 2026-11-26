# Profiling policy

Profiling answers "where did the time go", not "how long did it take". It is therefore
*selective* and always recorded as a selection.

## Policies

| `execution.profile_policy` | Selection | Use |
|---|---|---|
| `none` | nothing | plan / estimate / report requests, and any run where the user did not ask for traces |
| `representative` (default) | one passing task per `(EP size, level, operator)`, at most 24 | the normal choice: every level and every adapter gets a trace, without profiling a 20k-task matrix |
| `full` | every passing task | a deliberate deep dive; expensive |

Only `passed` tasks are selected. A trace of a failed task explains a failure, not a
performance, and the profiling stage requires a passing measurement to exist.

## Why the workflow calls `profile` itself

`pipeline` profiles a *representative* task only when the task's configuration has
`profiling.enabled: true`. That flag differs between the EP size configurations, so a
pipeline run can profile EP8 and silently skip EP64. The workflow therefore calls
`profile` with explicit filters and records the coverage it asked for:

```sh
python run_benchmark.py profile --run-id RUN --task-ids T1,T2        # precise selection
python run_benchmark.py profile --run-id RUN --cases C --operators O --ep-size 8,64
python run_benchmark.py profile --run-id RUN                          # policy full
```

The per-EP `profiling.enabled` flags are still recorded in `profiling_summary.json`, for
transparency, but they do not decide the coverage.

## What a profile must leave behind

* the raw trace per rank,
* `profile_manifest.json` with a sha256 per artifact,
* an entry in the task's `profiling_artifacts` with its status,
* `profiling_summary.json` in the workflow directory: selection, collected, failed,
  per-EP coverage.

## Gate

The phase passes when every selected task produced a parsable artifact. It returns
`partial` when some attempts failed or when nothing could be parsed — in both cases the
overlap and kernel claims stay unverified, and the report says so. It never invents a
kernel name or an overlap ratio from a missing trace.
