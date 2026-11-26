# Report acceptance

A report is accepted when it describes the run that actually happened, not when it looks
complete. `scripts/workflow/reporting.py` enforces the mechanical part; this file states
the whole rule so a human reviewer applies the same standard.

## Required artifacts

| File | Why |
|---|---|
| `benchmark_report.md` | the main analysis report (sections 0-11) |
| `benchmark_summary.json` / `.csv` | the machine-readable result index |
| `coverage_matrix.md` / `.csv` / `.json` | which planned point produced which status |
| `operator_coverage.md` | per EP size and adapter: supported / executed / passed |
| `ep_size_comparison.md` | cross-EP comparison, with its comparability verdicts |
| `variant_coverage.md` | planned vs measured per parameter variant |
| `failure_summary.md` | every non-passing task, with its request and reason |
| `l0_report.md` | only when the run measured L0 |

## Accounting gates

```
sum(status buckets)      == planned_tasks
not_run                  == 0                     (otherwise the run is incomplete)
coverage.accounted       == coverage.planned       (per EP x case x operator x variant)
every relative link      resolves
```

A failure of the first three is `failed`; `not_run` present is `failed` for the gate but the
workflow is `partial` because the data that exists is still reported. The distinction is
deliberate: an incomplete run is never silently delivered as complete.

## Status vocabulary in the report

The report distinguishes `passed`, `failed`, `blocked` (dependency/environment/source),
`unsupported`, `not_applicable`, `not_implemented` and `not_run`. `completion_status:
complete` in `ep_summary.json` means every task reached a terminal status — it never means
every task passed.

## What the delivery must state, in order

1. what passed (with the sample counts and the comparable groups),
2. what is blocked and why (each with the reason recorded by the run),
3. which comparisons are actually comparable (and which are single measurements),
4. where the raw records live.

## Claims the workflow never makes

* a backend ranking from one measurement,
* a scaling trend from one EP size,
* a device overlap from a latency ratio,
* a mechanism's performance from an unverified L0 path,
* a CPU result as an Ascend performance result,
* a kernel or overlap metric from a missing trace,
* a physical link bandwidth from a byte model.
