# Reporting

The report system turns one run directory into one readable analysis report plus the
specialty reports that back it up. This document states what each artifact promises, how
the analysis rules are defined, and what a report is not allowed to say.

## 1. Artifacts of one `report` phase

```sh
python run_benchmark.py report --run-id RUN_ID          # add --compare-run OTHER to join runs
```

| File | Content |
|---|---|
| `reports/benchmark_report.md` | **the main report**: sections 0-11, readable on its own |
| `reports/l0_report.md` | complete L0 hardware baseline (every curve, every verification row) |
| `reports/l0_metrics.json`, `l0_curves.csv` | the L0 result set, machine readable |
| `reports/operator_comparison.md` | every backend-comparison group and the excluded records |
| `reports/bandwidth_analysis.md` | payload bandwidth with scope/source, and the four reasons a cell is empty |
| `reports/overlap_analysis.md` | profiler evidence and latency-derived fusion gains, kept apart |
| `reports/scalability_analysis.md` | EP-scaling series plus the per-operator EP comparison |
| `reports/load_balance_analysis.md` | expert/rank distribution from the routing trace |
| `reports/planner_training.md` | planner stages, weight prefetch, gradient reduction, redundant weights |
| `reports/dedup_comparison.md` | rank/server dedup comparison |
| `reports/failure_summary.md` | **complete** failure list (the main report shows the first 200) |
| `reports/coverage_matrix.{md,csv,json}` | `EP size x case` status matrix |
| `reports/case_operator_coverage.csv` | `EP size x case x operator` cells |
| `reports/variant_coverage.{md,csv}` | planned vs measured per parameter variant |
| `reports/operator_coverage.md` | per EP size and adapter: supported / executed / passed |
| `reports/ep_size_comparison.md`, `l0_scaleout.md` | cross-EP and L0 scale-out tables |
| `reports/benchmark_summary.json`, `.csv` | the result index (`status_counts` + every record) |
| `reports/charts/*.png` | the charts that had data |

The `report` phase only **reads** `metrics.json`, `correctness.json`, `samples.jsonl`,
`routing_trace.json` and the profiler artifacts. Re-running it on the same inputs produces
byte-identical Markdown/CSV/JSON.

## 2. The main report

```
0. Executive Summary          outcomes, correctness, findings, bottlenecks, unverified capabilities
1. Environment and Hardware   provenance of every number, topology, capability matrix
2. Test Coverage Summary      status vocabulary, per level/EP/backend, case x EP, case x backend, variants
3. L0 Hardware Baseline       communication, mechanism, compute/copy, verification, scale-out
4. L1-L4 Operator Performance one table set per level: latency, then rates and quality
5. Backend Comparison         actual comparisons, single measurements, repeat measurements
6. EP Scalability             strong/weak series and efficiency
7. Communication and Compute  profiler evidence, then serial/fused benchmark
8. Load Balance and Planner   expert/rank distribution, dedup, planner stages
9. Failures and Limitations   every non-passing task with its planned request
10. Evidence and Raw Data     links to every raw file, profiler artifact and chart
11. Recommendations           follow-ups, each with the observation it follows from
```

Every section that cannot analyse anything renders the same structure (`reporting/empty.py`):
`Status` / `Current Findings` / `Why Analysis Is Unavailable` / `Required Conditions` /
`Available Alternative Evidence` / `Conclusion`. A report file never contains only a heading.

## 3. Data contract

One result record is written by `schemas.result()` *before* the case can fail, so a failure
keeps the identity of the point that was planned:

| Field | Meaning |
|---|---|
| `task_id`, `case_id`, `benchmark_level`, `operation`, `backend_id` | identity of the planned point |
| `category` | case category from the catalog (`transport`, `compute`, `memory`, ...) |
| `ep_size` | EP size of the resolved configuration |
| `case_parameters` | the planned parameters (`variant`, `message_bytes`, `mechanism`, ...) |
| `status` / `error_reason` | raw outcome and its reason |
| `latency`, `bandwidth`, `throughput`, `compute`, `memory`, `load_balance`, `overlap` | measurements, `None` when not measured |
| `correctness`, `verification`, `observed`, `requested`, `completion`, `observation` | what ran and what it may claim |
| `profiling_artifacts`, `logs`, `attempt` | evidence locations |

`case_parameters` is the standard field. The legacy `parameters` key is only *read*
(`analysis.comparability.case_parameters`), never written.

Raw fields are never removed or rewritten to make a report simpler.

## 4. Comparability

`analysis/comparability.py` owns the only definition of "these two numbers answer the same
question". `analysis/comparison.py` and `reporting/coverage.py` both reuse it.

A comparison key contains the **test conditions**: case identity, level, operation, the whole
case-parameter set (minus the explanatory `reason`/`notes` keys), the environment (device,
NPU model, CANN version, packages), topology, EP/TP size, model shape and dtype, the
per-rank workload, routing strategy/seed/trace, the timing boundary (mode plus latency and
bandwidth unit/scope), the execution mode (dedup, load balance) and the **completion level**
(a remote-visible measurement is not a fenced one).

It contains none of: `task_id`, `run_id`, backend id (that is the compared dimension),
latency, throughput or any other measured result.

Three verdicts come out of a group:

| Verdict | Meaning |
|---|---|
| `single_measurement` | one valid backend: a measured value, never an advantage |
| `repeated_measurement` | several samples of one backend: not a comparison |
| `actual_comparison` | at least two backends under identical conditions: rankable |

Speedup, latency delta and percentage difference are computed **only** for
`actual_comparison`.

For EP scaling (`scaling_key`) the EP size, the topology derived from it and the global
expert count are excluded, because they *are* the scaling dimension; the workload is
normalised by the suite's `scaling.mode` (strong: global work constant, weak: per-rank work
constant). A series needs two distinct EP sizes.

L0 adds one gate: only a `verified` execution path may be compared across mechanisms. An
unverified number stays in the report, marked, and is not attributed to the mechanism it
requested.

## 5. Status vocabulary

`schemas.STATUSES` is the raw vocabulary; a report never renames a raw status.
`schemas.STATUS_GROUPS` is the only classification used by summary tables:

| Bucket | Raw statuses |
|---|---|
| Passed | `passed` |
| Failed | `correctness_failed`, `runtime_failed`, `timeout`, `oom` |
| Unsupported | `unsupported`, `not_implemented` |
| Dependency Missing | `dependency_missing` |
| Environment Unavailable | `environment_unavailable` |
| Source Unavailable | `source_unavailable` |
| Not Run | `not_run` |
| Not Applicable | `not_applicable` |
| Missing | (report-level) a planned task with no result record |

## 6. Formatting rules

`reporting/formatting.py` is the only place a number becomes text.

* three significant digits: `1.011065` -> `1.01`, `126599.18` -> `1.27e+05`, `0.000123456` -> `0.000123`;
* percentages are percentages, latency is ms, bandwidth is GB/s, compute is TFLOPS,
  memory is MiB/GiB, and the unit is named in the column header;
* exact counts (bytes, iterations, tasks) keep their digits;
* `None` -> `-` ("not recorded"); a metric that does not apply -> `N/A (<reason>)`
  ("not a communication payload", "no bandwidth model for this case",
  "no compute model for this case"); a failed task has no metric at all -> `-`;
* no value is ever converted to `0`, and `payload_bandwidth_GBps` is always labelled with its
  scope, source and traffic model (`per_rank` / `global_ep_group`, `routing_trace_model` /
  `declared_payload_model`). It is a **byte model divided by a measured latency**, never a
  physical link bandwidth;
* `effective_tflops` is effective FLOPs over measured latency, not a hardware counter;
* host issue, host wait and output ready are reported as three separate numbers; their sum is
  not device execution time;
* the iteration count is shown next to every performance table, and a two-iteration P95/P99 is
  described as a weak estimate rather than a tail-latency conclusion.

## 7. Charts

`reporting/charts.py` returns a `ChartReport`:

```python
report = create_charts(destination, rows, analysis)
report.created      # charts that exist (label + relative path)
report.skipped      # [(name, reason)] for every plot that had no data
report.available    # False when matplotlib is missing
```

A missing plotting dependency never fails the report: section 10.6 prints
`Charts Not Generated`, the missing dependency and the skipped reasons, and no image link is
emitted for a file that does not exist. A chart is generated only when the data can support
it (two backends for a latency chart, two EP sizes for a scaling chart, a parsed profiler
artifact for the idle chart).

## 8. Environment and provenance

`common/environment.py` resolves the CANN version in this order and records the source:

1. `ASCEND_TOOLKIT_VERSION`;
2. `$ASCEND_HOME_PATH/version.cfg`, `ascend_toolkit_install.info`, `version.info`
   (also for `ASCEND_TOOLKIT_HOME` and `ASCEND_OPP_PATH`);
3. the standard toolkit roots (`/usr/local/Ascend/ascend-toolkit/latest`, ...).

When nothing is readable the report prints `unavailable` **and** the list of locations that
were tried. The driver version is read from `npu-smi info` when the run probed it. Neither
probe is required for a CPU-only run.

## 9. Determinism

`report` sorts every collection it emits, never embeds a timestamp and never iterates a set.
Two runs over the same manifest, metrics and analysis inputs produce identical files; the
tests assert this for the whole `reports/` directory.

## 10. Where the pieces live

```
analysis/comparability.py   comparison keys, eligibility, group kinds, exclusion reasons
analysis/comparison.py      comparison groups, EP scaling, fusion/planner/dedup joins
analysis/l0.py              L0 tables shared by the main report and l0_report.md
analysis/generate_report.py analysis entry point (writes analysis/results.json)
reporting/formatting.py     number/unit/N-A formatting and Markdown tables
reporting/empty.py          the mandatory structure of an unavailable analysis
reporting/context.py        the Report Context: status counts, environment, coverage, L0
reporting/sections/*.py     one module per main-report section
reporting/specialty.py      the eight specialty reports
reporting/coverage.py       coverage matrices and the per-operator EP comparison
reporting/l0_report.py      the complete L0 report
reporting/charts.py         chart generation with explicit degradation
reporting/report.py         orchestration: coverage -> charts -> specialty -> main report
```

`schemas/__init__.py` owns the status vocabulary, the status buckets and the initial result
record.

## Coverage metrics and capability gaps (section 2.8 / 2.9)

The coverage chapter reports four numbers separately -- planning, execution,
correctness and verified performance -- because a mechanism without an implementation
can plan 100% and still never reach verified coverage. `not_applicable` leaves the
execution denominator; `unverified`, `blocked_topology`, `not_run` and the failure
statuses lower the number they belong to.

Section 2.9 groups every point whose status means "this capability is not available
here" (`not_implemented`, `unsupported`, `blocked_topology`, `dependency_missing`,
`environment_unavailable`, `source_unavailable`, `unverified`) by status and recorded
reason, with counts and examples. It states explicitly that `unverified` does not count
as verified coverage, and when nothing was missing it says so instead of leaving an
empty table.
