# Reporting refactor

Record of the change that turned the report system from a result index into an analysis
report. Written for a reviewer: what was wrong, what the new architecture is, what changed in
the data contract, what the tests show, and what is still open.

Scope: `ascend_ep/ep/04_scripts/{analysis,reporting,common,schemas,tests}`. The L0-L4
execution logic, the dispatch/combine/GMM measurement code, the timing collection, the
correctness validation and the raw metric definitions were **not** changed.

## 1. Problems in the previous implementation

1. **The main report was an index.** `benchmark_report.md` was one 12-column table of every
   row plus a list of links. It never said what the run covered, what passed, what the
   performance difference was, or what remained unverified. Reading it alone was impossible.
2. **Measured data was collected but not shown.** `latency.rank_min/rank_mean/rank_max`,
   `rank_avg_p50_ms`, `cv`, `p95_p50_ratio`, `host_issue_p50_ms`, `host_wait_p50_ms`,
   `throughput`, `compute.effective_tflops`, `verification`, `completion` and
   `observation` were persisted and then ignored by the report.
3. **Three different comparability rules.** `analysis/comparison.py`,
   `reporting/coverage.py` and the chart/template code each had their own idea of a comparable
   group. The same two rows could be "comparable" in one report and not in another, one
   backend could be listed under a "Comparison" heading, and a difference of unrelated config
   fields silently split or merged groups.
4. **Specialty reports could be empty shells.** Matplotlib missing produced no charts and no
   explanation; a run without profiler data produced `overlap_analysis.md` containing a title
   and a table header; `scalability_analysis.md` was 58 bytes.
5. **L0 and L1-L4 conclusions were separated.** The main report linked `l0_report.md` and
   said nothing about verification, so the only place a mechanism claim could be checked was
   another file.
6. **Failed tasks lost their identity.** `category` and `parameters` were written at the *end*
   of a successful `run_task`; a failed or unsupported L0 case therefore had no category, and
   `reporting/l0_report.py` (which grouped the transport table by category) could not show it.
   The L0 mechanisms and scale-out tables silently lost exactly the rows that failed.
7. **Environment, units and methodology were thin.** CANN came only from
   `ASCEND_TOOLKIT_VERSION`; every number was printed at full float precision with no unit
   scope; `payload_bandwidth_GBps` was shown without saying it is a byte model over a measured
   latency; a CPU run printed "CPU results validate framework behavior" and an NPU report
   would have printed it too; nothing said how many iterations the percentiles came from.

## 2. New architecture

```
schemas/__init__.py          status vocabulary, status buckets, the initial result record
common/environment.py        provenance capture, CANN version fallback chain
common/runner.py             (unchanged execution) no longer writes a second parameter field

analysis/comparability.py    comparison keys, eligibility, group kinds, incompatibility reasons
analysis/comparison.py       groups, EP scaling, fusion/planner/dedup joins (uses comparability)
analysis/l0.py               every L0 table, shared by the main report and l0_report.md
analysis/generate_report.py  analysis entry point -> analysis/results.json

reporting/formatting.py      the only place a number becomes text
reporting/empty.py           the mandatory structure of an unavailable analysis
reporting/context.py         Report Context: status counts, environment, coverage, L0 tables
reporting/sections/*.py      one module per main-report section (11 sections)
reporting/specialty.py       the eight specialty reports
reporting/coverage.py        coverage matrices + per-operator EP comparison (reuses comparability)
reporting/l0_report.py       the complete L0 report (reuses analysis/l0)
reporting/charts.py          charts with an explicit ChartReport degradation
reporting/report.py          orchestration: coverage -> charts -> specialty -> main report
```

Data flow of one `report` phase:

```
metrics.json / manifest.json / environment.json / capabilities.json
        |
        v
common.controller.results()  -> result records (category, parameters, ep_size already set)
        |
        v
analysis.generate_report.analyze_run()  -> comparison_details, scaling, fusion, profiles, ...
        |
        v
reporting.report.report()
        |-- reporting.coverage.write()        coverage_matrix.*, operator_coverage, variant_coverage
        |-- reporting.charts.create_charts()  ChartReport (created / skipped / unavailable)
        |-- reporting.specialty.REPORTS       eight specialty reports
        |-- reporting.l0_report.write()       l0_report.md, l0_metrics.json, l0_curves.csv
        `-- sections 0..11                    benchmark_report.md   (written last: links exist)
```

`benchmark_report.md` is written **last** on purpose: every link it emits already points at a
file that exists, which is why the link check is part of the test suite.

## 3. Main code changes

### 3.1 Data contract (P0)

* `schemas/__init__.py` now builds the record with `category`, `ep_size`, `case_parameters`,
  `operation`, `benchmark_level`, `backend_id` and `task_id` **before** the case runs, so a
  failure keeps them. The category comes from `task["case_category"]` (stamped by the
  scheduler) or, for an older manifest, from the case catalog — never from ID text.
* `schemas/__init__.py` also owns `STATUS_GROUPS` / `status_group()` / `group_totals()`, the
  single classification every report uses.
* `common/runner.py`: removed the duplicate `record["parameters"]` write. `case_parameters` is
  the only standard field; the legacy `parameters` key is read for old results only.
* `schemas/result-v1.json` documents the new fields as optional properties, so old records
  still validate.

### 3.2 Comparability (P0)

`analysis/comparability.py` (new) defines:

* `eligibility(record)` — is this an accepted measurement of its own case (status, source,
  positive P50, correctness);
* `comparable(record)` — eligibility plus the L0 `verified` gate for cross-mechanism use;
* `exclusion_reason(record)` — a human-readable reason;
* `backend_comparison_key` / `scaling_key` — the condition fingerprints;
* `incompatibility(left, right)` — every dimension that differs, with a readable message;
* `group_kind` — `single_measurement` / `repeated_measurement` / `actual_comparison`.

`comparison.py` and `coverage.py` now call these instead of defining their own. Speedups and
percentage differences are computed only for `actual_comparison`.

### 3.3 L0 analysis (P0)

`analysis/l0.py` (new) owns the L0 grouping and verification reading used by both the main
report and `l0_report.md`: `measurements`, `mechanism_rows`, `baseline_rows`,
`verification_table`, `scale_out`, `summary`, `analyse`. Requested values come from the plan;
observed values come from evidence; a missing observation is `unavailable`; a record with no
verification block is labelled `unverified (from status)` and never `verified`.

`reporting/l0_report.py` was rewritten on top of it. It keeps its public `write()` signature
and its `l0_metrics.json` / `l0_curves.csv` products, adds the failed rows to the transport
table and to a dedicated failure section, and extends the CSV with the requested/observed
mechanism and the verification source.

### 3.4 Main report (P0)

`reporting/report.py` builds a `ReportContext` once and renders twelve sections from
`reporting/sections/`. `reporting/formatting.py` supplies the number/unit/N-A rules.
`reporting/context.py` computes the status summary, so the executive summary, the coverage
section and the recommendations all quote the same counts.

### 3.5 Specialty reports (P1)

`reporting/specialty.py` renders the eight reports; three of them reuse the matching
main-report section body verbatim, so the two can never disagree. `reporting/empty.py` gives
every unavailable analysis the same six-part structure with a specific cause each time:
no profiler, no planner, no dedup, one EP size, one backend, no payload, missing matplotlib.

### 3.6 Charts (P1)

`reporting/charts.py` returns a `ChartReport` instead of returning `None`:

* `created` — `(label, relative path)` for each generated chart;
* `skipped` — `(name, reason)` for each plot without data;
* `available=False`, `missing_dependency="matplotlib"` when the extra is absent.

The main report prints `Charts Not Generated` plus the missing dependency, lists the skip
reasons, and never links a file that does not exist.

### 3.7 Environment (P1)

`common/environment.py` resolves CANN from `ASCEND_TOOLKIT_VERSION`, then from
`version.cfg` / `ascend_toolkit_install.info` / `version.info` under `ASCEND_HOME_PATH`,
`ASCEND_TOOLKIT_HOME`, `ASCEND_OPP_PATH` and the standard toolkit roots. The source is
recorded (`cann_version_source`); a failure records the locations that were tried
(`cann_version_error`). The driver version is parsed from `npu-smi info` when the run probed
it. Neither probe is required for a CPU-only report.

## 4. Data contract changes

| Field | Before | After |
|---|---|---|
| `category` | written only at the end of a successful L0 run | written when the record is created, from the task metadata or the catalog |
| `case_parameters` | written when the record is created | unchanged (still the standard field) |
| `parameters` | written at the end of a successful L0 run | no longer written; still **read** for old results |
| `ep_size` | absent (derived from `config` everywhere) | written when the record is created |
| status classification | three separate tuples (`NOT_RUN_STATUSES`, `FAILED_STATUSES`, `SEVERITY`) | `schemas.STATUS_GROUPS` + `status_group()`, with the raw status kept |
| `analysis/results.json` | `comparison_groups` (dict of lists) | `comparison_details` (groups with kind, conditions, members, speedup) |

No measured field was removed and no metric definition changed.

## 5. Test results

```
$ python -m pytest -q
1 failed, 412 passed, 6 skipped
```

The one failure is pre-existing and unrelated to this change:
`tests/dependencies/test_doctor.py::test_torch_is_probed_by_import` fails because this host has
`torch 2.7.1+cpu` while `dependencies/manifest.yaml` pins `==2.6.0`. Before the refactor the
same suite reported `1 failed, 340 passed, 6 skipped`, so 72 tests were added.

| New test file | What it locks down |
|---|---|
| `tests/test_reporting.py` (20) | all twelve sections present; NPU/CPU wording; outcome and correctness counts; a real backend pair is ranked while a single measurement is not; scalability refused for one EP size; overlap refused without a profiler; failed/unsupported/missing rows keep their request; every link resolves; charts section; report determinism; raw `metrics.json` unchanged; a run without any result still reports |
| `tests/test_comparability.py` (14) | two backends comparable; one backend is a single measurement; repeats are not a comparison; different workload / timing / variant / EP size / environment / completion semantics are not comparable; L0 unverified is excluded; eligibility reasons; legacy `parameters`; non-semantic parameters do not split a group; scaling key drops EP size and keeps the backend |
| `tests/test_l0_reporting.py` (10) | LINK-001 `passed` / `runtime_failed` / `unsupported` / `dependency_missing` all keep `category` and `case_parameters`; the transport table and the failure section still show the failed cases; verification labels come from evidence or from the status; requested values are not promoted to observed; `l0_metrics.json` and `l0_curves.csv` keep the requested/observed mechanism; a run without L0 explains itself; `ring` and `cross_server` stay separate variants |
| `tests/test_coverage_reporting.py` (9) | every coverage product is written by the report phase; two EP sizes form a comparison group and one EP size does not; missing tasks are counted; `compare_ep_sizes: false` removes the file and the link; unsupported is never counted as passed; aliases and `out_of_scope` stay visible; the CSV keeps the reason |
| `tests/test_chart_fallback.py` (5) | missing matplotlib is reported, not raised; no data yields skip reasons; chart links point at real files; the report still writes every section without matplotlib and contains no `charts/` link; a comparison group produces a latency chart |
| `tests/test_formatting.py` (14) | three significant digits, scientific notation, `None` never becomes `0`, percentages, MiB/GiB, `N/A (reason)`, table escaping, empty table |

### Offline regeneration

The `report` phase was re-run on every run directory in `05_results/raw` that has a manifest,
re-using the stored metrics without executing any benchmark:

| Run | Rows | Report files | Dangling links | Non-deterministic files |
|---|---|---|---|---|
| `local-hardware-cpu` | 13 | 23 | 0 | 0 |
| `local-smoke-gloo` | 12 | 20 | 0 | 0 |
| `local-smoke-initial` | 12 | 20 | 0 | 0 |
| `local-smoke-verification` | 6 | 20 | 0 | 0 |
| `local-final-ep-smoke` | 5 | 20 | 0 | 0 |
| `local-fixed-owner-final` | 9 | 20 | 0 | 0 |
| `local-npu-preflight-final` | 1 | 20 | 0 | 0 |
| `local-relocated-ep8-plan` | 1431 | 23 | 0 | 0 |
| `local-coverage-final` | 1431 | 23 | 0 | 0 |

`local-relocated-ep8-plan` is the plan-only NPU matrix (EP8, L0-L4, 37 adapters, 1431 tasks,
893 unsupported, 538 not run). It exercises the "nothing ran" path: the report states that no
backend can be ranked, no scaling trend exists, no L0 mechanism is verified, and it still
lists every planned request in section 9 and in `failure_summary.md`.

## 6. Known gaps and deliberate limits

1. **No complete A5 NPU run exists in this repository.** `05_results/raw` holds partial CPU
   runs and one plan-only NPU matrix (`950dt_ep8_bf16` is a legacy import without a manifest).
   The offline regeneration above therefore covers CPU runs and a planned matrix, not a full
   L0-L4 NPU execution. Re-running the report on a real NPU run directory is still required
   for the final end-to-end acceptance, and it needs no code change.
2. **`benchmark_summary.json` still embeds every result record** (38 MB for the 1431-task
   matrix). It is the historical machine-readable index and its shape is unchanged on purpose;
   splitting it would break existing consumers.
3. **An L0 record without a verification block reports a status-derived label.** A
   `runtime_failed` LINK point shows `failed (from status)`; that is a classification of the
   outcome, not evidence about the execution path. Only the `observed` column is evidence.
4. **Completion semantics split comparison groups.** Two adapters that promise different
   completion levels (`remote_visible` vs `local_complete`) are never ranked against each
   other. This reduces the number of comparisons; it is intentional, because a fenced sample
   and a completed transfer are different measurements.
5. **Main-report tables are capped.** 150 rows per level and 200 rows in section 9, with a
   pointer to `benchmark_summary.csv`, `coverage_matrix.json` and `failure_summary.md`, which
   hold the complete set. A 1431-task matrix otherwise produces a 600 KB Markdown file.
6. **`reporting/coverage.py` still computes the coverage matrices.** Its functions are heavily
   tested and imported by `tests/test_matrix.py`; the refactor reused them instead of moving
   them into `analysis/`. They now depend on `analysis.comparability` and `analysis.l0` for the
   rules, so there is still only one definition of comparability.
7. **Chart labels are English and the font is matplotlib's default.** Chart *content* is
   consistent with the Markdown (same units, same definitions); localisation is not done.
8. **The tests are CPU-only.** The `npu`-marked tests still need an Ascend host; nothing in
   the new report code requires a device.
9. **`dedup_opportunity_ratio` and every cross-server byte count remain routing-trace models.**
   They are labelled with their scope and source and are never presented as link counters.
10. **A profiler-based overlap claim still needs a profiling run.** The report refuses the
    claim without a parsed artifact; it cannot manufacture the data.
