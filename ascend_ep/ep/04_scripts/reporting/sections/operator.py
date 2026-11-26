"""Section 4: L1-L4 Operator Performance.

One table set per benchmark level.  The levels measure different things (L1
communication, L2 fusion, L3 whole-MoE, L4 router/planner/training), so mixing them in
one table would silently compare unrelated numbers.  Failed tasks keep their row: a
failure is a result of the level, not a reason to hide it.
"""
from collections import defaultdict

from analysis import comparability
from reporting import formatting as F
from reporting.sections import common

LEVELS = ("L1", "L2", "L3", "L4")

#: Rows shown per level table in the *main* report.  A full plan expands thousands of
#: tasks; the complete list stays in ``benchmark_summary.csv`` and the coverage
#: products, so the main report stays readable.
MAX_ROWS = 150


def render(ctx):
    lines = ["## 4. L1-L4 Operator Performance", ""]
    lines += ["Every row is one task. `-` means the metric was not recorded for that task and "
              "`N/A (<reason>)` means it does not apply to it; neither is a zero. Rank Min / Mean "
              "/ Max are the medians across iterations of the per-iteration cross-rank values, "
              "and the headline P50/P95/P99 are percentiles of the per-iteration **rank max**.",
              ""]
    for level in LEVELS:
        rows = ctx.rows_at_level(level)
        lines += [f"### 4.{LEVELS.index(level) + 1} {common.level_name(level)}", ""]
        if not rows:
            lines += [_absence(ctx, level), ""]
            continue
        lines += _level_tables(rows, level)
    return lines


def _absence(ctx, level):
    planned = [task for task in ctx.manifest.get("tasks") or () if task.get("level") == level]
    if not planned:
        return f"_No {level} task was planned for this run, so this level is out of scope._"
    statuses = defaultdict(int)
    for task in planned:
        row = next((item for item in ctx.rows if item.get("task_id") == task.get("task_id")), None)
        statuses[row.get("status") if row else "missing"] += 1
    return ("_No result record exists for this level. Planned tasks: "
            f"{F.count(len(planned))} ({F.text(F.status_counts(dict(statuses)))}). "
            "See section 9 for the per-task reason._")


def _level_tables(rows, level):
    rows = sorted(rows, key=lambda row: (row.get("case_id") or "", row.get("backend_id") or "",
                                         comparability.ep_size(row) or 0))
    shown = rows[:MAX_ROWS]
    lines = ["#### Latency", ""]
    lines += F.table(("Case", "Operator", "EP", "Status", "P50 ms", "P95 ms", "P99 ms",
                      "Rank min ms", "Rank mean ms", "Rank max ms", "Rank avg P50 ms", "CV",
                      "P95/P50", "Host issue ms", "Host wait ms"),
                     (_latency_row(row) for row in shown))
    lines += ["#### Throughput, bandwidth and compute", ""]
    lines += F.table(("Case", "Operator", "EP", "Status", "Throughput tok/s", "Payload GB/s",
                      "Effective TFLOPS", "Verification", "Correctness", "Iterations",
                      "Timing mode", "Bandwidth scope"),
                     (_rate_row(row) for row in shown))
    if len(rows) > len(shown):
        lines += [f"_{len(shown)} of {len(rows)} {level} task(s) are shown. The complete list, "
                  f"including every failed and unsupported task, is in `benchmark_summary.csv`, "
                  f"`coverage_matrix.json` and section 9._", ""]
    return lines


def _latency_row(row):
    latency = row.get("latency") or {}
    return [row.get("case_id"), row.get("backend_id"),
            _ep(row), F.text(row.get("status")),
            F.number(latency.get("p50")), F.number(latency.get("p95")),
            F.number(latency.get("p99")),
            F.number(common.median(latency.get("rank_min"))),
            F.number(common.median(latency.get("rank_mean"))),
            F.number(common.median(latency.get("rank_max"))),
            F.number(latency.get("rank_avg_p50_ms")), F.number(latency.get("cv")),
            F.number(latency.get("p95_p50_ratio")),
            F.number(latency.get("host_issue_p50_ms")),
            F.number(latency.get("host_wait_p50_ms"))]


def _rate_row(row):
    throughput = row.get("throughput") or {}
    return [row.get("case_id"), row.get("backend_id"), _ep(row), F.text(row.get("status")),
            F.number(throughput.get("global_source_tokens_per_second")),
            common.payload_bandwidth(row), common.effective_tflops(row),
            common.verification_cell(row), common.correctness_cell(row),
            common.iterations(row), common.timing_mode(row), common.payload_scope(row)]


def _ep(row):
    size = comparability.ep_size(row)
    return f"EP{size}" if size else F.MISSING
