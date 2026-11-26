"""Write the main report, the specialty reports, CSV and JSON from one result set.

Order matters and is fixed:

1. the coverage products (``coverage.write``) -- the main report links them;
2. the charts -- the main report links the ones that exist and explains the rest;
3. the specialty reports -- the main report links them;
4. ``benchmark_report.md`` last, so every link it emits already points at a file.

Every step only *reads* the result records. A report pass never rewrites
``metrics.json``, ``samples.jsonl`` or a profiler artifact, and running it twice on the
same inputs produces the same files.
"""
import csv
import json

from common.storage import write_json
from reporting import charts as charting
from reporting import specialty
from reporting.context import build
from reporting.coverage import write as write_coverage
from reporting.l0_report import write as write_l0
from reporting.sections import (
    comparison, coverage as coverage_section, environment, evidence, failures, l0 as l0_section,
    load_balance, operator, overlap, recommendations, scalability, summary,
)

#: Section renderers, in report order.
SECTIONS = (
    summary.render,        # 0. Executive Summary
    environment.render,    # 1. Environment and Hardware Capability
    coverage_section.render,  # 2. Test Coverage Summary
    l0_section.render,     # 3. L0 Hardware Baseline
    operator.render,       # 4. L1-L4 Operator Performance
    comparison.render,     # 5. Backend Comparison
    scalability.render,    # 6. EP Scalability
    overlap.render,        # 7. Communication and Compute Overlap
    load_balance.render,   # 8. Load Balance and Planner
    failures.render,       # 9. Failures and Limitations
    evidence.render,       # 10. Evidence and Raw Data
    recommendations.render,  # 11. Recommendations and Next Steps
)

#: The columns of ``benchmark_summary.csv``; kept stable for existing consumers.
SUMMARY_COLUMNS = ("task_id", "backend_id", "case_id", "benchmark_level", "device", "status",
                   "p50_ms", "p95_ms", "p99_ms", "tokens_per_second", "payload_GBps",
                   "error_reason")


def _summary_row(row):
    delay = row.get("latency") or {}
    rate = row.get("throughput") or {}
    traffic = row.get("bandwidth") or {}
    return (row["task_id"], row["backend_id"], row["case_id"], row["benchmark_level"],
            (row.get("config") or {}).get("device"), row.get("status"), delay.get("p50"),
            delay.get("p95"), delay.get("p99"), rate.get("global_source_tokens_per_second"),
            traffic.get("payload_bandwidth_GBps"), row.get("error_reason"))


def write_summary(destination, manifest, rows, analysis):
    """``benchmark_summary.json`` / ``.csv``: the machine-readable result index."""
    from collections import Counter
    statuses = Counter(row.get("status") for row in rows)
    write_json(destination / "benchmark_summary.json", {
        "schema_version": 1, "run_id": manifest["run_id"], "status_counts": dict(statuses),
        "results": rows, "analysis": analysis,
    })
    with (destination / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(_summary_row(row) for row in rows)


def _capabilities(root):
    path = root / "capabilities.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": 1, "source": "unavailable",
                "error": "capabilities.json exists but could not be parsed",
                "channels": [], "engines": [], "accelerators": [], "probes": []}


def _header(ctx):
    return [
        f"# EP benchmark report — run `{ctx.run_id}`",
        "",
        "This is the main analysis report of one benchmark run. It is meant to be read on "
        "its own: it states the test scope, the task outcomes, the hardware baseline, the "
        "operator performance, the comparison and scaling verdicts, the failures and the "
        "evidence files. Specialty reports hold the detailed tables.",
        "",
        f"- Device: {ctx.device_wording}",
        f"- EP size(s): {', '.join(f'EP{size}' for size in ctx.ep_sizes()) or 'none'} · "
        f"planned tasks: {ctx.status.planned} · result records: {ctx.status.records}",
        "",
        "Every number is shown with three significant digits, `-` means "
        "\"not recorded\" and `N/A (reason)` means \"does not apply to this case\"; neither is a "
        "zero. Sample counts are shown next to every performance table, because a percentile of "
        "a handful of iterations is a weak estimate.",
        "",
        "## Contents",
        "",
        "0. Executive Summary · 1. Environment and Hardware Capability · 2. Test Coverage "
        "Summary · 3. L0 Hardware Baseline · 4. L1-L4 Operator Performance · 5. Backend "
        "Comparison · 6. EP Scalability · 7. Communication and Compute Overlap · 8. Load Balance "
        "and Planner · 9. Failures and Limitations · 10. Evidence and Raw Data · "
        "11. Recommendations and Next Steps",
        "",
    ]


def report(root, manifest, rows, analysis, capabilities=None):
    """Generate every report artifact for one run and return the reports directory."""
    rows = list(rows)
    destination = root / "reports"
    destination.mkdir(parents=True, exist_ok=True)

    write_summary(destination, manifest, rows, analysis)

    coverage_paths = write_coverage(root, manifest, rows)
    coverage_data = {}
    try:
        coverage_data = json.loads(
            (destination / "coverage_matrix.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        coverage_data = {}

    charts = charting.create_charts(destination, rows, analysis)
    if capabilities is None:
        capabilities = _capabilities(root)

    ctx = build(root, manifest, rows, analysis, capabilities=capabilities,
                coverage=coverage_paths, coverage_data=coverage_data,
                destination=destination, charts=charts)
    ctx.charts = charts

    l0_present = bool(ctx.l0["summary"]["records"])
    if l0_present:
        write_l0(root, manifest, rows, capabilities, analysis=ctx.l0, charts=charts)

    for name, renderer in specialty.REPORTS:
        lines = renderer(ctx)
        (destination / name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    lines = _header(ctx)
    for render in SECTIONS:
        lines += render(ctx)
    (destination / "benchmark_report.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return destination
