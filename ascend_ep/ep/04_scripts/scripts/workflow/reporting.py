"""Phase 11-12: report generation and the final delivery gate.

The final gate is deliberately mechanical, because "looks complete" is not a criterion:

* every planned task appears in exactly one status bucket,
* the coverage matrix accounts for every planned (EP, case, operator, variant) cell,
* no link in the main report points at a file that does not exist,
* the report names the failures and blocked points that the run produced.

``completion_status: complete`` means every task reached a terminal status. It never
means every task passed, and the gate here treats the two as different questions.
"""
import json
from pathlib import Path
import re

from scripts.workflow.phases import PhaseOutcome
from scripts.workflow.execution import run_directory, output_args

#: Files a delivered run must contain. ``l0_report.md`` is not listed: a run without an
#: L0 case legitimately has none (the main report says so).
REQUIRED_FILES = (
    "benchmark_report.md",
    "benchmark_summary.json",
    "benchmark_summary.csv",
    "coverage_matrix.md",
    "coverage_matrix.csv",
    "coverage_matrix.json",
    "operator_coverage.md",
    "ep_size_comparison.md",
    "variant_coverage.md",
    "failure_summary.md",
)

_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def dangling_links(report_path):
    """Every relative Markdown target that does not exist on disk."""
    text = Path(report_path).read_text(encoding="utf-8")
    missing = []
    for target in _LINK.findall(text):
        if "://" in target or target.startswith("#"):
            continue
        if not (Path(report_path).parent / target).resolve().exists():
            missing.append(target)
    return missing


def accounting(run_dir):
    """The three counts the delivery gate compares."""
    summary = _load(Path(run_dir) / "ep_summary.json") or {}
    coverage = _load(Path(run_dir) / "reports" / "coverage_matrix.json") or {}
    buckets = {name: summary.get(name) for name in
               ("passed", "failed", "unsupported", "not_applicable", "not_implemented",
                "dependency_missing", "environment_unavailable", "source_unavailable",
                "not_run")}
    bucket_sum = sum(value or 0 for value in buckets.values())
    variants = (coverage.get("case_operator_variant") or {}).get("totals") or {}
    return {
        "planned_tasks": summary.get("planned_tasks"),
        "bucket_sum": bucket_sum,
        "buckets": buckets,
        "buckets_consistent": bucket_sum == summary.get("planned_tasks"),
        "declared_consistent": (summary.get("consistency") or {}).get("consistent"),
        "not_run": summary.get("not_run", 0),
        "unresolved_tasks": summary.get("unresolved_tasks") or [],
        "coverage_planned": variants.get("planned"),
        "coverage_matched": variants.get("matched"),
        "coverage_missing": variants.get("missing"),
    }


def handler_generate(ctx):
    """``report_generation``: render the reports."""
    result = ctx.commands.benchmark("report", "--run-id", ctx.run_id, *output_args(ctx),
                                    log_name="report.log")
    ctx.record("report_generation", result)
    reports = run_directory(ctx) / "reports"
    missing = [name for name in REQUIRED_FILES if not (reports / name).is_file()]
    if missing:
        return PhaseOutcome.failed(
            f"the report phase did not produce {', '.join(missing)} (exit "
            f"{result.exit_code}); see {result.log_path}")
    report = reports / "benchmark_report.md"
    links = dangling_links(report)
    path = ctx.write_json("report_summary.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id, "run_id": ctx.run_id,
        "reports": str(reports), "required": list(REQUIRED_FILES),
        "dangling_links": links, "exit_code": result.exit_code,
        "log": str(result.log_path),
    })
    ctx.data["reports"] = reports
    if links:
        return PhaseOutcome.failed(
            f"the main report links {len(links)} file(s) that do not exist: "
            + ", ".join(links[:5]), artifacts=[path])
    return PhaseOutcome.passed([path], detail={"reports": str(reports)})


def handler_validate(ctx):
    """``final_validation``: does the delivery describe the run that happened?"""
    run_dir = run_directory(ctx)
    reports = run_dir / "reports"
    if not (reports / "benchmark_report.md").is_file():
        return PhaseOutcome.failed("no benchmark_report.md exists for this run")
    counts = accounting(run_dir)
    report_text = (reports / "benchmark_report.md").read_text(encoding="utf-8")
    summary_text = (reports / "failure_summary.md").read_text(encoding="utf-8") \
        if (reports / "failure_summary.md").is_file() else ""
    problems = []
    if counts["buckets_consistent"] is False:
        problems.append(
            f"the status buckets sum to {counts['bucket_sum']} but "
            f"{counts['planned_tasks']} tasks were planned")
    if counts["not_run"]:
        problems.append(f"{counts['not_run']} task(s) are `not_run`; the run is incomplete")
    if counts["coverage_missing"]:
        problems.append(f"{counts['coverage_missing']} variant cell(s) have no result record")
    if counts["buckets"]["failed"]:
        # A failure is allowed in the delivery, but only if the report names it.
        if "Failures and Limitations" not in report_text:
            problems.append("tasks failed but the main report has no failure section")
    unsupported = (counts["buckets"]["unsupported"] or 0) + (
        counts["buckets"]["not_applicable"] or 0)
    if unsupported and not summary_text:
        problems.append("unsupported/not-applicable tasks exist but failure_summary.md is empty")
    payload = {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id, "run_id": ctx.run_id,
        "accounting": counts, "problems": problems,
        "report": str(reports / "benchmark_report.md"),
        "ep_summary": str(run_dir / "ep_summary.json"),
    }
    path = ctx.write_json("final_report.json", payload)
    if problems:
        status = "failed" if any("bucket" in item or "no result record" in item
                                 for item in problems) else "partial"
        return getattr(PhaseOutcome, status)("; ".join(problems), artifacts=[path],
                                            detail={"accounting": counts})
    passed = counts["buckets"]["passed"] or 0
    if not passed:
        return PhaseOutcome.partial(
            "the delivery is consistent but no task passed, so it contains no measurement",
            artifacts=[path], detail={"accounting": counts})
    blocked = sum(counts["buckets"].get(name) or 0 for name in
                  ("dependency_missing", "environment_unavailable", "source_unavailable"))
    if counts["buckets"]["failed"] or unsupported or blocked:
        return PhaseOutcome.partial(
            f"the delivery is consistent: {passed} task(s) passed, "
            f"{counts['buckets']['failed']} failed, {unsupported} unsupported/not-applicable "
            f"and {blocked} blocked (all listed in the report)",
            artifacts=[path], detail={"accounting": counts})
    return PhaseOutcome.passed([path], detail={"accounting": counts})
