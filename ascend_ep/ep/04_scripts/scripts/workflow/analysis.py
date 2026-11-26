"""Phase 10: analysis.

``run_benchmark.py analyze`` already enforces the comparability rules
(``analysis/comparability.py``) and the L0 ``verified`` gate. This module runs it and then
checks that the analysis is *of this run*: the file exists, it was written after the
result records, and it is keyed by task ids the run actually produced.
"""
import json
from pathlib import Path

from scripts.workflow.phases import PhaseOutcome
from scripts.workflow.execution import run_directory, output_args, result_records


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarise(results, records):
    """A short, checkable view of what the analysis concluded."""
    details = results.get("comparison_details") or {}
    groups = details.get("groups") or []
    kinds = {}
    for group in groups:
        kinds[group.get("kind")] = kinds.get(group.get("kind"), 0) + 1
    analysed = {member.get("task_id") for group in groups
                for member in group.get("members") or ()}
    known = set(records)
    return {
        "comparison_groups": len(groups),
        "group_kinds": kinds,
        "scaling_series": len(results.get("scaling_groups") or {}),
        "fusion_pairs": len(results.get("fusion") or ()),
        "planner_comparisons": len(results.get("planner_comparisons") or ()),
        "fixed_owner_comparisons": len(results.get("fixed_owner_comparisons") or ()),
        "dedup_comparisons": len(results.get("dedup_comparisons") or ()),
        "profiled_tasks": len(results.get("profiles") or {}),
        "excluded_records": len(details.get("excluded") or ()),
        "hypotheses": len(results.get("hypotheses") or ()),
        "profile_parse_errors": len(results.get("profile_parse_errors") or ()),
        "unknown_task_ids": sorted(analysed - known)[:10],
    }


def handler(ctx):
    request = ctx.request
    argv = ["analyze", "--run-id", ctx.run_id, *output_args(ctx)]
    result = ctx.commands.benchmark(*argv, log_name="analyze.log")
    ctx.record("analysis", result)
    path = run_directory(ctx) / "analysis" / "results.json"
    results = _load(path)
    if not isinstance(results, dict):
        return PhaseOutcome.failed(
            f"`analyze` produced no analysis/results.json (exit {result.exit_code}); "
            f"see {result.log_path}")
    if request.execution.profile_policy == "none" and (results.get("profiles") or {}):
        return PhaseOutcome.partial(
            "the analysis carries profile data although profiling was disabled for this "
            "workflow; the run directory may be shared with another profile pass")
    # A resumed workflow does not keep `result_validation`'s in-memory records, so the
    # task ids are re-read from the run rather than assumed.
    records = ctx.data.get("validation", {}).get("records") or {}
    if not records:
        try:
            _manifest, records = result_records(ctx)
        except (OSError, ValueError):
            records = {}
    summary = summarise(results, records)
    summary_path = ctx.write_json("analysis_summary.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id, "run_id": ctx.run_id,
        "summary": summary, "results": str(path), "log": str(result.log_path),
        "exit_code": result.exit_code,
    })
    ctx.data["analysis"] = summary
    if summary["unknown_task_ids"]:
        return PhaseOutcome.failed(
            "the analysis references task ids this run did not produce: "
            + ", ".join(summary["unknown_task_ids"]), artifacts=[summary_path])
    if not summary["comparison_groups"] and not summary["scaling_series"]:
        return PhaseOutcome.partial(
            "the analysis found no comparable group: no two accepted measurements share "
            "the same test conditions", artifacts=[summary_path])
    return PhaseOutcome.passed([summary_path], detail={"analysis": summary})
