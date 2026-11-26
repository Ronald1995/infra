"""Phase 9: profiling coverage.

Profiling a full matrix is never the goal, so the workflow decides *which* measured tasks
get a trace and records that decision:

* ``none``           -- the phase is skipped with the request as the reason;
* ``representative`` -- one passing task per (EP size, level, operator), capped;
* ``full``           -- every passing task.

Only ``passed`` tasks are selected, because a trace of a failed task explains the failure
and not the performance. The selection is passed to ``profile`` through its explicit
filters, so the run's ``profiling.enabled`` flags cannot change the coverage behind the
workflow's back.
"""
import json
from pathlib import Path

from scripts.workflow.phases import PhaseOutcome
from scripts.workflow.execution import run_directory, result_records, output_args

#: Upper bound for a representative selection.
MAX_REPRESENTATIVE = 24


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def select(records, policy):
    """The tasks to profile, as a sorted list of task ids, with their identity."""
    passing = [record for record in records.values() if record.get("status") == "passed"]
    passing.sort(key=lambda record: (record.get("ep_size") or 0,
                                     record.get("benchmark_level") or "",
                                     record.get("backend_id") or "",
                                     record.get("case_id") or "",
                                     record.get("task_id") or ""))
    if policy == "none":
        return []
    if policy == "full":
        return [{"task_id": record["task_id"], "case_id": record.get("case_id"),
                 "operator": record.get("backend_id"), "level": record.get("benchmark_level"),
                 "ep_size": record.get("ep_size")} for record in passing]
    selected, seen = [], set()
    # Every EP size gets a trace before the budget is spent on more points of one
    # size: with five sizes in the matrix a plain (size, level, operator) sweep
    # would exhaust the cap inside the first three sizes.
    for size in sorted({record.get("ep_size") for record in passing}, key=lambda item: item or 0):
        for record in passing:
            if record.get("ep_size") != size:
                continue
            seen.add((record.get("ep_size"), record.get("benchmark_level"),
                      record.get("backend_id")))
            selected.append(_selection(record))
            break
    for record in passing:
        key = (record.get("ep_size"), record.get("benchmark_level"), record.get("backend_id"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(_selection(record))
        if len(selected) >= MAX_REPRESENTATIVE:
            break
    return selected[:max(MAX_REPRESENTATIVE, len({item["ep_size"] for item in selected}))]


def _selection(record):
    return {"task_id": record["task_id"], "case_id": record.get("case_id"),
            "operator": record.get("backend_id"),
            "level": record.get("benchmark_level"),
            "ep_size": record.get("ep_size")}


def _profile_argv(ctx, selected, records, policy):
    argv = ["profile", "--run-id", ctx.run_id, *output_args(ctx)]
    if policy == "full":
        return argv
    if not selected:
        return argv
    cases = sorted({item["case_id"] for item in selected if item["case_id"]})
    operators = sorted({item["operator"] for item in selected if item["operator"]})
    sizes = sorted({str(item["ep_size"]) for item in selected if item["ep_size"]})
    if len(selected) <= 40:
        argv += ["--task-ids", ",".join(item["task_id"] for item in selected)]
        return argv
    if cases:
        argv += ["--cases", ",".join(cases)]
    if operators:
        argv += ["--operators", ",".join(operators)]
    if sizes:
        argv += ["--ep-size", ",".join(sizes)]
    return argv


def coverage(records):
    """Collected / failed / missing profile coverage over the selected tasks."""
    output = {}
    for task_id, record in records.items():
        artifacts = record.get("profiling_artifacts") or ()
        if not artifacts:
            continue
        entry = output.setdefault(record.get("ep_size"), {"collected": 0, "failed": 0})
        if any(item.get("status") == "passed" for item in artifacts):
            entry["collected"] += 1
        else:
            entry["failed"] += 1
    return output


def handler(ctx):
    request = ctx.request
    policy = request.execution.profile_policy
    directory = run_directory(ctx)
    manifest = _load(directory / "manifest.json")
    if not isinstance(manifest, dict):
        return PhaseOutcome.failed(f"run `{ctx.run_id}` has no manifest; there is nothing "
                                   "to profile")
    _manifest, records = result_records(ctx)
    passing = [record for record in records.values() if record.get("status") == "passed"]
    if policy == "none":
        return PhaseOutcome.skipped(
            "the request set execution.profile_policy to `none`, so no trace was collected")
    if not passing:
        return PhaseOutcome.skipped("no task passed, so there is no performance to profile")
    selected = select(records, policy)
    if not selected:
        return PhaseOutcome.skipped("the selection produced no task to profile")
    argv = _profile_argv(ctx, selected, records, policy)
    result = ctx.commands.benchmark(*argv, log_name="profile.log")
    ctx.record("profiling", result, policy=policy, selected=len(selected))
    _manifest, after = result_records(ctx)
    collected = [item["task_id"] for item in selected
                 if any(artifact.get("status") == "passed"
                        for artifact in (after.get(item["task_id"], {}).get(
                            "profiling_artifacts") or ()))]
    failed = [item["task_id"] for item in selected
              if after.get(item["task_id"], {}).get("profiling_artifacts") and
              item["task_id"] not in collected]
    configured = {
        str(size): ((entry or {}).get("profiling") or {}).get("enabled")
        for size, entry in sorted({
            str(task.get("ep_size")): (task.get("config") or {})
            for task in manifest.get("tasks") or ()}.items())
    }
    path = ctx.write_json("profiling_summary.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id, "run_id": ctx.run_id,
        "policy": policy, "selected": selected, "collected": collected,
        "failed": failed, "coverage": coverage(after),
        "profile_enabled_flags": configured,
        "log": str(result.log_path), "exit_code": result.exit_code,
    })
    ctx.data["profiling"] = {"selected": selected, "collected": collected,
                             "failed": failed, "path": path}
    if not collected:
        return PhaseOutcome.partial(
            f"profiling selected {len(selected)} task(s) but collected no parsable artifact "
            f"(exit {result.exit_code}); overlap and kernel claims stay unverified",
            artifacts=[path])
    if failed or len(collected) < len(selected):
        return PhaseOutcome.partial(
            f"{len(collected)} of {len(selected)} selected task(s) produced a profile; "
            f"{len(failed)} attempt(s) failed (each reason is in the summary)",
            artifacts=[path])
    return PhaseOutcome.passed([path], detail={"collected": len(collected)})
