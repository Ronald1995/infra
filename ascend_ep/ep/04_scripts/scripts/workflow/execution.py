"""Phase 7-8: execute the matrix and validate the results.

Two rules decide this module:

1. **one run id, one run.** The workflow never runs ``run`` and then ``pipeline``: the
   execution path is fixed to ``run`` -> ``profile`` -> ``analyze`` -> ``report`` so every
   phase can be recorded and resumed independently. A completed run is never re-executed;
   an incomplete one is continued with ``resume``, which keeps the same directory.
2. **an exit code is not a task status.** ``run_benchmark.py run`` exits non-zero when a
   task failed or when a task never reached a terminal status. If every task is terminal,
   the workflow keeps going and the report says what failed; only a run that produced no
   usable record fails the phase.
"""
import json
from pathlib import Path

from common.controller import result_path
from scripts.workflow.phases import PhaseOutcome
from scripts.workflow import planning


def run_directory(ctx):
    """The official run directory of this workflow."""
    return ctx.run_output_dir / ctx.run_id


def output_args(ctx):
    """``--output`` pointing at the run root, used by every run-scoped command."""
    return ["--output", str(ctx.run_output_dir)]


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def plan_hash_of_manifest(manifest):
    return planning.plan_hash(manifest)


def _prior_manifest(ctx):
    path = ctx.output / "plan_manifest.json"
    data = _load(path) or {}
    counts = data.get("counts") or {}
    return data.get("plan_hash"), counts


def handler_run(ctx):
    """``benchmark_execution``: run or resume the single official run."""
    request = ctx.request
    directory = run_directory(ctx)
    summary_path = directory / "ep_summary.json"
    prior_summary = _load(summary_path)
    completed = bool(prior_summary and prior_summary.get("completion_status") == "complete")

    if completed:
        # The same request never re-runs a completed matrix.
        ctx.state.run_id = ctx.run_id
        return PhaseOutcome.skipped(
            f"run `{ctx.run_id}` is already complete; its results are reused")

    resume = bool(prior_summary) or request.execution.resume
    if resume and not (directory / "manifest.json").is_file():
        # Nothing exists to resume: a resumed workflow that never reached the run must
        # start it, not fail trying to continue a directory that is not there.
        resume = False
    if resume:
        argv = ["resume", "--run-id", ctx.run_id, *output_args(ctx)]
        argv += planning.launch_args(ctx)
        mode = "resume"
    else:
        argv = planning.expand_argv(ctx, "run")
        argv += ["--run-id", ctx.run_id, *output_args(ctx)]
        argv += planning.launch_args(ctx)
        mode = "run"
    result = ctx.commands.benchmark(*argv, log_name=f"{mode}.log")
    ctx.record("benchmark_execution", result, mode=mode, run_id=ctx.run_id)
    ctx.state.run_id = ctx.run_id
    ctx.store.save(ctx.state)

    manifest = _load(directory / "manifest.json")
    if not isinstance(manifest, dict):
        return PhaseOutcome.failed(
            f"run `{ctx.run_id}` produced no manifest at {directory} "
            f"(exit {result.exit_code}); see {result.log_path}")
    expected_hash, _counts = _prior_manifest(ctx)
    actual_hash = plan_hash_of_manifest(manifest)
    if expected_hash and actual_hash != expected_hash:
        return PhaseOutcome.failed(
            "the executed manifest differs from the validated plan preview; refusing to "
            "measure a task list that was never checked (re-run the workflow to "
            "regenerate the plan)")
    summary = _load(summary_path)
    if not isinstance(summary, dict):
        return PhaseOutcome.failed(
            f"run `{ctx.run_id}` has no ep_summary.json (exit {result.exit_code}); "
            f"see {result.log_path}")
    payload = {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id, "run_id": ctx.run_id,
        "run_directory": str(directory), "mode": mode,
        "exit_code": result.exit_code, "timed_out": result.timed_out,
        "log": str(result.log_path), "summary": summary,
        "plan_hash_match": bool(not expected_hash or actual_hash == expected_hash),
    }
    path = ctx.write_json("execution_summary.json", payload)
    ctx.data["execution"] = payload
    if result.timed_out:
        return PhaseOutcome.partial(
            f"the run was interrupted by the command timeout; resume the workflow to "
            f"continue run `{ctx.run_id}` from its recorded state", artifacts=[path])
    if summary.get("completion_status") != "complete":
        remaining = summary.get("unresolved_tasks") or []
        return PhaseOutcome.partial(
            f"{len(remaining)} task(s) never reached a terminal status "
            f"({summary.get('completion_status')}); the run can be resumed", artifacts=[path],
            detail={"unresolved_tasks": remaining[:20]})
    if summary.get("consistency", {}).get("consistent") is False:
        return PhaseOutcome.failed(
            "ep_summary.json is internally inconsistent (the status buckets do not sum to "
            "the planned task count)", artifacts=[path])
    if summary.get("passed", 0) == 0:
        return PhaseOutcome.partial(
            "the run is complete but no task passed; every point is blocked, unsupported "
            "or failed", artifacts=[path])
    return PhaseOutcome.passed([path], detail={"run_id": ctx.run_id,
                                               "passed": summary.get("passed", 0)})


def result_records(ctx):
    """Every result record of the run, keyed by task id."""
    directory = run_directory(ctx)
    manifest = _load(directory / "manifest.json") or {}
    records = {}
    for task in manifest.get("tasks") or ():
        path = Path(result_path(directory, task))
        record = _load(path)
        if record is None:
            record = {
                "task_id": task.get("task_id"), "case_id": task.get("case_id"),
                "backend_id": task.get("backend_id"),
                "benchmark_level": task.get("level"), "status": "missing",
                "correctness": None,
                "case_parameters": task.get("parameters") or {},
                "ep_size": task.get("ep_size"), "error_reason": None,
            }
        else:
            # A record written by an older version may lack these; the report needs them.
            record.setdefault("case_parameters", task.get("parameters") or {})
            record.setdefault("ep_size", task.get("ep_size"))
            record.setdefault("category", task.get("case_category"))
        records[task.get("task_id")] = record
    return manifest, records


def handler_validate(ctx):
    """``result_validation``: correctness evidence and status accounting."""
    directory = run_directory(ctx)
    summary = _load(directory / "ep_summary.json")
    if not isinstance(summary, dict):
        return PhaseOutcome.failed(
            f"run `{ctx.run_id}` has no ep_summary.json; nothing can be validated")
    manifest, records = result_records(ctx)
    passed = [record for record in records.values() if record.get("status") == "passed"]
    without_correctness = [record["task_id"] for record in passed
                           if not (record.get("correctness") or {}).get("passed")]
    failed_correctness = [record["task_id"] for record in records.values()
                          if record.get("status") == "correctness_failed"]
    missing = [task_id for task_id, record in records.items()
               if record.get("status") == "missing"]
    retryable = [record["task_id"] for record in records.values()
                 if record.get("status") in ("runtime_failed", "timeout", "oom")]
    totals = {
        "planned_tasks": summary.get("planned_tasks"),
        "completed_tasks": summary.get("completed_tasks"),
        "passed": len(passed),
        "failed": summary.get("failed", 0),
        "unsupported": summary.get("unsupported", 0),
        "blocked": summary.get("blocked", 0),
        "not_run": summary.get("not_run", 0),
        "not_applicable": summary.get("not_applicable", 0),
        "passed_without_correctness": len(without_correctness),
        "correctness_failed": len(failed_correctness),
        "missing_records": len(missing),
        "retryable": len(retryable),
    }
    # The records and the summary must agree; a disagreement means one of them is stale.
    recorded_not_run = len([record for record in records.values()
                            if record.get("status") == "not_run"])
    if recorded_not_run > totals["not_run"]:
        totals["not_run"] = recorded_not_run
    path = ctx.write_json("result_validation.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id, "run_id": ctx.run_id,
        "totals": totals,
        "consistency": summary.get("consistency") or {},
        "passed_without_correctness": without_correctness[:50],
        "correctness_failed": failed_correctness[:50],
        "missing_records": missing[:50],
        "retryable": retryable[:50],
    })
    ctx.data["validation"] = {"totals": totals, "records": records,
                              "manifest": manifest}
    if without_correctness:
        return PhaseOutcome.failed(
            f"{len(without_correctness)} task(s) are `passed` without a correctness record; "
            "a pass without evidence is not a measurement", artifacts=[path],
            detail={"totals": totals})
    if missing:
        return PhaseOutcome.partial(
            f"{len(missing)} planned task(s) have no result record; the run is incomplete",
            artifacts=[path], detail={"totals": totals})
    if failed_correctness:
        return PhaseOutcome.partial(
            f"{len(failed_correctness)} task(s) failed their correctness check; the cause must "
            "be fixed, not retried blindly", artifacts=[path], detail={"totals": totals})
    if retryable:
        return PhaseOutcome.partial(
            f"{len(retryable)} task(s) failed with a retryable status and will be retried by "
            "`resume`", artifacts=[path], detail={"totals": totals})
    if totals["passed"] == 0:
        return PhaseOutcome.partial(
            "no task passed, so there is no measurement to validate", artifacts=[path],
            detail={"totals": totals})
    if totals["not_run"]:
        return PhaseOutcome.partial(
            f"{totals['not_run']} task(s) never reached a terminal status; the run must be "
            f"resumed before the result is complete", artifacts=[path],
            detail={"totals": totals})
    if totals["failed"] or totals["unsupported"] or totals["blocked"]:
        return PhaseOutcome.partial(
            f"{totals['passed']} task(s) passed; {totals['failed']} failed, "
            f"{totals['unsupported']} unsupported, {totals['blocked']} blocked (each keeps its "
            f"reason in the report)", artifacts=[path], detail={"totals": totals})
    return PhaseOutcome.passed([path], detail={"totals": totals})
