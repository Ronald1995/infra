"""Phase 3: static and runtime capability probe.

Two different questions, two different answers:

* **static** -- which operator/case/EP combinations the catalog declares executable. The
  command is ``run_benchmark.py capabilities --plan P --json``;
* **runtime** -- whether a communication channel is *usable now* on this SoC. The staged
  probe in ``baselines/l0/topology/runtime_probe.py`` answers that, and only a runtime
  answer may mark a transport mechanism supported. A static descriptor is never
  promoted to a runtime verification.

The phase only probes the requested scope, so an unrelated channel cannot occupy a
device resource that a measured point needs.
"""
from scripts.workflow.phases import PhaseOutcome

#: Statuses that mean "the case may still be measured".
RUNNABLE_STATUSES = ("not_run",)


def _capability_args(ctx, runtime=False):
    request = ctx.request
    argv = ["capabilities"]
    if request.is_plan_file:
        argv += ["--plan", str(_path(ctx, request.plan))]
    else:
        argv += ["--config", str(_path(ctx, request.config))]
    argv += _scope_args(ctx)
    if runtime:
        argv.append("--runtime")
    argv.append("--json")
    return argv


def _path(ctx, value):
    path = ctx.root / value
    return path if path.exists() else value


def _scope_args(ctx):
    request = ctx.request
    argv = []
    if request.scope.levels:
        argv += ["--levels", ",".join(request.scope.levels)]
    if request.scope.cases:
        argv += ["--cases", ",".join(request.scope.cases)]
    if isinstance(request.scope.operators, list):
        argv += ["--operators", ",".join(request.scope.operators)]
    if len(request.scope.ep_sizes) == 1:
        argv += ["--ep-size", str(request.scope.ep_sizes[0])]
    return argv


def summarise_static(summary):
    """Totals per EP size from the ``capabilities --json`` payload."""
    output = {}
    for size, entry in (summary or {}).items():
        statuses = dict(entry.get("statuses") or {})
        runnable = sum(count for status, count in statuses.items()
                       if status in RUNNABLE_STATUSES)
        output[str(size)] = {
            "tasks": entry.get("tasks", 0),
            "statuses": statuses,
            "runnable": runnable,
            "operators": len(entry.get("operators") or {}),
            "cases": len(entry.get("cases") or {}),
        }
    return output


def summarise_runtime(payload):
    """Per-status and per-EP-size view of the runtime probe payload."""
    if not isinstance(payload, dict):
        return None
    runtime = payload.get("runtime") or {}
    records = runtime.get("records") or []
    per_ep = {}
    for record in records:
        size = record.get("ep_size")
        entry = per_ep.setdefault(str(size), {"channels": 0, "supported": 0,
                                              "statuses": {}})
        entry["channels"] += 1
        entry["supported"] += 1 if record.get("supported") else 0
        status = record.get("status") or "unknown"
        entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
    return {"stages": runtime.get("stages") or [], "records": records,
            "summary": runtime.get("summary") or {}, "per_ep_size": per_ep,
            "released": all(record.get("released", True) for record in records)}


def handler(ctx):
    request = ctx.request
    static = ctx.commands.benchmark(*_capability_args(ctx), log_name="capabilities.json.log")
    ctx.record("capability_probe", static, probe="static")
    payload = static.json()
    if not isinstance(payload, dict):
        return PhaseOutcome.failed(
            "`capabilities --json` produced no JSON (exit "
            f"{static.exit_code}); see {static.log_path}")

    # A plan answers "which tasks can run"; a single configuration answers "which L0
    # channels exist on this machine".  They are different payloads, so they are read
    # differently instead of pretending one shape fits both.
    if request.is_plan_file:
        static_summary = summarise_static(payload)
        matrix_summary = None
    else:
        static_summary = {}
        matrix_summary = {"source": payload.get("source"),
                          "channels": len(payload.get("channels") or ()),
                          "engines": len(payload.get("engines") or ()),
                          "accelerators": len(payload.get("accelerators") or ())}

    runtime_payload, runtime_summary = None, None
    if request.environment.device == "npu":
        runtime = ctx.commands.benchmark(*_capability_args(ctx, runtime=True),
                                         log_name="capabilities-runtime.json.log")
        ctx.record("capability_probe", runtime, probe="runtime")
        runtime_payload = runtime.json()
        runtime_summary = summarise_runtime(runtime_payload)
        if runtime_summary is None:
            # A probe that cannot run is a recorded limitation, not a crash: the static
            # matrix stays usable and no runtime claim is made.
            runtime_summary = {"records": [], "per_ep_size": {}, "error":
                               f"runtime probe produced no JSON (exit {runtime.exit_code})"}

    capacity = ctx.data.get("capacity") or {}
    runnable_sizes, blocked_sizes = {}, {}
    for ep_size in request.scope.ep_sizes:
        capacity_entry = capacity.get(ep_size) or {}
        if capacity_entry.get("executable") is False:
            blocked_sizes[ep_size] = capacity_entry.get("reason")
            continue
        if not request.is_plan_file:
            # Support is decided by the plan expansion; here only the host capacity is known.
            runnable_sizes[ep_size] = {"tasks": None, "runnable": None}
            continue
        static_entry = static_summary.get(str(ep_size)) or {}
        if static_entry.get("tasks") and static_entry.get("runnable"):
            runnable_sizes[ep_size] = static_entry
        elif static_entry.get("tasks"):
            blocked_sizes[ep_size] = "no runnable task for this EP size"
        else:
            blocked_sizes[ep_size] = "the plan expands no task for this EP size"

    path = ctx.write_json("capability_report.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id,
        "static": {"exit_code": static.exit_code, "log": str(static.log_path),
                   "summary": static_summary, "matrix": matrix_summary},
        "runtime": {"summary": runtime_summary},
        "runnable_ep_sizes": sorted(runnable_sizes),
        "blocked_ep_sizes": blocked_sizes,
        "task_support_validated": bool(request.is_plan_file),
    })
    ctx.data["capability"] = {"static": static_summary, "matrix": matrix_summary,
                              "runtime": runtime_summary,
                              "runnable_ep_sizes": sorted(runnable_sizes),
                              "blocked_ep_sizes": blocked_sizes}
    if not runnable_sizes:
        return PhaseOutcome.blocked(
            "no requested EP size has a runnable task: "
            + "; ".join(f"EP{size}: {reason}" for size, reason in blocked_sizes.items()),
            artifacts=[path])
    if blocked_sizes:
        return PhaseOutcome.partial(
            "some EP sizes are blocked by capability: "
            + "; ".join(f"EP{size}: {reason}" for size, reason in blocked_sizes.items()),
            artifacts=[path], detail={"blocked_ep_sizes": blocked_sizes})
    return PhaseOutcome.passed([path], detail={"ep_sizes": sorted(runnable_sizes)})


