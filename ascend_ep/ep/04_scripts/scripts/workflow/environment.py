"""Phase 1: environment discovery and rank capacity.

The doctor is the single source of environment facts -- this module only runs it,
computes how many ranks the host can actually provide, and decides which requested EP
sizes are therefore executable. It never invents a device: an EP size that needs more
ranks than the host (plus any declared multi-node nodes) has is *blocked*, not silently
run on a smaller group.
"""
from scripts.workflow.phases import PhaseOutcome

#: Processes a single CPU host may launch for a Gloo run before it stops being a
#: meaningful smoke test. This is a *policy* limit, not a measurement.
CPU_PROCESS_LIMIT = 64


def doctor_profile(request):
    """The dependency profile that answers "can this request run here?"."""
    if request.environment.device == "cpu":
        return "cpu"
    if request.execution.nnodes > 1:
        return "all"
    return "all"


def detected_devices(report, request):
    """How many local accelerator devices the host provides, with its source."""
    if request.environment.local_devices:
        return int(request.environment.local_devices), "request"
    platform = (report or {}).get("platform") or {}
    smi = platform.get("npu_smi") or {}
    rows = smi.get("chip_rows")
    if isinstance(rows, int) and rows > 0:
        return rows, "npu-smi info"
    nodes = [path for path in (platform.get("devices") or ())
             if not path.endswith(("davinci_manager", "devmm_svm"))]
    if nodes:
        return len(nodes), "device nodes"
    visible = ((platform.get("env") or {}).get("ASCEND_RT_VISIBLE_DEVICES")
               or (platform.get("env") or {}).get("ASCEND_VISIBLE_DEVICES"))
    if visible:
        entries = [item for item in str(visible).split(",") if item.strip()]
        if entries:
            return len(entries), "ASCEND_RT_VISIBLE_DEVICES"
    return 0, "unavailable"


def rank_capacity(report, request):
    """``{ep_size: {"ranks_needed", "ranks_available", "executable", "reason"}}``."""
    tp = 1
    devices, device_source = detected_devices(report, request)
    per_node = request.execution.nproc_per_node or devices
    nodes = max(1, request.execution.nnodes)
    available = per_node * nodes
    capacity = {}
    for ep_size in request.scope.ep_sizes:
        needed = ep_size * tp
        if request.environment.device == "cpu":
            executable = needed <= CPU_PROCESS_LIMIT * nodes
            reason = None if executable else (
                f"EP{ep_size} needs {needed} ranks, above the {CPU_PROCESS_LIMIT * nodes} "
                f"process policy limit for a CPU run")
            source = "cpu processes"
        elif available <= 0:
            executable, source = False, device_source
            reason = (f"no Ascend device was detected (source: {device_source}), so "
                      f"EP{ep_size} cannot run")
        elif needed > available:
            executable, source = False, device_source
            if nodes == 1:
                reason = (f"EP{ep_size} needs {needed} ranks but this host offers "
                          f"{available} ({device_source}); request multi-node launch "
                          f"(execution.nnodes > 1) to run it")
            else:
                reason = (f"EP{ep_size} needs {needed} ranks but {nodes} node(s) x "
                          f"{per_node} device(s) = {available}")
        else:
            executable, reason, source = True, None, device_source
        capacity[ep_size] = {"ranks_needed": needed, "ranks_available": available,
                             "device_source": source, "executable": executable,
                             "reason": reason}
    return capacity


def unavailable_dependencies(report):
    """Dependency ids the doctor could not satisfy, with their blocking groups."""
    failing = []
    for check in (report or {}).get("checks") or ():
        if check.get("status") == "fail":
            failing.append({"id": check.get("id"), "detail": check.get("detail"),
                            "action": check.get("suggested_action"),
                            "blocking_for": list(check.get("blocking_for") or ()),
                            "category": check.get("category")})
    return failing


def handler(ctx):
    request = ctx.request
    result = ctx.commands.benchmark("doctor", "--profile", doctor_profile(request), "--json",
                                    log_name="doctor.json.log")
    ctx.record("environment_discovery", result)
    report = result.json()
    if not isinstance(report, dict):
        return PhaseOutcome.failed(
            "`run_benchmark.py doctor --json` produced no JSON report"
            f" (exit {result.exit_code}); see {result.log_path}")
    capacity = rank_capacity(report, request)
    report_path = ctx.write_json("environment.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id,
        "doctor": report, "rank_capacity": capacity,
        "device": request.environment.device,
        "soc": request.environment.soc or ((report.get("platform") or {}).get("soc")),
        "doctor_exit_code": result.exit_code,
        "log": str(result.log_path),
    })
    ctx.data["doctor"] = report
    ctx.data["capacity"] = capacity
    executable = [size for size, entry in capacity.items() if entry["executable"]]
    blocked = {size: entry["reason"] for size, entry in capacity.items()
               if not entry["executable"]}
    if not executable:
        return PhaseOutcome.blocked(
            "no requested EP size can run on this host: "
            + "; ".join(f"EP{size}: {reason}" for size, reason in blocked.items()),
            artifacts=[report_path], detail={"capacity": capacity})
    detail = {"capacity": capacity, "blocked_ep_sizes": blocked}
    if blocked:
        return PhaseOutcome.partial(
            "some EP sizes are not executable here: "
            + "; ".join(f"EP{size}: {reason}" for size, reason in blocked.items()),
            artifacts=[report_path], detail=detail)
    return PhaseOutcome.passed([report_path], detail=detail)
