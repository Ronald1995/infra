"""Phase 5-6: plan generation, scope validation and resource estimation.

The plan file is the single source of the matrix, so the workflow never builds a task
list itself. It either

* reuses the plan as written (when the request asked for exactly what the plan declares),
  or
* writes a **temporary** plan next to the workflow output that narrows the declared
  scope and points every layer file at its original absolute path.

The repository's own configuration files are never modified. The preview manifest is
written to the workflow directory, so the official run id stays free for the run itself
(``create_manifest`` refuses to reuse a directory).
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import yaml

from common.storage import fingerprint
from scripts.workflow.phases import PhaseOutcome
from scripts.workflow.request import is_plan_document

#: Bounded scan of historical metrics used for the runtime estimate.
HISTORY_SCAN_LIMIT = 400
#: Per-task overhead factor applied to the measured P50 (setup, barrier, validation).
SETUP_FACTOR = 1.35


def _read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve(value, base):
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (Path(base) / path).resolve()


def plan_source(ctx):
    """``(path, kind)`` of the configuration the request expands."""
    return ctx.root / (ctx.request.config or ctx.request.plan), (
        "config" if ctx.request.config else "plan")


def request_matches_plan(request, plan_path):
    """Whether the request is exactly the plan's own declared scope."""
    try:
        data = _read_yaml(plan_path)
    except (OSError, yaml.YAMLError):
        return False
    if not is_plan_document(data):
        return False
    spec = data.get("plan") or data.get("matrix") or {}
    suites = spec.get("suites") or ["all"]
    if isinstance(suites, str):
        suites = [suites]
    operators = spec.get("operators", "all")
    return (sorted(int(size) for size in (spec.get("ep_sizes") or []))
            == sorted(request.scope.ep_sizes)
            and sorted(spec.get("levels") or []) == sorted(request.scope.levels)
            and sorted(suites) == sorted(request.scope.suites)
            and operators == request.scope.operators
            and not request.scope.cases)


def narrowed_plan(ctx, ep_sizes=None):
    """Write the temporary plan for this request and return its path.

    Every layer file is rewritten to an absolute path so the temporary plan can live
    under the workflow output without breaking the composition.  ``ep_sizes`` narrows the
    plan to the EP sizes this host can actually run; an EP size that is blocked by the
    host is never measured on a smaller group.
    """
    source, kind = plan_source(ctx)
    if kind != "plan":
        raise ValueError("a single configuration needs no generated plan")
    data = deepcopy(_read_yaml(source))
    spec = data.get("plan") or data.get("matrix") or {}
    base_dir = Path(source).parent
    spec["ep_sizes"] = list(ep_sizes or ctx.request.scope.ep_sizes)
    spec["levels"] = list(ctx.request.scope.levels)
    spec["suites"] = list(ctx.request.scope.suites)
    spec["operators"] = ctx.request.scope.operators
    profile = getattr(ctx.request.scope, "execution_profile", "")
    if profile:
        spec["profile"] = profile
    if ctx.request.scope.cases:
        spec["cases"] = list(ctx.request.scope.cases)
    for key in ("base_config",):
        if spec.get(key):
            spec[key] = str(_resolve(spec[key], base_dir))
    for key in ("ep_configs", "suite_configs"):
        if isinstance(spec.get(key), dict):
            spec[key] = {name: str(_resolve(value, base_dir))
                         for name, value in spec[key].items()}
    target = ctx.output / "plan" / "request_plan.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def planned_ep_sizes(ctx):
    """The EP sizes this host can run, in the order the request named them."""
    capability = (ctx.data.get("capability") or {})
    runnable = capability.get("runnable_ep_sizes")
    requested = list(ctx.request.scope.ep_sizes)
    if not runnable:
        return requested
    allowed = {int(size) for size in runnable}
    return [size for size in requested if size in allowed]


def scope_args(ctx):
    """The CLI arguments that narrow the plan to the requested scope."""
    request = ctx.request
    argv = []
    if request.scope.levels and len(request.scope.levels) < 5:
        argv += ["--levels", ",".join(request.scope.levels)]
    if request.scope.cases:
        argv += ["--cases", ",".join(request.scope.cases)]
    if isinstance(request.scope.operators, list):
        argv += ["--operators", ",".join(request.scope.operators)]
    profile = getattr(request.scope, "execution_profile", "")
    if profile:
        argv += ["--execution-profile", profile]
    return argv


def expand_argv(ctx, command):
    """The documented CLI arguments that expand the request into tasks."""
    request = ctx.request
    argv = [command]
    if request.is_plan_file:
        plan_path = ctx.root / request.plan
        sizes = planned_ep_sizes(ctx)
        if request_matches_plan(request, plan_path) and sizes == request.scope.ep_sizes:
            argv += ["--plan", str(plan_path)]
        else:
            argv += ["--plan", str(narrowed_plan(ctx, sizes))]
    else:
        argv += ["--config", str(ctx.root / request.config)]
        if request.scope.ep_sizes:
            argv += ["--ep-size", ",".join(str(size) for size in request.scope.ep_sizes)]
    argv += scope_args(ctx)
    return argv


def launch_args(ctx):
    """Multi-node launch arguments, or an empty list for a single node."""
    execution = ctx.request.execution
    if not execution.multinode:
        return []
    argv = ["--nnodes", str(execution.nnodes)]
    if execution.nproc_per_node:
        argv += ["--nproc-per-node", str(execution.nproc_per_node)]
    if execution.master_port:
        argv += ["--master-port", str(execution.master_port)]
    if execution.session:
        argv += ["--session", execution.session]
    argv += ["--master-addr", execution.master_addr]
    return argv


def _preview_command(ctx):
    return expand_argv(ctx, "plan")


def manifest_counts(manifest):
    """Counts and breakdowns a plan gate and a report can both use."""
    tasks = manifest.get("tasks") or []
    statuses = Counter(task.get("status") for task in tasks)
    executable = [task for task in tasks if task.get("status") == "not_run"
                  and not task.get("reason")]
    by_ep = defaultdict(lambda: {"planned": 0, "executable": 0, "statuses": Counter()})
    by_level = defaultdict(lambda: {"planned": 0, "executable": 0})
    by_operator = defaultdict(lambda: {"planned": 0, "executable": 0})
    by_case = defaultdict(lambda: {"planned": 0, "executable": 0})
    variants = set()
    for task in tasks:
        size = str(task.get("ep_size") or (task.get("config") or {}).get(
            "distributed", {}).get("ep_size"))
        runnable = task.get("status") == "not_run" and not task.get("reason")
        entry = by_ep[size]
        entry["planned"] += 1
        entry["executable"] += 1 if runnable else 0
        entry["statuses"][task.get("status")] += 1
        by_level[task.get("level")]["planned"] += 1
        by_level[task.get("level")]["executable"] += 1 if runnable else 0
        by_operator[task.get("backend_id")]["planned"] += 1
        by_operator[task.get("backend_id")]["executable"] += 1 if runnable else 0
        by_case[task.get("case_id")]["planned"] += 1
        by_case[task.get("case_id")]["executable"] += 1 if runnable else 0
        variants.add((task.get("case_id"), task.get("backend_id"),
                      (task.get("parameters") or {}).get("variant")))
    return {
        "planned_tasks": len(tasks),
        "executable_tasks": len(executable),
        "blocked_tasks": len([task for task in tasks if task.get("reason")]),
        "statuses": dict(sorted(statuses.items())),
        "variants": len(variants),
        "by_ep_size": {size: {"planned": entry["planned"],
                              "executable": entry["executable"],
                              "statuses": dict(sorted(entry["statuses"].items()))}
                       for size, entry in sorted(by_ep.items(), key=lambda item: str(item[0]))},
        "by_level": dict(sorted(by_level.items())),
        "by_operator": dict(sorted((name, dict(entry))
                                   for name, entry in by_operator.items())),
        "by_case": dict(sorted(by_case.items())),
        "suites": sorted({suite for task in tasks for suite in (task.get("suite_ids") or ())}),
        "topology": (manifest.get("matrix") or {}).get("topology") or {},
        "expected_cases": (manifest.get("matrix") or {}).get("expected_cases") or [],
    }


def plan_hash(manifest):
    """A fingerprint of *which* points the plan expands, not of their status."""
    tasks = manifest.get("tasks") or []
    return fingerprint(sorted((task.get("task_id"),
                               task.get("case_id"),
                               task.get("backend_id"),
                               json.dumps(task.get("parameters") or {}, sort_keys=True))
                              for task in tasks))


def scope_problems(request, counts, capacity, manifest=None, ep_sizes=None):
    """Everything that makes the expanded plan unusable for this request."""
    problems = []
    if not counts["planned_tasks"]:
        problems.append("the plan expanded zero tasks")
    for size in (ep_sizes if ep_sizes is not None else request.scope.ep_sizes):
        entry = (counts["by_ep_size"].get(str(size)) or {})
        blocked = (capacity.get(size) or {}).get("executable") is False
        if not entry.get("planned"):
            if not blocked:
                problems.append(f"EP{size} has no task in the plan")
        elif not entry.get("executable") and not blocked:
            problems.append(f"EP{size} has no executable task")
    missing_levels = [level for level in request.scope.levels
                      if not (counts["by_level"].get(level) or {}).get("planned")]
    if missing_levels:
        problems.append("the plan has no task for level(s) " + ", ".join(missing_levels))
    if manifest is not None:
        unreasoned = [task.get("task_id") for task in manifest.get("tasks") or ()
                      if task.get("status") != "not_run" and not task.get("reason")]
        if unreasoned:
            problems.append(
                f"{len(unreasoned)} task(s) are non-executable without a reason "
                f"(for example {unreasoned[0]})")
    return problems


def handler_plan(ctx):
    """``plan_generation``: expand the request into a validated manifest."""
    request = ctx.request
    requested = list(request.scope.ep_sizes)
    effective = planned_ep_sizes(ctx)
    dropped = [size for size in requested if size not in effective]
    preview_id = f"{ctx.state.workflow_id}-plan"
    preview_dir = ctx.output / "preview"
    manifest_path = preview_dir / preview_id / "manifest.json"
    command_argv = _preview_command(ctx) + ["--output", str(preview_dir),
                                            "--run-id", preview_id]
    if not manifest_path.is_file():
        result = ctx.commands.benchmark(*command_argv, log_name="plan-preview.log")
        ctx.record("plan_generation", result, preview_run_id=preview_id)
        if not manifest_path.is_file():
            return PhaseOutcome.failed(
                f"the plan preview produced no manifest (exit {result.exit_code}); "
                f"see {result.log_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest_counts(manifest)
    capacity = ctx.data.get("capacity") or {}
    problems = scope_problems(request, counts, capacity, manifest,
                              ep_sizes=effective or requested)
    path = ctx.write_json("plan_manifest.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id,
        "preview_run_id": preview_id, "manifest": str(manifest_path),
        "plan_hash": plan_hash(manifest), "counts": counts,
        "requested_ep_sizes": requested, "planned_ep_sizes": effective,
        "blocked_ep_sizes": {size: (capacity.get(size) or {}).get("reason")
                             for size in dropped},
        "problems": problems,
    })
    ctx.state.plan_hash = plan_hash(manifest)
    ctx.data["plan"] = {"manifest": manifest, "counts": counts, "path": manifest_path,
                        "preview_run_id": preview_id, "planned_ep_sizes": effective,
                        "blocked_ep_sizes": {
                            size: (capacity.get(size) or {}).get("reason")
                            for size in dropped}}
    if problems:
        return PhaseOutcome.failed(
            "the expanded plan does not cover the request: " + "; ".join(problems),
            artifacts=[path])
    runnable_eps = [size for size, entry in counts["by_ep_size"].items()
                    if entry.get("executable")]
    blocked_eps = {size: entry.get("statuses") for size, entry in counts["by_ep_size"].items()
                   if not entry.get("executable")}
    detail = {"counts": counts, "runnable_ep_sizes": runnable_eps,
              "blocked_ep_sizes": ctx.data["plan"]["blocked_ep_sizes"]}
    if dropped:
        return PhaseOutcome.partial(
            "EP size(s) " + ", ".join(f"EP{size}" for size in dropped)
            + " cannot run on this host and were left out of the plan: "
            + "; ".join(f"EP{size}: {ctx.data['plan']['blocked_ep_sizes'][size]}"
                        for size in dropped),
            artifacts=[path], detail=detail)
    if blocked_eps and runnable_eps:
        return PhaseOutcome.partial(
            f"EP size(s) {', '.join(sorted(blocked_eps))} have no executable task and are "
            f"blocked; {len(runnable_eps)} EP size(s) remain runnable",
            artifacts=[path], detail=detail)
    if not runnable_eps:
        return PhaseOutcome.blocked(
            "no EP size in the plan has an executable task; every task is blocked or "
            "unsupported", artifacts=[path], detail=detail)
    return PhaseOutcome.passed([path], detail=detail)


# ------------------------------------------------------------------ estimate --

def historical_timings(root):
    """Per-case median P50 from earlier runs, with its sample count.

    Bounded on purpose: the estimate may not walk a 20k-task history on every plan.
    """
    samples = defaultdict(list)
    scanned = 0
    results = Path(root).parent / "05_results" / "raw"
    for path in sorted(results.glob("*/operators/*/*/metrics.json")):
        scanned += 1
        if scanned > HISTORY_SCAN_LIMIT:
            break
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("status") != "passed":
            continue
        p50 = ((record.get("latency") or {}).get("p50"))
        family = record.get("case_id")
        if p50 and family:
            samples[(family, record.get("backend_id"))].append(float(p50))
    medians = {}
    for key, values in samples.items():
        values.sort()
        middle = len(values) // 2
        medians[key] = {"p50_ms": values[middle] if len(values) % 2 else
                        (values[middle - 1] + values[middle]) / 2,
                        "samples": len(values)}
    return medians


def estimate(ctx, counts, history):
    """Runtime and storage estimate, with the basis of every number."""
    request = ctx.request
    iterations = 1
    warmup = 1
    plan = ctx.data.get("plan") or {}
    manifest = plan.get("manifest") or {}
    for task in manifest.get("tasks") or ():
        benchmark = (task.get("config") or {}).get("benchmark") or {}
        if benchmark.get("iterations"):
            iterations = int(benchmark["iterations"])
            warmup = int(benchmark.get("warmup") or 0)
            break
    per_ep = {}
    unknown = 0
    for size, entry in counts["by_ep_size"].items():
        if not entry.get("executable"):
            per_ep[size] = {"executable": 0, "estimate_seconds": 0.0, "basis": "not runnable"}
            continue
        seconds = 0.0
        covered = 0
        for task in manifest.get("tasks") or ():
            task_size = str(task.get("ep_size"))
            if task_size != size or task.get("status") != "not_run" or task.get("reason"):
                continue
            key = (task.get("case_id"), task.get("backend_id"))
            record = history.get(key)
            if not record:
                unknown += 1
                continue
            seconds += record["p50_ms"] / 1000 * (iterations + warmup) * SETUP_FACTOR
            covered += 1
        per_ep[size] = {
            "executable": entry.get("executable"),
            "estimate_seconds": round(seconds, 2) if covered else None,
            "covered_tasks": covered,
            "uncovered_tasks": entry.get("executable") - covered,
            "basis": (f"{covered} task(s) x measured P50 from earlier runs x "
                      f"({iterations} iterations + {warmup} warmup) x {SETUP_FACTOR} setup")
            if covered else "no historical timing for these tasks",
        }
    total = sum(entry["estimate_seconds"] or 0 for entry in per_ep.values())
    known = any(entry["estimate_seconds"] for entry in per_ep.values())
    storage = {}
    for size, entry in counts["by_ep_size"].items():
        ranks = int(size) if str(size).isdigit() else 1
        storage[size] = {"bytes": entry["executable"] * (
            2048 + (iterations + warmup) * ranks * 220)}
    budget = request.execution.time_budget_minutes
    over_budget = bool(budget and known and total > budget * 60)
    return {
        "iterations": iterations, "warmup": warmup, "setup_factor": SETUP_FACTOR,
        "per_ep_size": per_ep,
        "total_estimate_seconds": round(total, 2) if known else None,
        "uncovered_tasks": unknown,
        "basis": ("historical_p50" if known else "unavailable"),
        "uncertainty": ("the estimate covers only tasks with a comparable historical "
                        f"measurement; {unknown} executable task(s) have none"
                        if known else
                        "no historical measurement exists, so no runtime estimate is made"),
        "storage_bytes": sum(entry["bytes"] for entry in storage.values()),
        "storage_per_ep_size": storage,
        "time_budget_minutes": budget,
        "over_budget": over_budget,
    }


def handler_estimate(ctx):
    """``resource_estimation``: counts, runtime and storage, with their basis."""
    counts = (ctx.data.get("plan") or {}).get("counts")
    if not counts:
        return PhaseOutcome.failed("the plan counts are missing; `plan_generation` "
                                   "did not run")
    history = historical_timings(ctx.root)
    estimate_data = estimate(ctx, counts, history)
    path = ctx.write_json("resource_estimate.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id,
        "counts": counts, "estimate": estimate_data,
        "history_tasks": len(history),
    })
    ctx.data["estimate"] = estimate_data
    if estimate_data["over_budget"]:
        return PhaseOutcome.blocked(
            f"the estimate ({estimate_data['total_estimate_seconds']} s) exceeds the "
            f"requested budget of {estimate_data['time_budget_minutes']} minute(s)",
            artifacts=[path], detail={"estimate": estimate_data})
    if estimate_data["basis"] == "unavailable":
        return PhaseOutcome.partial(
            "no runtime estimate is possible: no historical timing matches these tasks",
            artifacts=[path], detail={"estimate": estimate_data})
    return PhaseOutcome.passed([path], detail={"estimate": estimate_data})
