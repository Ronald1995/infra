"""NPU smoke: environment gate plus one real NPU run per operator.

Two stages, both on real Ascend NPU:

Stage 1 ``environment``
    The public L0 hardware baselines (P2P, collectives, memory, GMM, SwiGLU,
    communication/compute overlap).  It answers "can this host run anything at
    all", and it is what gives L0 real coverage -- no operator profile repeats it.

Stage 2 ``operators``
    For every operator, the case declared in ``operators/smoke_profiles.yaml``,
    executed through the normal ``run_benchmark`` planner and runner.  A profile
    that claims a backward pass runs the backward case too.

The driver itself does not schedule, launch or time anything: it builds tasks with
``common.scheduler.make_plan``, executes them with ``common.controller`` and reads
the results back.  What it does own is the *policy*: the whole run is NPU_ONLY
(``common.execution_mode``), a hardware shortfall is recorded as a blocker instead
of being downgraded to CPU, and the report separates operator coverage, level
coverage and execution-path coverage, reporting ``COVERAGE_GAP`` for every
required path that no passed profile exercised.

Exit codes
    0  every required operator and required path is covered and passed
    1  the run completed but coverage is incomplete or something failed
    2  the request or the profile table is invalid (nothing was executed)
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import execution_mode, readiness, smoke_profile, status_map  # noqa: E402
from common.case_spec import catalog  # noqa: E402
from common.config import load_configs  # noqa: E402
from common.controller import create_manifest, execute, results  # noqa: E402
from common.environment import capture  # noqa: E402
from common.scheduler import make_plan  # noqa: E402

SCHEMA_VERSION = 1
#: Stage 1 case selection: one representative (or a small set) per L0 family.
#: Derived from the case catalog and validated by :func:`stage1_cases`, so a
#: renamed or re-categorised case fails loudly instead of silently shrinking the
#: gate.
STAGE1_CASES = {
    "collective": ("HW-001", "HW-004", "HW-006"),   # p2p unidirectional, alltoall, gather/reduce
    "compute": ("HW-010", "HW-011"),                # grouped matmul, swiglu
    "overlap": ("HW-012", "OVERLAP-001"),           # concurrent gmm+comm, concurrent streams
    "memory": ("MEM-001",),                         # local HBM copy
    "transport": ("LINK-001",),                     # native UB transport, one mechanism
    "execution": ("EXEC-001",),                     # hccl_test accelerator path
}
#: Categories Stage 1 must exercise. Transport/execution are reported but not
#: required: their native bridges and the hccl_test binary are optional
#: dependencies, and their absence is already a recorded blocker.
STAGE1_REQUIRED = ("collective", "compute", "overlap")
STAGE1_REPORTED = ("collective", "compute", "overlap", "transport", "memory", "execution")
ENVIRONMENT_CONFIG = ROOT / "configs" / "smoke_environment.yaml"


def stage1_cases(cases=None):
    """Validate the Stage-1 case list against the catalog and flatten it."""
    cases = catalog() if cases is None else cases
    selected, problems = [], []
    for category, ids in STAGE1_CASES.items():
        for case_id in ids:
            case = cases.get(case_id)
            if case is None:
                problems.append(f"{category}: unknown case {case_id}")
            elif case.category != category:
                problems.append(f"{case_id} is {case.category}, not {category}")
            elif case.level != "L0":
                problems.append(f"{case_id} is {case.level}, not L0")
            else:
                selected.append(case_id)
    if problems:
        raise ValueError("invalid Stage 1 case selection: " + "; ".join(problems))
    return selected


# ---------------------------------------------------------------------------
# environment probe
# ---------------------------------------------------------------------------

def device_probe():
    """Ask torch/torch_npu what the host actually offers.

    Never raises: a probe failure is itself the finding, and it must surface as a
    blocker with a reason rather than as a traceback.
    """
    info = {"npu_available": False, "device_count": 0, "models": [],
            "hccl_backend": None, "error": None}
    try:
        import torch
        import torch_npu  # noqa: F401  (registers the npu device type)
        info["torch"] = torch.__version__
        info["torch_npu"] = getattr(torch_npu, "__version__", None)
        info["npu_available"] = bool(torch.npu.is_available())
        info["hccl_backend"] = bool(torch.distributed.is_backend_available("hccl"))
        if info["npu_available"]:
            info["device_count"] = int(torch.npu.device_count())
            info["models"] = [torch.npu.get_device_name(index)
                              for index in range(info["device_count"])]
            props = []
            for index in range(info["device_count"]):
                try:
                    props.append(getattr(torch.npu.get_device_properties(index),
                                         "total_memory", None))
                except (RuntimeError, OSError):
                    props.append(None)
            info["hbm_bytes"] = props
    except Exception as exc:  # noqa: BLE001 - the probe reports, it never crashes the run
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def environment_record(probe):
    record = {"probe": probe, "capture": capture(probe_npu=False)}
    return record


def dependency_view(soc_model=None, profile="all"):
    """Per-operator readiness from the manifest, classified against this SoC.

    Returns ``(readiness_map, soc, error)``.  A failure here must never stop a run:
    the dependency view refines the report, it does not gate it.
    """
    try:
        from scripts import dependency_report, doctor

        report = dependency_report.build(profile=profile)
        data = doctor.load_manifest(ROOT / "dependencies" / "manifest.yaml")
        entries = {entry["id"]: entry for entry in data["dependencies"]}
        soc = readiness.normalise_soc(soc_model)
        return readiness.classify(report["operators"], entries, soc), soc, None
    except Exception as exc:  # noqa: BLE001 - a dependency view is advisory
        return {}, None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def environment_tasks():
    """Stage 1: the public L0 hardware baseline tasks.

    Cheap by construction: ``configs/smoke_environment.yaml`` sweeps one value per
    dimension, and only the representative cases are selected.
    """
    config = load_configs(ENVIRONMENT_CONFIG)[0]
    return make_plan([config], levels=["L0"], cases=stage1_cases())


def selected_profiles(table, operators=None):
    if not operators:
        return list(table)
    wanted = [name.strip() for name in operators if name.strip()]
    unknown = sorted(set(wanted) - set(table.operators()))
    if unknown:
        raise SystemExit(f"unknown operator(s) in --operators: {', '.join(unknown)}")
    return [table.get(name) for name in wanted]


def profile_tasks(profile, direction="both"):
    tasks = list(profile.forward_tasks())
    if direction in ("both", "backward"):
        tasks.extend(profile.backward_tasks())
    return tasks


def build_plan(table, stage="all", operators=None, direction="both"):
    """Return ``(tasks, owners)`` where owners maps task_id -> profile label."""
    tasks, owners = [], {}
    if stage in ("all", "environment"):
        for task in environment_tasks():
            task.setdefault("smoke_stage", "environment")
            tasks.append(task)
    if stage in ("all", "operators"):
        for profile in selected_profiles(table, operators):
            for task in profile_tasks(profile, direction):
                task["smoke_stage"] = "operators"
                owners[task["task_id"]] = profile.label
                tasks.append(task)
    return tasks, owners


def apply_npu_only(mode, tasks):
    """Refuse a non-NPU task while the formal mode is active."""
    for task in tasks:
        execution_mode.assert_formal(
            mode, task["config"],
            f"{task['backend_id']}:{task['case_id']}")


def apply_preflight(probe, tasks):
    """Turn a hardware shortfall into a recorded blocker on the task itself.

    ``execute`` only launches a task whose status is ``not_run`` and whose reason
    is empty, so writing the blocker here is what stops the launch.  The kind is
    kept in a side map because the raw status vocabulary has no
    "hardware_shortfall" value.
    """
    reasons = {}
    for task in tasks:
        config = task["config"]
        ranks = config["distributed"]["ep_size"] * config["distributed"].get("tp_size", 1)
        verdict = execution_mode.preflight(probe["device_count"], ranks,
                                           device_reachable=probe["npu_available"])
        if verdict["status"]:
            task["status"] = verdict["status"]
            task["reason"] = verdict["reason"]
            reasons[task["task_id"]] = verdict["blocked_kind"]
    return reasons


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def launch_args(run_id, nproc_per_node=None):
    return SimpleNamespace(nnodes=1, nproc_per_node=nproc_per_node, node_rank=0,
                           master_addr="127.0.0.1", master_port=None, session=run_id,
                           cases=None, operators=None, ep_size=None, task_ids=None,
                           fail_fast=False)


#: Marker written into the preflight reason, so a resumed run can tell "the host
#: had too few cards last time" apart from any other environment blocker and
#: re-evaluate it instead of inheriting a stale verdict.
_SHORTFALL_MARKER = "never replaced by CPU execution"


def reset_preflight_blocks(tasks):
    """Clear a previous hardware verdict so a resume can re-probe the host.

    A task blocked because the cards were busy must be retried when the cards come
    back; every other blocker (a missing package, a platform mismatch) is left
    alone, because nothing about it changes between runs.
    """
    cleared = 0
    for task in tasks:
        if (task.get("status") == "environment_unavailable"
                and _SHORTFALL_MARKER in (task.get("reason") or "")):
            task["status"] = "not_run"
            task["reason"] = None
            cleared += 1
    return cleared


def open_run(output, tasks, run_id, resume=False):
    """Create a new run directory, or reopen an existing one for a resume."""
    from common.controller import checked_id
    from common.storage import load_json

    root = Path(output).resolve() / checked_id(run_id)
    manifest_path = root / "manifest.json"
    if not resume:
        if manifest_path.is_file():
            raise ValueError(
                f"{root} already holds a run; pass --resume to continue it "
                f"(results already on disk are reused) or choose a new --run-id")
        return create_manifest(output, tasks, run_id)
    if not manifest_path.is_file():
        raise ValueError(f"cannot resume: no manifest at {manifest_path}")
    manifest = load_json(manifest_path)
    planned = {task["task_id"] for task in manifest["tasks"]}
    wanted = {task["task_id"] for task in tasks}
    if planned != wanted:
        stale = len(planned - wanted)
        new = len(wanted - planned)
        raise ValueError(
            f"the stored plan does not match this request ({stale} task(s) no longer "
            f"planned, {new} new task(s)); rerun without --resume into a fresh run-id")
    cleared = reset_preflight_blocks(manifest["tasks"])
    if cleared:
        print(f"resume: re-evaluating {cleared} task(s) that were blocked by a "
              f"hardware shortfall", flush=True)
    return root, manifest


def run_tasks(output, tasks, run_id, nproc_per_node=None, resume=False):
    root, manifest = open_run(output, tasks, run_id, resume=resume)
    code = execute(root, manifest, launch_args(run_id, nproc_per_node), resume=resume)
    return root, manifest, code


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _task_rows(root, manifest):
    rows = {}
    for record in results(root, manifest):
        rows[record["task_id"]] = record
    return rows


def _status_of(rows, task):
    record = rows.get(task["task_id"])
    if record is None:
        return task["status"] or "not_run", task.get("reason") or ""
    return record.get("status") or "not_run", record.get("error_reason") or ""


def profile_outcome(profile, tasks, rows, kinds, mode="npu_only", readiness_map=None):
    """Collapse a profile's tasks into one bucket.

    A profile passes only when *every* task it expanded to passed on the NPU; one
    blocked step means the operator's main path was not proven, whatever the other
    tasks did.
    """
    buckets, detail = [], []
    for task in tasks:
        status, reason = _status_of(rows, task)
        bucket = status_map.bucket(status, reason, kinds.get(task["task_id"]))
        bucket = readiness.refine_bucket(bucket, profile.operator, readiness_map)
        if bucket == status_map.PASSED:
            config = task["config"]
            if not execution_mode.counts_as_operator_evidence(mode, config, status):
                bucket = status_map.UNKNOWN
        buckets.append((bucket, task, status, reason))
        detail.append({"task_id": task["task_id"], "case_id": task["case_id"],
                       "level": task["level"], "status": status, "bucket": bucket,
                       "reason": reason})
    order = [status_map.FAILED, status_map.BUILD_FAILED, status_map.BLOCKED_DEPENDENCY,
             status_map.BLOCKED_HARDWARE, status_map.BLOCKED_PLATFORM,
             status_map.BLOCKED_ENVIRONMENT, status_map.UNKNOWN, status_map.PASSED]
    chosen = status_map.PASSED
    for candidate in order:
        if any(item[0] == candidate for item in buckets):
            chosen = candidate
            break
    worst = next((item for item in buckets if item[0] == chosen), None)
    return chosen, (worst[3] if worst else ""), detail


def build_report(table, tasks, owners, rows, kinds, probe, stage, run_id, mode="npu_only",
                 readiness_map=None, soc=None):
    operator_rows, outcomes = [], {}
    for profile in table:
        own = [task for task in tasks if owners.get(task["task_id"]) == profile.label]
        if not own:
            continue
        bucket, reason, detail = profile_outcome(profile, own, rows, kinds, mode,
                                                 readiness_map)
        outcomes[profile.label] = bucket
        ready = (readiness_map or {}).get(profile.operator) or {}
        operator_rows.append({
            "operator": profile.operator, "family": profile.family,
            "level": profile.level, "case": profile.case, "ep_size": profile.ep_size,
            "direction": profile.direction, "validation": profile.validation,
            "paths": list(profile.paths), "bucket": bucket, "reason": reason,
            "tasks": detail, "note": profile.note,
            "readiness": ready.get("state"),
            "unsupported_dependencies": ready.get("unsupported") or [],
        })

    stage1 = {"required": list(STAGE1_REQUIRED), "categories": {}, "passed": None}
    if stage in ("all", "environment"):
        cases = catalog()
        categories = {}
        for task in tasks:
            if task.get("smoke_stage") != "environment":
                continue
            categories.setdefault(cases[task["case_id"]].category, []).append(task)
        for name in STAGE1_REPORTED:
            group = categories.get(name) or []
            if not group:
                stage1["categories"][name] = {"total": 0, "passed": 0, "failed": 0,
                                              "blocked": 0, "bucket": None, "ok": False}
                continue
            buckets = []
            for task in group:
                status, reason = _status_of(rows, task)
                buckets.append(status_map.bucket(status, reason, kinds.get(task["task_id"])))
            passed = sum(1 for bucket in buckets if bucket == status_map.PASSED)
            failed = sum(1 for bucket in buckets if bucket == status_map.FAILED)
            blocked = len(buckets) - passed - failed
            # A family is exercised when at least one point really ran and nothing
            # crashed.  Unavailable variants are reported as blocked; they are not a
            # crash, but they also never count as passed.
            ok = passed > 0 and failed == 0
            worst = next((bucket for bucket in (status_map.FAILED,
                                                status_map.BLOCKED_DEPENDENCY,
                                                status_map.BLOCKED_HARDWARE,
                                                status_map.BLOCKED_PLATFORM,
                                                status_map.BLOCKED_ENVIRONMENT)
                          if bucket in buckets), None)
            stage1["categories"][name] = {
                "total": len(group), "passed": passed, "failed": failed,
                "blocked": blocked,
                "bucket": status_map.PASSED if ok and blocked == 0 else (worst or status_map.PASSED),
                "ok": ok,
            }
        stage1["passed"] = all(
            (stage1["categories"].get(name) or {}).get("ok") for name in STAGE1_REQUIRED)

    level_evidence = {}
    if stage1["passed"] is not None:
        # Report L0 either way: a failed gate must read "the evidence did not
        # pass", not "nobody planned L0".
        level_evidence["L0"] = {"passed": bool(stage1["passed"]),
                                "source": "Stage 1 environment smoke"}
    coverage = table.coverage(outcomes, level_evidence=level_evidence)

    counts = {name: 0 for name in status_map.BUCKETS}
    for row in operator_rows:
        counts[row["bucket"]] += 1

    requirements = smoke_profile.coverage_requirements(table, None)
    gaps = list(coverage["gaps"])
    for operator in requirements["missing"]:
        gaps.append({"kind": "COVERAGE_GAP", "scope": "operator", "path": operator,
                     "reason": "no smoke profile declared", "operators": [operator]})

    operator_ok = (coverage["operator_coverage"]["passed"] ==
                   coverage["operator_coverage"]["total"] == requirements["required"])
    paths_ok = coverage["path_coverage"]["covered"] == coverage["path_coverage"]["total"]
    levels_ok = coverage["level_coverage"]["covered"] == coverage["level_coverage"]["total"]
    env_ok = stage1["passed"] is not False
    if not probe["npu_available"]:
        conclusion = "BLOCKED"
    elif operator_ok and paths_ok and levels_ok and env_ok and not gaps:
        conclusion = "PASS"
    else:
        conclusion = "INCOMPLETE"

    readiness_counts = {}
    for row in operator_rows:
        key = row.get("readiness") or "unreported"
        readiness_counts[key] = readiness_counts.get(key, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "execution_mode": mode,
        "execution_mode_description": execution_mode.describe(mode),
        "stage": stage,
        "environment": environment_record(probe),
        "stage1": stage1,
        "operators": operator_rows,
        "coverage": coverage,
        "counts": counts,
        "requirements": requirements,
        "gaps": gaps,
        "readiness": {"soc": soc, "counts": readiness_counts,
                      "operators": readiness_map or {}},
        "conclusion": conclusion,
    }


RECOMMENDED = {
    status_map.BLOCKED_DEPENDENCY: "install the dependency and rerun the same profile",
    status_map.BLOCKED_HARDWARE: "run this EP size on a host with enough cards",
    status_map.BLOCKED_PLATFORM: "the adapter targets other silicon; do not retry here",
    status_map.BLOCKED_ENVIRONMENT: "fix the host environment (driver/CANN/devices)",
    status_map.BUILD_FAILED: "fix the kernel build and rerun",
}


def render_markdown(report):
    probe = report["environment"]["probe"]
    capture_info = report["environment"]["capture"]
    lines = [
        f"# NPU smoke report: {report['run_id']}",
        "",
        f"- execution mode: **{report['execution_mode']}** "
        f"({report['execution_mode_description']})",
        f"- stage: {report['stage']}",
        f"- conclusion: **{report['conclusion']}**",
        "",
        "## Environment",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| NPU available | {probe['npu_available']} |",
        f"| device count | {probe['device_count']} |",
        f"| devices | {', '.join(sorted(set(probe['models']))) or '-'} |",
        f"| HCCL backend | {probe['hccl_backend']} |",
        f"| torch / torch_npu | {probe.get('torch')} / {probe.get('torch_npu')} |",
        f"| CANN | {(capture_info.get('cann_version') or {}).get('version') if isinstance(capture_info.get('cann_version'), dict) else capture_info.get('cann_version')} |",
        f"| driver | {capture_info.get('driver_version')} |",
    ]
    if probe.get("error"):
        lines.append(f"| probe error | `{probe['error']}` |")

    stage1 = report["stage1"]
    if stage1["categories"]:
        lines += ["", "## Stage 1: environment smoke (L0)", "",
                  "| category | tasks | passed | blocked | failed | required | ok |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        for name in STAGE1_REPORTED:
            entry = stage1["categories"].get(name)
            if not entry:
                continue
            lines.append(f"| {name} | {entry['total']} | {entry['passed']} | "
                         f"{entry.get('blocked', 0)} | {entry.get('failed', 0)} | "
                         f"{'yes' if name in STAGE1_REQUIRED else 'no'} | "
                         f"{entry.get('ok')} |")
        lines.append("")
        lines.append(f"Stage 1 passed: **{stage1['passed']}**")

    lines += ["", "## Stage 2: operator smoke (real NPU main path)", "",
              "| operator | family | level | case | EP | direction | status | reason |",
              "| --- | --- | --- | --- | ---: | --- | --- | --- |"]
    for row in report["operators"]:
        reason = (row["reason"] or "").replace("|", "/")
        if len(reason) > 90:
            reason = reason[:90] + "..."
        lines.append(f"| `{row['operator']}` | {row['family']} | {row['level']} | "
                     f"{row['case']} | {row['ep_size']} | {row['direction']} | "
                     f"**{row['bucket']}** | {reason} |")

    coverage = report["coverage"]
    lines += ["", "## Coverage", "",
              "| dimension | covered | total |", "| --- | ---: | ---: |",
              f"| operators (NPU main path passed) | {coverage['operator_coverage']['passed']} "
              f"| {coverage['operator_coverage']['total']} |",
              f"| levels L0-L4 | {coverage['level_coverage']['covered']} "
              f"| {coverage['level_coverage']['total']} |",
              f"| execution paths | {coverage['path_coverage']['covered']} "
              f"| {coverage['path_coverage']['total']} |", ""]
    lines += ["| level | profiles | passed | covered | source |", "| --- | ---: | ---: | --- | --- |"]
    for row in coverage["levels"]:
        lines.append(f"| {row['level']} | {row['total']} | {row['passed']} | "
                     f"{row['covered']} | {row.get('source', '-')} |")

    gaps = report["gaps"]
    lines += ["", f"## COVERAGE_GAP ({len(gaps)})", ""]
    if not gaps:
        lines.append("None: every required path, level and operator was covered by a passed NPU run.")
    else:
        lines += ["| scope | target | reason |", "| --- | --- | --- |"]
        for gap in gaps:
            lines.append(f"| {gap['scope']} | {gap['path']} | {gap['reason']} |")

    readiness_info = report.get("readiness") or {}
    if readiness_info.get("counts"):
        lines += ["", "## Dependency readiness", "",
                  f"host SoC: `{readiness_info.get('soc') or 'unknown'}`", "",
                  "| state | operators |", "| --- | ---: |"]
        for state in readiness.STATES + ("unreported",):
            if readiness_info["counts"].get(state):
                lines.append(f"| {state} | {readiness_info['counts'][state]} |")
        unsupported = [row["operator"] for row in report["operators"]
                       if row.get("readiness") == readiness.UNSUPPORTED_PLATFORM]
        if unsupported:
            lines += ["", "These adapters declare no support for this SoC, so installing "
                          "their dependency would not make them runnable:", "",
                      ", ".join(f"`{name}`" for name in unsupported)]

    lines += ["", "## Outcome buckets", "",
              "| bucket | count | next action |", "| --- | ---: | --- |"]
    for name in status_map.BUCKETS:
        if report["counts"].get(name):
            lines.append(f"| {name} | {report['counts'][name]} | "
                         f"{RECOMMENDED.get(name, '-')} |")
    return "\n".join(lines) + "\n"


def write_report(root, report):
    path = Path(root) / "smoke_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = Path(root) / "smoke_report.md"
    markdown.write_text(render_markdown(report), encoding="utf-8")
    return path, markdown


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common(command):
    command.add_argument("--profiles", type=Path, default=None,
                         help="smoke profile table (default operators/smoke_profiles.yaml)")
    command.add_argument("--operators", type=csv_values, default=None)


def csv_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan_command = commands.add_parser("plan", help="expand profiles into tasks without running")
    add_common(plan_command)
    plan_command.add_argument("--stage", choices=("all", "environment", "operators"),
                              default="all")
    plan_command.add_argument("--direction", choices=("forward", "both"), default="both")
    plan_command.add_argument("--json", action="store_true")

    run_command = commands.add_parser("run", help="execute the smoke on real Ascend NPU")
    add_common(run_command)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--run-id", required=True)
    run_command.add_argument("--stage", choices=("all", "environment", "operators"),
                             default="all")
    run_command.add_argument("--direction", choices=("forward", "both"), default="both")
    run_command.add_argument("--execution-mode", choices=execution_mode.EXECUTION_MODES,
                             default=None)
    run_command.add_argument("--nproc-per-node", type=int, default=None)
    run_command.add_argument("--dry-run", action="store_true",
                             help="plan and gate only; create no run directory")
    run_command.add_argument("--resume", action="store_true",
                             help="continue an existing --run-id: reuse every result already "
                                  "on disk and retry only what is not terminal")
    run_command.add_argument("--no-readiness", action="store_true",
                             help="skip the per-operator dependency readiness view (it runs "
                                  "the dependency doctor, which is slow on a real host)")

    report_command = commands.add_parser(
        "report", help="re-render the report of an existing run from its persisted results")
    add_common(report_command)
    report_command.add_argument("--output", type=Path, required=True)
    report_command.add_argument("--run-id", required=True)
    report_command.add_argument("--no-readiness", action="store_true")
    report_command.add_argument("--live-probe", action="store_true",
                                help="re-probe the device instead of reusing the probe "
                                     "recorded by the run (a past run is reported under the "
                                     "conditions it actually ran with)")

    gate_command = commands.add_parser(
        "gate", help="print the operators a smoke run passed, for a Full Benchmark --operators")
    gate_command.add_argument("--report", type=Path, required=True,
                              help="path to smoke_report.json")
    gate_command.add_argument("--format", choices=("csv", "lines", "json"), default="csv")
    gate_command.add_argument("--strict", action="store_true",
                              help="require the smoke report's own conclusion to be PASS")

    deps_command = commands.add_parser(
        "deps", help="per-operator dependency readiness on this SoC, and how to install")
    add_common(deps_command)
    deps_command.add_argument("--json", action="store_true")
    deps_command.add_argument("--profile", default="all",
                              choices=("cpu", "npu", "communication", "profiling",
                                       "multinode", "all"))
    deps_command.add_argument("--install", action="store_true",
                              help="run the repo's installer for the operators that can be "
                                   "installed here (never for a platform-unsupported one)")
    deps_command.add_argument("--dry-run", action="store_true",
                              help="with --install: print the installer plan, change nothing")
    deps_command.add_argument("--soc", default=None,
                              help="override the SoC family (e.g. Ascend950) when no device "
                                   "is reachable, so the classification can be inspected")
    return parser


def command_plan(args):
    table = smoke_profile.load(args.profiles)
    tasks, owners = build_plan(table, args.stage, args.operators, args.direction)
    if args.json:
        print(json.dumps({"tasks": [
            {"task_id": task["task_id"], "case": task["case_id"], "level": task["level"],
             "backend": task["backend_id"], "stage": task.get("smoke_stage"),
             "ep_size": task["config"]["distributed"]["ep_size"], "profile": owners.get(task["task_id"])}
            for task in tasks]}, indent=2))
        return 0
    print(f"{len(tasks)} tasks from {len(table)} profiles "
          f"({sum(1 for t in tasks if t.get('smoke_stage') == 'environment')} environment)")
    print(f"{'stage':12s} {'profile':38s} {'level':5s} {'case':10s} {'EP':>3s} status")
    for task in tasks:
        owner = owners.get(task["task_id"], "environment")
        print(f"{task.get('smoke_stage', '-'):12s} {owner:38s} {task['level']:5s} "
              f"{task['case_id']:10s} {task['config']['distributed']['ep_size']:3d} "
              f"{task['status']}")
    return 0


def command_run(args):
    mode = execution_mode.resolve(args.execution_mode)
    try:
        table = smoke_profile.load(args.profiles)
    except smoke_profile.ProfileError as exc:
        print("invalid smoke profile table:", file=sys.stderr)
        for item in str(exc).split("; "):
            print(f"  - {item}", file=sys.stderr)
        return 2

    tasks, owners = build_plan(table, args.stage, args.operators, args.direction)
    apply_npu_only(mode, tasks)

    probe = device_probe()
    if not probe["npu_available"] and execution_mode.is_formal(mode):
        print(f"NPU_ONLY: no Ascend NPU available ({probe['error'] or 'torch.npu.is_available() is False'}); "
              f"refusing to run operator smoke on CPU", file=sys.stderr)
    kinds = apply_preflight(probe, tasks)

    if args.dry_run:
        runnable = sum(1 for task in tasks if task["status"] == "not_run")
        print(f"mode={mode} tasks={len(tasks)} runnable={runnable} "
              f"blocked={len(tasks) - runnable} devices={probe['device_count']}")
        for task in tasks:
            if task["status"] != "not_run":
                print(f"  blocked {task['case_id']:10s} {task['backend_id']:34s} "
                      f"{task['reason']}")
        return 0 if runnable else 1

    try:
        root, manifest, code = run_tasks(args.output, tasks, args.run_id,
                                         args.nproc_per_node, resume=args.resume)
    except ValueError as exc:
        print(f"cannot start the run: {exc}", file=sys.stderr)
        return 2
    rows = _task_rows(root, manifest)
    readiness_map, soc, readiness_error = (
        ({}, None, "disabled") if getattr(args, "no_readiness", False)
        else dependency_view(probe["models"][0] if probe.get("models") else None))
    if readiness_error and readiness_error != "disabled":
        print(f"warning: dependency readiness unavailable ({readiness_error})",
              file=sys.stderr)
    report = build_report(table, tasks, owners, rows, kinds, probe, args.stage,
                          args.run_id, mode, readiness_map, soc)
    json_path, md_path = write_report(root, report)
    print(render_markdown(report))
    print(f"report: {json_path}")
    print(f"report: {md_path}")
    verdict = 0 if report["conclusion"] == "PASS" else 1
    return verdict if code == 0 else max(verdict, 1)


def owners_from_manifest(table, tasks):
    """Re-attach each manifest task to the profile that planned it.

    Recomputed from ``(backend_id, case_id)`` instead of re-planning, so a report
    can be rebuilt from a persisted run even if the profile table moved on.
    """
    index = {}
    for profile in table:
        index[(profile.operator, profile.case)] = profile.label
        if profile.backward_case:
            index[(profile.operator, profile.backward_case)] = profile.label
    return {task["task_id"]: index.get((task["backend_id"], task["case_id"]))
            for task in tasks}


def command_report(args):
    """Rebuild the report from an existing run directory; nothing is executed."""
    from common.controller import checked_id
    from common.storage import load_json

    root = Path(args.output).resolve() / checked_id(args.run_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = load_json(manifest_path)
    table = smoke_profile.load(args.profiles)
    tasks = manifest["tasks"]
    owners = owners_from_manifest(table, tasks)
    stage = ("all" if any(task.get("smoke_stage") == "environment" for task in tasks)
             else "operators")
    previous = root / "smoke_report.json"
    if previous.is_file() and not getattr(args, "live_probe", False):
        # Re-render under the conditions the run actually saw.  Re-probing here
        # would flip a completed NPU run to BLOCKED just because the cards are busy
        # now, which is not what the artifact is about.
        probe = (json.loads(previous.read_text(encoding="utf-8"))
                 .get("environment", {}).get("probe") or device_probe())
    else:
        probe = device_probe()
    # Re-apply the preflight so a hardware shortfall keeps its kind; the tasks are
    # already blocked, so this only recomputes the reason classification.
    kinds = apply_preflight(probe, tasks)
    rows = _task_rows(root, manifest)
    mode = execution_mode.resolve(None)
    readiness_map, soc, readiness_error = (
        ({}, None, "disabled") if getattr(args, "no_readiness", False)
        else dependency_view(probe["models"][0] if probe.get("models") else None))
    if readiness_error and readiness_error != "disabled":
        print(f"warning: dependency readiness unavailable ({readiness_error})",
              file=sys.stderr)
    report = build_report(table, tasks, owners, rows, kinds, probe, stage, args.run_id,
                          mode, readiness_map, soc)
    json_path, md_path = write_report(root, report)
    print(render_markdown(report))
    print(f"report: {json_path}")
    print(f"report: {md_path}")
    return 0 if report["conclusion"] == "PASS" else 1


def command_gate(args):
    """Emit the operator allowlist for a Full Benchmark run.

    The gate is operator-level on purpose: a run where some adapters are blocked by
    a missing dependency must still let the passed ones go on to be measured. Only
    a ``PASSED`` operator enters the benchmark, and ``--strict`` additionally
    demands that the smoke run itself concluded ``PASS``.
    """
    path = Path(args.report)
    if not path.is_file():
        print(f"no smoke report at {path}", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    passed = [row["operator"] for row in report.get("operators", [])
              if row.get("bucket") == status_map.PASSED]
    blocked = {row["operator"]: row.get("bucket") for row in report.get("operators", [])
               if row.get("bucket") != status_map.PASSED}
    conclusion = report.get("conclusion")

    if args.format == "json":
        print(json.dumps({"conclusion": conclusion, "passed": passed, "blocked": blocked},
                         indent=2))
    elif args.format == "lines":
        print("\n".join(passed))
    else:
        print(",".join(passed))

    if args.strict and conclusion != "PASS":
        print(f"strict gate: the smoke run concluded {conclusion}, not PASS",
              file=sys.stderr)
        return 1
    return 0 if passed else 1


def command_deps(args):
    """Report per-operator dependency readiness, optionally driving the installer.

    This never re-implements installation: it classifies (``common/readiness.py``)
    and then hands the subset that is installable *on this SoC* to the repo's own
    ``scripts/install_deps.sh``.  Platform-unsupported operators are excluded from
    the install set, because installing their package cannot make them run.
    """
    table = None
    try:
        table = smoke_profile.load(args.profiles)
    except smoke_profile.ProfileError as exc:
        print(f"warning: smoke profiles unavailable ({exc}); reporting registry-wide",
              file=sys.stderr)
    selected = None
    if args.operators:
        selected = sorted(args.operators)
    elif table is not None:
        selected = table.operators()
    else:
        from common.registry import registry
        selected = sorted(registry())

    probe = device_probe()
    model = args.soc or (probe["models"][0] if probe.get("models") else None)
    readiness_map, soc, error = dependency_view(model, profile=args.profile)
    if error:
        print(f"cannot build the dependency view: {error}", file=sys.stderr)
        return 2

    rows = [(name, readiness_map.get(name) or {}) for name in selected]
    counts = {}
    for _, row in rows:
        key = row.get("state") or "unreported"
        counts[key] = counts.get(key, 0) + 1

    if args.json:
        print(json.dumps({"soc": soc, "counts": counts,
                          "operators": {name: row for name, row in rows}}, indent=2))
    else:
        print(f"host SoC: {soc or 'unknown'}  (from {model or 'no device probe'})")
        print(f"{'operator':36s} {'state':22s} blockers")
        print("-" * 120)
        for name, row in rows:
            blockers = ", ".join(row.get("blockers") or []) or "-"
            print(f"{name:36s} {row.get('state') or 'unreported':22s} {blockers[:60]}")

    installable = [name for name, row in rows
                   if row.get("state") in (readiness.MISSING, readiness.VERSION_MISMATCH,
                                           readiness.SOURCE_UNAVAILABLE,
                                           readiness.BUILD_FAILED)]
    unsupported = [name for name, row in rows
                   if row.get("state") == readiness.UNSUPPORTED_PLATFORM]
    if args.install and installable:
        return _run_installer(installable, unsupported, args)
    if args.install:
        print(f"nothing to install: {len(unsupported)} operator(s) are unsupported on this "
              f"SoC and {len(rows) - len(unsupported)} are ready or unreported",
              file=sys.stderr)
        return 0 if not unsupported else 1
    return 0


def _run_installer(installable, unsupported, args):
    """Hand the installable subset to scripts/install_deps.sh."""
    import subprocess

    script = ROOT / "scripts" / "install_deps.sh"
    command = ["bash", str(script), "--profile", args.profile]
    if args.dry_run:
        command.append("--dry-run")
    print(f"installing for {len(installable)} operator(s): {', '.join(installable)}",
          file=sys.stderr)
    if unsupported:
        print(f"skipping {len(unsupported)} platform-unsupported operator(s): "
              f"{', '.join(unsupported)}", file=sys.stderr)
    result = subprocess.run(command, cwd=ROOT, text=True)
    return result.returncode


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        return command_plan(args)
    if args.command == "report":
        return command_report(args)
    if args.command == "gate":
        return command_gate(args)
    if args.command == "deps":
        return command_deps(args)
    return command_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
