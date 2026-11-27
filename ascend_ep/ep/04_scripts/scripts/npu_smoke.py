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

from common import execution_mode, smoke_profile, status_map  # noqa: E402
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


def run_tasks(output, tasks, run_id, nproc_per_node=None):
    root, manifest = create_manifest(output, tasks, run_id)
    code = execute(root, manifest, launch_args(run_id, nproc_per_node))
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


def profile_outcome(profile, tasks, rows, kinds, mode="npu_only"):
    """Collapse a profile's tasks into one bucket.

    A profile passes only when *every* task it expanded to passed on the NPU; one
    blocked step means the operator's main path was not proven, whatever the other
    tasks did.
    """
    buckets, detail = [], []
    for task in tasks:
        status, reason = _status_of(rows, task)
        bucket = status_map.bucket(status, reason, kinds.get(task["task_id"]))
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


def build_report(table, tasks, owners, rows, kinds, probe, stage, run_id, mode="npu_only"):
    operator_rows, outcomes = [], {}
    for profile in table:
        own = [task for task in tasks if owners.get(task["task_id"]) == profile.label]
        if not own:
            continue
        bucket, reason, detail = profile_outcome(profile, own, rows, kinds, mode)
        outcomes[profile.label] = bucket
        operator_rows.append({
            "operator": profile.operator, "family": profile.family,
            "level": profile.level, "case": profile.case, "ep_size": profile.ep_size,
            "direction": profile.direction, "validation": profile.validation,
            "paths": list(profile.paths), "bucket": bucket, "reason": reason,
            "tasks": detail, "note": profile.note,
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

    report_command = commands.add_parser(
        "report", help="re-render the report of an existing run from its persisted results")
    add_common(report_command)
    report_command.add_argument("--output", type=Path, required=True)
    report_command.add_argument("--run-id", required=True)
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

    root, manifest, code = run_tasks(args.output, tasks, args.run_id, args.nproc_per_node)
    rows = _task_rows(root, manifest)
    report = build_report(table, tasks, owners, rows, kinds, probe, args.stage,
                          args.run_id, mode)
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
    probe = device_probe()
    # Re-apply the preflight so a hardware shortfall keeps its kind; the tasks are
    # already blocked, so this only recomputes the reason classification.
    kinds = apply_preflight(probe, tasks)
    rows = _task_rows(root, manifest)
    mode = execution_mode.resolve(None)
    report = build_report(table, tasks, owners, rows, kinds, probe, stage, args.run_id, mode)
    json_path, md_path = write_report(root, report)
    print(render_markdown(report))
    print(f"report: {json_path}")
    print(f"report: {md_path}")
    return 0 if report["conclusion"] == "PASS" else 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        return command_plan(args)
    if args.command == "report":
        return command_report(args)
    return command_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
