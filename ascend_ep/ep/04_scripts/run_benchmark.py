"""Plan, run, profile, analyze and report expert-parallel benchmarks.

Two entry points exist side by side:

``--config FILE``  single-configuration execution (kept for compatibility)
``--plan FILE``    unified benchmark plan: EP8/EP64/EP128 x L0-L4 x operators
"""
import argparse
import json
from pathlib import Path
import sys

from common.config import LEVELS, load_configs
from common.controller import DEFAULT_OUTPUT, checked_id, create_manifest, execute, results, wait_for
from common.plan import expand as expand_plan, load_plan, matrix_metadata
from common.registry import CAPABILITIES, registry
from common.scheduler import make_plan
from common.storage import load_json, write_json
from common.suites import normalize
from schemas import SEVERITY

COMMANDS_WITH_PLAN = ("plan", "run")


def csv_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def add_launch(parser):
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--session", help="unique ID shared by all node controllers for this invocation")


def add_scope(parser):
    parser.add_argument("--config", type=Path, help="single configuration file")
    parser.add_argument("--plan", type=Path, help="benchmark plan covering several EP sizes")
    parser.add_argument("--suite", help="select one suite (all/full expands the complete catalog)")
    parser.add_argument("--operators", type=csv_values)
    parser.add_argument("--levels", type=csv_values)
    parser.add_argument("--cases", type=csv_values)
    parser.add_argument("--ep-size", type=int, help="select one point of an EP scaling matrix")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--capabilities", action="store_true")
    doctor_command = commands.add_parser("doctor", help="check every third-party dependency")
    doctor_command.add_argument("--profile", default="all",
                                choices=("cpu", "npu", "communication", "profiling",
                                         "multinode", "all"))
    doctor_command.add_argument("--json", action="store_true",
                               help="machine-readable report on stdout")
    doctor_command.add_argument("--wheelhouse", type=Path,
                               help="check that an offline wheelhouse satisfies the pins")
    capability = commands.add_parser("capabilities", help="print the benchmark capability matrix")
    add_scope(capability)
    capability.add_argument("--matrix", type=Path, help="use a recorded L0 matrix instead of probing")
    capability.add_argument("--runtime", action="store_true",
                            help="also run the staged runtime capability probe")
    capability.add_argument("--json", action="store_true")
    for name in ("plan", "run", "baseline", "pipeline"):
        command = commands.add_parser(name)
        add_scope(command)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--run-id")
        command.add_argument("--fail-fast", action="store_true",
                             help="stop at the first failing case instead of continuing")
        if name != "plan":
            add_launch(command)
    for name in ("resume", "profile", "analyze", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if name in ("resume", "profile"):
            add_launch(command)
        if name == "profile":
            command.add_argument("--cases", type=csv_values)
        if name in ("analyze", "report"):
            command.add_argument("--compare-run", action="append", default=[],
                                 help="include another run from the same output root in comparative analysis")
    imported = commands.add_parser("import-ascendep")
    imported.add_argument("source", type=Path)
    imported.add_argument("--source-commit", required=True)
    imported.add_argument("--destination", type=Path, required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("manifest", type=Path)
    worker.add_argument("task_id")
    worker.add_argument("directory", type=Path)
    worker.add_argument("--profile-only", action="store_true")
    return parser


def doctor(args=None):
    """Environment doctor: installed dependency and device availability.

    Delegates to :mod:`scripts.doctor`, which probes the real system and answers per
    benchmark family rather than with one global PASS/FAIL.  A missing dependency is
    reported as a failure with an action, never as a silent skip.
    """
    from scripts.doctor import render, run_doctor

    report = run_doctor(profile=getattr(args, "profile", "all"),
                        wheelhouse=getattr(args, "wheelhouse", None))
    if getattr(args, "json", False):
        print(report.as_json())
    else:
        print(render(report))
    return report.exit_code


def _require_source(args):
    if bool(args.config) == bool(args.plan):
        raise ValueError("choose exactly one of --config or --plan")
    if args.plan and args.command not in COMMANDS_WITH_PLAN + ("capabilities",):
        raise ValueError("--plan is supported by the plan, run and capabilities commands")
    if args.suite:
        normalize(args.suite)
    if args.levels and set(args.levels) - set(LEVELS):
        raise ValueError("expected benchmark levels L0 through L4")


def build_tasks(args):
    """Expand either a benchmark plan or a single configuration."""
    _require_source(args)
    if args.plan:
        plan = load_plan(args.plan)
        run = expand_plan(
            plan,
            ep_sizes=[args.ep_size] if args.ep_size else None,
            suites=[args.suite] if args.suite else None,
            levels=args.levels,
            cases=args.cases,
            operators=args.operators,
        )
        matrix = run.metadata()
        matrix["selected_ep_sizes"] = sorted({task["ep_size"] for task in run.tasks})
        return run.tasks, matrix
    configs = load_configs(args.config)
    if args.suite:
        suite_id = normalize(args.suite)
        configs = [{**config, "suite_id": suite_id} for config in configs]
    if args.ep_size is not None:
        configs = [cfg for cfg in configs if cfg["distributed"]["ep_size"] == args.ep_size]
    if args.command != "plan" and getattr(args, "nnodes", 1) > 1 and len({
        (cfg["distributed"]["ep_size"], cfg["distributed"]["tp_size"]) for cfg in configs
    }) > 1:
        raise ValueError("multi-node scaling runs require --ep-size to select one point")
    tasks = make_plan(configs, args.operators, args.levels, args.cases,
                      args.command == "baseline",
                      suites=[args.suite] if args.suite else None)
    if args.command == "pipeline" and not (args.cases or args.levels or args.suite) and any(
        config["device"] == "npu" for config in configs
    ):
        public = make_plan([config for config in configs if config["device"] == "npu"], baselines=True)
        tasks = list({task["task_id"]: task for task in [*tasks, *public]}.values())
        tasks.sort(key=lambda task: not task["baseline"])
    return tasks, matrix_metadata(tasks)


def capability_summary(tasks):
    """Per EP size / operator / case status counts without running anything."""
    summary = {}
    for task in tasks:
        size = task["ep_size"]
        entry = summary.setdefault(str(size), {"tasks": 0, "statuses": {},
                                               "operators": {}, "cases": {}})
        entry["tasks"] += 1
        entry["statuses"][task["status"]] = entry["statuses"].get(task["status"], 0) + 1
        operator = entry["operators"].setdefault(task["backend_id"], {})
        operator[task["status"]] = operator.get(task["status"], 0) + 1
        cases = entry["cases"].setdefault(task["case_id"], {})
        cases[task["backend_id"]] = task["status"]
    return summary


def print_capabilities(args, tasks, as_json):
    summary = capability_summary(tasks)
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    cases = sorted({task["case_id"] for task in tasks})
    operators = sorted({task["backend_id"] for task in tasks})
    print(f"EP sizes: {', '.join(summary)}  cases: {len(cases)}  operators: {len(operators)}")
    header = "operator".ljust(34) + "".join(size.rjust(12) for size in summary)
    print(header)
    for operator in operators:
        row = operator.ljust(34)
        for size in summary:
            statuses = summary[size]["operators"].get(operator)
            row += ("-" if not statuses else str(sum(statuses.values()))).rjust(12)
        print(row)
    print("\nstatus totals")
    for size, entry in summary.items():
        counts = ", ".join(f"{name}={count}" for name, count in
                           sorted(entry["statuses"].items(), key=lambda item: _severity(item[0])))
        print(f"EP{size}: {entry['tasks']} tasks ({counts})")


def _severity(status):
    return SEVERITY.index(status) if status in SEVERITY else len(SEVERITY)


def plan_summary(tasks, matrix):
    sizes = matrix.get("selected_ep_sizes") or matrix.get("ep_sizes") or []
    print(f"plan: {matrix.get('name')}  mode: {matrix.get('mode', 'catalog')}  "
          f"EP sizes: {', '.join(map(str, sizes))}")
    print(f"suites ({len(matrix.get('suites') or [])}): {', '.join(matrix.get('suites') or [])}")
    print(f"cases: {len({task['case_id'] for task in tasks})}  tasks: {len(tasks)}")
    for size in sizes:
        subset = [task for task in tasks if task["ep_size"] == size]
        counts = {}
        for task in subset:
            counts[task["status"]] = counts.get(task["status"], 0) + 1
        levels = {task["level"] for task in subset}
        detail = ", ".join(f"{name}={count}" for name, count in
                           sorted(counts.items(), key=lambda item: _severity(item[0])))
        suites = {suite for task in subset for suite in task.get("suite_ids", ())}
        print(f"  EP{size}: {len(subset)} tasks, {len(levels)} levels, {len(suites)} suites "
              f"({detail})")
    known = matrix.get("catalog") or {}
    aliases = matrix.get("case_aliases") or {}
    if known:
        print(f"catalog: {len(known)} cases ({len(aliases)} legacy aliases resolved)")
    topology = matrix.get("topology") or {}
    for size, record in sorted(topology.items(), key=lambda item: str(item[0])):
        print(f"  topology EP{size}: source={record.get('source')} "
              f"verified={record.get('verified')} servers={record.get('servers')} "
              f"supernodes={record.get('supernodes')}")


def run_command(args):
    if args.command == "list":
        for spec in registry().values():
            entry = {"id": spec.id, "status": spec.status}
            if args.capabilities:
                entry.update(capabilities={key: key in spec.capabilities for key in CAPABILITIES},
                             constraints=spec.constraints, source=spec.source, commit=spec.commit,
                             native_api=spec.api, dependencies=spec.dependencies, reason=spec.reason)
            print(json.dumps(entry, ensure_ascii=False))
        return 0
    if args.command == "doctor":
        return doctor(args)
    if args.command == "capabilities":
        if args.plan:
            tasks, _ = build_tasks(args)
            print_capabilities(args, tasks, args.json)
            return 0
        from baselines.l0.topology import discovery
        cfg = load_configs(args.config)[0] if args.config else None
        matrix = discovery.load(cfg, override_path=args.matrix)
        if getattr(args, "runtime", False):
            # The staged probe answers "is this channel usable *now*", which a
            # compiled-in channel does not.  It is opt-in because it opens a bridge
            # session and can be slow on a busy machine.
            from baselines.l0.topology import runtime_probe
            descriptor = next((probe.get("detail") for probe in matrix.get("probes", ())
                               if probe.get("name") == "transport_bridge" and probe.get("ok")), {})
            matrix = {**matrix, "runtime": runtime_probe.as_dict(
                runtime_probe.probe(cfg or {}, descriptor=descriptor or {},
                                    device=matrix.get("device")))}
        print(discovery.dump(matrix) if args.json else discovery.describe(matrix))
        return 0
    if args.command == "_worker":
        from common.runner import execute_worker
        return execute_worker(args.manifest, args.task_id, args.directory, args.profile_only)
    if args.command == "import-ascendep":
        from analysis.import_ascendep_results import import_results
        write_json(args.destination, import_results(args.source, args.source_commit))
        return 0
    if hasattr(args, "nnodes"):
        if args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes or (
            args.nproc_per_node is not None and args.nproc_per_node < 1
        ):
            raise ValueError("invalid torchrun node configuration")
        if args.nnodes > 1 and not (args.run_id and args.session and args.master_port):
            raise ValueError("multi-node launch requires shared --run-id, --session and --master-port")
        if args.session:
            checked_id(args.session)
    if args.command in {"plan", "run", "baseline", "pipeline"}:
        tasks, matrix = build_tasks(args)
        if not tasks:
            raise ValueError("configuration selected no benchmark cases")
        execution = matrix.get("execution") or {}
        if execution.get("fail_fast") or not execution.get("continue_on_error", True):
            args.fail_fast = True
        if args.command != "plan" and args.node_rank:
            root = args.output.resolve() / checked_id(args.run_id)
            manifest = wait_for(root / "manifest.json", 180)
            if manifest["tasks"] != tasks:
                raise ValueError("coordinator plan differs from this node's configuration")
        else:
            root, manifest = create_manifest(args.output, tasks, args.run_id, matrix=matrix)
        print(f"{root} ({len(tasks)} tasks)", flush=True)
        if args.command == "plan":
            if args.plan:
                plan_summary(tasks, matrix)
            return 0
    else:
        root = args.output.resolve() / checked_id(args.run_id)
        manifest = load_json(root / "manifest.json")
    if args.command in {"baseline", "run", "resume", "profile", "pipeline"}:
        status = execute(root, manifest, args, profile=args.command == "profile", resume=args.command == "resume")
        if args.command == "run":
            from reporting.coverage import write as write_coverage
            coverage = write_coverage(root, manifest, list(results(root, manifest)))
            print(f"{coverage.get('coverage', coverage['json'])}", flush=True)
        if args.command == "pipeline" and any(
            task["config"]["device"] == "npu" and task["config"]["profiling"].get("enabled", False)
            for task in manifest["tasks"]
        ):
            status |= execute(root, manifest, args, profile=True, representative=True)
        if args.command != "pipeline" or args.node_rank != 0:
            return status
    else:
        status = 0
    from analysis.generate_report import analyze_run
    rows = list(results(root, manifest))
    comparison_rows = []
    for run_id in getattr(args, "compare_run", []):
        other = args.output.resolve() / checked_id(run_id)
        if other != root:
            comparison_rows.extend(results(other, load_json(other / "manifest.json")))
    analysis = analyze_run(root, rows, comparison_rows)
    if args.command in {"report", "pipeline"}:
        from reporting.report import report
        print(report(root, manifest, rows, analysis))
    return status


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--list-backends"]:
        argv[0] = "list"
    elif argv[:1] == ["--config"]:
        argv.insert(0, "run")
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    try:
        return run_command(parser.parse_args(argv))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130


def single_operator(operator, argv=None):
    """Entry point for ``python -m operators.<id>.benchmark``.

    L0 is a shared hardware baseline and is not part of a single-operator run, so
    the operator entry point defaults to L1-L4 unless the caller selects levels.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--levels" not in args:
        args += ["--levels", "L1,L2,L3,L4"]
    return main(["run", *args, "--operators", operator])


if __name__ == "__main__":
    raise SystemExit(main())
