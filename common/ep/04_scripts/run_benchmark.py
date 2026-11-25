"""Plan, run, profile, analyze and report expert-parallel benchmarks."""
import argparse
import json
from pathlib import Path
import sys

from common.config import LEVELS, load_configs
from common.controller import DEFAULT_OUTPUT, checked_id, create_manifest, execute, results, wait_for
from common.registry import CAPABILITIES, registry
from common.scheduler import make_plan
from common.storage import load_json, write_json


def csv_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def add_launch(parser):
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--session", help="unique ID shared by all node controllers for this invocation")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--capabilities", action="store_true")
    commands.add_parser("doctor")
    for name in ("plan", "run", "baseline", "pipeline"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--operators", type=csv_values)
        command.add_argument("--levels", type=csv_values)
        command.add_argument("--cases", type=csv_values)
        command.add_argument("--ep-size", type=int, help="select one point of an EP scaling matrix")
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--run-id")
        if name != "plan":
            add_launch(command)
    for name in ("resume", "profile", "analyze", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if name in ("resume", "profile"):
            add_launch(command)
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


def doctor():
    import importlib.util
    import os
    from common.environment import capture
    from operators.ascendep import source_checkout
    from common.backend import BenchmarkUnavailable

    env = capture(probe_npu=True)
    env["dependencies"] = {name: importlib.util.find_spec(name) is not None
                           for name in ("torch", "torch_npu", "shmem", "deep_ep", "npu_patch",
                                        "cann_ops_transformer", "megatron", "fsdp_turbo")}
    try:
        import torch
        env["gloo_available"] = torch.distributed.is_gloo_available()
        if env["dependencies"]["torch_npu"]:
            import torch_npu  # noqa: F401
            env["npu_devices"] = torch.npu.device_count()
            env["npu_models"] = [torch.npu.get_device_name(i) for i in range(env["npu_devices"])]
            env["hccl_backend"] = torch.distributed.is_backend_available("hccl")
            if hasattr(torch.npu, "get_device_properties"):
                env["npu_total_hbm_bytes"] = [
                    getattr(torch.npu.get_device_properties(i), "total_memory", None)
                    for i in range(env["npu_devices"])
                ]
    except (ImportError, OSError, RuntimeError) as exc:
        env["torch_runtime_error"] = str(exc)
    try:
        env["ascendep_checkout"] = str(source_checkout(os.getenv("EP_ASCENDEP_ROOT")))
    except BenchmarkUnavailable as exc:
        env["ascendep_checkout"] = {"status": exc.status, "reason": str(exc)}
    print(json.dumps(env, indent=2))


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
        doctor()
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
        configs = load_configs(args.config)
        if args.ep_size is not None:
            configs = [cfg for cfg in configs if cfg["distributed"]["ep_size"] == args.ep_size]
        if args.command != "plan" and args.nnodes > 1 and len({
            (cfg["distributed"]["ep_size"], cfg["distributed"]["tp_size"]) for cfg in configs
        }) > 1:
            raise ValueError("multi-node scaling runs require --ep-size to select one point")
        if args.levels and set(args.levels) - set(LEVELS):
            raise ValueError("expected benchmark levels L0 through L4")
        tasks = make_plan(configs, args.operators, args.levels, args.cases, args.command == "baseline")
        if not tasks:
            raise ValueError("configuration selected no benchmark cases")
        if args.command != "plan" and args.node_rank:
            root = args.output.resolve() / checked_id(args.run_id)
            manifest = wait_for(root / "manifest.json", 180)
            if manifest["tasks"] != tasks:
                raise ValueError("coordinator plan differs from this node's configuration")
        else:
            root, manifest = create_manifest(args.output, tasks, args.run_id)
        print(f"{root} ({len(tasks)} tasks)", flush=True)
        if args.command == "plan":
            return 0
    else:
        root = args.output.resolve() / checked_id(args.run_id)
        manifest = load_json(root / "manifest.json")
    if args.command in {"baseline", "run", "resume", "profile", "pipeline"}:
        status = execute(root, manifest, args, profile=args.command == "profile", resume=args.command == "resume")
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
    args = list(sys.argv[1:] if argv is None else argv)
    return main(["run", *args, "--operators", operator])


if __name__ == "__main__":
    raise SystemExit(main())
