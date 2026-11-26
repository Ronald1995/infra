"""Autonomous benchmark workflow: one request in, one audited run out.

This program is the executable half of the ``ep-benchmark`` skill. It does not implement a
benchmark; it runs the documented CLI commands in the documented order, records every
phase, and stops when a gate fails instead of pretending the run succeeded.

```sh
# interpret a prompt (the skill calls this, or writes request.json itself)
python scripts/benchmark_workflow.py interpret "在 Ascend 950 上运行完整 EP Benchmark，EP8、EP64、EP128"

# plan + estimate only
python scripts/benchmark_workflow.py plan --request request.json

# the full flow
python scripts/benchmark_workflow.py execute --request request.json
python scripts/benchmark_workflow.py resume --workflow-id <id>
python scripts/benchmark_workflow.py status --workflow-id <id>
```

Exit code: 0 when the workflow is ``passed`` or ``partial`` (a partial delivery is a real
delivery: it reports what failed), 1 when it is ``blocked`` or ``failed``, 2 on a usage
error.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):  # allow `python scripts/benchmark_workflow.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.storage import write_json                      # noqa: E402
from scripts.workflow import handlers as build_handlers    # noqa: E402
from scripts.workflow import phases as phase_runner        # noqa: E402
from scripts.workflow import request as request_module     # noqa: E402
from scripts.workflow import state as state_module         # noqa: E402

DEFAULT_OUTPUT = "05_results/workflows"


def _repo_root():
    from common.registry import ROOT
    return ROOT


def _git_commit(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _output_root(root, override=None):
    base = Path(override) if override else Path(root).parent / DEFAULT_OUTPUT
    return base.resolve() if Path(base).is_absolute() else (Path(root).parent / base).resolve()


def _store(root, output):
    return state_module.WorkflowStore(_output_root(root, output), root=root)


def _context(state, request, store, root, run_output=None):
    return phase_runner.WorkflowContext(
        state=state, request=request, store=store,
        commands=phase_runner.CommandRunner(root, store.directory(state.workflow_id) / "logs"),
        root=root, output=store.directory(state.workflow_id),
        run_output=run_output)


def _prepare(args):
    """Resolve the request, create or load the workflow, apply revalidation."""
    root = _repo_root()
    if getattr(args, "request", None):
        request = request_module.load(args.request, root=root)
    else:
        request = request_module.interpret(args.prompt or "", root=root)
    store = _store(root, getattr(args, "output", None))
    repo_commit = _git_commit(root)
    if getattr(args, "workflow_id", None):
        try:
            state = store.load(args.workflow_id)
        except FileNotFoundError:
            # A named id that does not exist yet starts a new workflow under that name,
            # which keeps a scripted invocation deterministic.
            state = store.create(request, workflow_id=args.workflow_id,
                                 repo_commit=repo_commit)
        else:
            changed = state_module.needs_revalidation(state, request, repo_commit)
            if changed:
                state.invalidate_from(changed, "the request or the repository commit changed")
                store.event(state, event="revalidated", from_phase=changed,
                            request_hash=request.hash)
                store.save(state)
            else:
                store.event(state, event="resumed", request_hash=request.hash)
    else:
        state = store.create(request, repo_commit=repo_commit)
    _apply_run_id(state, getattr(args, "run_id", None))
    return root, request, store, state


def _apply_run_id(state, run_id):
    """A run id may be chosen before the matrix runs, never after."""
    if not run_id:
        return
    if state.phases["benchmark_execution"].status not in ("pending",):
        raise ValueError("the run id cannot change after the benchmark has started")
    state.run_id = run_id


def _summary(state):
    lines = [f"workflow : {state.workflow_id}",
             f"status   : {state.status}",
             f"run id   : {state.run_id}",
             f"phases   : {len(state.completed())}/{len(state_module.PHASES)} done"]
    for name in state_module.PHASES:
        record = state.phases[name]
        reason = f" — {record.reason}" if record.reason else ""
        lines.append(f"  {name:<22} {record.status}{reason}")
    pending = [name for name in state_module.PHASES
               if state.phases[name].status == "pending"]
    if pending and state.status in ("blocked", "failed"):
        lines.append(f"not reached: {len(pending)} phase(s) — fix the blocker, then "
                     f"`resume --workflow-id {state.workflow_id}`")
    return "\n".join(lines)


def command_interpret(args):
    root = _repo_root()
    request = request_module.interpret(args.prompt, root=root)
    payload = request.as_dict()
    if args.out:
        write_json(Path(args.out).resolve(), payload)
        print(f"request written to {Path(args.out).resolve()}")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def command_status(args):
    root = _repo_root()
    store = _store(root, getattr(args, "output", None))
    workflow_id = args.workflow_id
    if not workflow_id:
        latest = store.latest(unfinished=args.unfinished)
        if latest is None:
            print("no workflow has been created", file=sys.stderr)
            return 1
        workflow_id = latest.workflow_id
    state = store.load(workflow_id)
    if args.json:
        print(json.dumps(state.as_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(_summary(state))
    return 0


def _execute(args, intent_override=None):
    root, request, store, state = _prepare(args)
    if intent_override and request.intent != intent_override:
        request.intent = intent_override
        request_module.resolve(request, root=root)
        state.request = request.as_dict()
        state.request_hash = request.hash
        store.save(state)
    context = _context(state, request, store, root)
    final = phase_runner.execute_workflow(context, build_handlers())
    print(_summary(final))
    print(f"artifacts: {store.directory(final.workflow_id)}")
    if final.status in ("passed", "partial"):
        return 0
    return 1


def command_execute(args):
    return _execute(args)


def command_plan(args):
    return _execute(args, intent_override="plan_only")


def command_estimate(args):
    return _execute(args, intent_override="estimate")


def command_resume(args):
    root = _repo_root()
    store = _store(root, getattr(args, "output", None))
    workflow_id = args.workflow_id
    if not workflow_id:
        latest = store.latest(unfinished=True)
        if latest is None:
            print("no unfinished workflow to resume", file=sys.stderr)
            return 1
        workflow_id = latest.workflow_id
    state = store.load(workflow_id)
    request = request_module.build(state.request or {}, root=root)
    repo_commit = _git_commit(root)
    changed = state_module.needs_revalidation(state, request, repo_commit)
    if changed:
        state.invalidate_from(changed, "the request or the repository commit changed")
    store.event(state, event="resumed", workflow_id=workflow_id,
                from_phase=state.resume_point())
    store.save(state)
    context = _context(state, request, store, root)
    final = phase_runner.execute_workflow(context, build_handlers())
    print(_summary(final))
    print(f"artifacts: {store.directory(final.workflow_id)}")
    return 0 if final.status in ("passed", "partial") else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    interpret = commands.add_parser("interpret", help="turn a prompt into a request.json")
    interpret.add_argument("prompt")
    interpret.add_argument("--out", help="write the resolved request here")
    interpret.set_defaults(func=command_interpret)

    for name, function, help_text in (
            ("execute", command_execute, "run every phase the intent selects"),
            ("plan", command_plan, "plan and estimate only"),
            ("estimate", command_estimate, "estimate only")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--request", type=Path, help="a request.json")
        command.add_argument("--prompt", help="a natural-language instruction")
        command.add_argument("--workflow-id", help="resume this workflow id")
        command.add_argument("--run-id", help="use this benchmark run id")
        command.add_argument("--output", help=f"workflow root (default {DEFAULT_OUTPUT})")
        command.set_defaults(func=function)

    resume = commands.add_parser("resume", help="continue the last unfinished workflow")
    resume.add_argument("--workflow-id")
    resume.add_argument("--output")
    resume.set_defaults(func=command_resume)

    status = commands.add_parser("status", help="print one workflow's state")
    status.add_argument("--workflow-id")
    status.add_argument("--unfinished", action="store_true")
    status.add_argument("--output")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=command_status)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except request_module.RequestError as exc:
        print(f"request error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"workflow error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
