"""Phase table, command runner and the workflow loop.

There is no benchmark logic in this module: it decides *which* documented command runs,
records what happened, and applies the acceptance gate. The measurement itself stays in
``run_benchmark.py`` and the packages it already had.

Every phase is described by the same record, so a report can answer "what did this phase
need, what did it run, what did it produce, and why did it stop" without reading code:

* ``requires``  -- phases whose evidence this phase consumes,
* ``gate``      -- the acceptance condition, in words, mirrored by the handler,
* ``retryable`` -- whether a failed attempt may be repeated,
* ``skippable`` -- whether the workflow may continue without it (always with a reason).
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.workflow.state import (
    PHASES, STOPPING_STATUSES, DONE_PHASE_STATUSES, WorkflowState, should_run,
)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    title: str
    requires: tuple
    gate: str
    retryable: bool
    skippable: bool
    skip_reason: str | None = None


PHASE_TABLE = {
    "request_resolution": PhaseSpec(
        "request_resolution", "Resolve the natural-language request",
        (), "intent, scope, safety policy and every default resolved and recorded",
        False, False),
    "environment_discovery": PhaseSpec(
        "environment_discovery", "Discover the environment",
        ("request_resolution",),
        "a doctor report exists and names python, CANN, SoC, devices and the toolchain",
        True, False),
    "dependency_check": PhaseSpec(
        "dependency_check", "Check dependencies",
        ("environment_discovery",),
        "every dependency of the requested scope is classified (present / installable / "
        "host-provided / blocked)", True, False),
    "dependency_install": PhaseSpec(
        "dependency_install", "Install the permitted dependencies",
        ("dependency_check",),
        "either nothing installable was missing, or the installer ran and doctor was "
        "re-run afterwards", True, True,
        "the request did not authorise installing dependencies"),
    "capability_probe": PhaseSpec(
        "capability_probe", "Probe the static and runtime capabilities",
        ("environment_discovery",),
        "for every requested EP size the operator/case support and the runtime channel "
        "status are recorded", True, False),
    "kernel_build": PhaseSpec(
        "kernel_build", "Build the required kernels",
        ("capability_probe",),
        "each kernel engine is built (with its ABI manifest) or explicitly blocked",
        True, True,
        "no kernel source is present or the request did not ask for a build"),
    "plan_generation": PhaseSpec(
        "plan_generation", "Generate and validate the benchmark plan",
        ("capability_probe",),
        "the manifest covers every requested EP size / level / operator, and every "
        "non-executable task carries a reason", True, False),
    "resource_estimation": PhaseSpec(
        "resource_estimation", "Estimate runtime and storage",
        ("plan_generation",),
        "task counts, iteration counts, a runtime estimate with its basis and a storage "
        "estimate are recorded", False, True,
        "no estimate was requested and no historical timing exists"),
    "benchmark_execution": PhaseSpec(
        "benchmark_execution", "Execute the benchmark matrix",
        ("plan_generation", "resource_estimation"),
        "one run directory exists, every planned task is terminal, and the bucket sum "
        "equals planned_tasks", True, False),
    "result_validation": PhaseSpec(
        "result_validation", "Validate correctness and accounting",
        ("benchmark_execution",),
        "every passed task has a correctness record, no passed task lacks evidence and "
        "no failure was retried blindly", True, False),
    "profiling": PhaseSpec(
        "profiling", "Collect profiling evidence",
        ("result_validation",),
        "the selected representative tasks were profiled, or the request disabled "
        "profiling (recorded as skipped)", True, True,
        "the request disabled profiling or no task passed"),
    "analysis": PhaseSpec(
        "analysis", "Analyze the results",
        ("result_validation",),
        "analysis/results.json exists and was produced from this run's records",
        True, False),
    "report_generation": PhaseSpec(
        "report_generation", "Generate the reports",
        ("analysis",),
        "benchmark_report.md and the coverage products exist for this run", True, False),
    "final_validation": PhaseSpec(
        "final_validation", "Validate the delivery",
        ("report_generation",),
        "status buckets sum to planned_tasks, every blocked/failed task is explained, "
        "and every report link resolves", False, False),
}

#: Phases whose output is a *measurement*.  A changed request or commit invalidates them.
MEASUREMENT_PHASES = ("benchmark_execution", "result_validation", "profiling", "analysis",
                      "report_generation", "final_validation")

#: Which phases each intent runs.  Everything not listed is ``skipped`` with the reason
#: from :func:`skip_reason`.
INTENT_PHASES = {
    "full_benchmark": PHASES,
    "partial_benchmark": PHASES,
    "smoke_test": PHASES,
    "environment_check": ("request_resolution", "environment_discovery", "dependency_check",
                          "capability_probe"),
    "plan_only": ("request_resolution", "environment_discovery", "dependency_check",
                  "capability_probe", "plan_generation", "resource_estimation"),
    "estimate": ("request_resolution", "environment_discovery", "dependency_check",
                 "plan_generation", "resource_estimation"),
    "resume": ("request_resolution", "environment_discovery", "plan_generation",
               "benchmark_execution", "result_validation", "profiling", "analysis",
               "report_generation", "final_validation"),
    "profile": ("request_resolution", "profiling", "analysis", "report_generation",
                "final_validation"),
    "analyze": ("request_resolution", "analysis", "report_generation", "final_validation"),
    "report": ("request_resolution", "report_generation", "final_validation"),
}

_INTENT_SKIP = {
    "environment_check": "the request only asked for an environment check",
    "plan_only": "the request asked for a plan and an estimate only",
    "estimate": "the request asked for an estimate only",
    "resume": "the resumed phases are run after the existing plan is revalidated",
    "profile": "the request only asked for profiling of an existing run",
    "analyze": "the request only asked for analysis of an existing run",
    "report": "the request only asked for the report of an existing run",
}


def skip_reason(intent, phase):
    if phase in INTENT_PHASES.get(intent, PHASES):
        return None
    base = _INTENT_SKIP.get(intent, f"{intent} does not run this phase")
    return f"{base} (phase `{phase}` is not part of the {intent} path)"


def gate_for(phase):
    return PHASE_TABLE[phase].gate


# ------------------------------------------------------------------ outcome --

@dataclass
class PhaseOutcome:
    """What one phase produced."""

    status: str
    reason: str | None = None
    artifacts: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("passed", "partial", "blocked", "failed", "skipped"):
            raise ValueError(f"invalid outcome status {self.status!r}")
        if self.status == "skipped" and not self.reason:
            raise ValueError("a skipped phase must state why")

    @classmethod
    def passed(cls, artifacts=(), detail=None):
        return cls("passed", None, list(artifacts), dict(detail or {}))

    @classmethod
    def partial(cls, reason, artifacts=(), detail=None):
        return cls("partial", reason, list(artifacts), dict(detail or {}))

    @classmethod
    def blocked(cls, reason, artifacts=(), detail=None):
        return cls("blocked", reason, list(artifacts), dict(detail or {}))

    @classmethod
    def failed(cls, reason, artifacts=(), detail=None):
        return cls("failed", reason, list(artifacts), dict(detail or {}))

    @classmethod
    def skipped(cls, reason):
        return cls("skipped", reason)


# ------------------------------------------------------------------ command --

@dataclass
class CommandResult:
    """One executed command: argv, exit code, output and the log file."""

    argv: list
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    log_path: Path | None = None
    timed_out: bool = False

    @property
    def ok(self):
        return self.exit_code == 0

    def json(self):
        """The JSON payload on stdout, or ``None`` when there is none."""
        return parse_json(self.stdout)


def parse_json(text):
    """Parse a JSON document from command output, tolerating a text preamble."""
    if not text:
        return None
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    for opener in ("{", "["):
        start = stripped.find(opener)
        while start >= 0:
            candidate = stripped[start:]
            try:
                return json.loads(candidate)
            except ValueError:
                start = stripped.find(opener, start + 1)
    return None


class CommandRunner:
    """Run a documented CLI command and keep its output.

    The runner is the only place a subprocess starts, which is what lets the tests drive
    a complete workflow without a device: they pass a runner whose ``execute`` returns
    canned results.
    """

    def __init__(self, root, log_dir, python=None, env=None, timeout=None):
        self.root = Path(root)
        self.log_dir = Path(log_dir)
        self.python = python or sys.executable
        self.env = dict(env or {})
        self.timeout = timeout
        self._counter = 0

    # -- public -----------------------------------------------------------
    def run(self, argv, log_name=None, timeout=None, cwd=None, env=None):
        argv = [str(item) for item in argv]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / (log_name or self._default_log_name(argv))
        try:
            completed = subprocess.run(
                argv, cwd=str(cwd or self.root), capture_output=True, text=True,
                errors="replace", timeout=timeout or self.timeout,
                env={**os.environ, **self.env, **(env or {})},
            )
            result = CommandResult(argv=argv, exit_code=completed.returncode,
                                   stdout=completed.stdout, stderr=completed.stderr,
                                   log_path=log_path)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            result = CommandResult(argv=argv, exit_code=124, stdout=stdout,
                                   stderr=f"timed out after {timeout or self.timeout}s",
                                   log_path=log_path, timed_out=True)
        except OSError as exc:
            result = CommandResult(argv=argv, exit_code=127, stdout="", stderr=str(exc),
                                   log_path=log_path)
        lines = [f"$ {' '.join(result.argv)}", f"# exit code: {result.exit_code}", "",
                 "## stdout", result.stdout, "", "## stderr", result.stderr, ""]
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return result

    def benchmark(self, *args, **kwargs):
        """Run ``python run_benchmark.py ...`` and keep the exit code and log."""
        return self.run([self.python, str(self.root / "run_benchmark.py"), *args], **kwargs)

    # -- helpers ----------------------------------------------------------
    def _default_log_name(self, argv):
        self._counter += 1
        slug = "-".join(Path(item).name if i == 0 else item
                        for i, item in enumerate(argv[1:4]) if not item.startswith("-"))
        slug = "".join(character if character.isalnum() or character in "-_." else "-"
                       for character in slug)[:60] or "command"
        return f"{self._counter:03d}-{slug}.log"


# ---------------------------------------------------------------- context ----

@dataclass
class WorkflowContext:
    """Everything a phase may read or write."""

    state: WorkflowState
    request: object
    store: object
    commands: CommandRunner
    root: Path
    output: Path
    run_output: Path | None = None
    data: dict = field(default_factory=dict)

    @property
    def run_id(self):
        return self.state.run_id

    @property
    def log_dir(self):
        return self.output / "logs"

    @property
    def run_output_dir(self):
        """Where the benchmark run directory lives (``05_results/raw`` by default)."""
        if self.run_output is not None:
            return Path(self.run_output)
        from common.controller import DEFAULT_OUTPUT
        return Path(DEFAULT_OUTPUT)

    def path(self, name):
        return self.output / name

    def write_json(self, name, value):
        from common.storage import write_json
        path = self.output / name
        write_json(path, value)
        return path

    def record(self, phase, result, **detail):
        """Record one command on the state and on the event history."""
        self.state.add_command(phase, result.argv, result.exit_code, result.log_path,
                               detail=detail)
        self.store.event(self.state, event="command", phase=phase,
                         argv=result.argv, exit_code=result.exit_code,
                         log=str(result.log_path) if result.log_path else None, **detail)
        self.store.save(self.state)


# ------------------------------------------------------------------- loop ----

def execute_workflow(ctx, handlers):
    """Run every phase the intent selects, honouring gates and stop conditions.

    Returns the state. Unexpected exceptions become ``failed`` phases with the traceback
    preserved, never a silent crash: a workflow that dies without a record is the one
    failure mode this loop exists to prevent.
    """
    intent = ctx.request.intent
    selected = INTENT_PHASES.get(intent, PHASES)
    for name in PHASES:
        record = ctx.state.phases[name]
        reason = skip_reason(intent, name)
        if reason:
            if record.status == "pending":
                ctx.state.skip_phase(name, reason)
                ctx.store.event(ctx.state, event="phase_skipped", phase=name, reason=reason)
            continue
        if name not in selected:  # defensive: selected and skip_reason disagree
            continue
        if not should_run(record):
            continue
        ctx.state.start_phase(name)
        ctx.store.save(ctx.state)
        ctx.store.event(ctx.state, event="phase_started", phase=name,
                        attempt=record.attempts, gate=gate_for(name))
        try:
            outcome = handlers[name](ctx)
        except Exception as exc:  # a phase must never take the workflow down silently
            import traceback
            detail = traceback.format_exc()
            outcome = PhaseOutcome.failed(f"{type(exc).__name__}: {exc}",
                                          detail={"traceback": detail})
        if outcome.status == "skipped":
            ctx.state.skip_phase(name, outcome.reason)
        else:
            ctx.state.finish_phase(name, outcome.status, outcome.reason,
                                   outcome.artifacts, outcome.detail)
        ctx.store.event(ctx.state, event="phase_finished", phase=name,
                        status=ctx.state.phases[name].status,
                        reason=ctx.state.phases[name].reason)
        ctx.store.save(ctx.state)
        if outcome.status in STOPPING_STATUSES:
            break
    return finalize(ctx, handlers)


def finalize(ctx, handlers):
    """Derive the workflow status from the phase statuses and persist it."""
    statuses = [ctx.state.phases[name].status for name in PHASES]
    if "failed" in statuses:
        ctx.state.status = "failed"
    elif "blocked" in statuses:
        ctx.state.status = "blocked"
    elif "partial" in statuses:
        ctx.state.status = "partial"
    elif all(status in DONE_PHASE_STATUSES for status in statuses):
        ctx.state.status = "passed"
    elif any(status == "pending" for status in statuses):
        ctx.state.status = "pending" if all(
            status in ("pending", "skipped") for status in statuses) else "running"
    ctx.store.save(ctx.state)
    ctx.store.event(ctx.state, event="workflow_finished", status=ctx.state.status,
                    completed=ctx.state.completed(), blocked=ctx.state.blocked_phases())
    return ctx.state
