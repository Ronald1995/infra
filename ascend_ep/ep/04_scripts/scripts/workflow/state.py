"""Workflow state machine and its on-disk record.

The state is the reason a run can be resumed, audited and trusted:

* every phase has exactly one status from :data:`PHASE_STATUSES`;
* a phase is written to ``state.json`` as soon as it finishes, so a crash loses at most
  the phase that was running;
* every command a phase executed is recorded with its exit code and log path;
* the request hash, the plan hash and the repository commit are stored, because a
  workflow may only be resumed when the thing being measured has not changed.

``events.jsonl`` is the append-only history (one JSON object per line). It is never
rewritten, so a resume can be distinguished from a first run by reading it.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from common.storage import write_json

SCHEMA_VERSION = 1

PHASES = (
    "request_resolution",
    "environment_discovery",
    "dependency_check",
    "dependency_install",
    "capability_probe",
    "kernel_build",
    "plan_generation",
    "resource_estimation",
    "benchmark_execution",
    "result_validation",
    "profiling",
    "analysis",
    "report_generation",
    "final_validation",
)

#: Phase statuses.  ``partial`` means the phase produced usable output but did not
#: satisfy its strongest gate (for example one EP size was blocked); ``skipped`` always
#: carries a reason.
PHASE_STATUSES = ("pending", "running", "passed", "partial", "blocked", "failed", "skipped")
WORKFLOW_STATUSES = ("pending", "running", "passed", "partial", "blocked", "failed")
#: Statuses that count as "this phase delivered what it could".  ``partial`` belongs here:
#: a resumed workflow must not repeat an environment probe or a plan because one EP size
#: was blocked.
DONE_PHASE_STATUSES = ("passed", "partial", "skipped")
#: Phases whose ``partial`` state can still improve when the workflow is resumed (an
#: interrupted measurement, a profile worth retrying).  They are the only ones a resume
#: re-attempts.
PROGRESSABLE_PHASES = ("benchmark_execution", "result_validation", "profiling")
#: Statuses that stop the workflow (the phase cannot be skipped).
STOPPING_STATUSES = ("blocked", "failed")


def should_run(record):
    """Whether ``record`` needs another attempt.

    Completed phases are never repeated; a ``partial`` measurement phase is retried
    because resuming it can make it complete, while a ``partial`` infrastructure phase
    (a blocked EP size, a missing host dependency) is final.
    """
    if record.status in ("passed", "skipped"):
        return False
    if record.status == "partial":
        return record.name in PROGRESSABLE_PHASES
    return True


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PhaseRecord:
    """One phase's outcome, with the evidence that produced it."""

    name: str
    status: str = "pending"
    reason: str | None = None
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    artifacts: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in PHASE_STATUSES:
            raise ValueError(f"invalid phase status {self.status!r}")

    @property
    def done(self):
        """Whether the phase reached a status a resume does not need to revisit."""
        return self.status in DONE_PHASE_STATUSES

    @property
    def retryable(self):
        """Whether a resume should attempt this phase again."""
        return should_run(self)

    def as_dict(self):
        return {"name": self.name, "status": self.status, "reason": self.reason,
                "attempts": self.attempts, "started_at": self.started_at,
                "finished_at": self.finished_at, "artifacts": list(self.artifacts),
                "commands": [dict(command) for command in self.commands],
                "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, data):
        return cls(name=data["name"], status=data.get("status", "pending"),
                   reason=data.get("reason"), attempts=int(data.get("attempts", 0)),
                   started_at=data.get("started_at"), finished_at=data.get("finished_at"),
                   artifacts=list(data.get("artifacts") or ()),
                   commands=list(data.get("commands") or ()),
                   detail=dict(data.get("detail") or {}))


@dataclass
class WorkflowState:
    """The durable state of one workflow."""

    workflow_id: str
    status: str = "pending"
    current_phase: str | None = None
    run_id: str | None = None
    repo_commit: str | None = None
    request_hash: str | None = None
    plan_hash: str | None = None
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    last_error: str | None = None
    phases: dict = field(default_factory=dict)
    request: dict | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not self.phases:
            self.phases = {name: PhaseRecord(name=name) for name in PHASES}
        if self.status not in WORKFLOW_STATUSES:
            raise ValueError(f"invalid workflow status {self.status!r}")
        for name, record in list(self.phases.items()):
            if isinstance(record, dict):
                self.phases[name] = PhaseRecord.from_dict(record)

    # -- queries ----------------------------------------------------------
    @property
    def phase(self):
        return self.phases.get(self.current_phase) if self.current_phase else None

    def completed(self):
        return [name for name in PHASES if self.phases[name].status in DONE_PHASE_STATUSES]

    def blocked_phases(self):
        return [name for name in PHASES if self.phases[name].status == "blocked"]

    def executable_phases(self):
        """Phases that still need a run, in order."""
        return [name for name in PHASES if should_run(self.phases[name])]

    def resume_point(self):
        """The first phase that still needs to run."""
        remaining = self.executable_phases()
        return remaining[0] if remaining else None

    # -- transitions ------------------------------------------------------
    def start_phase(self, name):
        record = self.phases[name]
        record.status = "running"
        record.attempts += 1
        record.started_at = now()
        record.reason = None
        self.current_phase = name
        self.status = "running"
        self.updated_at = now()

    def finish_phase(self, name, status, reason=None, artifacts=(), detail=None):
        if status not in PHASE_STATUSES:
            raise ValueError(f"invalid phase status {status!r}")
        record = self.phases[name]
        record.status = status
        record.reason = reason
        record.finished_at = now()
        record.artifacts = [str(item) for item in artifacts]
        if detail:
            record.detail.update(detail)
        if status in STOPPING_STATUSES:
            self.status = status
            self.last_error = reason
        self.updated_at = now()

    def skip_phase(self, name, reason):
        record = self.phases[name]
        if not reason:
            raise ValueError("a skipped phase must state why")
        record.status = "skipped"
        record.reason = reason
        record.finished_at = now()
        self.updated_at = now()

    def invalidate_from(self, name, reason):
        """Mark ``name`` and every later phase as pending again, with the reason kept.

        Used when the repository commit, the request or the plan changed: the earlier
        *evidence* is still true of the old inputs, but the results may not be reused.
        """
        if name not in PHASES:
            raise ValueError(f"unknown phase {name!r}")
        for phase in PHASES[PHASES.index(name):]:
            record = self.phases[phase]
            if record.status in ("pending",):
                continue
            record.status = "pending"
            record.reason = f"invalidated: {reason}"
            record.finished_at = None
        self.updated_at = now()

    def add_command(self, name, argv, exit_code, log_path=None, detail=None):
        record = self.phases[name]
        record.commands.append({"argv": [str(item) for item in argv],
                                "exit_code": exit_code, "log": _relative_text(log_path),
                                "at": now(), "detail": dict(detail or {})})
        self.updated_at = now()

    # -- serialisation ----------------------------------------------------
    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "current_phase": self.current_phase,
            "completed_phases": self.completed(),
            "blocked_phases": self.blocked_phases(),
            "run_id": self.run_id,
            "repo_commit": self.repo_commit,
            "request_hash": self.request_hash,
            "plan_hash": self.plan_hash,
            "started_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
            "phases": {name: self.phases[name].as_dict() for name in PHASES},
            "request": self.request,
        }

    @classmethod
    def from_dict(cls, data):
        phases = {name: PhaseRecord.from_dict(item)
                  for name, item in (data.get("phases") or {}).items()}
        state = cls(workflow_id=data["workflow_id"], status=data.get("status", "pending"),
                    current_phase=data.get("current_phase"), run_id=data.get("run_id"),
                    repo_commit=data.get("repo_commit"),
                    request_hash=data.get("request_hash"), plan_hash=data.get("plan_hash"),
                    created_at=data.get("started_at") or now(),
                    updated_at=data.get("updated_at") or now(),
                    last_error=data.get("last_error"), phases=phases,
                    request=data.get("request"),
                    schema_version=int(data.get("schema_version", SCHEMA_VERSION)))
        return state


def _relative_text(path):
    return None if path is None else str(path)


class WorkflowStore:
    """Create, load, save and find workflows under ``05_results/workflows``."""

    def __init__(self, output_root, root=None):
        self.output_root = Path(output_root)
        self.root = Path(root) if root is not None else None

    # -- paths ------------------------------------------------------------
    def directory(self, workflow_id):
        return self.output_root / workflow_id

    def _state_path(self, workflow_id):
        return self.directory(workflow_id) / "state.json"

    def _events_path(self, workflow_id):
        return self.directory(workflow_id) / "events.jsonl"

    # -- lifecycle --------------------------------------------------------
    def new_id(self, intent="full"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"ep-{intent.replace('_', '-')}-{stamp}-{uuid4().hex[:4]}"

    def create(self, request, workflow_id=None, repo_commit=None):
        workflow_id = workflow_id or self.new_id(request.intent)
        directory = self.directory(workflow_id)
        if directory.exists():
            raise FileExistsError(f"workflow {workflow_id} already exists at {directory}")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "logs").mkdir(exist_ok=True)
        state = WorkflowState(workflow_id=workflow_id, run_id=request.run_id or workflow_id,
                              repo_commit=repo_commit, request_hash=request.hash,
                              request=request.as_dict())
        self.save(state)
        self.event(state, event="created", intent=request.intent,
                   request_hash=request.hash)
        write_json(directory / "request.json", request.as_dict())
        return state

    def load(self, workflow_id):
        path = self._state_path(workflow_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"no workflow state at {path}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"workflow state {path} is corrupt: {exc}") from exc
        return WorkflowState.from_dict(data)

    def save(self, state):
        state.updated_at = now()
        write_json(self._state_path(state.workflow_id), state.as_dict())

    def event(self, state, event, **detail):
        record = {"at": now(), "workflow_id": state.workflow_id, "event": event,
                  "phase": state.current_phase, **detail}
        path = self._events_path(state.workflow_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    # -- discovery --------------------------------------------------------
    def ids(self):
        if not self.output_root.is_dir():
            return []
        return sorted(path.name for path in self.output_root.iterdir()
                      if (path / "state.json").is_file())

    def latest(self, unfinished=False):
        """The most recently updated workflow, optionally the latest unfinished one."""
        candidates = []
        for workflow_id in self.ids():
            try:
                state = self.load(workflow_id)
            except (ValueError, FileNotFoundError):
                continue
            if unfinished and state.status in ("passed",):
                continue
            candidates.append((state.updated_at, workflow_id, state))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[-1][2]

    def events(self, workflow_id):
        path = self._events_path(workflow_id)
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]


def needs_revalidation(state, request, repo_commit=None):
    """Which earlier phase must be redone, or ``None`` when the inputs still match.

    A new request invalidates everything that depends on it; a new repository commit
    invalidates the plan and the measurement (the code that ran differs), but not the
    environment probes; a new plan hash invalidates the measurement only.
    """
    if state.request_hash and request.hash != state.request_hash:
        return "plan_generation"
    if repo_commit and state.repo_commit and repo_commit != state.repo_commit:
        return "plan_generation"
    return None
