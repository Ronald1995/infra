"""Structured benchmark request: the one interface between a prompt and the runner.

A request is what a natural-language instruction becomes after interpretation:

* it names an **intent** (run everything, run a subset, plan only, resume, profile, ...);
* it carries the **scope** that differs from the plan's own defaults;
* it carries the **safety policy** (may dependencies be installed, is there network, is
  sudo allowed) so a workflow never widens its own authority;
* every resolved field records **where the value came from** (``sources``), so a report
  can say "EP sizes came from the plan" instead of implying the user typed them.

Two rules matter:

1. defaults come from the plan file, never from constants in this module. The effective
   suites, EP sizes and levels are read from the plan the request points at.
2. an explicit user value only overrides its own dimension. Asking for "EP64, L1-L2"
   narrows the EP size and the level; it does not silently drop suites or operators.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

from common.storage import fingerprint
from common.registry import ROOT

SCHEMA_VERSION = 1

DEFAULT_PLAN = "configs/benchmark_plan.yaml"
DEFAULT_CONFIG = "configs/smoke_cpu.yaml"
LEVELS = ("L0", "L1", "L2", "L3", "L4")

INTENTS = (
    "full_benchmark", "partial_benchmark", "smoke_test", "plan_only", "estimate",
    "resume", "profile", "analyze", "report", "environment_check",
)
#: Intents that stop before executing the matrix.
PLANNING_INTENTS = ("plan_only", "estimate")
#: Intents that only touch an existing run.
EXISTING_RUN_INTENTS = ("resume", "profile", "analyze", "report")
#: Intents that never touch the network or install anything.
READ_ONLY_INTENTS = ("plan_only", "estimate", "analyze", "report", "environment_check")

DEPENDENCY_POLICIES = ("check_only", "install_missing")
PROFILE_POLICIES = ("none", "representative", "full")
DEVICES = ("npu", "cpu")

#: Default dependency policy per intent.  ``install_missing`` is only the default when
#: the instruction explicitly asks for installation; automation never installs on its
#: own from an ambiguous prompt.
INSTALLING_INTENTS = ("full_benchmark", "partial_benchmark", "smoke_test")


class RequestError(ValueError):
    """Raised when a request cannot be resolved against the repository."""


@dataclass
class Scope:
    ep_sizes: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    #: ``smoke`` / ``core`` / ``full`` / ``exhaustive``; empty means "the plan's own
    #: profile".  The workflow passes it to the plan expansion so a request can ask for
    #: a narrowed execution without pretending the matrix changed.
    execution_profile: str = ""
    suites: list = field(default_factory=list)
    operators: object = "all"
    cases: list = field(default_factory=list)

    def as_dict(self):
        return {"ep_sizes": list(self.ep_sizes), "levels": list(self.levels),
                "suites": list(self.suites), "operators": self.operators,
                "cases": list(self.cases),
                "execution_profile": self.execution_profile}


@dataclass
class Environment:
    device: str = "npu"
    soc: str | None = None
    dependency_policy: str = "check_only"
    allow_network: bool = False
    allow_sudo: bool = False
    wheelhouse: str | None = None
    local_devices: int | None = None

    def as_dict(self):
        return {"device": self.device, "soc": self.soc,
                "dependency_policy": self.dependency_policy,
                "allow_network": self.allow_network, "allow_sudo": self.allow_sudo,
                "wheelhouse": self.wheelhouse, "local_devices": self.local_devices}


@dataclass
class Execution:
    mode: str = "full"
    profile_policy: str = "representative"
    resume: bool = False
    time_budget_minutes: int | None = None
    nnodes: int = 1
    nproc_per_node: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: int | None = None
    session: str | None = None

    def as_dict(self):
        return {"mode": self.mode, "profile_policy": self.profile_policy,
                "resume": self.resume, "time_budget_minutes": self.time_budget_minutes,
                "nnodes": self.nnodes, "nproc_per_node": self.nproc_per_node,
                "master_addr": self.master_addr, "master_port": self.master_port,
                "session": self.session}

    @property
    def multinode(self):
        return self.nnodes > 1


@dataclass
class BenchmarkRequest:
    """One resolved benchmark request."""

    intent: str = "full_benchmark"
    plan: str | None = DEFAULT_PLAN
    config: str | None = None
    scope: Scope = field(default_factory=Scope)
    environment: Environment = field(default_factory=Environment)
    execution: Execution = field(default_factory=Execution)
    run_id: str | None = None
    workflow_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    #: field name -> "request" | "plan:<path>" | "default:<why>"
    sources: dict = field(default_factory=dict)
    prompt: str | None = None

    # -- serialisation ----------------------------------------------------
    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "intent": self.intent,
            "plan": self.plan,
            "config": self.config,
            "scope": self.scope.as_dict(),
            "environment": self.environment.as_dict(),
            "execution": self.execution.as_dict(),
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "sources": dict(self.sources),
            "prompt": self.prompt,
        }

    @property
    def hash(self):
        """Fingerprint of the resolved request (the run identity it implies)."""
        payload = self.as_dict()
        payload.pop("workflow_id", None)
        payload.pop("prompt", None)
        payload.pop("sources", None)
        return fingerprint(payload)

    @property
    def source_path(self):
        """The configuration file the plan phase expands."""
        return self.config or self.plan

    @property
    def is_plan_file(self):
        return self.config is None

    def resolved(self, name):
        return self.sources.get(name, "unresolved")


# --------------------------------------------------------------------- plan --

def _load_yaml(path):
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RequestError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RequestError(f"{path} is not valid YAML: {exc}") from exc


def is_plan_document(data):
    return isinstance(data, dict) and ("plan" in data or "matrix" in data)


def plan_defaults(path):
    """Read the defaults a plan (or a plain configuration) declares.

    A plan declares ``ep_sizes``, ``levels``, ``suites`` and ``operators``.  A single
    configuration declares only its own EP size and the levels it selects, so those are
    the defaults for a smoke run against it.
    """
    resolved = Path(path)
    if not resolved.is_file():
        resolved = ROOT / path
    data = _load_yaml(resolved)
    if is_plan_document(data):
        spec = data.get("plan") or data.get("matrix") or {}
        suites = spec.get("suites") or ["all"]
        if isinstance(suites, str):
            suites = [suites]
        return {
            "kind": "plan",
            "path": resolved,
            "ep_sizes": [int(size) for size in (spec.get("ep_sizes") or [])],
            "levels": list(spec.get("levels") or LEVELS),
            "suites": list(suites),
            "operators": spec.get("operators", "all"),
        }
    configs = data if isinstance(data, list) else [data]
    first = configs[0] if configs else {}
    distributed = (first or {}).get("distributed") or {}
    benchmark = (first or {}).get("benchmark") or {}
    return {
        "kind": "config",
        "path": resolved,
        "ep_sizes": [int(distributed["ep_size"])] if distributed.get("ep_size") else [],
        "levels": list(benchmark.get("levels") or LEVELS),
        "suites": [first.get("suite_id") or "all"] if isinstance(first, dict) else ["all"],
        "operators": (first or {}).get("operators", "all"),
    }


# ------------------------------------------------------------------ resolve --

def _requested(value):
    return value not in (None, "", [], {})


def resolve(request, root=None):
    """Fill every unset field from the plan and record where each value came from.

    An explicit value only overrides its own dimension. Missing dimensions fall back to
    the plan, in this order: the plan's own declaration, then a documented default.
    """
    root = Path(root or ROOT)
    if request.intent not in INTENTS:
        raise RequestError(f"unknown intent {request.intent!r}; choose from {INTENTS}")
    if request.environment.device not in DEVICES:
        raise RequestError(f"unknown device {request.environment.device!r}")
    if request.environment.dependency_policy not in DEPENDENCY_POLICIES:
        raise RequestError(
            f"unknown dependency policy {request.environment.dependency_policy!r}")
    if request.execution.profile_policy not in PROFILE_POLICIES:
        raise RequestError(f"unknown profile policy {request.execution.profile_policy!r}")

    sources = dict(request.sources)
    # A CPU host cannot run the NPU plan: a CPU smoke request runs the CPU configuration,
    # whose own EP size and levels are the defaults.  This never overrides a plan the user
    # named explicitly.
    untouched_plan = request.plan in (None, DEFAULT_PLAN)
    if (request.intent == "smoke_test" and request.environment.device == "cpu"
            and request.config is None and untouched_plan):
        request.plan = None
        request.config = DEFAULT_CONFIG
        sources["config"] = f"intent:CPU smoke ({DEFAULT_CONFIG})"
        request.scope.ep_sizes = request.scope.ep_sizes if sources.get("ep_sizes") == "prompt" \
            else []
        request.scope.levels = request.scope.levels if sources.get("levels") == "prompt" else []
    # The default plan only applies to intents that expand a matrix.  A profile/analyze/
    # report request works on an existing run and needs no plan.
    if request.plan is None and request.config is None and (
            request.intent in ("full_benchmark", "partial_benchmark", "smoke_test",
                               "plan_only", "estimate")):
        request.plan = DEFAULT_PLAN
        sources.setdefault("plan", f"default:{DEFAULT_PLAN}")
    if request.source_path is None and request.intent in EXISTING_RUN_INTENTS:
        return request

    try:
        defaults = plan_defaults(request.source_path)
    except RequestError:
        if request.intent in EXISTING_RUN_INTENTS:
            defaults = {"ep_sizes": [], "levels": [], "suites": [], "operators": "all"}
        else:
            raise

    scope = request.scope
    if not _requested(scope.ep_sizes):
        scope.ep_sizes = list(defaults.get("ep_sizes") or [])
        sources.setdefault("ep_sizes", f"plan:{defaults.get('path')}")
    else:
        sources["ep_sizes"] = "request"
    if not _requested(scope.levels):
        scope.levels = list(defaults.get("levels") or list(LEVELS))
        sources.setdefault("levels", f"plan:{defaults.get('path')}")
    else:
        sources["levels"] = "request"
    if not _requested(scope.suites):
        scope.suites = list(defaults.get("suites") or ["all"])
        sources.setdefault("suites", f"plan:{defaults.get('path')}")
    else:
        sources["suites"] = "request"
    if not _requested(scope.operators):
        scope.operators = defaults.get("operators", "all")
        sources.setdefault("operators", f"plan:{defaults.get('path')}")
    else:
        sources["operators"] = "request"
    if scope.cases:
        sources["cases"] = "request"
    if request.execution.nproc_per_node is None and request.execution.nnodes == 1:
        sources.setdefault("nproc_per_node", "default:all local devices")
    if request.environment.dependency_policy == "check_only":
        sources.setdefault("dependency_policy", "default:check_only (no install authorization)")
    else:
        sources["dependency_policy"] = "request"
    if request.execution.profile_policy == "representative":
        sources.setdefault("profile_policy", "default:representative")
    else:
        sources["profile_policy"] = "request"
    request.sources = sources
    request.scope.ep_sizes = sorted({int(size) for size in scope.ep_sizes})
    request.scope.levels = [level for level in LEVELS if level in set(scope.levels)] \
        if scope.levels else []
    return request


def build(data, root=None):
    """Build a resolved request from a mapping (a ``request.json`` body)."""
    if not isinstance(data, dict):
        raise RequestError("a request must be a JSON object")
    scope = Scope(**(data.get("scope") or {}))
    environment = Environment(**(data.get("environment") or {}))
    execution = Execution(**(data.get("execution") or {}))
    unknown = set(data) - {"schema_version", "intent", "plan", "config", "scope",
                           "environment", "execution", "run_id", "workflow_id",
                           "sources", "prompt"}
    if unknown:
        raise RequestError(f"unknown request fields: {sorted(unknown)}")
    request = BenchmarkRequest(
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        intent=data.get("intent", "full_benchmark"),
        plan=data.get("plan"), config=data.get("config"),
        scope=scope, environment=environment, execution=execution,
        run_id=data.get("run_id"), workflow_id=data.get("workflow_id"),
        sources=dict(data.get("sources") or {}), prompt=data.get("prompt"),
    )
    return resolve(request, root=root)


def load(path, root=None):
    import json
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RequestError(f"cannot read request {path}: {exc}") from exc
    except ValueError as exc:
        raise RequestError(f"request {path} is not valid JSON: {exc}") from exc
    return build(data, root=root)


# ------------------------------------------------------------- interpretation --

_EP = re.compile(r"\bEP\s*(\d{1,4})\b", re.IGNORECASE)
_LEVEL = re.compile(r"\bL([0-4])\b")
_LEVEL_RANGE = re.compile(r"\bL([0-4])\s*(?:-|–|—|~|to|到|至)\s*L?([0-4])\b", re.IGNORECASE)
_MINUTES = re.compile(r"(\d{1,4})\s*(?:分钟|minutes?|min\b|小时|hours?|h\b)", re.IGNORECASE)

_RESUME_WORDS = ("继续", "恢复", "接着", "上次", "resume", "continue", "unfinished")
_ESTIMATE_WORDS = ("估算", "预估", "estimate", "多久", "需要多少", "how long")
_PLAN_ONLY_WORDS = ("只生成计划", "只做计划", "不要执行", "不执行", "plan only", "dry run",
                    "dry-run", "只生成测试计划")
_PROFILE_WORDS = ("profiling", "profile", "性能剖析", "采集")
_ANALYZE_WORDS = ("分析已有", "只分析", "analyze only", "重新分析")
_REPORT_WORDS = ("只生成报告", "重新生成报告", "regenerate report", "report only")
_ENV_WORDS = ("环境检查", "检查环境", "environment check", "doctor only")
_PARTIAL_WORDS = ("只跑", "只测", "只想跑", "仅跑", "仅测", "only run", "only test")
_FULL_WORDS = ("完整", "全部", "full", "整个", "全量")
_SMOKE_WORDS = ("smoke", "冒烟", "小规模", "快速验证")
_INSTALL_WORDS = ("安装", "install", "缺少依赖就", "缺依赖")
_NETWORK_WORDS = ("允许网络", "联网", "allow network", "download", "下载")
_SUDO_WORDS = ("允许 sudo", "允许sudo", "allow sudo")
_CPU_WORDS = ("cpu", "gloo", "无 npu", "没有 npu", "no npu")
_SOC = re.compile(r"(?:Ascend|昇腾|ascend)\s*(\d{3}[A-Za-z0-9]*)", re.IGNORECASE)


def _levels_in(text):
    """Every level named, expanding an explicit range such as ``L0-L4``."""
    levels = {f"L{number}" for number in _LEVEL.findall(text)}
    for low, high in _LEVEL_RANGE.findall(text):
        start, end = int(low), int(high)
        if start > end:
            start, end = end, start
        levels.update(f"L{number}" for number in range(start, end + 1))
    return sorted(levels)


def interpret(prompt, root=None):
    """Best-effort interpretation of a natural-language instruction.

    The skill is allowed to override any field; this function exists so the common
    phrasings produce the same request every time (and so the skill's scenarios are
    testable).  It never guesses a value it cannot read: an unmentioned dimension stays
    unset and is filled from the plan by :func:`resolve`.
    """
    text = (prompt or "").strip()
    lowered = text.lower()
    request = BenchmarkRequest(prompt=text)

    def has(words):
        return any(word.lower() in lowered for word in words)

    # -- intent ---------------------------------------------------------
    if has(_RESUME_WORDS):
        intent = "resume"
    elif has(_ENV_WORDS):
        intent = "environment_check"
    elif has(_ESTIMATE_WORDS):
        intent = "estimate"
    elif has(_PLAN_ONLY_WORDS):
        intent = "plan_only"
    elif has(_PROFILE_WORDS) and not (has(_FULL_WORDS) or has(_PARTIAL_WORDS)):
        intent = "profile"
    elif has(_ANALYZE_WORDS) and not (has(_FULL_WORDS) or has(_PARTIAL_WORDS)):
        intent = "analyze"
    elif has(_REPORT_WORDS):
        intent = "report"
    elif has(_SMOKE_WORDS):
        intent = "smoke_test"
    elif has(_PARTIAL_WORDS):
        intent = "partial_benchmark"
    elif has(_FULL_WORDS):
        intent = "full_benchmark"
    else:
        intent = "partial_benchmark"
    request.intent = intent

    # -- scope ----------------------------------------------------------
    sizes = sorted({int(size) for size in _EP.findall(text)})
    if sizes:
        request.scope.ep_sizes = sizes
        request.sources["ep_sizes"] = "prompt"
    levels = _levels_in(text)
    if levels:
        request.scope.levels = levels
        request.sources["levels"] = "prompt"
    # Only explicit profile wording sets it: a profile narrows the execution, so it
    # must never be inferred from a vague word like "quick".
    for word, profile in (("exhaustive", "exhaustive"), ("smoke", "smoke"),
                          ("冒烟", "smoke"), ("core", "core")):
        if word in lowered:
            request.scope.execution_profile = profile
            request.sources["execution_profile"] = f"prompt:{word}"
            break
    if has(_CPU_WORDS):
        request.environment.device = "cpu"
        request.sources["device"] = "prompt"
        # A CPU host cannot measure Ascend performance.  A "full" request on CPU is a
        # framework validation, so it runs the CPU smoke configuration and says so.
        if request.intent in ("full_benchmark",):
            request.intent = "smoke_test"
            request.sources["intent"] = "prompt:CPU host means framework validation"
    soc = _SOC.search(text)
    if soc:
        request.environment.soc = f"Ascend{soc.group(1)}"
        request.sources["soc"] = "prompt"

    # -- safety ---------------------------------------------------------
    if has(_INSTALL_WORDS):
        request.environment.dependency_policy = "install_missing"
        request.sources["dependency_policy"] = "prompt"
    if has(_NETWORK_WORDS):
        request.environment.allow_network = True
        request.sources["allow_network"] = "prompt"
    if has(_SUDO_WORDS):
        request.environment.allow_sudo = True
        request.sources["allow_sudo"] = "prompt"

    # -- execution ------------------------------------------------------
    if request.intent in READ_ONLY_INTENTS:
        request.environment.dependency_policy = "check_only"
        request.sources["dependency_policy"] = "intent:never installs"
        request.execution.profile_policy = "none"
    if "profile" in lowered and any(word in lowered for word in ("全部", "full", "所有")):
        request.execution.profile_policy = "full"
    if request.intent == "profile":
        request.execution.profile_policy = (
            "full" if has(("全部", "full", "所有")) else "representative")
        request.sources["profile_policy"] = "prompt"
    minutes = _MINUTES.search(text)
    if minutes:
        value = int(minutes.group(1))
        if "小时" in minutes.group(0) or re.search(r"hours?|h$", minutes.group(0)):
            value *= 60
        request.execution.time_budget_minutes = value
        request.sources["time_budget_minutes"] = "prompt"
    if request.intent == "smoke_test":
        request.execution.profile_policy = "none"
        if not sizes:
            request.scope.ep_sizes = [8]
            request.sources["ep_sizes"] = "intent:smoke default EP8"
        if not levels:
            request.scope.levels = ["L1"]
            request.sources["levels"] = "intent:smoke default L1"
    return resolve(request, root=root)


def multinode_gaps(request):
    """The details a multi-node request must supply before it may run."""
    gaps = []
    if request.execution.nnodes > 1:
        if request.execution.master_port is None:
            gaps.append("master_port")
        if not request.execution.session:
            gaps.append("session")
        if request.execution.nproc_per_node is None:
            gaps.append("nproc_per_node")
    return gaps


def handler(ctx):
    """``request_resolution``: prove the request is complete before anything runs.

    Everything the later phases rely on is checked here: the configuration file exists,
    a matrix intent selected at least one EP size, and a multi-node request carries the
    details it must not invent. The resolved request is written next to the workflow
    state, so a resumed workflow revalidates the same values.
    """
    from scripts.workflow.phases import PhaseOutcome
    request = ctx.request
    problems = []
    if request.source_path is not None:
        candidate = Path(request.source_path)
        if not candidate.is_absolute():
            candidate = Path(ctx.root) / candidate
        if not candidate.is_file():
            problems.append(f"the configuration file {request.source_path} does not exist")
    matrix_intent = request.intent in ("full_benchmark", "partial_benchmark", "smoke_test",
                                       "plan_only", "estimate")
    if matrix_intent and not request.scope.ep_sizes:
        problems.append("no EP size was requested and the plan declares none")
    if matrix_intent and not request.scope.levels:
        problems.append("no benchmark level was requested and the plan declares none")
    gaps = multinode_gaps(request)
    if gaps:
        problems.append("a multi-node request is missing " + ", ".join(gaps))
    if request.environment.dependency_policy == "install_missing" and not (
            request.environment.allow_network or request.environment.wheelhouse):
        problems.append(
            "the request allows installing dependencies but neither enables the network "
            "nor supplies a wheelhouse; installation would be blocked")
    path = ctx.write_json("resolved_request.json", {
        "schema_version": SCHEMA_VERSION, "workflow_id": ctx.state.workflow_id,
        "request": request.as_dict(), "request_hash": request.hash,
        "multinode_gaps": gaps,
        "problems": problems,
    })
    if any("does not exist" in item or "declares none" in item or "missing" in item
           for item in problems):
        return PhaseOutcome.blocked("; ".join(problems), artifacts=[path])
    if problems:
        return PhaseOutcome.partial("; ".join(problems), artifacts=[path])
    return PhaseOutcome.passed([path], detail={"request_hash": request.hash,
                                               "intent": request.intent})
