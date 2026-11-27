"""Per-operator smoke profiles: load, validate and turn into planner tasks.

A smoke profile answers one question for one operator: *which* single case
exercises its main execution path on real silicon.  The declaration lives in
``operators/smoke_profiles.yaml``; this module is the only reader.

Design rules (kept here so they cannot rot in a document):

* The operator registry stays the single source of truth for capabilities.  A
  profile only chooses a case, an EP size and a workload; whether the case is
  legal for that adapter is decided by ``common.registry.check_constraints`` and
  the case's own ``requires`` tuple.
* A profile that the planner cannot turn into an executable task is a *defect in
  the table*, not a runtime surprise: :meth:`ProfileTable.errors` reports it and
  the caller refuses to start.
* Nothing here selects a device.  The resulting configuration is validated as a
  formal ``npu`` configuration, and ``common.execution_mode`` refuses anything
  else while the formal mode is active.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.case_spec import catalog
from common.config import load_configs, merge, validate
from common.registry import check_constraints, registry
from common.scheduler import make_plan

ROOT = Path(__file__).resolve().parents[1]
#: The table lives next to ``registry.yaml``: it is per-operator specification
#: data, not a benchmark configuration, so it must not sit in ``configs/`` where
#: every ``*.yaml`` is resolved as one.
DEFAULT_PROFILES = ROOT / "operators" / "smoke_profiles.yaml"

#: Case whose operation exercises a backward pass, per level.
BACKWARD_CASES = (("L3", "MOE-002"), ("L4", "LAYER-002"))

#: Adapter capabilities that make a backward case mandatory.
BACKWARD_CAPABILITY = "moe_backward"

#: case operation -> execution-path compute tag
_COMPUTE_TAGS = {
    "dispatch": "communication",
    "combine": "communication",
    "dispatch_combine": "communication",
    "cached_dispatch": "communication",
    "async_dispatch": "communication",
    "async_combine": "communication",
    "deduplicated_dispatch": "communication",
    "planned_dispatch": "communication",
    "training_tail_dispatch": "communication",
    "high_throughput_dispatch": "communication",
    "combine_add_rmsnorm": "fused",
    "dispatch_gmm": "gmm",
    "gmm_combine": "gmm",
    "serial_dispatch_gmm": "gmm",
    "serial_gmm_combine": "gmm",
    "chunk_pipeline": "fused",
    "tile_schedule": "gmm",
    "comm_resources": "communication",
    "quantized_fusion": "fused",
    "fused_backward": "fused",
    "forward": "fused",
    "backward": "fused",
    "forward_backward": "fused",
    "recompute": "fused",
    "concurrent": "communication",
}

_PROFILE_KEYS = {"family", "level", "case", "ep_size", "direction", "note", "paths",
                 "model", "workload", "distributed", "topology", "benchmark",
                 "load_balance", "dedup", "profiling", "validation", "backward"}


class ProfileError(ValueError):
    """A smoke profile table that cannot be executed as declared."""


@dataclass(frozen=True)
class SmokeProfile:
    operator: str
    family: str
    level: str
    case: str
    ep_size: int
    direction: str
    config: dict
    paths: tuple
    validation: dict
    note: str = ""
    backward_case: str | None = None
    backward_level: str | None = None

    @property
    def label(self):
        return f"{self.operator}[{self.level}/{self.case}@EP{self.ep_size}]"

    def forward_tasks(self):
        return make_plan([self.config], operators=[self.operator],
                         levels=[self.level], cases=[self.case])

    def backward_tasks(self):
        if not self.backward_case:
            return []
        return make_plan([self.config], operators=[self.operator],
                         levels=[self.backward_level], cases=[self.backward_case])


@dataclass
class ProfileTable:
    profiles: dict = field(default_factory=dict)
    required_paths: tuple = ()
    defaults: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    source: Path | None = None

    def __iter__(self):
        return iter(self.profiles.values())

    def __len__(self):
        return len(self.profiles)

    def get(self, operator):
        return self.profiles.get(operator)

    def operators(self):
        return sorted(self.profiles)

    def paths(self):
        covered = {}
        for profile in self.profiles.values():
            for tag in profile.paths:
                covered.setdefault(tag, []).append(profile.operator)
        return covered

    def coverage(self, outcomes, level_evidence=None):
        """Operator, level and path coverage from ``{profile_label: bucket}``.

        ``outcomes`` keys are :attr:`SmokeProfile.label`.  A missing key means
        the profile never ran, which is a gap -- never a pass.

        ``level_evidence`` carries results that do not come from an operator
        profile, keyed by level, e.g. ``{"L0": {"passed": True, "source":
        "environment smoke"}}``.  A level is covered when either a profile at that
        level passed or the supplied evidence says the level itself did.
        """
        level_evidence = level_evidence or {}
        operator_rows, gaps = [], []
        level_counts = {}
        for profile in self.profiles.values():
            bucket = outcomes.get(profile.label, "UNKNOWN")
            passed = bucket == "PASSED"
            operator_rows.append({
                "operator": profile.operator,
                "family": profile.family,
                "level": profile.level,
                "case": profile.case,
                "ep_size": profile.ep_size,
                "direction": profile.direction,
                "bucket": bucket,
                "passed": passed,
            })
            level_counts.setdefault(profile.level, {"total": 0, "passed": 0})
            level_counts[profile.level]["total"] += 1
            level_counts[profile.level]["passed"] += int(passed)
        covered_tags = {tag for profile in self.profiles.values()
                        for tag in profile.paths if outcomes.get(profile.label) == "PASSED"}
        path_rows = []
        for tag in self.required_paths:
            owners = [profile.operator for profile in self.profiles.values() if tag in profile.paths]
            covered = tag in covered_tags
            path_rows.append({"path": tag, "covered": covered, "operators": owners})
            if not covered:
                gaps.append({"kind": "COVERAGE_GAP", "scope": "execution_path", "path": tag,
                             "reason": ("no profile declares this path" if not owners else
                                        "every profile that declares this path is blocked or failed"),
                             "operators": owners})
        level_rows = []
        for level in ("L0", "L1", "L2", "L3", "L4"):
            entry = level_counts.get(level)
            extra = level_evidence.get(level) or {}
            if entry is None:
                covered = bool(extra.get("passed"))
                level_rows.append({"level": level, "total": 0, "passed": 0,
                                   "covered": covered,
                                   "source": extra.get("source") or "no profile"})
                if not covered:
                    reason = ("the supplied evidence for this level did not pass"
                              if extra else "no smoke profile targets this level")
                    gaps.append({"kind": "COVERAGE_GAP", "scope": "level", "path": level,
                                 "reason": reason, "operators": []})
            else:
                covered = entry["passed"] > 0 or bool(extra.get("passed"))
                level_rows.append({"level": level, "total": entry["total"],
                                   "passed": entry["passed"], "covered": covered,
                                   "source": extra.get("source") or "operator profiles"})
                if not covered:
                    gaps.append({"kind": "COVERAGE_GAP", "scope": "level", "path": level,
                                 "reason": "no profile at this level passed",
                                 "operators": [row["operator"] for row in operator_rows
                                               if row["level"] == level]})
        return {
            "operators": operator_rows,
            "levels": level_rows,
            "paths": path_rows,
            "gaps": gaps,
            "operator_coverage": {
                "total": len(operator_rows),
                "passed": sum(1 for row in operator_rows if row["passed"]),
            },
            "level_coverage": {
                "total": len(level_rows),
                "covered": sum(1 for row in level_rows if row["covered"]),
            },
            "path_coverage": {
                "total": len(path_rows),
                "covered": sum(1 for row in path_rows if row["covered"]),
            },
        }


def _ep_config_path(ep_size):
    path = ROOT / "configs" / f"ep{ep_size}_bf16.yaml"
    if path.is_file():
        return path
    raise ProfileError(
        f"EP{ep_size} has no environment file at {path.relative_to(ROOT)}; add one or "
        f"choose a declared EP size")


def _apply(base, entry, defaults):
    """Merge one operator entry onto an EP environment configuration."""
    changes = {}
    for key in ("model", "workload", "distributed", "topology", "benchmark",
                "load_balance", "dedup", "profiling"):
        value = entry.get(key)
        if value:
            changes[key] = value
    merged = merge(base, changes) if changes else dict(base)
    merged["benchmark"] = merge(merged.get("benchmark", {}), defaults.get("benchmark", {}))
    return validate(merged)


def _compute_tag(case):
    return _COMPUTE_TAGS.get(case.operation, "communication")


def _backward_choice(spec, level):
    if BACKWARD_CAPABILITY not in spec.capabilities:
        return None, None
    levels = [item for item in BACKWARD_CASES]
    # Prefer the same level as the forward case, otherwise the closest L3 case.
    for candidate_level, case_id in levels:
        if candidate_level == level:
            return case_id, candidate_level
    return levels[0][1], levels[0][0]


def _paths_for(spec, case, family, backward_case, level):
    tags = {f"backend:{family}", f"compute:{_compute_tag(case)}", "direction:forward"}
    if backward_case:
        tags.add("direction:backward")
    if {"dynamic_planner", "planned_dispatch"} & set(spec.capabilities):
        tags.add("routing:planner")
    else:
        tags.add("routing:static")
    for capability in ("rank_dedup", "server_dedup"):
        if capability in spec.capabilities:
            tags.add(f"dedup:{capability.split('_')[0]}")
    precision = "fp8" if case.id == "FUSE-008" else "bf16"
    tags.add(f"precision:{precision}")
    return tuple(sorted(tags))


def load(path=None):
    """Load and fully validate the profile table.

    Raises :class:`ProfileError` with every problem found, not just the first, so
    a bad table can be fixed in one pass.
    """
    path = Path(path) if path else DEFAULT_PROFILES
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: profile table must be a mapping")
    if raw.get("schema_version") != 1:
        raise ProfileError(f"{path}: unsupported schema_version {raw.get('schema_version')!r}")

    defaults = raw.get("defaults") or {}
    specs, cases = registry(), catalog()
    table = ProfileTable(required_paths=tuple(raw.get("required_paths") or ()),
                         defaults=defaults, source=path)
    errors = []

    for operator, entry in (raw.get("operators") or {}).items():
        where = f"{path.name}:{operator}"
        if operator not in specs:
            errors.append(f"{where}: unknown operator")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{where}: entry must be a mapping")
            continue
        unknown = set(entry) - _PROFILE_KEYS
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)}")
            continue
        spec = specs[operator]
        case_id = entry.get("case")
        if not case_id:
            errors.append(f"{where}: 'case' is required")
            continue
        case = cases.get(case_id)
        if case is None:
            errors.append(f"{where}: unknown case {case_id!r}")
            continue
        level = entry.get("level") or case.level
        if level != case.level:
            errors.append(f"{where}: level {level} does not match case {case_id} level {case.level}")
            continue
        missing = sorted(set(case.requires) - set(spec.capabilities))
        if missing:
            errors.append(f"{where}: adapter lacks {', '.join(missing)} required by {case_id}")
            continue
        if case.owner and case.owner != operator:
            errors.append(f"{where}: {case_id} is owned by {case.owner}")
            continue
        ep_size = int(entry.get("ep_size") or defaults.get("ep_size") or 8)
        try:
            config = _apply(load_configs(_ep_config_path(ep_size))[0], entry, defaults)
        except (ProfileError, ValueError) as exc:
            errors.append(f"{where}: {exc}")
            continue
        if config["distributed"]["ep_size"] != ep_size:
            errors.append(f"{where}: EP{ep_size} environment declares "
                          f"EP{config['distributed']['ep_size']}")
            continue
        if config.get("device") != "npu":
            errors.append(f"{where}: formal profiles must declare device: npu, "
                          f"got {config.get('device')!r}")
            continue
        bounds = check_constraints(spec, config)
        if bounds:
            errors.append(f"{where}: adapter rejected the configuration: {bounds}")
            continue

        declared = entry.get("direction") or ("both" if BACKWARD_CAPABILITY in spec.capabilities
                                              else "forward")
        backward_case = backward_level = None
        if declared not in ("forward", "backward", "both"):
            errors.append(f"{where}: direction must be forward, backward or both")
            continue
        if declared in ("backward", "both"):
            case_choice, level_choice = _backward_choice(spec, level)
            if case_choice is None:
                errors.append(f"{where}: direction {declared} but the adapter declares no "
                              f"{BACKWARD_CAPABILITY} capability")
                continue
            backward_case, backward_level = case_choice, level_choice

        profile = SmokeProfile(
            operator=operator, family=entry.get("family") or operator, level=level,
            case=case_id, ep_size=ep_size, direction=declared, config=config,
            paths=_paths_for(spec, case, entry.get("family") or operator,
                             backward_case, level),
            validation=merge(defaults.get("validation", {}), entry.get("validation") or {}),
            note=(entry.get("note") or "").strip(),
            backward_case=backward_case, backward_level=backward_level,
        )

        # The planner, not this module, decides whether the point can execute.
        try:
            forward = profile.forward_tasks()
        except ValueError as exc:
            errors.append(f"{where}: planner rejected the profile: {exc}")
            continue
        if not forward:
            errors.append(f"{where}: planner expanded no task for {case_id}")
            continue
        task = forward[0]
        # A case may sweep variants (e.g. FUSE-005 chunks_2/4/8): every expanded
        # task must be executable, and the profile only passes when all of them do.
        for item in forward:
            if item["status"] != "not_run" or item["reason"]:
                errors.append(f"{where}: planner marks {case_id} as {item['status']}"
                              f"{': ' + item['reason'] if item['reason'] else ''}. "
                              f"A profile must be executable; fix the declaration instead of "
                              f"relying on the run to discover it.")
                task = None
                break
        if task is None:
            continue
        if backward_case:
            backward = profile.backward_tasks()
            if not backward or any(item["status"] != "not_run" or item["reason"]
                                   for item in backward):
                detail = backward[0] if backward else {}
                errors.append(f"{where}: backward case {backward_case} is not executable "
                              f"({detail.get('status')}: {detail.get('reason')})")
                continue
        table.profiles[operator] = profile

    unlisted = sorted(set(specs) - set(table.profiles))
    if errors:
        raise ProfileError("; ".join(errors))
    table.warnings = [f"{op}: no smoke profile" for op in unlisted]
    return table


def coverage_requirements(table, spec_status):
    """Adapters that must appear in smoke, from the registry's own status field.

    ``historical_verified`` and ``integrated`` are the two statuses the registry
    uses for an adapter that is supposed to run.  Anything else is reported as a
    gap rather than silently excluded.
    """
    runnable = {op for op, spec in registry().items()
                if spec.status in ("integrated", "historical_verified")}
    missing = sorted(runnable - set(table.profiles))
    return {"required": len(runnable), "declared": len(table.profiles),
            "missing": missing}
