"""Construct a bounded, dependency-free benchmark manifest.

Expansion model (see ``docs/benchmark_matrix.md``):

``L0``    ``EP size x L0 case x parameter variant`` -- never multiplied by the
          EP-operator dimension, because a hardware baseline does not belong to
          an operator.
``L1-L4`` ``EP size x case x operator x parameter variant`` -- capability checked
          per case, not per level.

Every task carries its EP size, suite membership, effective configuration
fingerprint and resolved status, so the manifest itself is the coverage record.
"""
from copy import deepcopy

from common.cases import L0_PARAMETERS, canonical_cases, catalog, l0_family
from common.registry import check_constraints, registry
from common.suites import select as select_suite_cases, suite as suite_of
from common.storage import fingerprint
from common.variants import message_sizes, parameter_problems, task_parameters, variants_for

FIXED_OWNER = "fixed_owner"
FIXED_OWNER_CASES = {"LAYER-003", "LAYER-005", "LAYER-006"}

ALIAS_MODES = ("dedupe", "run")


def _l0_matrix(cfg, cases):
    """Load the capability matrix once when an L0 family needs it."""
    if not any(case.level == "L0" and case.category in ("transport", "execution") for case in cases):
        return None
    try:
        from baselines.l0.topology import discovery
        return discovery.load(cfg)
    except Exception as exc:  # discovery must never stop the rest of L0
        return {"schema_version": 1, "source": "unavailable", "error": str(exc),
                "channels": [], "engines": [], "accelerators": []}


def _l0_status(case, params, matrix):
    """Return ``(status, reason)`` for one L0 parameter set from the matrix.

    A legacy alias is checked against the capability of its canonical case, so an
    explicit ``--cases HW-008`` run reports the same hardware verdict as LINK-013.
    """
    if matrix is None:
        return None, None
    from baselines.l0.topology import capabilities as capability_table
    lookup = case.alias_of or case.id
    if case.category == "transport":
        settings = L0_PARAMETERS.get(lookup, {})
        mechanism = settings.get("mechanism")
        resource = params.get("execution_resource") or settings.get("execution_resource")
        supported, reason = capability_table.support(matrix, case_id=lookup,
                                                     mechanism=mechanism,
                                                     execution_resource=resource)
        return ("not_run", None) if supported else ("unsupported", reason)
    if case.category == "execution":
        mode = params.get("accelerator")
        entry = capability_table.index(matrix)["accelerator"].get(mode)
        if entry is None:
            return "unsupported", f"accelerator mode {mode!r} is not offered by the installed hccl_test"
        if not entry["supported"]:
            return "unsupported", entry.get("reason")
    return "not_run", None


def status_for(spec, case, cfg, baseline=False):
    """Return ``(status, reason)`` for one (operator, case, configuration).

    The decision is made per *case*: an adapter capability is only accepted when
    the case's own requirement is present, so a README that claims a level does
    not make every case of that level runnable.
    """
    if spec.status not in ("integrated", "historical_verified"):
        return spec.status, spec.reason
    if case.id == "MOE-007" and (cfg["device"] != "npu" or cfg["distributed"]["ep_size"] < 2):
        return "unsupported", "concurrent MoE requires at least two HCCL ranks"
    if case.id == "COMM-010":
        mode = cfg["dedup"]["mode"]
        if mode == "none":
            return ("not_applicable",
                    "deduplicated dispatch has no dedup path; the None control for this suite is "
                    "COMM-001 with dedup.mode=none")
        if f"{mode}_dedup" not in spec.capabilities:
            return "unsupported", f"no {mode} dedup path"
    layer_dedup = {"LAYER-005": "rank", "LAYER-006": "server"}.get(case.id)
    if layer_dedup is not None and layer_dedup != cfg["dedup"]["mode"]:
        return "not_applicable", f"this case measures {layer_dedup} dedup; the plan selects dedup.mode={cfg['dedup']['mode']}"
    if case.id == "LAYER-007" and cfg["workload"]["routing"] != "real_trace_replay":
        return "not_run", "L3 replay requires a real_trace_replay workload"
    if case.id in {"COMM-012", "LAYER-003", "LAYER-004", "LAYER-008"} and not cfg["load_balance"]["enabled"]:
        return "not_run", "choose a load_balance configuration that enables the planner"
    if case.level == "L4" and cfg["workload"]["routing"] == "real_trace_replay":
        return "not_run", "replay is measured in LAYER-007 with the L3 boundary"
    if cfg["benchmark"].get("phase") == "forward" and case.training:
        return "not_applicable", "forward-only configuration selects a training case"
    if case.level == "L1" and spec.id == "mc2_dispatch_cann":
        tokens = cfg["workload"]["tokens_per_rank"]
        if max(tokens if isinstance(tokens, list) else [tokens]) > 512:
            return "unsupported", "a single 950DT V2 invocation supports at most 512 tokens per rank"
    fixed_owner = case.id in FIXED_OWNER_CASES and spec.id == "alltoall_hccl"
    contract = cfg if not (baseline or fixed_owner) else {
        **cfg, "dedup": {"mode": "none"}, "load_balance": {"enabled": False},
    }
    bounds = check_constraints(spec, contract)
    if bounds:
        return "unsupported", bounds
    required = ("router_layer", "moe_backward") if fixed_owner else case.requires
    missing = sorted(set(required) - set(spec.capabilities))
    if missing:
        gaps = spec.constraints.get("unadapted", {})
        known = [gaps[name] for name in missing if name in gaps]
        if known:
            return "not_implemented", "; ".join(known)
        return "unsupported", f"operation requires {', '.join(missing)}"
    return "not_run", None


def prepare_case(config, case):
    cfg = deepcopy(config)
    work = cfg["workload"]
    if case.id == "MOE-005" and work["routing"] in {"balanced", "uniform_random"}:
        work["routing"] = "expert_hotspot"
    if case.id == "MOE-009":
        tokens = work["tokens_per_rank"]
        work.setdefault("token_steps", [tokens, [n // 2 for n in tokens] if isinstance(tokens, list)
                                        else max(1, tokens // 2)])
    return cfg


def _operator_ids(config, operators, specs):
    ids = operators or config["operators"]
    if isinstance(ids, str):
        ids = [ids]
    if ids == ["all"]:
        ids = list(specs)
    unknown = set(ids) - set(specs)
    if unknown:
        raise ValueError(f"unknown operators: {', '.join(sorted(unknown))}")
    return ids


def _selected_cases(config, levels, cases, suites):
    """Resolve the case set for one configuration.

    An explicit ``cases`` argument (CLI ``--cases``) is a deliberate narrowing by
    the user, so it is only level-filtered when ``--levels`` was also given.  A
    suite selection is always filtered by the configured levels.
    """
    all_cases = catalog()
    canonical = canonical_cases()
    explicit = list(cases or ())
    unknown = set(explicit) - set(all_cases)
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(sorted(unknown))}")
    if explicit:
        return [all_cases[case_id] for case_id in explicit
                if not levels or all_cases[case_id].level in levels]
    level_filter = levels or config["benchmark"]["levels"]
    return [canonical[case_id] for case_id in select_suite_cases(suites, level_filter)]


def _identity(config, case_id, operator_id, params):
    """Task identity: two suites with the same point must not duplicate work."""
    effective = {key: value for key, value in config.items() if key != "suite_id"}
    return {"config": effective, "backend": operator_id, "case": case_id,
            "parameters": params,
            "ep_size": config["distributed"]["ep_size"]}


def make_plan(configs, operators=None, levels=None, cases=None, baselines=False,
              suites=None, aliases="dedupe"):
    """Expand configurations into manifest tasks.

    ``aliases`` controls legacy L0 identifiers: ``dedupe`` (default) keeps them
    out of the task list because their canonical case is already covered, while
    ``run`` expands them as their own task.  Either way the manifest keeps the
    alias mapping so nothing is silently lost.
    """
    if aliases not in ALIAS_MODES:
        raise ValueError(f"aliases must be one of {ALIAS_MODES}")
    specs, all_cases = registry(), catalog()
    if cases and set(cases) - set(all_cases):
        raise ValueError(f"unknown cases: {', '.join(sorted(set(cases) - set(all_cases)))}")
    tasks = {}
    for config in configs:
        ids = _operator_ids(config, operators, specs)
        if baselines:
            ids = ["alltoall_hccl", "allgather_hccl"]
        ep_size = config["distributed"]["ep_size"]
        suite_ids = [suite_of(item).id for item in (suites or [config["suite_id"]])]
        selected_levels = levels or (["L0", "L1", "L2"] if baselines else config["benchmark"]["levels"])
        # `--cases` is an explicit narrowing: it is only level-filtered when the
        # user also passed `--levels`, never by the configuration default.
        selected = _selected_cases(config, levels if cases else selected_levels, cases, suite_ids)
        if baselines:
            selected = [case for case in selected if case.level in {"L0", "L1", "L2"}]
            if not cases:
                selected = [case for case in selected
                            if case.level != "L2" or case.id in {"FUSE-001", "FUSE-003"}]
        matrix = _l0_matrix(config, selected) if not baselines else None
        for case in selected:
            if case.alias_of and aliases != "run" and not (cases and case.id in cases):
                continue
            cfg = prepare_case(config, case)
            settings = cfg["benchmark"]
            for op_id in ["hardware"] if case.level == "L0" else ids:
                spec = specs.get(op_id)
                if spec is None and case.level != "L0":
                    continue
                status, reason = status_for(spec, case, cfg, baselines) if spec else ("not_run", None)
                parameter_sets = _parameters_for(case, cfg, ep_size, op_id, status, reason)
                for params in parameter_sets:
                    case_status, case_reason = status, reason
                    problem = parameter_problems(case, params, ep_size)
                    if problem and not case_reason:
                        case_status, case_reason = "unsupported", problem
                    if case.level == "L0" and case_reason is None:
                        derived_status, derived_reason = _l0_status(case, params, matrix)
                        if derived_status:
                            case_status, case_reason = derived_status, derived_reason
                    identity = _identity(cfg, case.id, op_id, params)
                    task_id = f"{case.id}-{fingerprint(identity)[:16]}"
                    if task_id in tasks:
                        existing = tasks[task_id]
                        existing["suite_ids"] = sorted(set(existing["suite_ids"]) | set(suite_ids))
                        continue
                    tasks[task_id] = {
                        "task_id": task_id, "case_id": case.id, "level": case.level,
                        "operation": case.operation, "backend_id": op_id, "config": cfg,
                        "parameters": params, "status": case_status, "reason": case_reason,
                        "source_versions": {"adapter_source": spec.source, "adapter_commit": spec.commit}
                        if spec else {},
                        "baseline": baselines or not spec,
                        "ep_size": ep_size,
                        "suite_ids": list(suite_ids),
                        "case_category": case.category,
                        "case_family": l0_family(case) or "operator",
                        "alias_of": case.alias_of,
                        "config_fingerprint": fingerprint(
                            {key: value for key, value in cfg.items() if key != "suite_id"}),
                    }
    return list(tasks.values())


def _parameters_for(case, cfg, ep_size, op_id, status, reason):
    """Parameter sets for one (case, operator) point.

    A runnable point expands into its case-specific variants; an unrunnable one
    keeps a single default parameter set so the manifest can record the explicit
    status and reason for that exact case.
    """
    settings = cfg["benchmark"]
    fixed_owner = case.id in FIXED_OWNER_CASES and op_id == "alltoall_hccl"
    if status not in (None, "not_run") or reason is not None:
        if case.level != "L0":
            return [{"variant": "default", "message_bytes": None}]
        return [{"variant": "default", "message_bytes": message_sizes(case, settings)[0]}]
    if case.level == "L0":
        return task_parameters(case, "hardware", settings, ep_size)
    variants = (FIXED_OWNER,) if fixed_owner else variants_for(case)
    return [{"variant": variant, "message_bytes": None} for variant in variants]


def merge_tasks(store, tasks):
    """Merge per-suite expansions, keeping one task per identical point."""
    for task in tasks:
        key = task["task_id"]
        if key in store:
            previous = store[key]
            previous["suite_ids"] = sorted(set(previous["suite_ids"]) | set(task["suite_ids"]))
            previous["config_source"] = sorted(set(previous.get("config_source") or ()) |
                                               set(task.get("config_source") or ()))
            continue
        store[key] = task
    return store


def case_index(tasks, show_aliases=True):
    """Every catalog case -> task IDs covering it, or its alias target.

    Aliases are reported as ``alias`` entries rather than expanded twice, so a
    coverage matrix can prove that a legacy ID is accounted for.
    """
    index = {}
    for case_id, case in catalog().items():
        if case.alias_of:
            if show_aliases:
                index[case_id] = {"status": "alias", "alias_of": case.alias_of, "tasks": []}
            continue
        index[case_id] = {"status": "task", "alias_of": None, "tasks": []}
    for task in tasks:
        entry = index.get(task["case_id"])
        if entry is None:
            entry = index[task["case_id"]] = {"status": "task", "alias_of": None, "tasks": []}
        entry["tasks"].append(task["task_id"])
    return {case_id: {"status": entry["status"], "alias_of": entry["alias_of"],
                      "task_count": len(entry["tasks"]), "tasks": entry["tasks"]}
            for case_id, entry in index.items()}


def coverage_by_ep_size(tasks):
    """Case ID set per EP size, used by the completeness check."""
    coverage = {}
    for task in tasks:
        coverage.setdefault(task["ep_size"], {})[task["case_id"]] = True
    return {ep: sorted(cases) for ep, cases in sorted(coverage.items())}


def check_completeness(tasks, ep_sizes=None, expected=None):
    """Raise when an EP size is missing a case another EP size has.

    The check is about *presence in the manifest*, not about runnability: an
    unsupported point still produces a task with an explicit reason.  ``expected``
    defaults to every case that the plan selected, so the same function validates
    a focused suite and the complete matrix.
    """
    coverage = coverage_by_ep_size(tasks)
    if not coverage:
        raise ValueError("the plan generated no tasks")
    if expected is None:
        expected = set().union(*(set(items) for items in coverage.values()))
    expected = set(expected)
    sizes = sorted(ep_sizes or coverage)
    problems = []
    for size in sizes:
        present = set(coverage.get(size, ()))
        missing = sorted(expected - present)
        if missing:
            problems.append(f"EP{size} is missing {len(missing)} case(s): {', '.join(missing)}")
    reference = set(coverage.get(sizes[0], ()))
    for size in sizes[1:]:
        if set(coverage.get(size, ())) != reference:
            missing = sorted(reference - set(coverage.get(size, ())))
            extra = sorted(set(coverage.get(size, ())) - reference)
            problems.append(f"EP{size} differs from EP{sizes[0]}: missing={missing} extra={extra}")
    if problems:
        raise ValueError("benchmark plan is incomplete: " + "; ".join(problems))
    return coverage
