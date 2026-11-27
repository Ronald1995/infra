"""Construct a bounded, dependency-free benchmark manifest.

Expansion model (see ``docs/benchmark_matrix.md``):

``L0``    ``EP size x L0 case x parameter variant`` -- never multiplied by the
          EP-operator dimension, because a hardware baseline does not belong to
          an operator.
``L1-L4`` ``EP size x case x operator x parameter variant`` -- capability checked
          per case, not per level.

Three invariants make the manifest comparable across EP sizes:

* **Case identity** comes from :mod:`common.case_spec` and nowhere else.
* **Parameter expansion never consults a capability verdict.**  A task's
  parameters are a property of the plan; the capability verdict is attached to
  the task afterwards.  Collapsing an unsupported point to a single parameter set
  used to make EP8 produce more tasks than EP64 for the same case, which silently
  changed the test scope with the EP size.
* **Every point stays in the manifest**, whatever its status.  An unsupported,
  not-applicable or not-implemented point is a recorded outcome with a reason,
  never a missing task.

Every task carries its EP size, suite membership, effective configuration
fingerprint and resolved status, so the manifest itself is the coverage record.
"""
from copy import deepcopy

from common.case_spec import (APPLICABILITY_RULES, ApplicabilityContext, FIXED_OWNER,
                              TRANSFORMS, canonical_specs, catalog, l0_family,
                              owned_by, spec as case_spec)
from common.registry import check_constraints, registry
from common.suites import select as select_suite_cases, suite as suite_of
from common.storage import fingerprint
from common.variants import (TRANSPORT_DISABLED, policy_problem, structural_problem,
                             task_parameters, transport_gate)

#: Cases measured on exactly one adapter; the scheduler relaxes the scenario
#: contract for them because the adapter implements only the router path.
FIXED_OWNER_CASES = frozenset(case_id for case_id, case in catalog().items()
                              if case.owner is not None)
#: Capabilities the fixed-owner contract requires regardless of the row's own
#: requirement (the measurement is the router + backward path).
FIXED_OWNER_CAPABILITIES = ("router_layer", "moe_backward")

ALIAS_MODES = ("dedupe", "run")

#: Adapter statuses that may execute a case.
EXECUTABLE_STATUSES = ("integrated", "historical_verified")


def _l0_matrix(cfg, cases):
    """Load the capability matrix once when an L0 family needs it."""
    if not any(case.level == "L0" and case.category in ("transport", "execution")
               for case in cases):
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
        settings = case_spec(lookup).requested()
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


def required_capabilities(case, operator_id=None):
    """Capability names an adapter must expose to run ``case`` for ``operator_id``."""
    if case.owner is not None and case.owner == operator_id:
        return FIXED_OWNER_CAPABILITIES
    return case.requires


def setup_capabilities(case, operator_id=None):
    """Capabilities the case needs in order to build its inputs, never timed.

    Kept separate from :func:`required_capabilities` because the distinction is
    real: MOE-002 builds its forward graph in setup and only measures the backward,
    and COMM-011 dispatches before it times ``rmsnorm_combine``.  Both still need the
    forward/dispatch path to exist, so both gate the case.
    """
    if case.owner is not None and case.owner == operator_id:
        return ()
    return case.setup_requires


def _contract(spec, case, cfg, baseline):
    """The configuration an adapter constraint is evaluated against.

    A baseline run and the fixed-owner contract both measure the plain path, so
    the planner/dedup scenario state is neutralised for them instead of being
    reported as an adapter limitation.
    """
    fixed_owner = case.owner is not None and case.owner == spec.id
    if not (baseline or fixed_owner):
        return cfg
    return {**cfg, "dedup": {"mode": "none"}, "load_balance": {"enabled": False}}


def status_for(spec, case, cfg, baseline=False):
    """Return ``(status, reason)`` for one (operator, case, configuration).

    The decision is made per *case*: an adapter capability is only accepted when
    the case's own requirement is present, so a README that claims a level does
    not make every case of that level runnable.  All case-specific conditions are
    named applicability rules declared on the ``CaseSpec``; this function only
    applies them in order and then evaluates the generic adapter contract.
    """
    if spec.status not in EXECUTABLE_STATUSES:
        return spec.status, spec.reason
    context = ApplicabilityContext(spec=spec, config=cfg, baseline=baseline)
    for name in case.applicability:
        verdict = APPLICABILITY_RULES[name](case, context)
        if verdict:
            return verdict
    bounds = check_constraints(spec, _contract(spec, case, cfg, baseline))
    if bounds:
        return "unsupported", bounds
    missing = sorted(set(required_capabilities(case, spec.id)) - set(spec.capabilities))
    if missing:
        gaps = spec.constraints.get("unadapted", {})
        known = [gaps[name] for name in missing if name in gaps]
        if known:
            return "not_implemented", "; ".join(known)
        return "unsupported", f"operation requires {', '.join(missing)}"
    # A case also needs whatever it takes to *build* its inputs -- the routing, the
    # forward graph, the handles.  Those are never timed, but an adapter that lacks
    # them cannot reach the timed call at all, so they gate the same way.
    setup_missing = sorted({name for name in setup_capabilities(case, spec.id)}
                           - set(spec.capabilities))
    if setup_missing:
        gaps = spec.constraints.get("unadapted", {})
        known = [gaps[name] for name in setup_missing if name in gaps]
        if known:
            return "not_implemented", "; ".join(known)
        return ("unsupported",
                f"case preparation requires {', '.join(setup_missing)} "
                f"(setup only, not timed)")
    return "not_run", None


def prepare_case(config, case):
    """Apply the case's declared workload transform to one configuration."""
    if case.transform is None:
        return deepcopy(config)
    return TRANSFORMS[case.transform](deepcopy(config), case)


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
    canonical = canonical_specs()
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
                # A baseline run measures the reference composition only; which
                # cases those are is declared on the case, not decided here.
                selected = [case for case in selected
                            if case.level != "L2" or case.baseline_reference]
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
                if case.level == "L0" and case.category == "transport":
                    # `benchmark.transport.enabled` decides whether the mechanism
                    # may execute at all.  The case stays in the manifest as an
                    # explicit `not_applicable`; it never falls back to another
                    # mechanism and never produces executable parameter variants.
                    gate = transport_gate(case, settings)
                    if gate:
                        status, reason = gate
                # Parameters are capability independent: a point with an
                # unsupported adapter keeps the *same* variant expansion, so the
                # test scope never shrinks with the EP size.
                for params in task_parameters(case, op_id, settings, ep_size):
                    case_status, case_reason = status, reason
                    if params.get(TRANSPORT_DISABLED):
                        case_status, case_reason = "not_applicable", params.get("reason")
                    if case_reason is None:
                        # An impossible parameter set is a property of the plan
                        # itself, so it is reported before the hardware verdict.
                        problem = structural_problem(case, params, ep_size)
                        if problem:
                            case_status, case_reason = problem
                    if case.level == "L0" and case_reason is None:
                        # Hardware capability is checked before the topology
                        # constraints: an unsupported mechanism stays
                        # `unsupported` even when the peer policy is N/A too.
                        derived_status, derived_reason = _l0_status(case, params, matrix)
                        if derived_status and derived_status != "not_run":
                            case_status, case_reason = derived_status, derived_reason
                    if case_reason is None:
                        problem = policy_problem(case, params, ep_size, cfg)
                        if problem:
                            case_status, case_reason = problem
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
                        "case_tags": list(case.tags),
                        "case_phase": case.phase,
                        "handler": case.handler,
                        "correctness_requirements": list(case.correctness),
                        "alias_of": case.alias_of,
                        "config_fingerprint": fingerprint(
                            {key: value for key, value in cfg.items() if key != "suite_id"}),
                    }
    return list(tasks.values())


def merge_tasks(store, tasks):
    """Merge per-suite expansions, keeping one task per identical point.

    Two suites that produce the *same* work collapse into one task and keep the
    union of their ``suite_ids``/composition points, so de-duplication never loses
    which suites asked for the measurement.  Tasks whose configuration or
    parameters differ keep distinct task IDs because the identity is part of the
    hash: ``decode`` and ``prefill`` are different work, not a duplicate.
    """
    for task in tasks:
        key = task["task_id"]
        if key in store:
            previous = store[key]
            previous["suite_ids"] = sorted(set(previous["suite_ids"]) | set(task["suite_ids"]))
            previous["config_source"] = sorted(set(previous.get("config_source") or ()) |
                                               set(task.get("config_source") or ()))
            points = set(previous.get("plan_points") or ()) | set(task.get("plan_points") or ())
            if points:
                previous["plan_points"] = sorted(points)
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


def _variant_signature(task):
    """The (case, level, operator, variant) point a task represents."""
    return (task["case_id"], task["level"], task["backend_id"],
            task["parameters"].get("variant"))


def check_parity(tasks, ep_sizes=None):
    """Raise when two EP sizes disagree about the parameter matrix.

    Case presence is checked by :func:`check_completeness`; this check is about
    the *variants*: EP8, EP64 and EP128 must produce the same multiset of
    ``(case, level, operator, variant)`` points.  A case that is runnable at one
    size and unsupported at another is legitimate; a case that produces three
    parameter variants at one size and one at another is a scope change and is
    rejected here.
    """
    counts = {}
    for task in tasks:
        signature = _variant_signature(task)
        entry = counts.setdefault(task["ep_size"], {})
        entry[signature] = entry.get(signature, 0) + 1
    sizes = sorted(ep_sizes or counts)
    if len(sizes) < 2:
        return {size: len(counts.get(size, {})) for size in sizes}
    problems = []
    reference = sizes[0]
    for size in sizes[1:]:
        left, right = counts.get(reference, {}), counts.get(size, {})
        missing = sorted(set(left) - set(right))
        extra = sorted(set(right) - set(left))
        if missing:
            problems.append(f"EP{size} is missing {len(missing)} variant point(s) that "
                            f"EP{reference} has: {missing[:5]}")
        if extra:
            problems.append(f"EP{size} has {len(extra)} variant point(s) EP{reference} does "
                            f"not: {extra[:5]}")
        changed = sorted(key for key in set(left) & set(right) if left[key] != right[key])
        if changed:
            problems.append(f"EP{size} expands {len(changed)} point(s) a different number of "
                            f"times than EP{reference}: {[(key, left[key], right[key]) for key in changed[:5]]}")
    if problems:
        raise ValueError("benchmark matrix is not EP-size invariant: " + "; ".join(problems))
    return {size: len(counts[size]) for size in sizes}


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
    check_parity(tasks, sizes)
    return coverage
