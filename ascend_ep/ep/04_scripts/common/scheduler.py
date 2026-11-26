"""Construct a bounded, dependency-free benchmark manifest."""
from copy import deepcopy

from common.cases import L0_PARAMETERS, catalog, l0_family
from common.registry import check_constraints, registry
from common.storage import fingerprint

SUITE_CASES = {
    "training_tail": {"COMM-007", "MOE-003"},
    "high_throughput": {"COMM-008", "MOE-003"},
    "weak_scaling": {"COMM-001", "COMM-003", "MOE-003"},
    "strong_scaling": {"COMM-001", "COMM-003", "MOE-003"},
    "topology": {"COMM-001", "COMM-003", "MOE-001"},
    "load_balance": {"COMM-012", "MOE-005", "LAYER-003", "LAYER-004", "LAYER-008"},
    "dedup": {"COMM-010", "LAYER-005", "LAYER-006"},
    "overlap": {"HW-012", "FUSE-001", "FUSE-002", "FUSE-003", "FUSE-004",
                "FUSE-005", "FUSE-006", "FUSE-007", "FUSE-009", "MOE-007"},
    "quantized": {"COMM-009", "FUSE-008", "MOE-006"},
    "prefill": {"MOE-001", "LAYER-001"},
    "decode": {"MOE-001", "LAYER-001"},
}
L0_VARIANTS = {
    "HW-006": ("allgather", "reduce_scatter"),
    "HW-007": ("mte",), "HW-008": ("urma", "udma"),
    "HW-010": ("gmm1", "gmm2", "gmm1_dgrad", "gmm1_wgrad", "gmm2_dgrad", "gmm2_wgrad"),
    "HW-011": ("forward", "backward"),
}
CASE_VARIANTS = {
    **L0_VARIANTS, "LAYER-003": ("fresh_plan", "cached_plan"),
    "FUSE-005": ("chunks_2", "chunks_4", "chunks_8"),
    "FUSE-006": ("tile_128_256_256", "tile_256_128_256", "tile_256_256_256"),
    "FUSE-007": ("copy_cores_1", "copy_cores_2", "copy_cores_4", "copy_cores_8"),
}
FIXED_OWNER = "fixed_owner"
FIXED_OWNER_CASES = {"LAYER-003", "LAYER-005", "LAYER-006"}



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
    """Return ``(status, reason)`` for one L0 parameter set from the matrix."""
    if matrix is None:
        return None, None
    from baselines.l0.topology import capabilities as capability_table
    if case.category == "transport":
        mechanism = L0_PARAMETERS.get(case.id, {}).get("mechanism")
        resource = params.get("execution_resource") or L0_PARAMETERS.get(case.id, {}).get(
            "execution_resource")
        supported, reason = capability_table.support(matrix, case_id=case.id,
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


def l0_parameter_sets(case, cfg, size, variant, matrix=None):
    """Expand one L0 case into parameter dictionaries."""
    settings = cfg["benchmark"]
    family = l0_family(case)
    static = dict(L0_PARAMETERS.get(case.id, {}))
    base = {"variant": variant, "message_bytes": size}
    if family == "transport":
        outstanding = settings.get("outstanding") or [1]
        peers = settings.get("peer_counts") or [0]
        if case.id == "HW-008":
            return [{"variant": engine, "message_bytes": size} for engine in ("mte", "udma", "urma")]
        if case.id in ("LINK-012", "LINK-013"):
            variants = ("default",) if case.id == "LINK-012" else ("device", "host")
            return [{**base, **static, "outstanding": count, "peer_count": peer_count,
                     "variant": item, "cores": settings.get("cores", [1])[-1],
                     "quiet_mode": (settings.get("quiet_modes") or [0])[-1],
                     "mode": settings.get("transport_mode", "write")}
                    for item in variants
                    for count in outstanding for peer_count in peers]
        return [{**base, **static, "outstanding": count, "peer_count": peer_count,
                 "mode": settings.get("transport_mode", "write")}
                for count in outstanding for peer_count in peers]
    if family == "execution":
        modes = settings.get("execution", {}).get("modes") or ["default"]
        return [{**base, **static, "accelerator": mode, "operation": static.get("operation",
                                                                                 case.operation)}
                for mode in modes]
    if family == "overlap":
        return [{**base, **static, "transport": static.get("mechanism", "auto"),
                 "compute": static.get("compute", "gmm")}]
    if family == "memory":
        return [{**base, "variant": "default"}]
    return [{**base}]

def status_for(spec, case, cfg, baseline=False):
    if spec.status not in ("integrated", "historical_verified"):
        return spec.status, spec.reason
    if case.id == "MOE-007" and (cfg["device"] != "npu" or cfg["distributed"]["ep_size"] < 2):
        return "unsupported", "concurrent MoE requires at least two HCCL ranks"
    if case.id == "COMM-010":
        mode = cfg["dedup"]["mode"]
        if mode == "none":
            return "not_run", "choose a dedicated rank or server dedup configuration"
        if f"{mode}_dedup" not in spec.capabilities:
            return "unsupported", f"no {mode} dedup path"
    layer_dedup = {"LAYER-005": "rank", "LAYER-006": "server"}.get(case.id)
    if layer_dedup is not None and layer_dedup != cfg["dedup"]["mode"]:
        return "not_run", f"select the {layer_dedup} dedup mode for this training case"
    if case.id == "LAYER-007" and cfg["workload"]["routing"] != "real_trace_replay":
        return "not_run", "L3 replay requires a real_trace_replay workload"
    if case.id in {"COMM-012", "LAYER-003", "LAYER-004", "LAYER-008"} and not cfg["load_balance"]["enabled"]:
        return "not_run", "choose a load_balance configuration that enables the planner"
    if case.level == "L4" and cfg["workload"]["routing"] == "real_trace_replay":
        return "not_run", "replay is measured in LAYER-007 with the L3 boundary"
    if cfg["benchmark"].get("phase") == "forward" and case.training:
        return "not_run", "forward-only configuration"
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


def make_plan(configs, operators=None, levels=None, cases=None, baselines=False):
    specs, all_cases = registry(), catalog()
    if cases and set(cases) - set(all_cases):
        raise ValueError(f"unknown cases: {', '.join(sorted(set(cases) - set(all_cases)))}")
    tasks = {}
    for config in configs:
        ids = operators or config["operators"]
        if ids == ["all"]:
            ids = list(specs)
        if baselines:
            ids = ["alltoall_hccl", "allgather_hccl"]
        unknown = set(ids) - set(specs)
        if unknown:
            raise ValueError(f"unknown operators: {', '.join(sorted(unknown))}")
        suite_cases = None if baselines else SUITE_CASES.get(config["suite_id"])
        selected_levels = levels or (["L0", "L1", "L2"] if baselines else config["benchmark"]["levels"])
        for case in all_cases.values():
            if case.level not in selected_levels and (case.id != "LAYER-007" or "L4" not in selected_levels):
                continue
            if (cases or suite_cases) and case.id not in (cases or suite_cases):
                continue
            if baselines and case.level == "L2" and case.id not in {"FUSE-001", "FUSE-003"} and not cases:
                continue
            if baselines and case.level not in {"L0", "L1", "L2"}:
                continue
            cfg = prepare_case(config, case)
            for op_id in ["hardware"] if case.level == "L0" else ids:
                spec = specs.get(op_id)
                status, reason = status_for(spec, case, cfg, baselines) if spec else ("not_run", None)
                message_sizes = cfg["benchmark"].get("message_bytes", [4096])
                if case.id in {"HW-010", "HW-011"} or spec:
                    message_sizes = [None]
                matrix = _l0_matrix(cfg, [case]) if case.level == "L0" else None
                parameter_sets = []
                if case.level == "L0":
                    if reason is None:
                        for size in message_sizes:
                            variants = CASE_VARIANTS.get(case.id, ("default",))
                            for variant in variants:
                                parameter_sets.extend(
                                    l0_parameter_sets(case, cfg, size, variant, matrix))
                    else:
                        parameter_sets = [{"variant": "default", "message_bytes": message_sizes[0]}]
                else:
                    for size in message_sizes:
                        variants = CASE_VARIANTS.get(case.id, ("default",)) if reason is None else ("default",)
                        if case.id in FIXED_OWNER_CASES and op_id == "alltoall_hccl" and reason is None:
                            variants = (FIXED_OWNER,)
                        for variant in variants:
                            parameter_sets.append({"variant": variant, "message_bytes": size})
                for params in parameter_sets:
                    case_status, case_reason = status, reason
                    if case.level == "L0" and reason is None:
                        derived_status, derived_reason = _l0_status(case, params, matrix)
                        if derived_status:
                            case_status, case_reason = derived_status, derived_reason
                    identity = dict(config=cfg, backend=op_id, case=case.id, parameters=params)
                    task_id = f"{case.id}-{fingerprint(identity)[:16]}"
                    tasks[task_id] = {
                        "task_id": task_id, "case_id": case.id, "level": case.level,
                        "operation": case.operation, "backend_id": op_id, "config": cfg,
                        "parameters": params, "status": case_status, "reason": case_reason,
                        "source_versions": {"adapter_source": spec.source, "adapter_commit": spec.commit}
                        if spec else {},
                        "baseline": baselines or not spec,
                    }
    return list(tasks.values())
