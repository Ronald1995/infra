"""Read the operator catalog without importing any device libraries."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ROOT / "operators"
ENVIRONMENT_MAP = ROOT / "dependencies" / "environments" / "map.yaml"
ENVIRONMENTS = ROOT / "dependencies" / "environments"
CAPABILITIES = (
    "dispatch", "combine", "cached_dispatch", "async_dispatch", "async_combine",
    "moe_forward", "moe_backward", "router_weight_grad", "expert_weight_grad",
    "dispatch_gmm", "gmm_combine", "chunk_pipeline", "tile_schedule", "comm_resources",
    "quantized_dispatch", "quantized_fusion", "fused_backward", "rank_dedup",
    "server_dedup", "dynamic_planner", "recompute", "concurrent", "router_layer", "combine_add_rmsnorm",
    "dynamic_shapes", "planned_dispatch", "dedup_training",
)


@dataclass(frozen=True)
class Operator:
    id: str
    status: str
    source: str
    commit: str | None
    api: str
    capabilities: tuple[str, ...]
    constraints: dict
    reason: str = ""
    traffic_model: str | None = None
    upstream: str | None = None
    dependencies: tuple[str, ...] = ()
    #: Which isolated dependency environment runs this operator
    #: (``dependencies/environments/``).  Two dependency families that install the same
    #: module name cannot share an interpreter, so the environment is part of the
    #: operator's contract, not an installation detail.
    environment: str = "base"

    def create(self):
        module = import_module(f"operators.{self.id}.backend")
        return module.create(self)


def environments():
    """The declared dependency environments, keyed by id."""
    data = {}
    for path in sorted(ENVIRONMENTS.glob("*.yaml")):
        if path.name == "map.yaml":
            continue
        entry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        identifier = entry.get("id") or path.stem
        data[identifier] = {**entry, "id": identifier, "file": path.name}
    return data


def environment_assignments():
    """``(default, {operator: environment})`` from ``environments/map.yaml``."""
    data = yaml.safe_load(ENVIRONMENT_MAP.read_text(encoding="utf-8")) or {}
    if data.get("schema_version") != 1:
        raise ValueError("unsupported dependency environment map version")
    return data.get("default") or "base", dict(data.get("operators") or {})


def registry():
    data = yaml.safe_load((OPERATORS / "registry.yaml").read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("unsupported operator registry version")
    declared = environments()
    default, assigned = environment_assignments()
    operators = {}
    for operator_id, entry in data["operators"].items():
        environment = entry.get("environment") or assigned.get(operator_id) or default
        if declared and environment not in declared:
            raise ValueError(f"{operator_id}: unknown dependency environment {environment!r}")
        spec = Operator(id=operator_id, **{
            **entry, "environment": environment,
            "capabilities": tuple(entry.get("capabilities", ())),
            "dependencies": tuple(entry.get("dependencies", ())),
        })
        if set(spec.capabilities) - set(CAPABILITIES):
            raise ValueError(f"{operator_id}: invalid capability")
        operators[operator_id] = spec
    return operators


def check_constraints(spec, config):
    d, w, m = (config[k] for k in ("distributed", "workload", "model"))
    limits = spec.constraints
    if config["device"] not in limits.get("devices", ["npu"]):
        return f"{spec.id} requires {limits.get('devices', ['npu'])}"
    if limits.get("hierarchical"):
        inner = d.get("hierarchical_inner_size", 0)
        if type(inner) is not int or inner < 2 or d["ep_size"] % inner or d["ep_size"] // inner < 2:
            return "native hierarchy requires at least two outer ranks, each with at least two inner ranks"
        if not config["topology"].get("server_ids"):
            return "native hierarchy requires an explicit rank-to-server map"
    if m["dtype"] not in limits.get("dtypes", [m["dtype"]]):
        return f"dtype {m['dtype']} outside adapter's supported dtypes {limits['dtypes']}"
    sizes = limits.get("ep_sizes")
    if sizes and d["ep_size"] not in sizes:
        return f"EP{d['ep_size']} outside adapter's supported EP sizes {sizes}"
    for key in ("top_k", "num_experts", "hidden_size", "swiglu_limit"):
        if key in limits and m[key] != limits[key]:
            return f"{key}={m[key]} differs from adapter constraint {limits[key]}"
    tokens = w["tokens_per_rank"]
    counts = tokens if isinstance(tokens, list) else [tokens]
    if min(counts) < limits.get("min_tokens", 0):
        return "zero-token ranks are outside this adapter's contract"
    if limits.get("equal_tokens") and len(set(counts)) != 1:
        return "adapter requires equal tokens per rank"
    if max(counts) > limits.get("max_tokens", float("inf")):
        return f"tokens per rank exceeds adapter capacity {limits['max_tokens']}"
    if limits.get("contiguous_placement") and w.get("expert_placement") is not None:
        return "adapter requires default contiguous expert placement"
    for key in ("hidden_size", "top_k", "num_experts"):
        lower, upper = limits.get(f"{key}_range", (None, None))
        if lower is not None and not lower <= m[key] <= upper:
            return f"{key}={m[key]} is outside the inclusive range [{lower}, {upper}]"
    routes = limits.get("routing")
    if routes and w["routing"] not in routes:
        return f"adapter requires routing from {routes}"
    if config["dedup"]["mode"] != "none" and f"{config['dedup']['mode']}_dedup" not in spec.capabilities:
        return "requested dedup mode has no adapter implementation"
    if config["load_balance"]["enabled"] and not set(spec.capabilities) & {"dynamic_planner", "planned_dispatch"}:
        return "requested planner has no adapter implementation"
    return None
