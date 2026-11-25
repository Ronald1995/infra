"""Read the operator catalog without importing any device libraries."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ROOT / "operators"
CAPABILITIES = (
    "dispatch", "combine", "cached_dispatch", "async_dispatch", "async_combine",
    "moe_forward", "moe_backward", "router_weight_grad", "expert_weight_grad",
    "dispatch_gmm", "gmm_combine", "chunk_pipeline", "tile_schedule", "comm_resources",
    "quantized_dispatch", "quantized_fusion", "fused_backward", "rank_dedup",
    "server_dedup", "dynamic_planner", "recompute", "concurrent", "router_layer", "combine_add_rmsnorm",
    "dynamic_shapes", "planned_dispatch",
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

    def create(self):
        module = import_module(f"operators.{self.id}.backend")
        return module.create(self)


def registry():
    data = yaml.safe_load((OPERATORS / "registry.yaml").read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("unsupported operator registry version")
    operators = {}
    for operator_id, entry in data["operators"].items():
        spec = Operator(id=operator_id, **{
            **entry, "capabilities": tuple(entry.get("capabilities", ())),
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
    routes = limits.get("routing")
    if routes and w["routing"] not in routes:
        return f"adapter requires routing from {routes}"
    if config["dedup"]["mode"] != "none" and f"{config['dedup']['mode']}_dedup" not in spec.capabilities:
        return "requested dedup mode has no adapter implementation"
    if config["load_balance"]["enabled"] and not set(spec.capabilities) & {"dynamic_planner", "planned_dispatch"}:
        return "requested planner has no adapter implementation"
    return None
