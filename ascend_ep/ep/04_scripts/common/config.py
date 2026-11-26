"""Configuration loading, validation and scaling matrix expansion."""
from copy import deepcopy
from pathlib import Path

import yaml

LEVELS = ("L0", "L1", "L2", "L3", "L4")
ROUTINGS = (
    "balanced", "uniform_random", "expert_hotspot", "rank_hotspot",
    "server_hotspot", "high_locality", "cross_server_heavy", "dedup_controlled",
    "real_trace_replay", "dynamic_trace",
)
DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2}
TIMINGS = ("output_ready", "host_api", "device", "async", "throughput", "cold_start")
DEFAULT = {
    "schema_version": 1, "device": "npu", "suite_id": "baseline",
    "distributed": {"ep_size": 1, "tp_size": 1, "timeout_seconds": 180},
    "model": {"hidden_size": 128, "intermediate_size": 64, "num_experts": 8,
              "top_k": 2, "dtype": "bfloat16", "swiglu_limit": 10.0},
    "workload": {"tokens_per_rank": 32, "routing": "balanced", "seed": 42},
    "benchmark": {"warmup": 2, "iterations": 5, "levels": ["L1", "L2", "L3"],
                  "timing": "output_ready", "rank_reduction": "max",
                  "retries": 0, "task_timeout_seconds": 3600,
                  "message_bytes": [4096], "outstanding": [1], "peer_counts": [],
                  "cores": [1], "quiet_modes": [0], "transport_mode": "write",
                  "transport": {"enabled": [], "library": None},
                  "execution": {"modes": ["default"]},
                  "verification": {"enabled": True, "require_verified_for_comparison": True,
                                   "trust_descriptor": False, "hccl_log_paths": [],
                                   "hcomm_log_paths": []},
                  "native_hccl": {}, "memory_library": None,
                  "shmem_udma_library": None, "shmem_rendezvous": None},
    "topology": {}, "profiling": {"warmup": 1, "active": 3},
    "load_balance": {"enabled": False}, "dedup": {"mode": "none"},
    "operators": ["alltoall_hccl"],
}


def merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def require_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate(config):
    d, m, w, b = (config[key] for key in ("distributed", "model", "workload", "benchmark"))
    if config["schema_version"] != 1 or config["device"] not in ("cpu", "npu"):
        raise ValueError("unsupported configuration schema or device")
    for section, keys in ((d, ("ep_size", "tp_size", "timeout_seconds")),
                          (m, ("hidden_size", "intermediate_size", "num_experts", "top_k")),
                          (b, ("iterations", "task_timeout_seconds")),
                          (config["profiling"], ("active",))):
        for key in keys:
            require_int(section[key], key)
    for key in ("warmup", "retries"):
        require_int(b[key], key, 0)
    if "planner_steps" in b:
        require_int(b["planner_steps"], "benchmark.planner_steps", 2)
    if "hierarchical_inner_size" in d:
        require_int(d["hierarchical_inner_size"], "distributed.hierarchical_inner_size", 2)
    require_int(config["profiling"]["warmup"], "profiling.warmup", 0)
    require_int(w["seed"], "seed", 0)
    if m["top_k"] > m["num_experts"] or m["dtype"] not in DTYPES:
        raise ValueError("invalid top_k or dtype")
    if not 0 < m["swiglu_limit"] < float("inf"):
        raise ValueError("swiglu_limit must be finite and positive")
    tokens = w["tokens_per_rank"]
    if not isinstance(tokens, list):
        tokens = [tokens] * d["ep_size"]
    if len(tokens) != d["ep_size"]:
        raise ValueError("tokens_per_rank must contain one count per EP rank")
    for count in tokens:
        require_int(count, "tokens_per_rank", 0)
    if not sum(tokens):
        raise ValueError("the EP group must have at least one source token")
    owner = w.get("expert_placement")
    if owner is None and m["num_experts"] % d["ep_size"]:
        raise ValueError("num_experts must divide EP evenly with default placement")
    if owner is not None and (len(owner) != m["num_experts"] or any(
        type(rank) is not int or not 0 <= rank < d["ep_size"] for rank in owner
    )):
        raise ValueError("expert_placement must map every expert to an EP rank")
    for key in ("server_ids", "supernode_ids"):
        ids = config["topology"].get(key)
        if ids is not None and (len(ids) != d["ep_size"] or any(
            type(domain) not in (str, int) for domain in ids
        )):
            raise ValueError(f"{key} must contain one domain identifier per EP rank")
    if w["routing"] not in ROUTINGS:
        raise ValueError(f"unknown routing strategy {w['routing']!r}")
    router_bias = w.get("router_bias")
    if router_bias not in (None, "rotating_hot_expert", "remote_rank"):
        raise ValueError("unknown router logit bias")
    if router_bias == "remote_rank" and len(set(config["topology"].get("server_ids", ()))) < 2:
        raise ValueError("remote-rank router logit bias requires at least two server domains")
    if w["routing"] == "real_trace_replay" and not w.get("trace_path"):
        raise ValueError("real_trace_replay requires workload.trace_path")
    if b["timing"] not in TIMINGS or b["rank_reduction"] != "max":
        raise ValueError("invalid timing or rank reduction")
    if not b["levels"] or set(b["levels"]) - set(LEVELS):
        raise ValueError("benchmark.levels must contain L0 through L4")
    if b.get("phase", "train") not in ("train", "forward"):
        raise ValueError("benchmark.phase must be train or forward")
    if config["dedup"]["mode"] not in ("none", "rank", "server"):
        raise ValueError("invalid dedup mode")
    validate_l0(b)
    return config


L0_MECHANISMS = ("ub_memory", "urma", "ub_ctp", "ub_rtp", "roce", "hccs", "uboe",
                 "shmem_mte", "shmem_udma")
L0_ACCELERATORS = ("default", "host_ts", "aicpu_ts", "aiv", "aiv_only", "ccu_ms",
                   "ccu_sched", "aicpu", "ccu_ms_only", "ccu_sched_only")


def validate_l0(settings):
    """Validate the L0 dimension sweeps without changing older configurations."""
    for name in ("message_bytes", "outstanding", "cores", "quiet_modes", "peer_counts"):
        values = settings.get(name, [])
        if not isinstance(values, list) or any(type(value) is not int or value < 0
                                               for value in values):
            raise ValueError(f"benchmark.{name} must be a list of non-negative integers")
    if not settings.get("message_bytes"):
        raise ValueError("benchmark.message_bytes must not be empty")
    if any(value <= 0 for value in settings["message_bytes"]):
        raise ValueError("benchmark.message_bytes entries must be positive")
    if not settings.get("outstanding"):
        raise ValueError("benchmark.outstanding must not be empty")
    if any(value <= 0 for value in settings["outstanding"]):
        raise ValueError("benchmark.outstanding entries must be positive")
    transport = settings.get("transport") or {}
    enabled = transport.get("enabled") or []
    if not isinstance(enabled, list) or set(enabled) - set(L0_MECHANISMS):
        raise ValueError(f"benchmark.transport.enabled must be a subset of {L0_MECHANISMS}")
    execution = settings.get("execution") or {}
    modes = execution.get("modes") or []
    if not isinstance(modes, list) or set(modes) - set(L0_ACCELERATORS):
        raise ValueError(f"benchmark.execution.modes must be a subset of {L0_ACCELERATORS}")
    if settings.get("transport_mode") not in ("write", "read", "get", "put"):
        raise ValueError("benchmark.transport_mode must be write, read, put or get")
    verification = settings.get("verification") or {}
    for key in ("enabled", "require_verified_for_comparison", "trust_descriptor"):
        if not isinstance(verification.get(key, False), bool):
            raise ValueError(f"benchmark.verification.{key} must be a boolean")
    for key in ("hccl_log_paths", "hcomm_log_paths"):
        paths = verification.get(key, [])
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError(f"benchmark.verification.{key} must be a list of paths")


def load_configs(path, ancestry=()):
    path = Path(path).resolve()
    if path in ancestry:
        raise ValueError(f"configuration inheritance cycle: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    parent = raw.pop("extends", None)
    matrix = raw.pop("matrix", [{}])
    if parent:
        base = load_configs(path.parent / parent, (*ancestry, path))
        if len(base) != 1:
            raise ValueError("extends must reference exactly one configuration")
        raw = merge(base[0], raw)
    if "backends" in raw:
        raw["operators"] = raw.pop("backends")
    if raw.get("workload", {}).get("routing") == "hot_expert":
        raw["workload"]["routing"] = "expert_hotspot"
    if raw.get("workload", {}).get("trace_path"):
        raw["workload"]["trace_path"] = str((path.parent / raw["workload"]["trace_path"]).resolve())
    if not isinstance(matrix, list) or not matrix or any(not isinstance(v, dict) for v in matrix):
        raise ValueError("matrix must be a nonempty list of configuration overrides")
    return [validate(merge(merge(DEFAULT, raw), item)) for item in matrix]
