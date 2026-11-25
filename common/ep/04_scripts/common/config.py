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
                  "retries": 0, "task_timeout_seconds": 3600},
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
    return config


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
