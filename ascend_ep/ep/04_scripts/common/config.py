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
L0_MECHANISMS = ("ub_memory", "urma", "ub_ctp", "ub_rtp", "roce", "hccs", "uboe",
                 "shmem_mte", "shmem_udma")
L0_ACCELERATORS = ("default", "host_ts", "aicpu_ts", "aiv", "aiv_only", "ccu_ms",
                   "ccu_sched", "aicpu", "ccu_ms_only", "ccu_sched_only")
PEER_POLICIES = ("ring", "same_server", "cross_server")
DEFAULT = {
    "schema_version": 1, "device": "npu", "suite_id": "baseline",
    "distributed": {"ep_size": 1, "tp_size": 1, "timeout_seconds": 180},
    "model": {"hidden_size": 128, "intermediate_size": 64, "num_experts": 8,
              "top_k": 2, "dtype": "bfloat16", "swiglu_limit": 10.0},
    "workload": {"tokens_per_rank": 32, "routing": "balanced", "seed": 42},
    "benchmark": {"warmup": 2, "iterations": 5, "levels": ["L0", "L1", "L2", "L3", "L4"],
                  "timing": "output_ready", "rank_reduction": "max",
                  "retries": 0, "task_timeout_seconds": 3600,
                  "message_bytes": [4096], "outstanding": [1], "peer_counts": [],
                  "cores": [1], "quiet_modes": [0], "transport_mode": "write",
                  # Every mechanism is enabled by default, which is the legacy
                  # "no filter" behaviour.  An explicit subset restricts the LINK
                  # cases; an explicit `[]` disables all transport mechanisms and
                  # makes the scheduler report every LINK case as `not_applicable`.
                  # `null` is still accepted and means the same as the default.
                  "transport": {"enabled": list(L0_MECHANISMS), "library": None,
                                "peer_policy": "ring"},
                  "execution": {"modes": ["default"]},
                  "compute": {"fixed_shape": {"m": 8192, "k": 7168, "n": 3072, "groups": 8}},
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


#: Sections that describe the EP environment rather than a suite scenario.  A
#: suite configuration may not override them when it is merged into a benchmark
#: plan, otherwise ``decode.yaml`` would silently drag EP8's model into EP64.
ENVIRONMENT_SECTIONS = frozenset({
    "schema_version", "extends", "matrix", "device", "distributed", "model",
    "topology", "launch",
})

#: ``matrix`` entries are meant to vary a *scenario* (token count, dedup mode,
#: planner phase).  When a plan composes them they may not smuggle the EP
#: environment back in, with one deliberate exception: strong/weak scaling is
#: defined in terms of the expert count, so ``model.num_experts`` stays legal.
MATRIX_MODEL_OVERRIDES = frozenset({"num_experts"})


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
    enabled = transport.get("enabled")
    # ``None`` keeps the legacy meaning ("no explicit subset, every mechanism is
    # allowed"); an explicit empty list is a real decision and disables transport.
    if enabled is not None and (not isinstance(enabled, list)
                               or set(enabled) - set(L0_MECHANISMS)):
        raise ValueError(f"benchmark.transport.enabled must be null or a subset of {L0_MECHANISMS}")
    execution = settings.get("execution") or {}
    modes = execution.get("modes") or []
    if not isinstance(modes, list) or set(modes) - set(L0_ACCELERATORS):
        raise ValueError(f"benchmark.execution.modes must be a subset of {L0_ACCELERATORS}")
    if settings.get("transport_mode") not in ("write", "read", "get", "put"):
        raise ValueError("benchmark.transport_mode must be write, read, put or get")
    policy = transport.get("peer_policy", "ring")
    if policy not in PEER_POLICIES:
        raise ValueError(f"benchmark.transport.peer_policy must be one of {PEER_POLICIES}")
    policies = transport.get("peer_policies", [])
    if not isinstance(policies, list) or set(policies) - set(PEER_POLICIES):
        raise ValueError(f"benchmark.transport.peer_policies must be a subset of {PEER_POLICIES}")
    shape = (settings.get("compute") or {}).get("fixed_shape") or {}
    for key in ("m", "k", "n", "groups"):
        if type(shape.get(key)) is not int or shape[key] < 1:
            raise ValueError(f"benchmark.compute.fixed_shape.{key} must be a positive integer")
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
    if "plan" in raw:
        raise ValueError(
            f"{path.name} is a benchmark plan; load it with --plan instead of --config")
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


def suite_scenario(raw):
    """Strip the EP environment from one suite configuration.

    A suite file is written as a complete configuration (``extends: ep8_bf16.yaml``)
    so it can still be used with ``--config``.  A benchmark plan only takes its
    scenario sections; EP size, TP size, device, model shape, rank mapping and
    launch environment stay owned by the EP size configuration.
    """
    return {key: deepcopy(value) for key, value in raw.items()
            if key not in ENVIRONMENT_SECTIONS}


def matrix_overlay(item):
    """Sanitise one ``matrix`` entry before it is merged into a plan point.

    ``matrix`` entries vary a scenario, not an environment.  ``distributed`` is
    read only to route the entry to one EP size (see
    :func:`load_suite_overlays`); ``device``, ``topology`` and ``launch`` are
    dropped, and ``model`` keeps nothing but the expert count that strong/weak
    scaling is defined by.
    """
    clean = {}
    for key, value in item.items():
        if key == "model" and isinstance(value, dict):
            value = {name: deepcopy(setting) for name, setting in value.items()
                     if name in MATRIX_MODEL_OVERRIDES}
            if not value:
                continue
        elif key in ENVIRONMENT_SECTIONS:
            continue
        if key == "backends":
            key = "operators"
        clean[key] = deepcopy(value)
    return clean


def load_suite_overlays(path, ep_size=None):
    """Return the suite-scoped layers declared by one configuration file.

    Only the *scenario* sections survive: EP size, device, model and topology
    belong to the EP size configuration and are never taken from a suite file,
    not even through a ``matrix`` entry.  A ``matrix`` list is expanded into one
    layer per point, and scaling matrices are filtered to the requested EP size
    (``distributed.ep_size`` is a selector here, not an override).
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    if "plan" in raw:
        raise ValueError(
            f"{path.name} is a benchmark plan; register it as a suite with --plan, not as a "
            "suite_configs entry")
    raw.pop("extends", None)
    matrix = raw.pop("matrix", [{}]) or [{}]
    if not isinstance(matrix, list) or any(not isinstance(item, dict) for item in matrix):
        raise ValueError("matrix must be a list of configuration overrides")
    base = suite_scenario(raw)
    if "backends" in base:
        base["operators"] = base.pop("backends")
    if base.get("workload", {}).get("routing") == "hot_expert":
        base["workload"]["routing"] = "expert_hotspot"
    points = []
    for item in matrix:
        declared = (item.get("distributed") or {}).get("ep_size")
        if declared is not None and ep_size is not None and declared != ep_size:
            continue
        points.append(merge(base, matrix_overlay(item)))
    return points or [base]
