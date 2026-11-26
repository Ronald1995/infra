"""Configuration loading, validation and scaling matrix expansion.

The resolver has one job: turn a stack of *raw* layers into one effective
configuration **and** a record of where every field came from.  Two rules make
that verifiable:

1. **Defaults are applied last, never first.**  Every layer is read raw (no
   schema defaults merged in), composed in priority order, and only the final
   configuration is validated.  A default can therefore never shadow an explicit
   value from a lower-priority file -- the class of bug that made an EP
   environment silently override the base measurement settings.
2. **A layer may not silently drop a field.**  A suite or a ``matrix`` point that
   declares an environment field (EP size aside, where it is a selector) is a
   configuration error with the field name and the file.  Before this change the
   resolver deleted those keys, so the file said one thing and the run did
   another.

Priority, lowest to highest::

    schema_default -> base_config -> EP environment -> suite -> matrix point -> CLI

``resolve_config`` returns ``(config, provenance)`` where ``provenance`` maps a
dotted field path to the layer/file that set its final value.  The plan writes
that map next to the task manifest, which is what lets an audit answer "why did
this task have ``cores=[2]``?" without re-reading every YAML file.
"""
from copy import deepcopy
from pathlib import Path

import yaml

LEVELS = ("L0", "L1", "L2", "L3", "L4")
ROUTINGS = (
    "balanced", "uniform_random", "expert_hotspot", "rank_hotspot",
    "server_hotspot", "high_locality", "cross_server_heavy", "dedup_controlled",
    "real_trace_replay", "dynamic_trace", "target_distribution",
)
DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2}
TIMINGS = ("output_ready", "host_api", "device", "async", "throughput", "cold_start")
from common.transport import IDS as L0_MECHANISMS  # noqa: E402  (single registry)
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
                  "dimension_policy": "all",
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


# ---------------------------------------------------------------------------
# Field schema: what a configuration file may declare
# ---------------------------------------------------------------------------
#: Fields that are legal but have no entry in ``DEFAULT`` because they are
#: optional switches consumed by one subsystem.  Anything outside ``DEFAULT`` and
#: this mapping is a configuration error instead of an ignored key.
OPTIONAL_FIELDS = {
    "benchmark": {
        "phase": True, "planner_steps": True, "native_hccl": True,
        "memory_library": True, "shmem_udma_library": True, "shmem_rendezvous": True,
        "verification": {"hccl_log_paths", "hcomm_log_paths"},
        "compute": {"fixed_shape"},
        # Optional switches inside a section the default already declares.
        "transport": {"peer_policies", "library"},
        "straggler": True,
        "dimension_policy": True,
        "execution": {"modes"},
    },
    "distributed": {"hierarchical_inner_size": True},
    "workload": {"expert_placement": True, "router_bias": True, "trace_path": True,
                 "preferred_route_ratio": True, "routing_control": True,
                 "hot_rank": True, "hot_expert": True},
    "profiling": {"enabled": True},
    "load_balance": {"redundant_experts": True},
    "topology": {"server_ids": True, "supernode_ids": True, "source": True,
                 "verified": True, "server_size": True},
    "scaling": {"name": True, "mode": True, "global_tokens": True, "num_experts": True,
                "tokens_per_rank": True, "experts_per_rank": True},
}
#: Sections that describe the EP environment.  A suite or a ``matrix`` point may
#: not set them: they belong to the environment layer, and ``distributed.ep_size``
#: is the one exception (it *routes* a scaling point to an EP size).
ENVIRONMENT_SECTIONS = frozenset({
    "schema_version", "extends", "matrix", "device", "distributed", "model",
    "topology", "launch",
})
#: ``matrix`` entries may vary a scenario.  ``model`` keeps only the expert count,
#: because strong/weak scaling are defined in terms of it.
MATRIX_MODEL_OVERRIDES = frozenset({"num_experts"})
#: Top-level keys accepted in a configuration file (``extends``/``matrix`` are
#: consumed by the loader, ``suite_id``/``operators``/``backends`` are selectors).
TOP_LEVEL_FIELDS = frozenset({
    "schema_version", "device", "suite_id", "distributed", "model", "workload",
    "benchmark", "topology", "profiling", "load_balance", "dedup", "operators",
    "backends", "scaling",
})


def _field_tree(default, extras, prefix=""):
    """Nested mapping of every allowed dotted field."""
    if isinstance(extras, (bool, type(None))):
        extras = {}
    if isinstance(extras, (set, frozenset)):
        extras = {name: {} for name in extras}
    tree = {}
    for key, value in default.items():
        if isinstance(value, dict):
            tree[key] = _field_tree(value, extras.get(key) or {})
        else:
            # ``None`` is the "scalar leaf" sentinel; ``{}`` means a free-form
            # mapping (the schema declares the section, not its children).
            tree[key] = None
    for key, value in extras.items():
        if isinstance(value, (set, frozenset)):
            value = {name: {} for name in value}
        if isinstance(value, dict):
            tree[key] = _field_tree(tree.get(key) or {}, value)
        else:
            tree.setdefault(key, {})
    return tree


FIELD_TREE = _field_tree(DEFAULT, OPTIONAL_FIELDS)


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
    from common import straggler as straggler_module

    straggler_module.validate(config)
    if w["routing"] == "target_distribution":
        from common.routing import STATISTICAL_METRICS, TARGET_METRICS

        control = w.get("routing_control")
        if not isinstance(control, dict) or not control:
            raise ValueError("target_distribution requires workload.routing_control")
        metric = control.get("target_metric")
        if metric in STATISTICAL_METRICS:
            raise ValueError(
                f"routing_control.target_metric {metric!r} is statistical only; this "
                f"generator controls {TARGET_METRICS}")
        if metric not in TARGET_METRICS:
            raise ValueError(f"routing_control.target_metric must be one of {TARGET_METRICS}")
        value = control.get("target_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("routing_control.target_value must be a non-negative number")
        if metric == "local_route_ratio" and value > 1:
            raise ValueError("routing_control.target_value for local_route_ratio must be <= 1")
        if metric != "local_route_ratio" and value == 0:
            raise ValueError("routing_control.target_value must be positive for max/mean targets")
        tolerance = (control.get("tolerance") or {}).get("relative", 0.02)
        if not 0 < float(tolerance) <= 1:
            raise ValueError("routing_control.tolerance.relative must be in (0, 1]")
        hotspot = control.get("hotspot") or {}
        if hotspot.get("mode", "fixed") not in ("fixed", "rotating"):
            raise ValueError("routing_control.hotspot.mode must be fixed or rotating")
        for key in ("on_infeasible", "on_target_miss"):
            allowed = (("not_applicable", "blocked")
                       if key == "on_infeasible" else ("failed", "not_applicable"))
            if control.get(key) is not None and control[key] not in allowed:
                raise ValueError(f"routing_control.{key} must be one of {allowed}")
    if b["timing"] not in TIMINGS or b["rank_reduction"] != "max":
        raise ValueError("invalid timing or rank reduction")
    if b.get("dimension_policy", "all") not in ("all", "first"):
        raise ValueError("benchmark.dimension_policy must be all or first")
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
#: (Declared above, next to the field schema.)


def flatten_paths(value, prefix=""):
    """Dotted paths of every leaf in a (nested) mapping."""
    if not isinstance(value, dict):
        yield prefix
        return
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        yield from flatten_paths(item, path)


def field_errors(config):
    """Unknown configuration fields, with the path that made them unknown.

    A typo such as ``benchmark.quiet_modes`` vs ``quiet_mode`` used to be
    ignored: the field was simply never read.  Now it fails validation and names
    the field, so a configuration cannot describe a dimension that the executor
    never sees.
    """
    errors = []

    def walk(node, tree, prefix):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).startswith("_"):
                continue
            if prefix == "" and key not in TOP_LEVEL_FIELDS:
                errors.append(f"unknown top-level field {key!r}")
                continue
            if key not in tree:
                errors.append(f"unknown configuration field {path!r}")
                continue
            schema = tree[key]
            if schema is None:
                # A scalar leaf cannot grow children.
                if isinstance(value, dict):
                    errors.append(f"configuration field {path!r} must not have child keys")
                continue
            if not isinstance(value, dict):
                if schema:
                    errors.append(f"configuration field {path!r} must be a mapping")
                continue
            # An empty schema node is a free-form mapping: any child is allowed.
            if schema:
                walk(value, schema, path)

    walk(config, FIELD_TREE, "")
    return errors


def validate_fields(config):
    errors = field_errors(config)
    if errors:
        raise ValueError("invalid configuration: " + "; ".join(errors))
    return config


def _own_fields(raw):
    """The fields one file declares itself (``extends``/``matrix`` excluded)."""
    return {key: value for key, value in raw.items()
            if key not in ("extends", "matrix")}


def load_raw(path, ancestry=()):
    """Read one configuration file without applying any schema default.

    Returns ``(raw, provenance)``: the merged raw mapping (following ``extends``)
    and a ``dotted path -> file name`` map naming the file that declared each
    final value.  Nothing is validated here; validation happens once, after every
    layer has been composed.
    """
    path = Path(path).resolve()
    if path in ancestry:
        raise ValueError(f"configuration inheritance cycle: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: configuration must be a YAML mapping")
    if "plan" in raw:
        raise ValueError(
            f"{path.name} is a benchmark plan; load it with --plan instead of --config")
    own = _own_fields(raw)
    parent = raw.get("extends")
    if parent:
        parent_raw, parent_provenance = load_raw(path.parent / parent, (*ancestry, path))
        merged = merge(parent_raw, own)
        provenance = dict(parent_provenance)
    else:
        merged, provenance = merge({}, own), {}
    provenance.update({field: path.name for field in flatten_paths(own)})
    if merged.get("backends") is not None:
        merged["operators"] = merged.pop("backends")
        provenance.pop("backends", None)
        provenance["operators"] = provenance.get("operators", path.name)
    workload = merged.get("workload") or {}
    if workload.get("routing") == "hot_expert":
        workload["routing"] = "expert_hotspot"
    if workload.get("trace_path"):
        workload["trace_path"] = str((path.parent / workload["trace_path"]).resolve())
        provenance["workload.trace_path"] = path.name
    return merged, provenance


def resolve_config(layers, schema_default=None):
    """Compose layers (lowest first) into ``(effective, provenance)``.

    ``layers`` is a sequence of ``(label, raw_mapping, provenance_mapping)``.
    The schema defaults are the lowest layer, so an explicit value always wins
    over a default; the returned provenance names ``schema_default`` for every
    field no layer set.
    """
    default = deepcopy(DEFAULT if schema_default is None else schema_default)
    provenance = {field: "schema_default" for field in flatten_paths(default)}
    effective = default
    for label, raw, layer_provenance in layers:
        if not raw:
            continue
        effective = merge(effective, raw)
        for field in flatten_paths(raw):
            provenance[field] = (layer_provenance or {}).get(field, label)
    return effective, provenance


def provenance_record(config, provenance):
    """The provenance a task/manifest carries: value + source per field."""
    record = {}
    for field in flatten_paths(config):
        node = config
        for part in field.split("."):
            node = node[part]
        if isinstance(node, dict):
            continue
        record[field] = {"value": node, "source": provenance.get(field, "unknown")}
    return record


def load_configs(path, ancestry=()):
    """Load one standalone configuration and its ``matrix`` points.

    Kept for ``--config`` runs: the file may declare a ``matrix`` list, and each
    point is resolved as its own layer (highest priority).  The effective
    configuration is validated exactly once, after the full composition.
    """
    path = Path(path).resolve()
    raw, provenance = load_raw(path, ancestry)
    raw_file = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    matrix = raw_file.get("matrix")
    if matrix is None and "matrix" in raw:
        matrix = raw["matrix"]
    matrix = matrix or [{}]
    if not isinstance(matrix, list) or not matrix or any(not isinstance(v, dict) for v in matrix):
        raise ValueError("matrix must be a nonempty list of configuration overrides")
    base_raw = {key: value for key, value in raw.items() if key != "matrix"}
    base_provenance = {field: source for field, source in provenance.items()}
    configs = []
    for item in matrix:
        layer_provenance = {field: f"{path.name} (matrix point)" for field in flatten_paths(item)}
        config, point_provenance = resolve_config([
            ("base_config", base_raw, base_provenance),
            ("matrix_point", item, layer_provenance),
        ])
        validate(config)
        validate_fields(config)
        config["_provenance"] = provenance_record(config, point_provenance)
        configs.append(config)
    return configs


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


def suite_scenario(raw, name=None):
    """Return the scenario sections of one suite configuration, strictly.

    A suite file describes *when* a case runs, never *where*: EP size, device,
    model shape, rank mapping and launch environment belong to the environment
    layer.  Declaring one here is an error with the field name and the file, not a
    silent deletion -- a dropped field means the file says one thing and the run
    does another.  ``model.num_experts`` is the one exception: strong/weak scaling
    are *defined* in terms of the expert count.
    """
    label = name or "suite configuration"
    errors = []
    for key, value in raw.items():
        if key == "model":
            errors.extend(f"model.{name}" for name in
                          sorted(set(value or {}) - MATRIX_MODEL_OVERRIDES))
        elif key in ENVIRONMENT_SECTIONS and key not in ("extends", "matrix", "schema_version"):
            errors.append(key)
    if errors:
        raise ValueError(
            f"{label} declares environment field(s) {errors}; a suite selects a scenario "
            f"only (EP size, device, model, topology and launch live in the EP environment "
            f"configuration, and model.num_experts is reserved for the scaling definition)")
    scenario = {key: deepcopy(value) for key, value in raw.items()
                if key not in ENVIRONMENT_SECTIONS}
    if raw.get("model"):
        # Only the scaling definition survives; the shape stays with the environment.
        scenario["model"] = {key: deepcopy(value) for key, value in raw["model"].items()
                             if key in MATRIX_MODEL_OVERRIDES}
    return scenario


def matrix_overlay(item, name=None):
    """Validate and sanitise one ``matrix`` entry.

    A ``matrix`` point varies a scenario (token count, dedup mode, planner phase).
    ``distributed.ep_size`` is accepted because it *routes* a scaling point to an
    EP size, and ``model.num_experts`` is accepted because strong/weak scaling are
    defined in terms of it.  Every other environment field is an error.
    """
    label = name or "matrix point"
    errors = []
    for key, value in item.items():
        if key == "distributed":
            extra = sorted(set(value or {}) - {"ep_size"})
            if extra:
                errors.append(f"distributed.{extra[0]}")
        elif key == "model":
            extra = sorted(set(value or {}) - MATRIX_MODEL_OVERRIDES)
            if extra:
                errors.append(f"model.{extra[0]}")
        elif key in ENVIRONMENT_SECTIONS:
            errors.append(key)
    if errors:
        raise ValueError(
            f"{label} varies the environment ({', '.join(errors)}); a matrix point may only "
            f"vary a scenario (distributed.ep_size is a selector, model.num_experts is the "
            f"scaling definition)")
    clean = {}
    for key, value in item.items():
        if key == "model":
            keep = {name: deepcopy(setting) for name, setting in (value or {}).items()
                    if name in MATRIX_MODEL_OVERRIDES}
            if keep:
                clean["model"] = keep
            continue
        if key in ENVIRONMENT_SECTIONS:
            continue
        if key == "backends":
            key = "operators"
        clean[key] = deepcopy(value)
    return clean


def load_suite_overlays(path, ep_size=None):
    """Return ``(overlay, provenance)`` layers declared by one suite configuration.

    Only the scenario sections survive (strictly: see :func:`suite_scenario`).  A
    ``matrix`` list becomes one layer per point, filtered to the requested EP size;
    the provenance names the suite file and, for a matrix point, the point itself.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: configuration must be a YAML mapping")
    if "plan" in raw:
        raise ValueError(
            f"{path.name} is a benchmark plan; register it as a suite with --plan, not as a "
            "suite_configs entry")
    matrix = raw.get("matrix") or [{}]
    if not isinstance(matrix, list) or any(not isinstance(item, dict) for item in matrix):
        raise ValueError(f"{path.name}: matrix must be a list of configuration overrides")
    base = suite_scenario(_own_fields(raw), path.name)
    workload = base.get("workload") or {}
    if workload.get("routing") == "hot_expert":
        workload["routing"] = "expert_hotspot"
    base_provenance = {field: path.name for field in flatten_paths(base)}
    points = []
    for index, item in enumerate(matrix):
        declared = (item.get("distributed") or {}).get("ep_size")
        if declared is not None and ep_size is not None and declared != ep_size:
            continue
        overlay = matrix_overlay(item, f"{path.name} matrix[{index}]")
        provenance = dict(base_provenance)
        provenance.update({field: f"{path.name} matrix[{index}]"
                           for field in flatten_paths(overlay)})
        points.append((merge(base, overlay), provenance))
    return points or [(base, base_provenance)]
