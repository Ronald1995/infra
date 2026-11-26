"""The one authoritative definition of every benchmark case.

Before this module the same facts lived in four places and drifted apart:

``common/cases.py``
    case ID, level, operation and capability requirements;
``common/variants.py``
    named parameter variants (``CASE_VARIANTS``) and per-case size windows;
``common/scheduler.py``
    per-case ``if case.id == ...`` status rules and the fixed-owner exception;
``benchmarks/operator_cases.py``
    which operation name maps to which timed block.

A :class:`CaseSpec` now carries all of it.  The layers keep separate jobs, and the
separation is enforced by the type:

* **catalog** (this module) owns identity, capability requirement, parameter
  variants, execution handler name and applicability rules.  It defines *what a
  case is*.
* **suites** (:mod:`common.suites`) only *select* cases.  A suite never redefines
  one, and a case that belongs to no suite is still reachable through ``all``.
* **variants** (:mod:`common.variants`) turn ``spec.variants`` into parameter
  dictionaries.  It decides *with which parameters* a case runs, never whether a
  case means something else.
* **handlers** (:mod:`benchmarks.cases`) execute ``spec.handler``.  Adding a case
  is: add one row here, add one handler function, register it in
  ``benchmarks/cases/__init__.py``.  No other module needs an ``if``/``elif``.

Applicability rules are named and registered here instead of being a chain of
``if case.id == ...`` inside the scheduler: a rule is a pure function of
(``CaseSpec``, :class:`ApplicabilityContext`) returning ``(status, reason)`` or
``None``.  A spec lists its rules in *precedence order*, which is what makes a
status reproducible.

Nothing in this module imports torch, a device library or a baseline module: a
manifest must be generable on a CPU-only host.
"""
from dataclasses import dataclass, replace
from typing import Callable

LEVELS = ("L0", "L1", "L2", "L3", "L4")

CATEGORIES = ("memory", "transport", "execution", "collective", "compute", "overlap",
              "operator")

#: Phases a case is meaningful for.  ``both`` means the measurement is the same
#: code path in either phase (a dispatch is a dispatch), which is why the
#: forward-only configuration filter only drops ``training`` cases.
PHASES = ("inference", "training", "both")

#: What a case has to prove before its result may be reported as correct.  A
#: handler verifies these tokens; the report can then say *what* was checked
#: instead of printing a bare `passed`.
CORRECTNESS_TOKENS = (
    "dispatch_rows", "dispatch_counts", "combine_output", "quantized_dequant",
    "dedup_route", "planned_dispatch", "moe_forward_output", "input_gradient",
    "expert_weight_gradient", "router_weight_gradient", "planner_stages",
    "recompute_gradient", "multilayer_output", "concurrent_pair", "gmm_output",
    "fused_output", "payload_bytes", "collective_payload", "byte_exact",
    "element_exact", "compute_output",
)

#: Fixed-owner parameter variant: the case is only ever measured on that adapter,
#: so its parameter set carries the owner name instead of an adapter sweep.
FIXED_OWNER = "fixed_owner"

#: Dimension sweeps a case may opt into.  Declared on the spec so the variant
#: layer never has to recognise a case ID to know what it sweeps.
SWEEPS = ("message_bytes", "outstanding", "peer_count", "peer_policy", "accelerator",
          "variant")

#: Message-size windows.  A case names the one that makes sense for it: a latency
#: case is not swept over 1 GiB and a throughput case is not swept over 4 KiB.
SIZE_WINDOWS = ("latency", "throughput", "memory", "compute", "overlap")


@dataclass(frozen=True)
class CaseSpec:
    """Everything that defines one benchmark case."""

    id: str
    level: str
    operation: str
    category: str = "operator"
    #: Adapter capabilities required before any operator may run this case.
    requires: tuple[str, ...] = ()
    #: Parameter variants.  ``("default",)`` means "one parameter set".
    variants: tuple[str, ...] = ("default",)
    #: Name of the execution handler, resolved by ``benchmarks.cases``.
    handler: str = ""
    #: Which dimensions this case sweeps over (documentation + reports + the
    #: variant generator).
    sweep: tuple[str, ...] = ()
    #: Message-size window name, see ``SIZE_WINDOWS``.
    size_window: str | None = None
    #: Correctness tokens the handler is expected to verify.
    correctness: tuple[str, ...] = ()
    #: Suites/reports select on these without hard-coding case IDs.
    tags: tuple[str, ...] = ()
    #: ``inference`` / ``training`` / ``both``.
    phase: str = "inference"
    #: Applicability rules to evaluate, in precedence order.
    applicability: tuple[str, ...] = ()
    #: Workload transform applied to the configuration before expansion, e.g. a
    #: case that only measures an imbalanced routing pattern selects one.
    transform: str | None = None
    #: Adapter that always owns this case, e.g. ``alltoall_hccl``.
    owner: str | None = None
    #: True when the case belongs to the *baseline comparison* set: the reference
    #: composition an adapter is compared against.  A baseline run selects these
    #: cases instead of hard-coding their identifiers in the scheduler.
    baseline_reference: bool = False
    #: Canonical case that superseded this legacy identifier, if any.
    alias_of: str | None = None
    #: Static per-case parameters, stored as items so the spec stays hashable.
    #: These are *requested* values only; evidence decides what was observed.
    parameters: tuple = ()

    @property
    def legacy(self):
        return self.alias_of is not None

    @property
    def training(self):
        """True when the case measures a training-only code path."""
        return self.phase == "training"

    @property
    def inference(self):
        return self.phase in ("inference", "both")

    def requested(self):
        """Static per-case request parameters as a fresh dict."""
        return dict(self.parameters)

    def sweeps(self, dimension):
        return dimension in self.sweep


# --------------------------------------------------------------------------
# Applicability rules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ApplicabilityContext:
    """Everything a rule may look at, and nothing else."""

    spec: object          # common.registry.Operator
    config: dict
    baseline: bool = False


Rule = Callable[[CaseSpec, ApplicabilityContext], "tuple[str, str] | None"]

APPLICABILITY_RULES: dict = {}


def rule(name):
    """Register an applicability rule under ``name``."""

    def register(function):
        if name in APPLICABILITY_RULES:
            raise ValueError(f"duplicate applicability rule {name!r}")
        APPLICABILITY_RULES[name] = function
        return function

    return register


@rule("concurrent_needs_two_ranks")
def _concurrent_needs_two_ranks(case, context):
    if context.config["device"] != "npu" or context.config["distributed"]["ep_size"] < 2:
        return "unsupported", "concurrent MoE requires at least two HCCL ranks"
    return None


@rule("dedup_mode")
def _dedup_mode(case, context):
    mode = context.config["dedup"]["mode"]
    if mode == "none":
        return ("not_applicable",
                "deduplicated dispatch has no dedup path; the None control for this suite is "
                "COMM-001 with dedup.mode=none")
    if f"{mode}_dedup" not in context.spec.capabilities:
        return "unsupported", f"no {mode} dedup path"
    return None


#: Which dedup deployment layer each layer case measures.
_DEDUP_LAYER = {"rank_dedup_layer": "rank", "server_dedup_layer": "server"}


@rule("layer_dedup_mode")
def _layer_dedup_mode(case, context):
    layer = _DEDUP_LAYER.get(case.operation)
    if layer is None or layer == context.config["dedup"]["mode"]:
        return None
    return "not_applicable", (
        f"this case measures {layer} dedup; the plan selects "
        f"dedup.mode={context.config['dedup']['mode']}")


@rule("trace_replay_workload")
def _trace_replay_workload(case, context):
    if context.config["workload"]["routing"] == "real_trace_replay":
        return None
    # The case measures replay of a real trace.  A configuration without a trace
    # does not select the test condition, so it is not a gap in the execution: it
    # is `not_applicable` and the reason says why.
    return "not_applicable", ("L3 replay needs a real_trace_replay workload; this "
                             "configuration does not select one")


@rule("planner_enabled")
def _planner_enabled(case, context):
    if context.config["load_balance"]["enabled"]:
        return None
    return "not_applicable", ("the planner is disabled in this configuration; the "
                              "load_balance suite measures this case with "
                              "load_balance.enabled=true")


@rule("l4_replay_boundary")
def _l4_replay_boundary(case, context):
    if case.level != "L4":
        return None
    if context.config["workload"]["routing"] != "real_trace_replay":
        return None
    return "not_applicable", "replay is measured in LAYER-007 with the L3 boundary"


@rule("training_phase")
def _training_phase(case, context):
    if context.config["benchmark"].get("phase") != "forward" or not case.training:
        return None
    return "not_applicable", "forward-only configuration selects a training case"


@rule("adapter_token_limit")
def _adapter_token_limit(case, context):
    if context.spec.id != "mc2_dispatch_cann" or case.level != "L1":
        return None
    tokens = context.config["workload"]["tokens_per_rank"]
    counts = tokens if isinstance(tokens, list) else [tokens]
    if max(counts) <= 512:
        return None
    return "unsupported", "a single 950DT V2 invocation supports at most 512 tokens per rank"


#: Rules that apply to every operator case, in precedence order, after the
#: case-specific rules.
GENERIC_RULES = ("l4_replay_boundary", "training_phase", "adapter_token_limit")


# --------------------------------------------------------------------------
# Workload transforms
# --------------------------------------------------------------------------

#: A transform rewrites the configuration so that a case actually measures the
#: code path it names.  It returns a new configuration and never mutates its
#: input, because the same configuration object expands several cases.
TRANSFORMS: dict = {}


def transform(name):
    """Register a workload transform under ``name``."""

    def register(function):
        if name in TRANSFORMS:
            raise ValueError(f"duplicate workload transform {name!r}")
        TRANSFORMS[name] = function
        return function

    return register


@transform("imbalanced_workload")
def _imbalanced_workload(config, case):
    """An imbalance case must select an imbalanced routing pattern."""
    workload = config["workload"]
    if workload["routing"] in {"balanced", "uniform_random"}:
        workload["routing"] = "expert_hotspot"
    return config


@transform("dynamic_token_steps")
def _dynamic_token_steps(config, case):
    """A dynamic-shape case measures two token counts in one run."""
    workload = config["workload"]
    tokens = workload["tokens_per_rank"]
    half = ([count // 2 for count in tokens] if isinstance(tokens, list)
            else max(1, tokens // 2))
    workload.setdefault("token_steps", [tokens, half])
    return config


# --------------------------------------------------------------------------
# The catalog
# --------------------------------------------------------------------------

#: L0 native request table.  ``mechanism``/``execution_resource`` are *requested*
#: values; the evidence collector decides what was observed.
L0_PARAMETERS = {
    "LINK-001": {"mechanism": "ub_memory", "execution_resource": "AIV", "engine": "AIV",
                 "protocol": "UB_MEM"},
    "LINK-002": {"mechanism": "urma", "execution_resource": "AIV", "engine": "AIV",
                 "protocol": "UB_CTP"},
    "LINK-003": {"mechanism": "ub_ctp", "execution_resource": "AIV", "engine": "AIV",
                 "protocol": "UB_CTP"},
    "LINK-004": {"mechanism": "ub_rtp", "execution_resource": "AIV", "engine": "AIV",
                 "protocol": "UB_RTP"},
    "LINK-005": {"mechanism": "urma", "execution_resource": "AICPU_TS", "engine": "AICPU_TS",
                 "protocol": "UB_CTP"},
    "LINK-006": {"mechanism": "ub_ctp", "execution_resource": "AICPU_TS", "engine": "AICPU_TS",
                 "protocol": "UB_CTP"},
    "LINK-007": {"mechanism": "ub_rtp", "execution_resource": "AICPU_TS", "engine": "AICPU_TS",
                 "protocol": "UB_RTP"},
    "LINK-008": {"mechanism": "urma", "execution_resource": "CCU", "engine": "CCU",
                 "protocol": "UB_CTP"},
    "LINK-009": {"mechanism": "hccs", "execution_resource": "AICPU_TS", "engine": "AICPU_TS",
                 "protocol": "HCCS"},
    "LINK-010": {"mechanism": "roce", "execution_resource": "AICPU_TS", "engine": "AICPU_TS",
                 "protocol": "ROCE"},
    "LINK-011": {"mechanism": "uboe", "execution_resource": "AICPU_TS", "engine": "AICPU_TS",
                 "protocol": "UBOE"},
    "LINK-012": {"mechanism": "shmem_mte", "execution_resource": "HOST_CPU", "backend": "shmem"},
    "LINK-013": {"mechanism": "shmem_udma", "execution_resource": "AIV", "backend": "shmem"},
    "EXEC-001": {"operation": "alltoall"},
    "EXEC-002": {"operation": "alltoallv"},
    "EXEC-003": {"operation": "allgather"},
    "EXEC-004": {"operation": "reducescatter"},
    "EXEC-005": {"operation": "allreduce"},
    "EXEC-006": {"operation": "p2p"},
    "OVERLAP-001": {"mechanism": "ub_memory", "execution_resource": "AIV", "compute": "gmm"},
    "OVERLAP-002": {"mechanism": "urma", "execution_resource": "AIV", "compute": "gmm"},
    "OVERLAP-003": {"mechanism": "urma", "execution_resource": "AICPU_TS", "compute": "gmm"},
    "OVERLAP-004": {"mechanism": "urma", "execution_resource": "CCU", "compute": "gmm"},
    "OVERLAP-005": {"mechanism": "ub_memory", "execution_resource": "AIV", "compute": "swiglu"},
    "OVERLAP-006": {"mechanism": "urma", "execution_resource": "CCU", "compute": "swiglu"},
    "OVERLAP-007": {"mechanism": "shmem_udma", "execution_resource": "AIV", "compute": "gmm"},
    "OVERLAP-008": {"mechanism": "auto", "execution_resource": None, "compute": "hbm_copy"},
    "OVERLAP-009": {"mechanism": "auto", "execution_resource": None, "compute": "gmm_swiglu"},
}

#: Legacy L0 identifiers and the canonical case they now describe.
LEGACY_ALIASES = {
    # Native SHMEM baselines moved into the LINK family.
    "HW-007": "LINK-012",
    "HW-008": "LINK-013",
    # The device-copy framework case is the contiguous HBM copy case.
    "HW-009": "MEM-001",
    # The concurrent communication/GMM case is now the first overlap case; the
    # remaining OVERLAP-* cases sweep the other mechanisms and compute kernels.
    "HW-012": "OVERLAP-001",
}

_L0_OPERATIONS = {
    "MEM": ("copy", "copy_local", "copy_local_ub", "copy_strided"),
    "LINK": ("put",) * 13,
    "EXEC": ("alltoall", "alltoallv", "allgather", "reducescatter", "allreduce", "p2p"),
    "OVERLAP": ("concurrent",) * 9,
}

_L0_CATEGORY = {"MEM": "memory", "LINK": "transport", "EXEC": "execution",
                "OVERLAP": "overlap"}

#: Link cases sweep outstanding requests and the peer fan-out; the SHMEM cases
#: exchange with the whole group and have nothing to sweep.
_LINK_SWEEP = ("message_bytes", "outstanding", "peer_count", "peer_policy")
_SHMEM_SWEEP = ("message_bytes",)

#: Message-size window per L0 case.  ``None`` means "use the family default".
_L0_WINDOW = {
    "LINK-001": "latency", "LINK-002": "latency", "LINK-003": "latency",
    "LINK-004": "latency", "LINK-005": "latency", "LINK-006": "latency",
    "LINK-007": "latency", "LINK-008": "latency",
    "LINK-009": "throughput", "LINK-010": "throughput", "LINK-011": "throughput",
    "LINK-012": "memory", "LINK-013": "memory",
}

#: L1-L4 rows: (operation, required capabilities, handler, correctness, tags).
_L1 = (
    ("dispatch", ("dispatch",), "l1.dispatch",
     ("dispatch_rows", "dispatch_counts"), ("decode", "prefill", "throughput")),
    ("cached_dispatch", ("dispatch", "cached_dispatch"), "l1.cached_dispatch",
     ("dispatch_rows", "dispatch_counts"), ("cached", "decode")),
    ("combine", ("dispatch", "combine"), "l1.combine",
     ("combine_output",), ("decode", "prefill")),
    ("dispatch_combine", ("dispatch", "combine"), "l1.dispatch_combine",
     ("combine_output",), ("roundtrip", "throughput")),
    ("async_dispatch", ("dispatch", "async_dispatch"), "l1.async_dispatch",
     ("dispatch_rows", "dispatch_counts"), ("async", "training_tail")),
    ("async_combine", ("dispatch", "combine", "async_combine"), "l1.async_combine",
     ("combine_output",), ("async", "training_tail")),
    ("training_tail_dispatch", ("dispatch",), "l1.training_tail_dispatch",
     ("dispatch_rows", "dispatch_counts"), ("training", "async")),
    ("high_throughput_dispatch", ("dispatch",), "l1.high_throughput_dispatch",
     ("dispatch_rows", "dispatch_counts"), ("throughput", "scaling")),
    ("quantized_dispatch", ("dispatch", "quantized_dispatch"), "l1.quantized_dispatch",
     ("quantized_dequant",), ("quantized",)),
    ("deduplicated_dispatch", (), "l1.deduplicated_dispatch",
     ("dedup_route",), ("dedup",)),
    ("combine_add_rmsnorm", ("combine_add_rmsnorm",), "l1.combine_add_rmsnorm",
     ("fused_output",), ("fused", "combine")),
    ("planned_dispatch", ("planned_dispatch",), "l1.planned_dispatch",
     ("planned_dispatch",), ("planner", "load_balance")),
)

_L2 = (
    ("serial_dispatch_gmm", ("dispatch",), "l2.serial_dispatch_gmm",
     ("gmm_output",), ("fused", "serial")),
    ("dispatch_gmm", ("dispatch_gmm",), "l2.dispatch_gmm",
     ("gmm_output",), ("fused",)),
    ("serial_gmm_combine", ("dispatch", "combine"), "l2.serial_gmm_combine",
     ("combine_output",), ("fused", "serial")),
    ("gmm_combine", ("gmm_combine",), "l2.gmm_combine",
     ("combine_output",), ("fused",)),
    ("chunk_pipeline", ("chunk_pipeline",), "l2.chunk_pipeline",
     ("moe_forward_output",), ("overlap", "pipeline")),
    ("tile_schedule", ("tile_schedule",), "l2.tile_schedule",
     ("gmm_output",), ("overlap", "tiling")),
    ("comm_resources", ("comm_resources",), "l2.comm_resources",
     ("gmm_output",), ("overlap", "resources")),
    ("quantized_fusion", ("quantized_fusion",), "l2.quantized_fusion",
     ("fused_output",), ("quantized", "fused")),
    ("fused_backward", ("fused_backward",), "l2.fused_backward",
     ("input_gradient", "expert_weight_gradient"), ("training", "fused")),
)

_L3 = (
    ("forward", ("moe_forward",), "l3.forward",
     ("moe_forward_output",), ("forward",)),
    ("backward", ("moe_backward",), "l3.backward",
     ("input_gradient", "expert_weight_gradient"), ("training", "backward")),
    ("forward_backward", ("moe_backward",), "l3.forward_backward",
     ("moe_forward_output", "input_gradient", "expert_weight_gradient"),
     ("training", "roundtrip")),
    ("recompute", ("moe_backward", "recompute"), "l3.recompute",
     ("recompute_gradient",), ("training", "recompute")),
    ("imbalanced", ("moe_forward",), "l3.imbalanced",
     ("moe_forward_output",), ("load_balance", "forward")),
    ("dtype", ("moe_forward",), "l3.dtype",
     ("moe_forward_output",), ("dtype", "forward")),
    ("concurrent", ("concurrent",), "l3.concurrent",
     ("concurrent_pair",), ("overlap", "concurrent")),
    ("multilayer", ("moe_forward",), "l3.multilayer",
     ("multilayer_output",), ("layers", "forward")),
    ("dynamic_tokens", ("moe_forward", "dynamic_shapes"), "l3.dynamic_tokens",
     ("moe_forward_output",), ("dynamic_shapes", "forward")),
)

_L4 = (
    ("router_forward", ("router_layer",), "l4.router_forward",
     ("moe_forward_output", "router_weight_gradient"), ("router",)),
    ("router_training", ("router_layer", "moe_backward"), "l4.router_training",
     ("input_gradient", "router_weight_gradient"), ("router", "training")),
    ("planner", ("dynamic_planner",), "l4.planner",
     ("planner_stages",), ("planner", "training")),
    ("redundant_experts", ("dynamic_planner",), "l4.redundant_experts",
     ("planner_stages",), ("planner", "training")),
    ("rank_dedup_layer", ("rank_dedup", "dedup_training"), "l4.rank_dedup_layer",
     ("input_gradient", "expert_weight_gradient"), ("rank", "dedup", "training")),
    ("server_dedup_layer", ("server_dedup", "dedup_training"), "l4.server_dedup_layer",
     ("input_gradient", "expert_weight_gradient"), ("server", "dedup", "training")),
    ("trace_replay", ("moe_forward",), "l4.trace_replay",
     ("moe_forward_output",), ("replay", "forward")),
    ("planner_training_steps", ("dynamic_planner",), "l4.planner_training_steps",
     ("planner_stages",), ("planner", "training")),
    ("router_large_training", ("router_layer", "moe_backward"), "l4.router_large_training",
     ("input_gradient", "router_weight_gradient"), ("router", "training")),
)

#: Named variants for cases whose operation has several distinct kernels or
#: stages.  This is the *only* place a variant list is declared.
_CASE_VARIANTS = {
    "LAYER-003": ("fresh_plan", "cached_plan"),
    "FUSE-005": ("chunks_2", "chunks_4", "chunks_8"),
    "FUSE-006": ("tile_128_256_256", "tile_256_128_256", "tile_256_256_256"),
    "FUSE-007": ("copy_cores_1", "copy_cores_2", "copy_cores_4", "copy_cores_8"),
}

_L0_VARIANTS = {
    # Two collective entry points share one case: AllGather and ReduceScatter.
    # The token is the canonical operation name so a result never claims a
    # collective it did not run.
    "HW-006": ("allgather", "reducescatter"),
    "HW-007": ("mte",),
    # The legacy SHMEM case swept the engine, which is why it keeps its own list
    # instead of inheriting LINK-013's execution-resource split.
    "HW-008": ("mte", "udma", "urma"),
    "HW-010": ("gmm1", "gmm2", "gmm1_dgrad", "gmm1_wgrad", "gmm2_dgrad", "gmm2_wgrad"),
    "HW-011": ("forward", "backward"),
    "LINK-013": ("device", "host"),
}

#: Cases whose applicability depends on named rules, in precedence order.  The
#: generic rules are appended by :func:`_operator_specs`.
_CASE_RULES = {
    "MOE-007": ("concurrent_needs_two_ranks",),
    "COMM-010": ("dedup_mode",),
    "LAYER-005": ("layer_dedup_mode",),
    "LAYER-006": ("layer_dedup_mode",),
    "LAYER-003": ("planner_enabled",),
    "LAYER-004": ("planner_enabled",),
    "LAYER-008": ("planner_enabled",),
    "LAYER-007": ("trace_replay_workload",),
    "COMM-012": ("planner_enabled",),
}

#: Adapter that always owns the case (its parameter set carries ``fixed_owner``).
_CASE_OWNER = {"LAYER-003": "alltoall_hccl", "LAYER-005": "alltoall_hccl",
               "LAYER-006": "alltoall_hccl"}

#: Cases that make up the baseline comparison set: the serial compositions an
#: adapter's fused entry point is compared against.  A baseline run selects these
#: instead of the whole L2 level, because only they measure the same work.
_BASELINE_REFERENCE = frozenset({"FUSE-001", "FUSE-003"})

#: Workload transform a case needs so that it measures the code path it names.
_CASE_TRANSFORM = {"MOE-005": "imbalanced_workload", "MOE-009": "dynamic_token_steps"}

#: Cases measured at a different level than their ID prefix suggests.
_LEVEL_OVERRIDES = {"LAYER-007": "L3"}

_TRAINING_OPERATIONS = {
    "backward", "forward_backward", "recompute", "fused_backward", "router_training",
    "planner", "redundant_experts", "planner_training_steps", "rank_dedup_layer",
    "server_dedup_layer", "router_large_training",
}

#: Framework collective rows of the pre-refactor L0 catalog, kept because the
#: legacy ``HW-001`` .. ``HW-006`` identifiers are still valid selectors.
_FRAMEWORK_COLLECTIVES = {
    "HW-001": "p2p_unidirectional",
    "HW-002": "p2p_bidirectional",
    "HW-003": "multi_peer",
    "HW-004": "alltoall",
    "HW-005": "alltoallv",
    "HW-006": "gather_reduce",
}


def _phase_for(level, operation):
    if level == "L0":
        return "both"
    return "training" if operation in _TRAINING_OPERATIONS else "inference"


def _items(case_id):
    return tuple(sorted((L0_PARAMETERS.get(case_id) or {}).items()))


def _l0_specs():
    specs = {}
    for prefix, operations in _L0_OPERATIONS.items():
        category = _L0_CATEGORY[prefix]
        for number, operation in enumerate(operations, 1):
            case_id = f"{prefix}-{number:03}"
            shmem = case_id in ("LINK-012", "LINK-013")
            specs[case_id] = CaseSpec(
                case_id, "L0", operation,
                category=category,
                variants=_L0_VARIANTS.get(case_id, ("default",)),
                handler=f"l0.{category}",
                sweep=_SHMEM_SWEEP if shmem else (
                    _LINK_SWEEP if prefix == "LINK" else ("message_bytes",)
                    if prefix != "EXEC" else ("message_bytes", "accelerator")),
                size_window=_L0_WINDOW.get(case_id),
                correctness=("payload_bytes",),
                tags=(category,),
                phase="both",
                parameters=_items(case_id),
            )
    for case_id, operation in _FRAMEWORK_COLLECTIVES.items():
        specs[case_id] = CaseSpec(
            case_id, "L0", operation,
            category="collective",
            variants=_L0_VARIANTS.get(case_id, ("default",)),
            handler="l0.collective",
            sweep=("message_bytes",),
            # The framework collectives were always swept over the latency window;
            # keeping it means a refactor cannot silently change the payload sizes
            # a historical result was measured at.
            size_window="latency",
            correctness=("byte_exact", "collective_payload"),
            tags=("collective", "framework"),
            phase="both",
        )
    specs["HW-007"] = CaseSpec(
        "HW-007", "L0", "shmem_mte", category="transport",
        variants=_L0_VARIANTS["HW-007"], handler="l0.transport",
        sweep=_SHMEM_SWEEP, size_window="memory",
        correctness=("byte_exact", "payload_bytes"), tags=("transport", "shmem"),
        phase="both",
    )
    specs["HW-008"] = CaseSpec(
        "HW-008", "L0", "shmem_scaleout", category="transport",
        variants=_L0_VARIANTS["HW-008"], handler="l0.transport",
        sweep=_SHMEM_SWEEP, size_window="memory",
        correctness=("byte_exact", "payload_bytes"), tags=("transport", "shmem"),
        phase="both",
    )
    specs["HW-009"] = CaseSpec(
        "HW-009", "L0", "device_copy", category="memory", handler="l0.memory",
        sweep=("message_bytes",), size_window="memory",
        correctness=("payload_bytes",), tags=("memory",), phase="both",
    )
    specs["HW-010"] = CaseSpec(
        "HW-010", "L0", "grouped_matmul", category="compute", handler="l0.compute",
        variants=_L0_VARIANTS["HW-010"], sweep=("message_bytes",),
        size_window="compute", correctness=("compute_output",),
        tags=("compute",), phase="both",
    )
    specs["HW-011"] = CaseSpec(
        "HW-011", "L0", "swiglu", category="compute", handler="l0.compute",
        variants=_L0_VARIANTS["HW-011"], sweep=("message_bytes",),
        size_window="compute", correctness=("compute_output",),
        tags=("compute",), phase="both",
    )
    specs["HW-012"] = CaseSpec(
        "HW-012", "L0", "concurrent_gmm_communication", category="overlap",
        handler="l0.overlap", sweep=("message_bytes",), size_window="overlap",
        correctness=("payload_bytes",), tags=("overlap",), phase="both",
    )
    return specs


def _operator_specs(rows, level, prefix):
    specs = {}
    for number, (operation, requires, handler, correctness, tags) in enumerate(rows, 1):
        case_id = f"{prefix}-{number:03}"
        effective_level = _LEVEL_OVERRIDES.get(case_id, level)
        specs[case_id] = CaseSpec(
            case_id, effective_level, operation,
            requires=tuple(requires),
            variants=_CASE_VARIANTS.get(case_id, ("default",)),
            handler=handler,
            sweep=("variant",),
            correctness=tuple(correctness),
            tags=tuple(tags),
            phase=_phase_for(effective_level, operation),
            applicability=(*_CASE_RULES.get(case_id, ()), *GENERIC_RULES),
            transform=_CASE_TRANSFORM.get(case_id),
            owner=_CASE_OWNER.get(case_id),
            baseline_reference=case_id in _BASELINE_REFERENCE,
        )
    return specs


def build_catalog():
    """Build the case catalog."""
    specs = {}
    specs.update(_l0_specs())
    specs.update(_operator_specs(_L1, "L1", "COMM"))
    specs.update(_operator_specs(_L2, "L2", "FUSE"))
    specs.update(_operator_specs(_L3, "L3", "MOE"))
    specs.update(_operator_specs(_L4, "L4", "LAYER"))
    for legacy, canonical in LEGACY_ALIASES.items():
        if legacy in specs:
            specs[legacy] = replace(specs[legacy], alias_of=canonical)
    return specs


_CACHE = {}


def specs():
    """The complete case catalog, keyed by case ID (legacy aliases included)."""
    if not _CACHE:
        catalog = build_catalog()
        _validate(catalog)
        _CACHE.update(catalog)
    return _CACHE


def spec(case_id):
    return specs()[case_id]


def catalog():
    """Every catalog case keyed by case ID, legacy aliases included."""
    return specs()


def canonical_specs():
    """Catalog entries without the legacy aliases, keyed by case ID."""
    return {case_id: item for case_id, item in specs().items() if item.alias_of is None}


#: Historic name for :func:`canonical_specs`.
canonical_cases = canonical_specs


def alias_map():
    """Legacy case ID -> canonical case ID."""
    return {item.id: item.alias_of for item in specs().values() if item.alias_of}


def resolve(case_id):
    """Return the canonical ID for ``case_id`` (itself when it is canonical)."""
    return specs()[case_id].alias_of or case_id


def case_counts():
    """Dynamic case counts by level, including the legacy aliases."""
    counts = {"total": 0, "canonical": 0}
    for item in specs().values():
        counts["total"] += 1
        if item.alias_of is None:
            counts["canonical"] += 1
        counts[item.level] = counts.get(item.level, 0) + 1
    return counts


def l0_family(item):
    """Return the L0 family for a case, or ``None`` outside L0."""
    if item.level != "L0":
        return None
    if item.category != "operator":
        return item.category
    return {"device_copy": "memory", "grouped_matmul": "compute", "swiglu": "compute",
            "concurrent_gmm_communication": "overlap",
            "shmem_mte": "transport", "shmem_scaleout": "transport"}.get(
                item.operation, "collective")


def handler_for(case_id):
    """The dotted handler name for ``case_id``."""
    return specs()[case_id].handler


def variants_for(case_id):
    """Named parameter variants for ``case_id``."""
    return specs()[case_id].variants


def owned_by(case_id):
    """The fixed-owner adapter for ``case_id``, or ``None``."""
    return specs()[case_id].owner


def _validate(catalog):
    """Fail fast when the catalog contradicts itself.

    Every rule name must exist, every handler must be non-empty, every variant
    list must be unique and the correctness tokens must come from the controlled
    vocabulary.  This runs when the catalog is first built, so a bad row can
    never reach a manifest.
    """
    problems = []
    for case_id, item in catalog.items():
        if item.level not in LEVELS:
            problems.append(f"{case_id}: unknown level {item.level!r}")
        if item.category not in CATEGORIES:
            problems.append(f"{case_id}: unknown category {item.category!r}")
        if item.phase not in PHASES:
            problems.append(f"{case_id}: unknown phase {item.phase!r}")
        if not item.handler:
            problems.append(f"{case_id}: no execution handler")
        if len(set(item.variants)) != len(item.variants):
            problems.append(f"{case_id}: duplicate variants")
        unknown = set(item.correctness) - set(CORRECTNESS_TOKENS)
        if unknown:
            problems.append(f"{case_id}: unknown correctness tokens {sorted(unknown)}")
        unknown_sweeps = set(item.sweep) - set(SWEEPS)
        if unknown_sweeps:
            problems.append(f"{case_id}: unknown sweeps {sorted(unknown_sweeps)}")
        if item.size_window is not None and item.size_window not in SIZE_WINDOWS:
            problems.append(f"{case_id}: unknown size window {item.size_window!r}")
        for name in item.applicability:
            if name not in APPLICABILITY_RULES:
                problems.append(f"{case_id}: unknown applicability rule {name!r}")
        if item.transform is not None and item.transform not in TRANSFORMS:
            problems.append(f"{case_id}: unknown workload transform {item.transform!r}")
    if problems:
        raise ValueError("invalid case catalog: " + "; ".join(problems))
    return True


def validate_catalog():
    """Validate the catalog and return ``True`` (raises on a bad row)."""
    return _validate(specs())

