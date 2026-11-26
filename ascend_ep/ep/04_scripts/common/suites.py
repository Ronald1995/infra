"""Suite definitions: how the case catalog is grouped into benchmark scenarios.

A suite is *only* a selection over the catalog plus an optional workload policy.
It never defines a case and it never decides what a case measures, so a case that
belongs to no business suite is still reachable through ``all``.

Two kinds of suite exist:

*L0 suites* select by ``Case.category`` (``hardware``, ``memory``, ``transport``,
``collectives``, ``overlap``) and are expanded without the EP-operator dimension.
*Business suites* select explicit case IDs and may carry a workload policy that
turns one case into different parameter sets per EP size (strong/weak scaling).
"""
from dataclasses import dataclass

from common.cases import catalog
from common.config import LEVELS

#: Suites that only make sense for L0; used by the report to separate hardware
#: baselines from operator measurements.
L0_SUITES = ("hardware", "memory", "transport", "collectives")

#: The business suites.  ``overlap`` intentionally appears in both worlds: it
#: selects the L0 overlap family *and* the fused operator cases.
BUSINESS_SUITES = (
    "decode", "prefill", "training_tail", "high_throughput", "strong_scaling",
    "weak_scaling", "topology", "load_balance", "dedup", "overlap", "quantized",
    "routing_imbalance", "straggler",
)

ALL_SUITES = ("all", *L0_SUITES, *BUSINESS_SUITES)

FUSE_CASES = frozenset(f"FUSE-{number:03}" for number in range(1, 10))


@dataclass(frozen=True)
class Suite:
    id: str
    description: str
    levels: tuple[str, ...] = LEVELS
    categories: tuple[str, ...] = ()
    case_ids: frozenset[str] = frozenset()
    #: ``all`` selects every canonical case regardless of the filters above.
    everything: bool = False
    #: Workload policy applied per EP size: ``strong`` or ``weak`` scaling.
    scaling: str | None = None

    def selects(self, case):
        if self.everything:
            return case.alias_of is None
        if case.level not in self.levels or case.alias_of is not None:
            return False
        if self.categories and case.category in self.categories:
            return True
        return case.id in self.case_ids

    def case_ids_for(self):
        return sorted(case.id for case in catalog().values() if self.selects(case))


SUITES = {
    # ------------------------------------------------------------------ all --
    "all": Suite("all", "complete case catalog across L0-L4", everything=True),
    # --------------------------------------------------------------- L0 ------
    "hardware": Suite(
        "hardware", "framework collectives and standalone compute baselines",
        levels=("L0",), categories=("collective", "compute")),
    "memory": Suite("memory", "local memory movement", levels=("L0",), categories=("memory",)),
    "transport": Suite("transport", "native transport mechanisms", levels=("L0",),
                       categories=("transport",)),
    "collectives": Suite("collectives", "native HCCL collective execution", levels=("L0",),
                         categories=("execution",)),
    # ------------------------------------------------------------ business ---
    "decode": Suite("decode", "small token, low latency serving",
                    levels=("L1", "L2", "L3", "L4"),
                    case_ids=frozenset({"COMM-001", "COMM-003", "MOE-001", "LAYER-001"})),
    "prefill": Suite("prefill", "large token prefill",
                     levels=("L1", "L2", "L3", "L4"),
                     case_ids=frozenset({"COMM-001", "COMM-003", "MOE-001", "LAYER-001"})),
    "training_tail": Suite("training_tail", "training tail: backward and asynchronous paths",
                           levels=("L1", "L3"),
                           case_ids=frozenset({"COMM-005", "COMM-006", "COMM-007",
                                               "MOE-002", "MOE-003"})),
    "high_throughput": Suite(
        "high_throughput", "large scale communication throughput",
        levels=("L1", "L2", "L3"),
        case_ids=frozenset({"COMM-001", "COMM-003", "COMM-004", "COMM-008",
                            "MOE-003", "MOE-009"})),
    "strong_scaling": Suite(
        "strong_scaling", "fixed global workload across EP sizes",
        levels=("L1", "L3", "L4"),
        case_ids=frozenset({"COMM-001", "COMM-003", "MOE-001", "MOE-002", "MOE-003",
                            "LAYER-001"}),
        scaling="strong"),
    "weak_scaling": Suite(
        "weak_scaling", "fixed per-rank workload across EP sizes",
        levels=("L1", "L3", "L4"),
        case_ids=frozenset({"COMM-001", "COMM-003", "MOE-001", "MOE-002", "MOE-003",
                            "LAYER-001"}),
        scaling="weak"),
    "topology": Suite("topology", "topology and routing locality",
                      levels=("L1", "L3"),
                      case_ids=frozenset({"COMM-001", "COMM-002", "COMM-003", "MOE-001"})),
    "load_balance": Suite("load_balance", "planner, redundant experts and dynamic routing",
                          levels=("L1", "L3", "L4"),
                          case_ids=frozenset({"COMM-012", "MOE-005", "LAYER-003", "LAYER-004",
                                              "LAYER-008"})),
    "dedup": Suite(
        "dedup", "rank/server dedup with an explicit no-dedup control group",
        levels=("L1", "L3", "L4"),
        # COMM-001 / MOE-001 are the measured `none` control: the dedup cases
        # themselves cannot run without a dedup implementation, so the control is
        # the same dispatch/forward workload without dedup (see coverage report).
        case_ids=frozenset({"COMM-010", "LAYER-005", "LAYER-006", "COMM-001", "MOE-001"})),
    "overlap": Suite(
        "overlap", "communication/compute overlap: L0 concurrency and fused operators",
        levels=("L0", "L2", "L3"),
        categories=("overlap",),
        case_ids=frozenset({*FUSE_CASES, "MOE-007"})),
    "routing_imbalance": Suite(
        "routing_imbalance", "generated load imbalance: rank/expert targets and locality",
        levels=("L1", "L2", "L3", "L4"),
        case_ids=frozenset({"COMM-001", "COMM-003", "COMM-004", "MOE-001", "MOE-003",
                            "LAYER-001", "LAYER-003"})),
    "straggler": Suite(
        "straggler", "slow-rank injection: selection, delay and calibration",
        levels=("L1", "L2", "L3", "L4"),
        case_ids=frozenset({"COMM-001", "COMM-003", "COMM-004", "MOE-001", "MOE-003",
                            "LAYER-001"})),
    "quantized": Suite("quantized", "quantized dispatch and fusion",
                       levels=("L1", "L2", "L3"),
                       case_ids=frozenset({"COMM-009", "FUSE-008", "MOE-006"})),
}

ALIASES = {
    "full": "all",
    # Configurations that do not name a suite run the whole catalog.  `baseline`
    # is the historical default suite_id and the smoke configs only narrow the
    # levels, so all three keep their previous meaning.
    "baseline": "all",
    "smoke_cpu": "all",
    "smoke_npu": "all",
}


def normalize(suite_id):
    suite_id = ALIASES.get(suite_id, suite_id)
    if suite_id not in SUITES:
        raise ValueError(f"unknown suite {suite_id!r}; expected one of {', '.join(ALL_SUITES)}")
    return suite_id


def suite(suite_id):
    return SUITES[normalize(suite_id)]


def select(suite_ids, levels=None):
    """Return the case IDs selected by ``suite_ids``, optionally level-filtered."""
    if isinstance(suite_ids, str):
        suite_ids = [suite_ids]
    selected = []
    for suite_id in suite_ids:
        selected.extend(suite(suite_id).case_ids_for())
    ordered = {case.id: case for case in catalog().values()}
    result = []
    for case_id in sorted(set(selected) | set(), key=lambda item: (ordered[item].level, item)):
        case = ordered[case_id]
        if levels and case.level not in levels:
            continue
        result.append(case_id)
    return result


def workload_policy(suite_id):
    return suite(suite_id).scaling


def apply_scaling(suite_id, config, ep_size):
    """Generate the strong/weak scaling workload for one EP size.

    The policy is declarative: the suite configuration carries a ``scaling``
    block, and the same block is interpreted differently by the two modes.
    Integer division must be exact, otherwise the plan raises instead of silently
    changing the global workload.
    """
    policy = workload_policy(suite_id)
    if policy is None:
        return config
    scaling = config.get("scaling") or {}
    if not scaling:
        raise ValueError(f"suite {suite_id} requires a scaling block in its configuration")
    result = {**config, "scaling": {**scaling, "mode": policy}}
    workload = dict(result["workload"])
    model = dict(result["model"])
    if policy == "strong":
        total = scaling.get("global_tokens")
        if total is None:
            raise ValueError("strong scaling requires scaling.global_tokens")
        if total % ep_size:
            raise ValueError(
                f"strong scaling global_tokens={total} is not divisible by EP{ep_size}")
        workload["tokens_per_rank"] = total // ep_size
        if scaling.get("num_experts"):
            model["num_experts"] = scaling["num_experts"]
    elif policy == "weak":
        tokens = scaling.get("tokens_per_rank")
        experts = scaling.get("experts_per_rank")
        if tokens is None or experts is None:
            raise ValueError("weak scaling requires scaling.tokens_per_rank and experts_per_rank")
        workload["tokens_per_rank"] = tokens
        model["num_experts"] = experts * ep_size
    result["workload"] = workload
    result["model"] = model
    return result
