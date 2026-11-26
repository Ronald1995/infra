"""Case catalog matching the methodology IDs and actual measurement boundaries.

L0 is split into six families, and the family is part of the case metadata so the
scheduler, the runner and the report can route a case without inspecting its ID:

``memory``      local memory movement (HBM, MTE, UB)
``transport``   one communication mechanism on one execution resource
``execution``   collective operations swept over HCCL accelerator modes
``collective``  framework-level collectives through torch HCCL (``HW-001``..)
``compute``     standalone GMM / SwiGLU
``overlap``     communication and compute submitted concurrently
"""
from dataclasses import dataclass

CATEGORIES = ("memory", "transport", "execution", "collective", "compute", "overlap",
              "operator")


@dataclass(frozen=True)
class Case:
    id: str
    level: str
    operation: str
    requires: tuple[str, ...]
    training: bool = False
    category: str = "operator"
    #: Which benchmark dimensions this case sweeps over.
    dimensions: tuple[str, ...] = ()


#: Static per-case parameters.  ``mechanism``/``execution_resource`` mirror the
#: native request table; they are recorded as *requested* values only.
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


def catalog():
    groups = {
        "L0": ("HW", [
            ("p2p_unidirectional",), ("p2p_bidirectional",), ("multi_peer",),
            ("alltoall",), ("alltoallv",), ("gather_reduce",), ("shmem_mte",),
            ("shmem_scaleout",), ("device_copy",), ("grouped_matmul",),
            ("swiglu",), ("concurrent_gmm_communication",),
        ]),
        "L1": ("COMM", [
            ("dispatch", "dispatch"), ("cached_dispatch", "dispatch", "cached_dispatch"),
            ("combine", "dispatch", "combine"), ("dispatch_combine", "dispatch", "combine"),
            ("async_dispatch", "dispatch", "async_dispatch"),
            ("async_combine", "dispatch", "combine", "async_combine"),
            ("training_tail_dispatch", "dispatch"), ("high_throughput_dispatch", "dispatch"),
            ("quantized_dispatch", "dispatch", "quantized_dispatch"),
            ("deduplicated_dispatch",),
            ("combine_add_rmsnorm", "combine_add_rmsnorm"),
            ("planned_dispatch", "planned_dispatch"),
        ]),
        "L2": ("FUSE", [
            ("serial_dispatch_gmm", "dispatch"), ("dispatch_gmm", "dispatch_gmm"),
            ("serial_gmm_combine", "dispatch", "combine"), ("gmm_combine", "gmm_combine"),
            ("chunk_pipeline", "chunk_pipeline"), ("tile_schedule", "tile_schedule"),
            ("comm_resources", "comm_resources"), ("quantized_fusion", "quantized_fusion"),
            ("fused_backward", "fused_backward"),
        ]),
        "L3": ("MOE", [
            ("forward", "moe_forward"), ("backward", "moe_backward"),
            ("forward_backward", "moe_backward"), ("recompute", "moe_backward", "recompute"),
            ("imbalanced", "moe_forward"), ("dtype", "moe_forward"),
            ("concurrent", "concurrent"), ("multilayer", "moe_forward"),
            ("dynamic_tokens", "moe_forward", "dynamic_shapes"),
        ]),
        "L4": ("LAYER", [
            ("router_forward", "router_layer"), ("router_training", "router_layer", "moe_backward"),
            ("planner", "dynamic_planner"), ("redundant_experts", "dynamic_planner"),
            ("rank_dedup_layer", "rank_dedup", "dedup_training"),
            ("server_dedup_layer", "server_dedup", "dedup_training"),
            ("trace_replay", "moe_forward"), ("planner_training_steps", "dynamic_planner"),
            ("router_large_training", "router_layer", "moe_backward"),
        ]),
    }
    training = {"backward", "forward_backward", "recompute", "fused_backward",
                "router_training", "planner", "redundant_experts", "planner_training_steps",
                "rank_dedup_layer", "server_dedup_layer", "router_large_training"}
    cases = {
        case_id: Case(case_id, "L3" if operation == "trace_replay" else level,
                      operation, tuple(capabilities), operation in training)
        for level, (prefix, cases) in groups.items()
        for number, (operation, *capabilities) in enumerate(cases, 1)
        for case_id in (f"{prefix}-{number:03}",)
    }
    cases.update(l0_cases(cases))
    return cases


def _category_for(case_id):
    prefix = case_id.split("-")[0]
    return {"MEM": "memory", "LINK": "transport", "EXEC": "execution",
            "OVERLAP": "overlap"}.get(prefix, "operator")


def l0_cases(base):
    """The six L0 families introduced by the transport refactor."""
    definitions = {
        "MEM": [
            ("copy", ()), ("copy_local", ()), ("copy_local_ub", ()), ("copy_strided", ()),
        ],
        "LINK": [
            ("put", ()), ("put", ()), ("put", ()), ("put", ()), ("put", ()), ("put", ()),
            ("put", ()), ("put", ()), ("put", ()), ("put", ()), ("put", ()), ("put", ()),
            ("put", ()),
        ],
        "EXEC": [
            ("alltoall", ()), ("alltoallv", ()), ("allgather", ()),
            ("reducescatter", ()), ("allreduce", ()), ("p2p", ()),
        ],
        "OVERLAP": [("concurrent", ())] * 9,
    }
    dimensions = {
        "MEM": ("message_bytes",),
        "LINK": ("message_bytes", "outstanding", "peer_count"),
        "EXEC": ("message_bytes", "accelerator"),
        "OVERLAP": ("message_bytes",),
    }
    result = {}
    for prefix, entries in definitions.items():
        for number, (operation, capabilities) in enumerate(entries, 1):
            case_id = f"{prefix}-{number:03}"
            result[case_id] = Case(case_id, "L0", operation, capabilities,
                                   category=_category_for(case_id),
                                   dimensions=dimensions[prefix])
    result["HW-009"] = Case("HW-009", "L0", "device_copy", (), category="memory")
    result["HW-010"] = Case("HW-010", "L0", "grouped_matmul", (), category="compute")
    result["HW-011"] = Case("HW-011", "L0", "swiglu", (), category="compute")
    for case_id in ("HW-001", "HW-002", "HW-003", "HW-004", "HW-005", "HW-006"):
        result[case_id] = Case(case_id, "L0", base[case_id].operation,
                               base[case_id].requires, category="collective")
    result["HW-012"] = Case("HW-012", "L0", "concurrent_gmm_communication", (),
                            category="overlap")
    for case_id in ("HW-007", "HW-008"):
        result[case_id] = Case(case_id, "L0", base[case_id].operation,
                               base[case_id].requires, category="transport")
    return result


def l0_family(case):
    """Return the L0 family for a case, or ``None`` outside L0."""
    if case.level != "L0":
        return None
    if case.category != "operator":
        return case.category
    return {"device_copy": "memory", "grouped_matmul": "compute", "swiglu": "compute",
            "concurrent_gmm_communication": "overlap",
            "shmem_mte": "transport", "shmem_scaleout": "transport"}.get(case.operation,
                                                                        "collective")
