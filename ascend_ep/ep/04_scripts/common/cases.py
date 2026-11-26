"""Case catalog matching the methodology IDs and actual measurement boundaries."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    id: str
    level: str
    operation: str
    requires: tuple[str, ...]
    training: bool = False


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
    return {
        case_id: Case(case_id, "L3" if operation == "trace_replay" else level,
                      operation, tuple(capabilities), operation in training)
        for level, (prefix, cases) in groups.items()
        for number, (operation, *capabilities) in enumerate(cases, 1)
        for case_id in (f"{prefix}-{number:03}",)
    }
