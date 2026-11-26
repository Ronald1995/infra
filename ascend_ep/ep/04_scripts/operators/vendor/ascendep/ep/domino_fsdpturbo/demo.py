"""FSDPTurbo native training dispatcher."""
import torch.distributed as dist
from fsdp_turbo.distributed.expert_parallel.domino_dispatcher import get_domino_experts_forward_fn
from ep.runner import run


def forward(layer, x, ids, weights):
    if not hasattr(layer, "native_forward"):
        layer.native_forward = get_domino_experts_forward_fn(dist.group.WORLD, fixed_router=False)
    return layer.native_forward(layer, x, ids.int(), weights)


if __name__ == "__main__":
    run(forward, "domino_fsdpturbo")
