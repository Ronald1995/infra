"""FSDPTurbo native training dispatcher."""
import torch.distributed as dist

from ep.runner import run


def forward(layer, x, ids, weights):
    from fsdp_turbo.distributed.expert_parallel import dispatcher as native
    if hasattr(native, "all2all_experts_forward_fn"):
        return native.all2all_experts_forward_fn(
            x, ids.int(), weights, (layer.gate_up_proj, layer.down_proj),
            layer.num_global_experts, layer.expert_ids_per_ep_rank,
            dist.group.WORLD, fused=True, act_limit=layer.limit)
    from fsdp_turbo.distributed.expert_parallel.dispatcher import get_experts_forward_fn
    if not hasattr(layer, "native_forward"):
        layer.native_forward = get_experts_forward_fn(dist.group.WORLD, fused=True, fixed_router=False)
    return layer.native_forward(layer, x, ids.int(), weights)


if __name__ == "__main__":
    run(forward, "fused_fsdpturbo")
