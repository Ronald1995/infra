import torch.distributed as dist

from ep.runner import run


def forward(layer, x, ids, weights):
    from fsdp_turbo.distributed.expert_parallel import dispatcher_mc2 as native
    if hasattr(native, "mc2_experts_forward_fn"):
        return native.mc2_experts_forward_fn(
            x, ids.int(), weights, (layer.gate_up_proj, layer.down_proj),
            layer.num_global_experts, layer.expert_ids_per_ep_rank,
            dist.group.WORLD, fused=True, act_limit=layer.limit)
    from fsdp_turbo.distributed.expert_parallel.dispatcher_mc2 import dispatch_mlp_combine
    return dispatch_mlp_combine(dist.group.WORLD, x, ids.int(), weights,
                                (layer.gate_up_proj, layer.down_proj), layer.act_fn,
                                layer.limit, layer.num_global_experts)


if __name__ == "__main__":
    run(forward, "mc2_fsdpturbo")
