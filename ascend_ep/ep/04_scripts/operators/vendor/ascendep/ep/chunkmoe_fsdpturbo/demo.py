import torch.distributed as dist
from fsdp_turbo.distributed.expert_parallel.dispatcher_chunkmoe import chunkmoe_experts_forward_fn
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig
from ep.runner import run


def forward(layer, x, ids, weights):
    return chunkmoe_experts_forward_fn(x, ids.int(), weights, (layer.gate_up_proj, layer.down_proj),
        layer.num_global_experts, layer.expert_ids_per_ep_rank, dist.group.WORLD, fused=True,
        act_limit=layer.limit, ep_balance_strategy=None, ep_plan=EPPlanConfig(dispatcher="chunkmoe", moe_num_chunks=2))


if __name__ == "__main__":
    run(forward, "chunkmoe_fsdpturbo")
