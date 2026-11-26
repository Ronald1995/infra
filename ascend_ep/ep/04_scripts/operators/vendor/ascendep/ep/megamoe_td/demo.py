"""Mega-MoE-TD native forward/backward on the fixed balanced benchmark trace."""
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ep.runner import run


@torch.no_grad()
def check_activation_range(layer, x, ids):
    # Native SwiGLU has no clipping switch. Establish equivalence for this fixed trace.
    world = dist.get_world_size()
    global_x = x.new_empty((world * len(x), x.shape[1]))
    global_ids = ids.new_empty((world * len(x), ids.shape[1]))
    dist.all_gather_into_tensor(global_x, x.contiguous())
    dist.all_gather_into_tensor(global_ids, ids.contiguous())
    maxima = []
    start = dist.get_rank() * len(layer.gate_up_proj)
    for i, weight in enumerate(layer.gate_up_proj):
        tokens = (global_ids == start + i).any(1)
        gate, up = F.linear(global_x[tokens], weight).chunk(2, -1)
        maxima.append(torch.maximum(gate.max(), up.abs().max()))
    largest = torch.stack(maxima).max()
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    if not largest.item() < layer.limit:
        raise ValueError("native TD SwiGLU exceeds the configured clipping limit on this trace")
    if dist.get_rank() == 0:
        print(f"TD fixed trace maximum before clipping {largest.item()}, limit {layer.limit}", flush=True)


def forward(layer, x, ids, weights):
    import _moe_testkit as kit
    from mega_moe import FusedMoEForward, MegaMoEFunction, MoEForwardConfig

    if not hasattr(layer, "td_op"):
        check_activation_range(layer, x, ids)
        rank, world = dist.get_rank(), dist.get_world_size()
        kit.init_aclshmem(rank, world, kit.get_ash_size_bytes(default_gb=6))
        peer = kit.make_moonep_backward_peer_mem(ids.numel() * world, ids.numel(), x.shape[1],
                                                 x.dtype, rank, dist.group.WORLD)
        layer.td_peer = peer
        layer.td_state = SimpleNamespace(signal_mem=None, epoch=0)
        layer.td_op = FusedMoEForward(dist.group.WORLD, max_tokens_per_rank=len(x),
                                      hidden_size=x.shape[1], top_k=ids.shape[1], num_experts=layer.num_global_experts,
                                      config=MoEForwardConfig(receive_capacity_factor=float(world), enable_moonep=False))

        def release():
            layer.td_op.finalize()
            kit.ash.aclshmem_free_tensor(peer)
            kit.ash.aclshmem_finalize()

        layer.release_buffer = release
    packed = layer.gate_up_proj.transpose(1, 2).contiguous()
    return MegaMoEFunction.apply(layer.td_op, x, weights.float(), ids.int(), packed, layer.down_proj,
                                 layer.td_peer, layer.td_state)


if __name__ == "__main__":
    run(forward, "megamoe_td")
