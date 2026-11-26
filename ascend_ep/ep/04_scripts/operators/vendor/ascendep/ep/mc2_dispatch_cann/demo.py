"""Chunked CANN DispatchV2/CombineV2 routed training layer for Ascend 950DT."""
import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional as F
from ep.allgather_hccl.demo import Gather
from ep.runner import run


class Exchange:
    def __init__(self, ids, experts, group):
        self.ids = ids.int().contiguous()
        world, rank = dist.get_world_size(), dist.get_rank()
        comm = group._get_backend(ids.device).get_hccl_comm_name(rank)
        self.config = comm, world, rank, experts
        self.batch = len(ids) * world
        # Pass exact integer-valued FP32 labels through the public per-route
        # scalar channel. A fresh dispatch may receive peers in a different order.
        self.labels = (torch.arange(ids.numel(), device=ids.device, dtype=torch.float32)
                       + rank * ids.numel()).view_as(ids)
        self.ones = torch.ones_like(self.labels)

    def dispatch(self, x):
        return torch_npu.npu_moe_distribute_dispatch_v2(
            x.contiguous(), self.ids, *self.config, expert_scales=self.labels,
            shared_expert_num=0, global_bs=self.batch, expert_token_nums_type=1)

    def combine(self, rows):
        return torch_npu.npu_moe_distribute_combine_v2(
            rows.contiguous(), self.ids, self.assist, self.peer_counts, self.ones,
            *self.config, shared_expert_num=0, global_bs=self.batch)


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exchange):
        rows, _, exchange.assist, sizes, exchange.peer_counts, _, labels = exchange.dispatch(x)
        exchange.sizes = sizes.tolist()
        exchange.used = sum(exchange.sizes)
        exchange.slots = labels[:exchange.used].long()
        exchange.capacity = len(rows)
        ctx.exchange = exchange
        return rows[:exchange.used]

    @staticmethod
    def backward(ctx, grad):
        exchange = ctx.exchange
        padded = F.pad(grad, (0, 0, 0, exchange.capacity - exchange.used))
        return exchange.combine(padded), None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, exchange):
        ctx.exchange = exchange
        return exchange.combine(F.pad(rows, (0, 0, 0, exchange.capacity - exchange.used)))

    @staticmethod
    def backward(ctx, dy):
        exchange = ctx.exchange
        grad, _, _, _, _, _, labels = exchange.dispatch(dy)
        inverse = torch.empty(exchange.batch * exchange.ids.shape[1], dtype=torch.long, device=dy.device)
        inverse[labels[:exchange.used].long()] = torch.arange(exchange.used, device=dy.device)
        return grad[inverse[exchange.slots]], None


def forward(layer, x, ids, weights):
    if not hasattr(layer, "mc2_group"):
        layer.mc2_group = dist.new_group(backend="hccl")
        layer.release_buffer = lambda: dist.destroy_process_group(layer.mc2_group)
    exchanges, batches = [], []
    for tokens, routes in zip(x.split(512), ids.split(512)):
        exchange = Exchange(routes, layer.num_global_experts, layer.mc2_group)
        packed = Dispatch.apply(tokens, exchange)
        batches.append(packed.split(exchange.sizes))
        exchanges.append(exchange)

    # Process each expert once for the full sequence, then restore each
    # communication block's native receive layout for CombineV2.
    expert_outputs = []
    for e, pieces in enumerate(zip(*batches)):
        gate, up = F.linear(torch.cat(pieces), layer.gate_up_proj[e]).chunk(2, dim=-1)
        activation = F.silu(gate.clamp(max=layer.limit)) * up.clamp(-layer.limit, layer.limit)
        values = F.linear(activation, layer.down_proj[e])
        expert_outputs.append(values.split([len(piece) for piece in pieces]))
    outputs = []
    for parts, exchange, probs in zip(zip(*expert_outputs), exchanges, weights.split(512)):
        scales = Gather.apply(probs).flatten()[exchange.slots]
        outputs.append(Combine.apply(torch.cat(parts) * scales[:, None], exchange))
    return torch.cat(outputs)


if __name__ == "__main__":
    run(forward, "mc2_dispatch_cann", small_dims=(64, 1024, 256))
