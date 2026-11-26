"""HCCL AllGather input/route weights and ReduceScatter combined expert output."""
import torch
import torch.distributed as dist
from torch.nn import functional as F
from ep.runner import run


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x.new_empty((len(x)*dist.get_world_size(), *x.shape[1:]))
        dist.all_gather_into_tensor(result, x.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad):
        return Scatter.apply(grad)


class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x.new_empty((len(x)//dist.get_world_size(), *x.shape[1:]))
        dist.reduce_scatter_tensor(result, x.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad):
        return Gather.apply(grad)


def forward(layer, x, ids, weights):
    x, ids, weights = Gather.apply(x), Gather.apply(ids), Gather.apply(weights)
    out = torch.zeros_like(x)
    local = len(layer.down_proj)
    for i in range(local):
        rows, slots = torch.where(ids == dist.get_rank()*local+i)
        gate, up = F.linear(x[rows], layer.gate_up_proj[i]).chunk(2, dim=-1)
        part = F.linear(F.silu(gate.clamp(max=10))*up.clamp(-10,10), layer.down_proj[i])
        out.index_add_(0, rows, part*weights[rows, slots, None])
    return Scatter.apply(out)


if __name__ == "__main__":
    run(forward, "allgather_hccl")
