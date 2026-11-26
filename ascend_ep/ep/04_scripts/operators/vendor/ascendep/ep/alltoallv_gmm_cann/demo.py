"""Routed MoE using the CANN AlltoAllvGMM and GMMAlltoAllv operators."""
import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional as F
from ep.runner import run


def weight_grad(left, right, counts):
    return torch_npu.npu_grouped_matmul(
        [left.T], [right], group_list=counts, split_item=3, group_type=2, group_list_type=1
    )[0]


class DispatchLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, comm, size, send, recv, counts):
        y, _, received = torch_npu.npu_alltoallv_gmm(
            x, w, comm, size, send, recv, trans_gmm_weight=True, permute_out_flag=True
        )
        ctx.save_for_backward(w, received, counts)
        ctx.exchange = comm, size, recv, send
        return y

    @staticmethod
    def backward(ctx, dy):
        w, received, counts = ctx.saved_tensors
        dx, _ = torch_npu.npu_gmm_alltoallv(dy.contiguous(), w, *ctx.exchange)
        return dx, weight_grad(dy, received, counts), None, None, None, None, None


class CombineLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, comm, size, recv, send, counts):
        y, _ = torch_npu.npu_gmm_alltoallv(x, w, comm, size, recv, send, trans_gmm_weight=True)
        ctx.save_for_backward(x, w, counts)
        ctx.exchange = comm, size, send, recv
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, counts = ctx.saved_tensors
        dx, _, received = torch_npu.npu_alltoallv_gmm(
            dy.contiguous(), w, *ctx.exchange, permute_out_flag=True
        )
        return dx, weight_grad(received, x, counts), None, None, None, None, None


def forward(layer, x, ids, weights):
    world = dist.get_world_size()
    comm = dist.group.WORLD._get_backend(x.device).get_hccl_comm_name(dist.get_rank())
    flat = ids.flatten()
    order = flat.argsort(stable=True)
    send = torch.bincount(flat, minlength=layer.num_global_experts)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    counts = recv.view(world, -1).sum(0)
    send, recv = send.tolist(), recv.tolist()
    packed = x[order // ids.shape[1]]
    gu = DispatchLinear.apply(packed, layer.gate_up_proj, comm, world, send, recv, counts)
    gate, up = gu.chunk(2, dim=-1)
    activation = F.silu(gate.clamp(max=layer.limit)) * up.clamp(-layer.limit, layer.limit)
    output = CombineLinear.apply(activation, layer.down_proj, comm, world, recv, send, counts)
    return (output[order.argsort()].view(*ids.shape, -1) * weights.unsqueeze(-1)).sum(1)


if __name__ == "__main__":
    run(forward, "alltoallv_gmm_cann")
