"""Lueelu UDMA dispatch/GMM and GMM/combine routed training layer."""
import torch
import torch.distributed as dist
from torch.nn import functional as F

from deep_ep_experimental.ep_gmm_fused import (
    alltoallv_gmm, destroy_udma_group, gmm_alltoallv, init_udma_group,
)
from ep.alltoallv_gmm_cann.demo import weight_grad
from ep.runner import run


class Route:
    def __init__(self, ids, experts):
        self.world = dist.get_world_size()
        self.send = torch.bincount(ids.flatten(), minlength=experts)
        self.recv = torch.empty_like(self.send)
        dist.all_to_all_single(self.recv, self.send)
        self.counts = self.recv.view(self.world, -1).sum(0)
        self.source_rows = ids.numel()
        self.received_rows = self.recv.sum().item()
        self.params = dict(ep_size=self.world, expert_num=experts)

    def dispatch(self, x, weights):
        return alltoallv_gmm(x.contiguous(), weights.contiguous(), self.send, self.recv,
                             received_rows=self.received_rows, output_all_to_allv=True, **self.params)

    def combine(self, x, weights):
        out = x.new_empty((self.source_rows, weights.shape[2]))
        return gmm_alltoallv(x.contiguous(), weights.contiguous(), self.send, self.recv, out=out, **self.params)


class DispatchLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, route):
        y, received = route.dispatch(x, weights.transpose(1, 2))
        ctx.route = route
        ctx.save_for_backward(weights, received)
        return y

    @staticmethod
    def backward(ctx, dy):
        weights, received = ctx.saved_tensors
        return ctx.route.combine(dy, weights), weight_grad(dy, received, ctx.route.counts), None


class CombineLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, route):
        ctx.route = route
        ctx.save_for_backward(x, weights)
        return route.combine(x, weights.transpose(1, 2))

    @staticmethod
    def backward(ctx, dy):
        x, weights = ctx.saved_tensors
        dx, received = ctx.route.dispatch(dy, weights)
        return dx, weight_grad(received, x, ctx.route.counts), None


def forward(layer, x, ids, weights):
    if not hasattr(layer, "release_buffer"):
        init_udma_group(group=dist.group.WORLD)
        layer.release_buffer = destroy_udma_group
    route = Route(ids, layer.num_global_experts)
    order = ids.flatten().argsort(stable=True)
    gate, up = DispatchLinear.apply(x[order // ids.shape[1]], layer.gate_up_proj, route).chunk(2, -1)
    activation = F.silu(gate.clamp(max=layer.limit)) * up.clamp(-layer.limit, layer.limit)
    returned = CombineLinear.apply(activation, layer.down_proj, route)
    return (returned[order.argsort()].view(*ids.shape, -1) * weights.unsqueeze(-1)).sum(1)


if __name__ == "__main__":
    # The UDMA session is shared by forward and backward on its owning host thread.
    with torch.autograd.set_multithreading_enabled(False):
        run(forward, "ep_gmm_fused_lueelu")
