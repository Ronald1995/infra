"""SGLang DeepEP intranode training with a differentiable route-weight gather."""
import torch
import torch.distributed as dist
from torch.nn import functional as F

from ep.allgather_hccl.demo import Gather
from ep.runner import run


class Route:
    def __init__(self, buffer, ids, experts):
        self.buffer = buffer
        rank_counts, _, expert_counts, in_rank, _ = buffer.get_dispatch_layout(ids, experts)
        self.kwargs = dict(topk_idx=ids, topk_weights=torch.ones_like(ids, dtype=torch.float32),
                           num_tokens_per_rank=rank_counts, num_tokens_per_expert=expert_counts,
                           is_token_in_rank=in_rank)

    def dispatch(self, x):
        return self.buffer.dispatch(x=x.contiguous(), **self.kwargs)


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, route):
        rows, _, _, counts, handle, _ = route.dispatch(x)
        route.handle, route.counts = handle, counts
        ctx.route = route
        return rows

    @staticmethod
    def backward(ctx, grad):
        dx, _, _ = ctx.route.buffer.combine(x=grad.contiguous(), handle=ctx.route.handle)
        return dx, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, route):
        ctx.route = route
        out, _, _ = route.buffer.combine(x=rows.contiguous(), handle=route.handle)
        return out

    @staticmethod
    def backward(ctx, grad):
        rows, *_ = ctx.route.dispatch(grad)
        return rows, None


def forward(layer, x, ids, weights):
    from deep_ep import Buffer

    if not hasattr(layer, "buffer"):
        # The native kernels use the group's communication window as scratch.
        # Keep the runner's barriers and gradient collectives on WORLD.
        layer.sgl_group = dist.new_group(backend="hccl")
        layer.buffer = Buffer(layer.sgl_group, int(2e9), num_qps_per_rank=1)

        def release():
            del layer.buffer
            dist.destroy_process_group(layer.sgl_group)

        layer.release_buffer = release
    route = Route(layer.buffer, ids, layer.num_global_experts)
    rows = Dispatch.apply(x, route)
    rank, token, slot = route.handle[3].reshape(-1, 3).long().unbind(1)
    probabilities = Gather.apply(weights)[rank * len(x) + token, slot]
    outputs = []
    for e, (chunk, p) in enumerate(zip(rows.split(route.counts), probabilities.split(route.counts))):
        gate, up = F.linear(chunk, layer.gate_up_proj[e]).chunk(2, -1)
        activation = F.silu(gate.clamp(max=layer.limit)) * up.clamp(-layer.limit, layer.limit)
        outputs.append(F.linear(activation, layer.down_proj[e]) * p[:, None])
    return Combine.apply(torch.cat(outputs), route)


if __name__ == "__main__":
    run(forward, "deepep_sglang", small_dims=(64, 1024, 256))
