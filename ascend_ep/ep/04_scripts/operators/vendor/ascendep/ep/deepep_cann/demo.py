"""CANN ElasticBuffer MoE EP dispatch/combine and their training adjoints."""
import torch
import torch.distributed as dist
from torch.nn import functional as F
from ep.runner import run


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ids, p, buffer, handles):
        rows, _, probs, handle = buffer.dispatch(x.contiguous(), topk_idx=ids.int(),
                                                 topk_weights=p.float(), num_experts=24)
        handles.append(handle)
        ctx.buffer, ctx.handle = buffer, handle
        return rows, probs

    @staticmethod
    def backward(ctx, grad, dp):
        dx, weights = ctx.buffer.combine(grad.contiguous(), ctx.handle, topk_weights=dp.float().contiguous())
        return dx, None, weights, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, buffer, handle):
        ctx.buffer, ctx.handle = buffer, handle
        return buffer.combine(rows.contiguous(), handle)[0]

    @staticmethod
    def backward(ctx, grad):
        rows, _, _, _ = ctx.buffer.dispatch(grad.contiguous(), handle=ctx.handle)
        return rows, None, None


def forward(layer, x, ids, p):
    from cann_ops_transformer.ops.mc2.common.elastic_buffer import ElasticBuffer

    if not hasattr(layer, "buffer"):
        group = dist.new_group(backend="hccl")
        layer.buffer = ElasticBuffer(group, num_max_tokens_per_rank=len(x), hidden=x.shape[1],
                                     num_topk=ids.shape[1], explicitly_destroy=True)

        def release():
            layer.buffer.destroy()
            dist.destroy_process_group(group)

        layer.release_buffer = release
    handles = []
    rows, probs = Dispatch.apply(x, ids, p, layer.buffer, handles)
    outputs = []
    # This trace has exactly the same number of received rows per expert.
    local = len(layer.down_proj)
    for i, (chunk, weights) in enumerate(zip(rows.chunk(local), probs.chunk(local))):
        gate, up = F.linear(chunk, layer.gate_up_proj[i]).chunk(2, -1)
        hidden = F.silu(gate.clamp(max=layer.limit)) * up.clamp(-layer.limit, layer.limit)
        outputs.append((F.linear(hidden, layer.down_proj[i]) * weights[:, None]).to(x.dtype))
    return Combine.apply(torch.cat(outputs), layer.buffer, handles[0])


if __name__ == "__main__":
    run(forward, "deepep_cann")
