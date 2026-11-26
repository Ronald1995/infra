"""Expanded DeepSeek EPBuffer dispatch/combine with reverse communication in autograd."""
import torch
import torch.distributed as dist
from torch.nn import functional as F
from ep.runner import run


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, ids, buffer, handles):
        rows, _, probs, handle, _ = buffer.dispatch(
            x.detach().contiguous(), topk_idx=ids.long(), topk_weights=p.detach().float(), num_experts=24,
            expert_alignment=1, do_expand=True, do_cpu_sync=True)
        handles.append(handle)
        ctx.buffer, ctx.handle = buffer, handle
        return rows, probs

    @staticmethod
    def backward(ctx, drows, dp):
        dx, dp, _ = ctx.buffer.combine(drows.contiguous(), ctx.handle, topk_weights=dp.contiguous())
        return dx, dp, None, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, buffer, handle):
        ctx.buffer, ctx.handle = buffer, handle
        y, _, _ = buffer.combine(rows.detach().contiguous(), handle)
        return y

    @staticmethod
    def backward(ctx, dy):
        rows, _, _, _, _ = ctx.buffer.dispatch(dy.contiguous(), handle=ctx.handle,
                                                do_expand=True, do_cpu_sync=False)
        return rows, None, None


def expanded_forward(layer, x, ids, p, buffer_type, **buffer_options):
    if not hasattr(layer, "buffer"):
        layer.buffer = buffer_type(dist.group.WORLD, num_max_tokens_per_rank=len(x),
                                hidden=x.shape[1], num_topk=6, explicitly_destroy=True, **buffer_options)
        layer.release_buffer = layer.buffer.destroy
    handles = []
    rows, probs = Dispatch.apply(x, p, ids, layer.buffer, handles)
    counts = handles[0].psum_num_recv_tokens_per_expert.tolist()
    start = 0
    outputs = []
    for e, end in enumerate(counts):
        g, u = F.linear(rows[start:end], layer.gate_up_proj[e]).chunk(2, -1)
        v = F.linear(F.silu(g.clamp(max=10)) * u.clamp(-10,10), layer.down_proj[e])
        outputs.append((v * probs[start:end,None]).to(x.dtype))
        start = end
    return Combine.apply(torch.cat(outputs), layer.buffer, handles[0])


def forward(layer, x, ids, p):
    from deep_ep import EPBuffer
    return expanded_forward(layer, x, ids, p, EPBuffer, allow_hybrid_mode=False)


if __name__ == "__main__":
    run(forward, "deepep_ascend")
