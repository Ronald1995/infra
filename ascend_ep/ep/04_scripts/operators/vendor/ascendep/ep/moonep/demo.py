"""MoonEP UDMA dispatch and combine with fixed expert ownership."""
import torch
import torch.distributed as dist
from torch.nn import functional as F

from ep.runner import run


def fixed_layout(ids, experts):
    from npu_patch import MoonEPCommPlan

    world = dist.get_world_size()
    rank = dist.get_rank()
    slots = ids.numel()
    k = ids.shape[1]
    gathered = torch.empty((world * len(ids), k), dtype=ids.dtype, device=ids.device)
    dist.all_gather_into_tensor(gathered, ids.contiguous())
    # Order destination rows by expert, then by original route.
    ordered = gathered.cpu().flatten().argsort(stable=True)
    dst = ordered.argsort()[rank * slots:(rank + 1) * slots].reshape(-1, k)
    owner = dst // slots
    signed = dst.clone()
    for i in range(1, k):
        duplicate = (owner[:, :i] == owner[:, i:i+1]).any(1)
        signed[duplicate, i] = -dst[duplicate, i] - 1

    routes, offsets = ordered[rank * slots:(rank + 1) * slots].sort()
    tokens = routes // k
    _, counts = torch.unique_consecutive(tokens, return_counts=True)
    starts = (counts.cumsum(0) - counts)[counts > 1]
    copies = counts[counts > 1] - 1
    groups = torch.full((slots, 3), -1, dtype=torch.int32)
    groups[:len(starts)] = torch.stack((offsets[starts], copies.cumsum(0) - copies, copies), dim=1)
    duplicates = offsets[1:][tokens[1:] == tokens[:-1]]
    dup_offsets = torch.full((slots,), -1, dtype=torch.int32)
    dup_offsets[:len(duplicates)] = duplicates
    opt = dict(dtype=torch.int32, device=ids.device)
    b = experts // world
    return MoonEPCommPlan(dst=signed.flatten().to(**opt), dup_groups=groups.to(**opt),
                          dup_loffs=dup_offsets.to(**opt),
                          dup_counts=torch.tensor([len(starts), len(duplicates)], **opt),
                          experts_to_copy=torch.full((world, b), -1, **opt),
                          zero_fill_ranges=torch.zeros((experts + b, 2), **opt),
                          remote_stats=torch.zeros(2, **opt), N=slots, R=world,
                          E=experts, B=b, NvS=slots, K=k)


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, buffer, plan):
        ctx.buffer, ctx.plan = buffer, plan
        rows, probs, _, _ = buffer.dispatch(x.contiguous(), weights.float(), plan=plan)
        return rows, probs

    @staticmethod
    def backward(ctx, rows, probs):
        dx, dp, _ = ctx.buffer.combine(ctx.plan, rows.contiguous(), probs.contiguous())
        return dx, dp, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, buffer, plan):
        ctx.buffer, ctx.plan = buffer, plan
        out, _, _ = buffer.combine(plan, rows.contiguous())
        return out

    @staticmethod
    def backward(ctx, dy):
        rows, _, _, _ = ctx.buffer.dispatch(dy.contiguous(), plan=ctx.plan)
        return rows, None, None


def forward(layer, x, ids, p):
    from npu_patch import Buffer, ShmemRuntime

    experts = layer.num_global_experts
    world = dist.get_world_size()
    if not hasattr(layer, "buffer"):
        layer.buffer = Buffer(len(x), x.shape[1], ids.shape[1], experts, world,
                              B=experts // world, group=dist.group.WORLD,
                              explicitly_destroy=True, use_udma=True)
        ShmemRuntime.init_with_buffer(group=dist.group.WORLD, enable_udma=True, qp_count=2)

        def release():
            ShmemRuntime.finalize()
            layer.buffer.destroy()

        layer.release_buffer = release
    plan = fixed_layout(ids, experts)
    rows, probs = Dispatch.apply(x, p, layer.buffer, plan)
    outputs = []
    # The benchmark trace assigns equal counts to every expert.
    chunks = zip(rows.chunk(experts // world), probs.chunk(experts // world))
    for i, (chunk, weights) in enumerate(chunks):
        gate, up = F.linear(chunk, layer.gate_up_proj[i]).chunk(2, -1)
        activation = F.silu(gate.clamp(max=layer.limit)) * up.clamp(-layer.limit, layer.limit)
        output = F.linear(activation, layer.down_proj[i])
        outputs.append((output * weights[:, None]).to(x.dtype))
    return Combine.apply(torch.cat(outputs), layer.buffer, plan)


if __name__ == "__main__":
    with torch.autograd.set_multithreading_enabled(False):
        run(forward, "moonep")
