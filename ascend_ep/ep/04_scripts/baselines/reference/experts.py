"""Serial expert math with an NPU grouped matmul and a CPU reference implementation."""
import torch


def _native_matmul(rows, weights, counts, group_type):
    import torch_npu

    return torch_npu.npu_grouped_matmul(
        [rows], [weights], group_list=counts, group_list_type=1,
        split_item=3, group_type=group_type,
    )[0]


class NativeGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, weights, counts):
        ctx.save_for_backward(rows, weights, counts)
        return _native_matmul(rows.contiguous(), weights.transpose(-1, -2).contiguous(), counts, 0)

    @staticmethod
    def backward(ctx, gradient):
        rows, weights, counts = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx = _native_matmul(gradient, weights.contiguous(), counts, 0) if ctx.needs_input_grad[0] else None
        dw = _native_matmul(gradient.T, rows.contiguous(), counts, 2) if ctx.needs_input_grad[1] else None
        return dx, dw, None


def grouped_linear(rows, weights, counts, context):
    if context.device.type == "npu" and len(rows) and len(counts):
        count_tensor = torch.tensor(counts, device=rows.device, dtype=torch.int64)
        return NativeGroupedLinear.apply(rows, weights, count_tensor)
    if not counts:
        return rows.new_empty((0, weights.shape[1])) + (rows.sum() + weights.sum()) * 0
    pieces = rows.split(counts)
    return torch.cat([torch.nn.functional.linear(part, weight) for part, weight in zip(pieces, weights)])


def grouped_linear_gradient(rows, weights, counts, gradient, variant, context):
    """Issue an isolated Dgrad or Wgrad without executing the other derivative."""
    dgrad = variant == "dgrad"
    if context.device.type == "npu" and len(rows) and len(counts):
        count_tensor = torch.tensor(counts, device=rows.device, dtype=torch.int64)
        if dgrad:
            return _native_matmul(gradient.contiguous(), weights.contiguous(), count_tensor, 0)
        return _native_matmul(gradient.T, rows.contiguous(), count_tensor, 2)
    if not counts:
        return torch.zeros_like(rows if dgrad else weights)
    updates = gradient.split(counts)
    if dgrad:
        return torch.cat([dy @ w for dy, w in zip(updates, weights)])
    return torch.stack([dy.T @ x for dy, x in zip(updates, rows.split(counts))])


def gmm1(work, rows, counts, context):
    weights = torch.cat((work.w_gate, work.w_up), dim=1)
    with torch.profiler.record_function("ep_gmm1"):
        return grouped_linear(rows, weights, counts, context)


def activate(gate_up, limit):
    gate, up = gate_up.chunk(2, dim=-1)
    with torch.profiler.record_function("ep_swiglu"):
        return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def gmm2(work, rows, counts, context):
    with torch.profiler.record_function("ep_gmm2"):
        return grouped_linear(rows, work.w_down, counts, context)


def expert_forward(work, dispatched, context):
    counts = dispatched.expert_token_counts
    gate_up = gmm1(work, dispatched.expert_input, counts, context)
    return gmm2(work, activate(gate_up, work.limit), counts, context)
