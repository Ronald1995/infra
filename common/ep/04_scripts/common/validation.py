"""Numerical and structural checks independent of measured operations."""
from dataclasses import replace

import numpy as np
import torch

from baselines.reference.collectives import CollectiveBackend
from common.routing import generate
from common.workload import activation_inputs, expert_parameters, router


class CorrectnessError(Exception):
    pass


def error(actual, expected, threshold):
    if actual is None or expected is None or actual.shape != expected.shape:
        raise CorrectnessError("missing tensor or tensor shape mismatch")
    left = actual.detach().reshape(-1).cpu()
    right = expected.detach().reshape(-1).cpu()
    delta = sum((a.float() - b.float()).double().square().sum().item()
                for a, b in zip(left.split(1 << 20), right.split(1 << 20)))
    scale = sum(b.double().square().sum().item() for b in right.split(1 << 20))
    relative = delta**0.5 / max(1e-9, scale**0.5)
    if not np.isfinite(relative) or relative > threshold:
        raise CorrectnessError(f"relative L2 {relative:.6g} exceeds {threshold:g}")
    return relative


def validate_dispatch(work, result, ctx, trace, threshold=0):
    expected = tuple(trace["expert_counts"][e] for e in work.local_experts)
    if result.expert_token_counts != expected or len(result.expert_input) != sum(expected):
        raise CorrectnessError(f"expert counts differ: {result.expert_token_counts} vs {expected}")
    if result.expert_ids is None:
        raise CorrectnessError("dispatch adapter must expose expert IDs")
    if result.routing_handle is None:
        raise CorrectnessError("dispatch did not supply a combine handle")
    # The reference exchanges activation rows only for the same logical routes.
    ref_backend = CollectiveBackend(None, "alltoall")
    ref_backend.initialize(ctx)
    ref = ref_backend.dispatch(work)
    if result.route_slots is None:
        if result.layout != "local_expert_contiguous_opaque_reverse_route":
            raise CorrectnessError("dispatch adapter must expose inverse route slots")
        if not torch.equal(result.expert_ids, ref.expert_ids):
            raise CorrectnessError("expert boundaries disagree with reference")

        def canonical(rows):
            # Lexicographic matching is only used for APIs whose reverse map is opaque.
            data = rows.detach().cpu().numpy().astype(np.float32)
            order = np.lexsort(data.T[::-1])
            return torch.from_numpy(data[order])

        checks = [error(canonical(actual.float()), canonical(expected.float()), 0)
                  for actual, expected in zip(result.expert_input.split(result.expert_token_counts),
                                              ref.expert_input.split(ref.expert_token_counts))]
        return {"dispatch_relative_l2": max(checks, default=0), "route_validation": "per_expert_token_multiset"}
    observed = result.route_slots.argsort()
    wanted = ref.route_slots.argsort()
    if not torch.equal(result.route_slots[observed], ref.route_slots[wanted]) or not torch.equal(
        result.expert_ids[observed], ref.expert_ids[wanted]
    ):
        raise CorrectnessError("native dispatch route slots/expert IDs disagree with the reference")
    return {"dispatch_relative_l2": error(result.expert_input[observed], ref.expert_input[wanted], threshold)}


def operation_gradients(work, output, use_router=False):
    values = (work.hidden_states, work.router_weight if use_router else work.topk_weights,
              work.w_gate, work.w_up, work.w_down)
    return torch.autograd.grad(output, values, work.grad_output, allow_unused=True)


def cpu_oracle(ctx, work, step, use_router, training):
    cfg = ctx.config
    dtype = work.hidden_states.dtype
    inputs = [activation_inputs(torch, cfg, r, step, n) for r, n in enumerate(work.route.tokens)]
    x = torch.cat([pair[0] for pair in inputs]).to(dtype).float().requires_grad_()
    dy = torch.cat([pair[1] for pair in inputs]).to(dtype).float()
    params = [expert_parameters(torch, cfg, e) for e in range(len(work.route.placement))]
    gate, up, down = [torch.stack(items).to(dtype).float().requires_grad_() for items in zip(*params)]
    routing = [generate(cfg, rank, step) for rank in range(ctx.size)]
    ids = torch.from_numpy(np.concatenate([r.ids for r in routing]))
    weights = torch.from_numpy(np.concatenate([r.weights for r in routing])).to(dtype).float().requires_grad_()
    router_weight = work.router_weight.detach().cpu().float().requires_grad_()
    if use_router:
        # The router is replicated. Its local gradient is measured before any
        # optimizer-level replica reduction, while experts see all source ranks.
        logits = torch.cat([
            part @ (router_weight if rank == ctx.rank else router_weight.detach()).T
            for rank, part in enumerate(x.split(work.route.tokens))
        ]).to(dtype).float()
        logits, ids = logits.topk(ids.shape[-1])
        weights = logits.softmax(-1).to(dtype).float()
    output = torch.zeros_like(x)
    for expert in range(len(params)):
        rows, cols = torch.where(ids == expert)
        g = torch.nn.functional.linear(x[rows], gate[expert])
        u = torch.nn.functional.linear(x[rows], up[expert])
        out = torch.nn.functional.linear(
            torch.nn.functional.silu(g.clamp(max=work.limit)) * u.clamp(-work.limit, work.limit),
            down[expert],
        )
        output = output.index_add(0, rows, out * weights[rows, cols, None])
    start = sum(work.route.tokens[:ctx.rank])
    stop = start + work.route.tokens[ctx.rank]
    result = [output[start:stop]]
    if training:
        grad = torch.autograd.grad(output, (x, router_weight if use_router else weights, gate, up, down), dy)
        result.extend((grad[0][start:stop], grad[1] if use_router else grad[1][start:stop]))
        result.extend(value[list(work.local_experts)] for value in grad[2:])
    return result


def validate_moe(ctx, work, output, grads, step=0, use_router=False):
    shape = ctx.config["model"]
    elements = (sum(work.route.tokens) * shape["hidden_size"] +
                3 * shape["hidden_size"] * shape["intermediate_size"] * shape["num_experts"])
    if elements <= 2_000_000:
        expected = cpu_oracle(ctx, work, step, use_router, grads is not None)
        scope = "cpu_fp32_oracle"
    else:
        ref = CollectiveBackend(None, "alltoall")
        ref.initialize(ctx)
        ref_work = router(work) if use_router else work
        ref_output = ref.moe_forward(ref_work)
        expected = [ref_output]
        if grads is not None:
            expected.extend(operation_gradients(work, ref_output, use_router))
        scope = "target_shape_collective_reference"
    names = ["output", "input_grad", "router_grad" if use_router else "routing_weight_grad",
             "gate_grad", "up_grad", "down_grad"]
    threshold = 0.04 if shape["dtype"] == "bfloat16" else 0.012 if shape["dtype"] == "float16" else 2e-5
    actual = [output, *(grads if grads is not None else ())]
    errors = {}
    for name, got, want in zip(names, actual, expected):
        try:
            errors[name] = error(got, want, threshold)
        except CorrectnessError as exc:
            raise CorrectnessError(f"{name}: {exc}") from exc
    return {"scope": scope, "relative_l2": errors, "threshold": threshold}
