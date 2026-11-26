"""Autograd and symmetric-memory projection handling for MoonEP dev_npu."""
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from baselines.reference.experts import expert_forward
from common.backend import DispatchResult
from common.validation import CorrectnessError
from common.workload import router


@dataclass
class PlannerRoute:
    plan: object = None
    offsets: object = None


@dataclass
class TrainingStep:
    output: object
    gradients: tuple
    stages: dict
    received_rows: int
    replica_plan: object

    def timing_ns(self):
        return {name: first.elapsed_time(last) * 1e6 for name, (first, last) in self.stages.items()}


@contextmanager
def stage(name, stages):
    with torch.profiler.record_function(f"ep_{name}"):
        if stages is not None:
            first = torch.npu.Event(enable_timing=True)
            last = torch.npu.Event(enable_timing=True)
            first.record()
            stages[name] = first, last
        yield
        if stages is not None:
            last.record()


def local_expert_counts(offsets, experts, rank, local):
    """The native prefix covers E global home positions followed by B local slots."""
    if len(offsets) != experts + local:
        raise CorrectnessError("planner returned a different number of home and replica offsets")
    ends = [0, *offsets]
    counts = [stop - start for start, stop in zip(ends, ends[1:])]
    start = rank * local
    stop = start + local
    if any(n < 0 for n in counts) or any(counts[:start]) or any(counts[stop:experts]):
        raise CorrectnessError("planner placed receive rows in a non-local home group")
    return tuple(counts[start:stop] + counts[experts:])


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probabilities, backend, work, route):
        if route.plan is None:
            rows, weights, route.offsets, route.plan = backend.planned_dispatch(work)
        else:
            rows, weights, _, _ = backend.buffer.dispatch(
                x.detach().contiguous(), probabilities.detach().float().contiguous(), plan=route.plan
            )
        ctx.buffer = backend.buffer
        ctx.plan = route.plan
        ctx.weight_dtype = probabilities.dtype
        return rows, weights

    @staticmethod
    def backward(ctx, grad_rows, grad_weights):
        with torch.profiler.record_function("ep_dispatch_backward"):
            dx, dp, _ = ctx.buffer.combine(
                ctx.plan, grad_rows.contiguous(), route_weights_nvs=grad_weights.contiguous()
            )
        return dx, dp.to(ctx.weight_dtype), None, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, buffer, plan):
        ctx.buffer, ctx.plan = buffer, plan
        return buffer.combine(plan, rows.contiguous())[0]

    @staticmethod
    def backward(ctx, grad):
        with torch.profiler.record_function("ep_combine_backward"):
            rows, _, _, _ = ctx.buffer.dispatch(grad.contiguous(), plan=ctx.plan)
        return rows, None, None


class Projections:
    """Create virtual E+B projections while physically backing only home and slot rows."""
    def __init__(self, work, ctx):
        from moonep.buffer import (
            create_full_grad_and_reduce_buffer,
            create_full_weight,
            projection_home_slice,
            projection_slot_slice,
        )

        self.home = projection_home_slice
        self.slots = projection_slot_slice
        self.weights = []
        self.grad_pairs = []
        experts, local = len(work.route.placement), len(work.local_experts)
        for weight in (work.w_gate, work.w_up, work.w_down):
            m, n = weight.shape[-2:]
            # The upstream constructors read only the home slice and this
            # rank's slots. Broadcast a zero scalar without allocating a
            # physical global expert table.
            weight_table = weight.new_zeros(()).expand(experts, m, n)
            pool = create_full_weight(
                experts, local, m, n, ctx.rank, ctx.size, weight_table,
                group=ctx.ep_group, device=ctx.local_rank,
            )
            self.home(pool).copy_(weight.detach())
            zero = weight.new_zeros((), dtype=torch.float32)
            grad, reduce = create_full_grad_and_reduce_buffer(
                experts, local, m, n, ctx.rank, ctx.size,
                zero.expand(experts, m, n), zero.expand(ctx.size, local, m, n),
                group=ctx.ep_group, device=ctx.local_rank,
            )
            self.weights.append(pool)
            self.grad_pairs.append((grad, reduce))

    def grad_inputs(self):
        homes, slots = zip(*(
            (self.home(pool).detach().requires_grad_(), self.slots(pool).detach().requires_grad_())
            for pool in self.weights
        ))
        return (*homes, *slots)

    def pack_gradients(self, gradients):
        for index, (grad, _) in enumerate(self.grad_pairs):
            self.home(grad).copy_(gradients[index])
            self.slots(grad).copy_(gradients[index + len(self.weights)])

    def home_gradients(self):
        return tuple(self.home(grad) for grad, _ in self.grad_pairs)


def prepare_route(backend, work):
    routed = router(work)
    _, _, offsets, plan = backend.planned_dispatch(routed)
    return PlannerRoute(plan, offsets)


def run_training_step(backend, work, cached=None, collect_stages=False):
    ctx, buffer, projections = backend.context, backend.buffer, backend.projections
    events = {} if collect_stages else None
    with torch.profiler.record_function("ep_forward"):
        with torch.profiler.record_function("ep_router"):
            routed = router(work)
        route = PlannerRoute() if cached is None else cached
        with stage("planner_dispatch", events):
            rows, weights = Dispatch.apply(routed.hidden_states, routed.topk_weights, backend, routed, route)
            counts = local_expert_counts(route.offsets.tolist(), len(work.route.placement),
                                         ctx.rank, len(work.local_experts))
        with stage("prefetch_weight", events):
            buffer.prefetch_projections(route.plan, projections.weights)
        actual = sum(counts)
        if actual > rows.shape[0] or actual > weights.shape[0]:
            raise CorrectnessError("planner receive offsets exceed buffer capacity")
        grad_inputs = projections.grad_inputs()
        gate, up, down = [torch.cat((home, slot)) for home, slot in zip(grad_inputs[:3], grad_inputs[3:])]
        inputs = replace(work, w_gate=gate, w_up=up, w_down=down)
        packed = DispatchResult(rows[:actual], counts, route.plan, "home_then_slots")
        values = expert_forward(inputs, packed, ctx)
        applied = (values * weights[:actual, None]).to(values.dtype)
        capacity_rows = torch.nn.functional.pad(applied, (0, 0, 0, rows.shape[0] - actual))
        with torch.profiler.record_function("ep_combine"):
            result = Combine.apply(capacity_rows, buffer, route.plan)

    with torch.profiler.record_function("ep_backward"):
        gradients = torch.autograd.grad(
            result, (work.hidden_states, work.router_weight, *grad_inputs),
            work.grad_output, allow_unused=False,
        )
        with stage("reduce_grad", events):
            projections.pack_gradients(gradients[2:])
            buffer.reduce_projection_grads(route.plan, projections.grad_pairs)
        reduced = projections.home_gradients()
    return TrainingStep(result, (*gradients[:2], *reduced), events or {}, actual, route.plan.experts_to_copy)
