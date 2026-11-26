"""L4: router, planner and training-level capabilities.

L4 does not reuse the L3 forward unchanged.  A router case gates the input through
the router layer *inside* the timed block, and the planner cases measure the
planner-driven training step, which is a different code path with different
gradients (``prepare_route`` / ``run_training_step``).

Two variants measure different things on purpose:

``fresh_plan``
    the planner runs every step.
``cached_plan``
    the route is planned once, outside the timer, and the measured step reuses it.

The fixed-owner variant is the exception: the owning adapter implements only the
plain MoE path, so that variant measures the plain path and says so.
"""
from common.prepared import PreparedCase
from common.validation import validate_moe

from .common import CaseRequest, moe_case, reference_dispatch, uses_planner


def router_forward(request):
    """LAYER-001: router gate + MoE forward, forward only."""
    return moe_case(request, forward_only=True, router_gate=True)


def router_training(request):
    """LAYER-002: router gate + forward + backward, router weight gradient included."""
    return moe_case(request, forward_only=False, router_gate=True)


def trace_replay(request):
    """LAYER-007: replay a recorded trace through the L3 boundary."""
    return moe_case(request, forward_only=True, router_gate=False)


def router_large_training(request):
    """LAYER-009: the large-batch router training path."""
    return moe_case(request, forward_only=False, router_gate=True)


def _planned(request):
    """The planner-driven training step shared by the planner cases."""
    backend, works = request.backend, request.works
    if not uses_planner(request):
        # The fixed-owner variant measures the plain MoE path; routing it to the
        # planner would compare two different implementations.
        return moe_case(request, forward_only=False, router_gate=True)

    from operators.moonep_dynamic_planner.training import prepare_route, run_training_step

    cached = prepare_route(backend, works[0]) if request.variant == "cached_plan" else None
    collect_stages = request.operation == "redundant_experts"

    def run(_):
        return [run_training_step(backend, work, cached=cached, collect_stages=collect_stages)
                for work in works]

    def validate():
        checked = []
        for step, work in enumerate(works):
            value = run_training_step(backend, work, cached=cached)
            check = validate_moe(request.ctx, work, value.output, value.gradients, step,
                                 use_router=True)
            rank_counts = request.ctx.gather(value.received_rows)
            copies = int((value.replica_plan[request.ctx.rank] >= 0).sum().item())
            copied = sum(request.ctx.gather(copies))
            check.update(
                planned_receive_rows=rank_counts,
                planned_rank_max_mean=max(rank_counts) * request.ctx.size / sum(rank_counts),
                planned_expert_copies=copied,
                planned_redundant_weight_bytes=copied * sum(
                    tensor[0].numel() * tensor.element_size()
                    for tensor in (work.w_gate, work.w_up, work.w_down)
                ),
            )
            checked.append(check)
        return {"steps": checked,
                "plan_mode": "cached_plan" if cached is not None else "fresh_plan"}

    return PreparedCase(run=run, wait=lambda value: value, validate=validate,
                        sample_metrics=lambda values: values[0].timing_ns(),
                        metadata={"operation": request.operation,
                                  "plan_mode": "cached_plan" if cached is not None else "fresh_plan",
                                  "correctness": ["input_gradient", "expert_weight_gradient",
                                                  "planner_stages"]})


HANDLERS = {
    "l4.router_forward": router_forward,
    "l4.router_training": router_training,
    "l4.trace_replay": trace_replay,
    "l4.router_large_training": router_large_training,
    "l4.planner": _planned,
    "l4.redundant_experts": _planned,
    "l4.planner_training_steps": _planned,
    "l4.rank_dedup_layer": _planned,
    "l4.server_dedup_layer": _planned,
}
