"""Shared machinery for the L1-L4 case handlers.

A handler receives one :class:`CaseRequest` and returns a
:class:`~common.prepared.PreparedCase`.  It owns the timed block and the untimed
correctness preparation for exactly one case; nothing here branches on a case ID.

Two helpers keep the timing boundary honest:

:func:`dispatch` / :func:`combine`
    submit and wait for one communication step, so a handler never measures a
    submit-only path by accident;
:func:`moe_case`
    the generic MoE forward/backward block.  ``MOE-002`` builds its graph in
    ``setup`` so only the backward pass is timed, and ``MOE-003`` runs forward and
    backward in the *same* call instead of adding two measurements together.
"""
from dataclasses import dataclass

import torch

from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import activate, gmm1, gmm2
from common.case_spec import FIXED_OWNER
from common.prepared import Operation, PreparedCase
from common.validation import error, opaque_row_order, operation_gradients, validate_dispatch, validate_moe
from common.workload import router

#: Cases whose measurement is the planner-driven training step rather than a
#: plain forward.  The *operation* decides this, not the case ID.
PLANNER_TRAINING = frozenset({"planner", "redundant_experts", "planner_training_steps",
                              "rank_dedup_layer", "server_dedup_layer"})


@dataclass(frozen=True)
class CaseRequest:
    """Everything one handler needs, and nothing it does not."""

    ctx: object
    case: object
    backend: object
    works: list
    traces: list
    parameters: dict
    output: object = None

    @property
    def work(self):
        return self.works[0]

    @property
    def trace(self):
        return self.traces[0]

    @property
    def variant(self):
        return self.parameters.get("variant") or "default"

    @property
    def level(self):
        return self.case.level

    @property
    def operation(self):
        return self.case.operation


def threshold_for(work):
    """Numerical tolerance for the workload's dtype.

    These thresholds are a property of the reference implementation and of the
    input dtype; a backend may not widen them, so they live in exactly one place.
    """
    if work.hidden_states.dtype == torch.bfloat16:
        return 0.04
    if work.hidden_states.dtype == torch.float16:
        return 0.012
    return 2e-5


def uses_planner(request):
    """True when this (case, variant) measures the planner-driven training step.

    The fixed-owner variant exists precisely because the owning adapter implements
    only the plain MoE path, so it must not be routed to the planner handler body.
    """
    return request.operation in PLANNER_TRAINING and request.variant != FIXED_OWNER


def dispatch(backend, work, cached=None):
    """Submit one dispatch and wait until its result is usable."""
    with torch.profiler.record_function("ep_dispatch"):
        result = backend.dispatch(work, cached)
    return backend.wait_until_ready(result)


def combine(backend, work, result, rows):
    """Submit one combine and wait until the output is materialised."""
    with torch.profiler.record_function("ep_combine"):
        output = backend.combine(work, result, rows)
    return backend.wait_until_ready(output).output


def combine_weights(work):
    """The local weighting a combine applies to the packed expert outputs."""
    scale = len(work.route.placement)
    weights = ((work.topk_ids.float() + 1) / scale).to(work.hidden_states.dtype)
    return (weights * work.topk_weights).sum(-1, keepdim=True).expand_as(work.hidden_states)


def preset_expert_output(work, cached):
    """Build the untimed expert output a combine consumes.

    The value is derived from the dispatch result (``cached``) so a combine case
    never claims to measure routing: the routing trace is pre-generated.
    """
    if cached is None:
        return None
    scale = len(work.route.placement)
    return ((cached.expert_ids.float() + 1) / scale).to(
        work.hidden_states.dtype).unsqueeze(1).expand(
            -1, work.hidden_states.shape[1]).contiguous()


def reference_dispatch(ctx, work):
    """The reference collective dispatch used as the correctmess oracle."""
    reference = CollectiveBackend(None, "alltoall")
    reference.initialize(ctx)
    return reference, reference.dispatch(work)


def gmm_reference(work, layout, ctx):
    """Reference GMM-1 output for a dispatch layout."""
    return gmm1(work, layout.expert_input, layout.expert_token_counts, ctx)


def moe_case(request, *, forward_only=None, router_gate=None):
    """The generic MoE block shared by L3 forward/backward and L4 router cases.

    ``forward_only`` and ``router_gate`` default to the case's own declaration:
    a training case measures gradients, an L4 case gates its input through the
    router layer first.
    """
    ctx, backend = request.ctx, request.backend
    op = request.operation
    threshold = threshold_for(request.work)
    use_router = request.level == "L4" if router_gate is None else router_gate
    use_backward = request.case.training if forward_only is None else not forward_only

    def forward(value):
        with torch.profiler.record_function("ep_forward"):
            training = ctx.config["benchmark"].get("phase", "train") == "train"
            with torch.set_grad_enabled(training):
                if use_router:
                    value = router(value)
                return backend.moe_forward(value)

    def gradients(value, output):
        with torch.profiler.record_function("ep_backward"):
            return operation_gradients(value, output, use_router)

    def call(item):
        index, prepared = item
        value = request.works[index]
        if op == "backward":
            # `prepared` was built in setup(), so the timed block is the backward
            # pass alone: MOE-002 must not include the forward in its latency.
            return gradients(value, prepared)
        if op == "recompute":
            from torch.utils.checkpoint import checkpoint
            output = checkpoint(lambda x: forward(value.with_hidden(x)), value.hidden_states,
                                use_reentrant=False)
        elif op == "multilayer":
            first = forward(value)
            return forward(value.with_hidden(first))
        else:
            output = forward(value)
        return gradients(value, output) if use_backward else output

    def setup(index):
        index %= len(request.works)
        # A backward graph can only be differentiated once, so it is rebuilt for
        # every iteration -- always outside the timer.
        return index, forward(request.works[index]) if op == "backward" else None

    def validate():
        checks = []
        for step, value in enumerate(request.works):
            output = forward(value)
            grads = gradients(value, output) if use_backward else None
            if hasattr(backend, "validate_forward") and not use_backward and not use_router:
                checks.append(backend.validate_forward(value, output))
            else:
                checks.append(validate_moe(ctx, value, output, grads, step, use_router))
            if op == "recompute":
                measured = call((step, None))
                checks[-1]["recompute_gradient_relative_l2"] = [
                    error(actual, reference, threshold)
                    for actual, reference in zip(measured, grads)
                ]
            if op == "multilayer":
                reference, _layout = reference_dispatch(ctx, value)
                expected = reference.moe_forward(value.with_hidden(reference.moe_forward(value)))
                checks[-1]["two_layer_relative_l2"] = error(call((step, None)), expected, threshold)
        return {"steps": checks}

    return PreparedCase(setup=setup, run=call, wait=lambda value: value, validate=validate,
                        metadata={"operation": op, "use_router": use_router,
                                  "use_backward": use_backward})


def as_operation(prepared):
    """Project a :class:`PreparedCase` onto the legacy ``Operation`` shape."""
    return Operation(prepared.setup, prepared.run, prepared.validate, prepared.wait,
                     prepared.sample_metrics)


__all__ = [
    "CaseRequest", "Operation", "PLANNER_TRAINING", "PreparedCase", "activate",
    "as_operation", "combine", "combine_weights", "dispatch", "error", "gmm1", "gmm2",
    "gmm_reference", "moe_case", "opaque_row_order", "preset_expert_output",
    "reference_dispatch", "router", "threshold_for", "uses_planner", "validate_dispatch",
    "validate_moe",
]
