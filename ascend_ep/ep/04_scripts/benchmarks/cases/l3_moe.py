"""L3: complete MoE execution.

The three numbered cases are deliberately distinct measurements and must not be
derived from one another:

``MOE-001`` forward
    one MoE forward inside the timer.
``MOE-002`` backward only
    the forward graph is built in ``setup`` (untimed) and only the backward pass is
    measured.
``MOE-003`` forward + backward
    forward *and* backward run inside a single timed call.  It is not the sum of
    ``MOE-001`` and ``MOE-002`` and the report must never add them.

Correctness covers the forward output, the input gradient and every expert weight
gradient; the router weight gradient is added when the case gates through a router.
"""
import torch

from common.prepared import PreparedCase
from common.validation import validate_moe

from .common import CaseRequest, moe_case


def forward(request):
    """MOE-001: one MoE forward, no gradients."""
    return moe_case(request, forward_only=True, router_gate=False)


def backward(request):
    """MOE-002: backward only; the forward graph is built in setup."""
    return moe_case(request, forward_only=False, router_gate=False)


def forward_backward(request):
    """MOE-003: forward and backward in the same timed call."""
    return moe_case(request, forward_only=False, router_gate=False)


def recompute(request):
    """Checkpointed forward: the measured block includes the recomputation."""
    return moe_case(request, forward_only=False, router_gate=False)


def imbalanced(request):
    """Forward under a skewed routing pattern selected by the case transform."""
    return moe_case(request, forward_only=True, router_gate=False)


def dtype(request):
    """Forward at the configuration's dtype; the tolerance follows the dtype."""
    return moe_case(request, forward_only=True, router_gate=False)


def multilayer(request):
    """Two stacked MoE layers in one timed call."""
    return moe_case(request, forward_only=False, router_gate=False)


def dynamic_tokens(request):
    """Forward over several per-rank token counts in one run."""
    return moe_case(request, forward_only=True, router_gate=False)


def concurrent(request):
    """MOE-007: two independent requests submitted concurrently.

    Both requests are inside the same timed block, so the sample is the slower of
    two overlapping forwards rather than two serialised forwards.
    """
    backend, work = request.backend, request.work
    backend.prepare_concurrent()
    second = work.with_hidden(work.hidden_states.detach().clone())

    def run(_):
        with torch.no_grad(), torch.profiler.record_function("ep_concurrent_forward"):
            return backend.concurrent_forward(work, second)

    def validate():
        first_output, second_output = run(None)
        return {"requests": [validate_moe(request.ctx, work, first_output, None),
                             validate_moe(request.ctx, second, second_output, None)]}

    return PreparedCase(run=run, wait=lambda value: value, validate=validate,
                        metadata={"operation": "concurrent", "requests_per_sample": 2})


HANDLERS = {
    "l3.forward": forward,
    "l3.backward": backward,
    "l3.forward_backward": forward_backward,
    "l3.recompute": recompute,
    "l3.imbalanced": imbalanced,
    "l3.dtype": dtype,
    "l3.concurrent": concurrent,
    "l3.multilayer": multilayer,
    "l3.dynamic_tokens": dynamic_tokens,
}
