"""Standalone compute baselines: GMM forward/backward and SwiGLU forward/backward.

The cases keep the measurement boundary of the original ``HW-010``/``HW-011``:
no router, no dispatch, no combine.  Each variant selects exactly one kernel
group (``gmm1``, ``gmm2``, and ``dgrad``/``wgrad`` for either), so a GMM result is
never mixed with an EP operator's communication cost.
"""
import torch

from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import activate, gmm1, gmm2, grouped_linear_gradient
from baselines.l0.contract import Prepared
from baselines.l0.verification.vocabulary import requested
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError, error

VARIANTS = ("gmm1", "gmm2", "gmm1_dgrad", "gmm1_wgrad", "gmm2_dgrad", "gmm2_wgrad",
            "forward", "backward")


def prepare(ctx, case, work, size, variant):
    """Return ``(run, wait, validate, cleanup, evidence, payload)``."""
    if work is None:
        raise BenchmarkUnavailable("environment_unavailable", "compute baselines need a workload")
    if case.id == "HW-011":
        expected, run = _swiglu(ctx, work, variant)
        payload = None
        operation = "compute"
    else:
        expected, run = _gmm(ctx, work, variant)
        payload = None
        operation = "compute"

    def validate():
        got = run()
        expected_values = expected if isinstance(expected, tuple) else (expected,)
        got_values = got if isinstance(got, tuple) else (got,)
        threshold = 0.04 if work.hidden_states.dtype != torch.float32 else 2e-5
        errors = [error(a, b, threshold) for a, b in zip(got_values, expected_values)]
        return {"relative_l2": errors}

    # No profiler evidence here, so the execution resource stays unobserved.
    request = requested(backend="torch_memory", operation=operation,
                        transport_protocol="auto")
    return Prepared(run=run, validate=validate, request=request, payload=payload)


def _dispatched(ctx, work):
    reference = CollectiveBackend(None, "alltoall")
    reference.initialize(ctx)
    dispatched = reference.dispatch(work)
    rows = dispatched.expert_input.detach().requires_grad_()
    return rows, dispatched.expert_token_counts


def _swiglu(ctx, work, variant):
    rows, counts = _dispatched(ctx, work)
    gate_up = gmm1(work, rows, counts, ctx).detach().requires_grad_()
    activation = activate(gate_up, work.limit).detach().requires_grad_()
    act = lambda: activate(gate_up, work.limit)
    upstream = torch.ones_like(activation)
    expected = torch.nn.functional.silu(gate_up.float().chunk(2, dim=-1)[0].clamp(max=work.limit))
    expected = expected * gate_up.float().chunk(2, dim=-1)[1].clamp(-work.limit, work.limit)
    if variant == "backward":
        return (torch.autograd.grad(expected, gate_up, upstream.float())[0],
                lambda: torch.autograd.grad(act(), gate_up, upstream)[0])
    return expected, act


def _gmm(ctx, work, variant):
    rows, counts = _dispatched(ctx, work)
    activation = activate(gmm1(work, rows, counts, ctx).detach().requires_grad_(),
                          work.limit).detach().requires_grad_()
    first = variant.startswith("gmm1")
    x = rows if first else activation
    weights = (work.w_gate, work.w_up) if first else (work.w_down,)
    cpu_x = x.detach().cpu().float().requires_grad_()
    cpu_weights = [weight.detach().cpu().float().requires_grad_() for weight in weights]
    packed = torch.cat(cpu_weights, dim=1) if first else cpu_weights[0]
    if counts:
        expected = torch.cat([torch.nn.functional.linear(piece, weight)
                              for piece, weight in zip(cpu_x.split(counts), packed)])
    else:
        expected = cpu_x.new_empty((0, packed.shape[1])) + (cpu_x.sum() + packed.sum()) * 0
    forward = lambda: (gmm1 if first else gmm2)(work, x, counts, ctx)
    if not variant.endswith(("dgrad", "wgrad")):
        return expected, forward
    derivative = variant.split("_")[-1]
    reference_inputs = (cpu_x,) if derivative == "dgrad" else (packed,)
    native_weights = torch.cat(weights, dim=1) if first else weights[0]
    upstream = x.new_ones(expected.shape)
    gradient = torch.autograd.grad(expected.sum(), reference_inputs)
    return gradient, (lambda: grouped_linear_gradient(x, native_weights, counts, upstream,
                                                      derivative, ctx))


def request_for(case, params):
    return requested(backend="torch_memory", operation="compute", transport_protocol="auto")
