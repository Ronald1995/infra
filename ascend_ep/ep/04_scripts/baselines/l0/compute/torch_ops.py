"""Standalone compute baselines: GMM forward/backward and SwiGLU forward/backward.

The cases keep the measurement boundary of the original ``HW-010``/``HW-011``:
no router, no dispatch, no combine.  Each variant selects exactly one kernel
group (``gmm1``, ``gmm2``, and ``dgrad``/``wgrad`` for either), so a GMM result is
never mixed with an EP operator's communication cost.

Two shape modes exist:

``workload``
    use the expert rows the configured routing actually produces.  This measures
    compute efficiency in the real EP scenario and therefore differs per EP size.
``fixed``
    use ``benchmark.compute.fixed_shape`` (M/N/K and group count) at every EP
    size.  This measures the kernel itself; the shape is recorded with the
    result so the two modes are never compared against each other.
"""
from types import SimpleNamespace

import torch

from baselines.l0.contract import Prepared
from baselines.l0.verification.path_verifier import Evidence
from baselines.l0.verification.vocabulary import requested
from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import activate, gmm1, gmm2, grouped_linear_gradient
from common.backend import BenchmarkUnavailable
from common.validation import error

VARIANTS = ("gmm1", "gmm2", "gmm1_dgrad", "gmm1_wgrad", "gmm2_dgrad", "gmm2_wgrad",
            "forward", "backward")


def prepare(ctx, case, work, size, variant, params=None):
    """Return a :class:`Prepared` for one standalone compute case."""
    if work is None:
        raise BenchmarkUnavailable("environment_unavailable", "compute baselines need a workload")
    params = params or {}
    shape_mode = params.get("shape_mode") or "workload"
    if case.id == "HW-011":
        expected, run, shape = _swiglu(ctx, work, variant, shape_mode, params)
    else:
        expected, run, shape = _gmm(ctx, work, variant, shape_mode, params)
    payload = None
    operation = "compute"

    def validate():
        got = run()
        expected_values = expected if isinstance(expected, tuple) else (expected,)
        got_values = got if isinstance(got, tuple) else (got,)
        threshold = 0.04 if work.hidden_states.dtype != torch.float32 else 2e-5
        errors = [error(a, b, threshold) for a, b in zip(got_values, expected_values)]
        return {"relative_l2": errors, "shape_mode": shape_mode, "shape": shape}

    evidence = [Evidence(source="native_descriptor",
                         claim={"execution_resource": "AIV"},
                         detail="torch_npu GMM/SwiGLU kernels on the AIV vector cores")]
    request = requested(backend="torch_memory", operation=operation,
                        execution_resource="AIV", transport_protocol="auto")
    return Prepared(run=run, validate=validate, request=request, evidence=evidence,
                    payload=payload)


def _dispatched(ctx, work):
    reference = CollectiveBackend(None, "alltoall")
    reference.initialize(ctx)
    dispatched = reference.dispatch(work)
    rows = dispatched.expert_input.detach().requires_grad_()
    return rows, dispatched.expert_token_counts


def _fixed(ctx, work, params):
    """Synthetic fixed-shape input, independent of the routing result."""
    shape = dict((params or {}).get("shape") or {})
    missing = [key for key in ("m", "k", "n", "groups") if not shape.get(key)]
    if missing:
        raise BenchmarkUnavailable("unsupported",
                                   f"fixed compute shape is missing {', '.join(missing)}")
    m, k, n, groups = (int(shape[key]) for key in ("m", "k", "n", "groups"))
    if groups > m:
        raise BenchmarkUnavailable("unsupported",
                                   "fixed compute shape needs at least one row per group")
    dtype = work.hidden_states.dtype
    seed = int(ctx.config["workload"]["seed"])
    generator = torch.Generator(device="cpu").manual_seed(seed + 20250)
    rows = (torch.randn((m, k), generator=generator, dtype=torch.float32) * k ** -0.5).to(
        ctx.device, dtype).requires_grad_()
    gate = (torch.randn((groups, n, k), generator=generator, dtype=torch.float32) *
            k ** -0.5).to(ctx.device, dtype).requires_grad_()
    up = (torch.randn((groups, n, k), generator=generator, dtype=torch.float32) *
          k ** -0.5).to(ctx.device, dtype).requires_grad_()
    down = (torch.randn((groups, k, n), generator=generator, dtype=torch.float32) *
            n ** -0.5).to(ctx.device, dtype).requires_grad_()
    counts = [m // groups + (1 if index < m % groups else 0) for index in range(groups)]
    weights = SimpleNamespace(w_gate=gate, w_up=up, w_down=down, limit=work.limit)
    shape = {"m": m, "k": k, "n": n, "groups": groups, "mode": "fixed"}
    return weights, rows, counts, shape


def _workload_shape(rows, counts, ctx):
    model = ctx.config["model"]
    return {"m": int(rows.shape[0]), "k": int(model["hidden_size"]),
            "n": int(model["intermediate_size"]), "groups": len(counts),
            "mode": "workload"}


def _reference(rows, counts, weights, first):
    cpu_x = rows.detach().cpu().float().requires_grad_()
    cpu_weights = [weight.detach().cpu().float().requires_grad_() for weight in weights]
    packed = torch.cat(cpu_weights, dim=1) if first else cpu_weights[0]
    if counts:
        expected = torch.cat([torch.nn.functional.linear(piece, weight)
                              for piece, weight in zip(cpu_x.split(counts), packed)])
    else:
        expected = cpu_x.new_empty((0, packed.shape[1])) + (cpu_x.sum() + packed.sum()) * 0
    return expected, cpu_x, cpu_weights, packed


def _swiglu(ctx, work, variant, shape_mode, params):
    if shape_mode == "fixed":
        holder, rows, counts, shape = _fixed(ctx, work, params)
    else:
        rows, counts = _dispatched(ctx, work)
        holder, shape = work, _workload_shape(rows, counts, ctx)
    gate_up = gmm1(holder, rows, counts, ctx).detach().requires_grad_()
    activation = activate(gate_up, holder.limit).detach().requires_grad_()
    act = lambda: activate(gate_up, holder.limit)
    upstream = torch.ones_like(activation)
    cpu = gate_up.detach().cpu().float()
    expected = torch.nn.functional.silu(cpu.chunk(2, dim=-1)[0].clamp(max=holder.limit))
    expected = expected * cpu.chunk(2, dim=-1)[1].clamp(-holder.limit, holder.limit)
    if variant == "backward":
        gradient = torch.autograd.grad(expected, cpu, upstream.float().cpu())[0]
        return gradient, (lambda: torch.autograd.grad(act(), gate_up, upstream)[0]), shape
    return expected, act, shape


def _gmm(ctx, work, variant, shape_mode, params):
    if shape_mode == "fixed":
        holder, rows, counts, shape = _fixed(ctx, work, params)
    else:
        rows, counts = _dispatched(ctx, work)
        holder, shape = work, _workload_shape(rows, counts, ctx)
    activation = activate(gmm1(holder, rows, counts, ctx).detach().requires_grad_(),
                          holder.limit).detach().requires_grad_()
    first = variant.startswith("gmm1")
    x = rows if first else activation
    weights = (holder.w_gate, holder.w_up) if first else (holder.w_down,)
    expected, cpu_x, cpu_weights, packed = _reference(x, counts, weights, first)
    forward = lambda: (gmm1 if first else gmm2)(holder, x, counts, ctx)
    if not variant.endswith(("dgrad", "wgrad")):
        return expected, forward, shape
    derivative = variant.split("_")[-1]
    reference_inputs = (cpu_x,) if derivative == "dgrad" else (packed,)
    native_weights = torch.cat(weights, dim=1) if first else weights[0]
    upstream = x.new_ones(expected.shape)
    gradient = torch.autograd.grad(expected.sum(), reference_inputs)
    return (gradient, (lambda: grouped_linear_gradient(x, native_weights, counts, upstream,
                                                       derivative, ctx)), shape)


def request_for(case, params):
    return requested(backend="torch_memory", operation="compute",
                     execution_resource="AIV", transport_protocol="auto")


__all__ = ["VARIANTS", "prepare", "request_for"]
