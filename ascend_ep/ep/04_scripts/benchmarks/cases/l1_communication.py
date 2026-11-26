"""L1: standalone communication operators.

L1 measures communication and nothing else.  A case here never computes a router,
an expert GMM or an FFN: the routing trace is pre-generated and the expert output a
combine consumes is pre-constructed outside the timer.  That boundary is what makes
``COMM-001`` .. ``COMM-012`` comparable across adapters at all.

``COMM-001`` .. ``COMM-006`` are the core cross-adapter comparison (dispatch,
cached dispatch, combine, dispatch+combine, async dispatch, async combine);
``COMM-007`` .. ``COMM-012`` are specialised measurements reported separately.

Every handler also validates the dispatch layout whenever the case depends on one,
because a combine measured over a wrong layout is not a combine measurement.
"""
import torch

from common.prepared import PreparedCase
from common.validation import error, validate_dispatch

from .common import (CaseRequest, combine, combine_weights, dispatch, preset_expert_output,
                     threshold_for, validate_moe)


def _completion(backend):
    level, detail = backend.completion_semantics()
    return {"completion": level, "completion_detail": detail}


def _plain(request):
    backend, work = request.backend, request.work

    def run(_):
        return dispatch(backend, work)

    def validate():
        return validate_dispatch(work, run(None), request.ctx, request.trace)

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "dispatch", "correctness": ["dispatch_rows"]},
                        **_completion(backend))


def _cached(request):
    backend, work = request.backend, request.work
    cached = dispatch(backend, work)

    def run(_):
        return dispatch(backend, work, cached.routing_handle)

    def validate():
        return validate_dispatch(work, run(None), request.ctx, request.trace)

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "cached_dispatch",
                                  "correctness": ["dispatch_rows"]},
                        **_completion(backend))


def _combine(request):
    backend, work = request.backend, request.work
    cached = dispatch(backend, work)
    preset = preset_expert_output(work, cached)
    expected = combine_weights(work)

    def run(_):
        return combine(backend, work, cached, preset)

    def validate():
        checks = validate_dispatch(work, cached, request.ctx, request.trace)
        checks["combine_relative_l2"] = error(run(None), expected, threshold_for(work))
        return checks

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "combine",
                                  "correctness": ["dispatch_rows", "combine_output"]},
                        **_completion(backend))


def _dispatch_combine(request):
    backend, work = request.backend, request.work
    cached = dispatch(backend, work)
    preset = preset_expert_output(work, cached)
    expected = combine_weights(work)

    def run(_):
        # Both transfers are inside the timed block and both are completed before
        # `wait` returns, so the sample is a dispatch+combine round trip.
        fresh = dispatch(backend, work)
        return combine(backend, work, fresh, preset)

    def validate():
        checks = validate_dispatch(work, cached, request.ctx, request.trace)
        checks["combine_relative_l2"] = error(run(None), expected, threshold_for(work))
        return checks

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "dispatch_combine",
                                  "correctness": ["dispatch_rows", "combine_output"]},
                        **_completion(backend))


def _async_dispatch(request):
    backend, work = request.backend, request.work

    def run(_):
        return backend.issue_dispatch(work)

    def wait(handle):
        # The adapter returns a deferred callback; invoking it is the completion.
        return handle()

    def validate():
        return validate_dispatch(work, wait(run(None)), request.ctx, request.trace)

    return PreparedCase(run=run, wait=wait, validate=validate,
                        completion="local_complete",
                        completion_detail="adapter deferred completion callback",
                        metadata={"operation": "async_dispatch",
                                  "correctness": ["dispatch_rows"]})


def _async_combine(request):
    backend, work = request.backend, request.work
    cached = dispatch(backend, work)
    preset = preset_expert_output(work, cached)
    expected = combine_weights(work)

    def run(_):
        return backend.issue_combine(work, cached, preset)

    def wait(callback):
        return callback().output

    def validate():
        checks = validate_dispatch(work, cached, request.ctx, request.trace)
        checks["combine_relative_l2"] = error(wait(run(None)), expected, threshold_for(work))
        return checks

    return PreparedCase(run=run, wait=wait, validate=validate,
                        completion="local_complete",
                        completion_detail="adapter deferred completion callback",
                        metadata={"operation": "async_combine",
                                  "correctness": ["dispatch_rows", "combine_output"]})


def _combine_add_rmsnorm(request):
    backend, work = request.backend, request.work
    native = dispatch(backend, work)
    rows = native.expert_input.detach().clone()
    residual = work.grad_output.unsqueeze(1).contiguous()
    gamma = work.hidden_states.new_ones(work.hidden_states.shape[-1])

    def run(_):
        with torch.profiler.record_function("ep_combine_add_rmsnorm"):
            return backend.rmsnorm_combine(work, native, rows, residual, gamma)

    def validate():
        checks = validate_dispatch(work, native, request.ctx, request.trace)
        combined = work.hidden_states.float() * work.topk_weights.float().sum(-1, keepdim=True)
        added = combined.unsqueeze(1) + residual.float()
        reciprocal = (added.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        expected = (added * reciprocal * gamma, reciprocal, added)
        checks.update({name: error(actual, want, threshold_for(work)) for name, actual, want
                       in zip(("normalization", "reciprocal_std", "residual_add"),
                              run(None), expected)})
        return checks

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "combine_add_rmsnorm",
                                  "correctness": ["dispatch_rows", "fused_output"]},
                        **_completion(backend))


def _deduplicated_dispatch(request):
    backend, work = request.backend, request.work

    def run(_):
        with torch.profiler.record_function("ep_dispatch"):
            return backend.deduplicated_dispatch(work)

    return PreparedCase(run=run, validate=lambda: backend.validate_dedup(work, run(None)),
                        metadata={"operation": "deduplicated_dispatch",
                                  "correctness": ["dedup_route"]},
                        **_completion(backend))


def _quantized_dispatch(request):
    backend, work = request.backend, request.work

    def run(_):
        with torch.profiler.record_function("ep_dispatch"):
            return backend.quantized_dispatch(work)

    return PreparedCase(
        run=run,
        validate=lambda: backend.validate_quantized_dispatch(work, run(None), request.trace),
        metadata={"operation": "quantized_dispatch",
                  "correctness": ["quantized_dequant"]},
        **_completion(backend))


def _planned_dispatch(request):
    backend, work = request.backend, request.work

    def run(_):
        with torch.profiler.record_function("ep_planner_dispatch"):
            return backend.planned_dispatch(work)

    return PreparedCase(
        run=run, validate=lambda: backend.validate_planned_dispatch(work, run(None)),
        metadata={"operation": "planned_dispatch", "correctness": ["planned_dispatch"]},
        **_completion(backend))


#: Handler name -> implementation.  The catalog names one of these per case.
HANDLERS = {
    "l1.dispatch": _plain,
    # These two measure the same code path as COMM-001 under a different suite
    # workload; they share the implementation instead of copying it.
    "l1.training_tail_dispatch": _plain,
    "l1.high_throughput_dispatch": _plain,
    "l1.cached_dispatch": _cached,
    "l1.combine": _combine,
    "l1.dispatch_combine": _dispatch_combine,
    "l1.async_dispatch": _async_dispatch,
    "l1.async_combine": _async_combine,
    "l1.combine_add_rmsnorm": _combine_add_rmsnorm,
    "l1.deduplicated_dispatch": _deduplicated_dispatch,
    "l1.quantized_dispatch": _quantized_dispatch,
    "l1.planned_dispatch": _planned_dispatch,
}
