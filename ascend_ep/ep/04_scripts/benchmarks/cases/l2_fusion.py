"""L2: communication and compute fusion.

A fused case measures dispatch and GMM in one timed block.  The timer boundary is
declared by the handler, not inferred from the backend name, so two adapters are
only compared when both include the same work:

``serial_*``
    the handler issues dispatch and GMM itself, so the reference and the native
    adapter execute the same composition.
``dispatch_gmm`` / ``gmm_combine``
    the adapter exposes one fused entry point; the reference is the same
    composition measured through the reference collective.
"""
import torch

from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import activate, gmm1, gmm2
from common.prepared import PreparedCase
from common.validation import error, opaque_row_order, validate_dispatch

from .common import CaseRequest, combine, dispatch, threshold_for, validate_moe


def _reference_layout(request):
    """The reference dispatch layout every fused case is compared against."""
    reference = CollectiveBackend(None, "alltoall")
    reference.initialize(request.ctx)
    layout = reference.dispatch(request.work)
    gate_up = gmm1(request.work, layout.expert_input, layout.expert_token_counts, request.ctx)
    return reference, layout, gate_up


def _serial_dispatch_gmm(request):
    """``serial_dispatch_gmm``: dispatch() followed by GMM-1 in the same block."""
    backend, work = request.backend, request.work
    _reference, layout, gate_up = _reference_layout(request)
    native = dispatch(backend, work)
    if native.route_slots is None:
        sorted_native = opaque_row_order(native.expert_input, native.expert_token_counts)
        sorted_reference = opaque_row_order(layout.expert_input, layout.expert_token_counts)
    else:
        sorted_native = native.route_slots.argsort()
        sorted_reference = layout.route_slots.argsort()
    expected = gate_up[sorted_reference]

    def run(_):
        result = dispatch(backend, work)
        return gmm1(work, result.expert_input, result.expert_token_counts, request.ctx)

    def validate():
        checks = validate_dispatch(work, native, request.ctx, request.trace)
        checks["relative_l2"] = error(run(None)[sorted_native], expected, threshold_for(work))
        return checks

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "serial_dispatch_gmm",
                                  "timing_includes": ["dispatch", "gmm1"]})


def _dispatch_gmm(request):
    """``dispatch_gmm``: one adapter entry point that fuses dispatch and GMM-1."""
    backend, work = request.backend, request.work
    _reference, _layout, gate_up = _reference_layout(request)

    def run(_):
        return backend.dispatch_gmm(work)

    def validate():
        return {"relative_l2": error(run(None), gate_up, threshold_for(work))}

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "dispatch_gmm",
                                  "timing_includes": ["dispatch", "gmm1"]})


def _serial_gmm_combine(request):
    """``serial_gmm_combine``: GMM-2 followed by combine() in the same block."""
    backend, work = request.backend, request.work
    reference, layout, gate_up = _reference_layout(request)
    activation = activate(gate_up, work.limit).detach()
    expected = reference.combine(work, layout, gmm2(work, activation,
                                                    layout.expert_token_counts, request.ctx)).output
    native = dispatch(backend, work)
    if native.route_slots is None:
        validate_dispatch(work, native, request.ctx, request.trace)
        native_activation = activate(gmm1(work, native.expert_input, native.expert_token_counts,
                                          request.ctx), work.limit).detach()
    else:
        native_activation = activation[layout.route_slots.argsort()][
            native.route_slots.argsort().argsort()]

    def run(_):
        value = gmm2(work, native_activation, native.expert_token_counts, request.ctx)
        return combine(backend, work, native, value)

    def validate():
        return {"relative_l2": error(run(None), expected, threshold_for(work))}

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "serial_gmm_combine",
                                  "timing_includes": ["gmm2", "combine"]})


def _gmm_combine(request):
    """``gmm_combine``: one adapter entry point that fuses GMM-2 and combine."""
    backend, work = request.backend, request.work
    reference, layout, gate_up = _reference_layout(request)
    activation = activate(gate_up, work.limit).detach()
    expected = reference.combine(work, layout, gmm2(work, activation,
                                                    layout.expert_token_counts, request.ctx)).output

    def run(_):
        return backend.gmm_combine(work, activation)

    def validate():
        return {"relative_l2": error(run(None), expected, threshold_for(work))}

    return PreparedCase(run=run, validate=validate,
                        metadata={"operation": "gmm_combine",
                                  "timing_includes": ["gmm2", "combine"]})


def _chunk_pipeline(request):
    backend, work = request.backend, request.work
    chunks = int(request.variant.rsplit("_", 1)[1])

    def run(_):
        with torch.profiler.record_function("ep_chunk_forward"):
            return backend.chunk_forward(work, chunks)

    return PreparedCase(run=run, validate=lambda: validate_moe(request.ctx, work, run(None), None),
                        metadata={"operation": "chunk_pipeline", "chunks": chunks})


def _tuned(request):
    """``tile_schedule`` / ``comm_resources``: a native dispatch with tuned tiling.

    The tuning variant is a *parameter*, so the same handler measures both cases;
    which scheduling knob it turns is recorded in the metadata.
    """
    backend, work = request.backend, request.work
    run_native = backend.tuned_dispatch(work, request.variant)
    reference = CollectiveBackend(None, "alltoall")
    reference.initialize(request.ctx)
    layout = reference.dispatch(work)
    expected = gmm1(work, layout.expert_input, layout.expert_token_counts, request.ctx)

    def run(_):
        with torch.profiler.record_function("ep_dispatch_gmm"):
            return run_native(work)

    return PreparedCase(run=run,
                        validate=lambda: {"relative_l2": error(run(None), expected,
                                                               threshold_for(work))},
                        metadata={"operation": request.operation,
                                  "variant_tunes": request.variant})


def _quantized_fusion(request):
    backend, work = request.backend, request.work

    def run(_):
        with torch.profiler.record_function("ep_quantized_dispatch_gmm"):
            return backend.quantized_fusion(work)

    return PreparedCase(
        run=run, validate=lambda: backend.validate_quantized_fusion(work, run(None)),
        metadata={"operation": "quantized_fusion", "timing_includes": ["dispatch", "gmm1"]})


def _fused_backward(request):
    """``fused_backward``: the training-side fused path.

    The forward graph the adapter differentiates is built by the backend itself, so
    the timed block contains the backward pass only.
    """
    from .l3_moe import forward_backward

    prepared = forward_backward(request)
    prepared.metadata.update({"operation": "fused_backward",
                              "timing_includes": ["backward"]})
    return prepared


HANDLERS = {
    "l2.serial_dispatch_gmm": _serial_dispatch_gmm,
    "l2.dispatch_gmm": _dispatch_gmm,
    "l2.serial_gmm_combine": _serial_gmm_combine,
    "l2.gmm_combine": _gmm_combine,
    "l2.chunk_pipeline": _chunk_pipeline,
    "l2.tile_schedule": _tuned,
    "l2.comm_resources": _tuned,
    "l2.quantized_fusion": _quantized_fusion,
    "l2.fused_backward": _fused_backward,
}
