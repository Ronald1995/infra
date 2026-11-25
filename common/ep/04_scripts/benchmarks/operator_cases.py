"""Actual timed operations and their untimed correctness preparations."""
from dataclasses import dataclass, replace

import torch

from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import activate, gmm1, gmm2
from common.backend import BenchmarkUnavailable
from common.validation import error, operation_gradients, validate_dispatch, validate_moe
from common.workload import router


@dataclass
class Operation:
    setup: object
    call: object
    validate: object
    wait: object = lambda value: value


def operator_case(ctx, case, backend, works, traces):
    op = case.operation
    work, trace = works[0], traces[0]
    untouched = lambda _: None
    threshold = 0.04 if work.hidden_states.dtype == torch.bfloat16 else 0.012 if (
        work.hidden_states.dtype == torch.float16
    ) else 2e-5

    def dispatch(work, cached=None):
        with torch.profiler.record_function("ep_dispatch"):
            result = backend.dispatch(work, cached)
        return backend.wait_until_ready(result)

    def combine(work, result, rows):
        with torch.profiler.record_function("ep_combine"):
            output = backend.combine(work, result, rows)
        return backend.wait_until_ready(output).output

    if op == "combine_add_rmsnorm":
        native = dispatch(work)
        rows = native.expert_input.detach().clone()
        residual = work.grad_output.unsqueeze(1).contiguous()
        gamma = work.hidden_states.new_ones(work.hidden_states.shape[-1])

        def call(_):
            with torch.profiler.record_function("ep_combine_add_rmsnorm"):
                return backend.rmsnorm_combine(work, native, rows, residual, gamma)

        def validate():
            checks = validate_dispatch(work, native, ctx, trace)
            combined = work.hidden_states.float() * work.topk_weights.float().sum(-1, keepdim=True)
            added = combined.unsqueeze(1) + residual.float()
            reciprocal = (added.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            expected = (added * reciprocal * gamma, reciprocal, added)
            checks.update({name: error(actual, want, threshold) for name, actual, want in
                           zip(("normalization", "reciprocal_std", "residual_add"), call(None), expected)})
            return checks

        return Operation(untouched, call, validate)

    if op == "deduplicated_dispatch":
        def call(_):
            with torch.profiler.record_function("ep_dispatch"):
                return backend.deduplicated_dispatch(work)
        return Operation(untouched, call, lambda: backend.validate_dedup(work, call(None)))

    if op == "quantized_dispatch":
        def call(_):
            with torch.profiler.record_function("ep_dispatch"):
                return backend.quantized_dispatch(work)
        return Operation(untouched, call, lambda: backend.validate_quantized_dispatch(work, call(None), trace))

    if op == "planned_dispatch":
        def call(_):
            with torch.profiler.record_function("ep_planner_dispatch"):
                return backend.planned_dispatch(work)
        return Operation(untouched, call, lambda: backend.validate_planned_dispatch(work, call(None)))

    if case.level == "L1":
        cached = dispatch(work) if op in {"cached_dispatch", "combine", "dispatch_combine", "async_combine"} else None
        expert_scale = len(work.route.placement)
        preset = ((cached.expert_ids.float() + 1) / expert_scale).to(work.hidden_states.dtype).unsqueeze(1).expand(
            -1, work.hidden_states.shape[1]
        ).contiguous() if cached is not None else None
        wait = lambda value: value
        if op == "cached_dispatch":
            call = lambda _: dispatch(work, cached.routing_handle)
        elif op == "async_dispatch":
            call = lambda _: backend.issue_dispatch(work)
            wait = lambda callback: callback()
        elif op == "async_combine":
            call = lambda _: backend.issue_combine(work, cached, preset)
            wait = lambda callback: callback().output
        elif op == "combine":
            call = lambda _: combine(work, cached, preset)
        elif op == "dispatch_combine":
            def call(_):
                fresh = dispatch(work)
                return combine(work, fresh, preset)
        else:
            call = lambda _: dispatch(work)

        def validate():
            checked = dispatch(work)
            results = validate_dispatch(work, checked, ctx, trace)
            if op in {"combine", "dispatch_combine", "async_combine"}:
                weights = ((work.topk_ids.float() + 1) / expert_scale).to(work.hidden_states.dtype)
                expected = (weights * work.topk_weights).sum(-1, keepdim=True).expand_as(work.hidden_states)
                results["combine_relative_l2"] = error(wait(call(None)), expected, threshold)
            elif op in {"cached_dispatch", "async_dispatch"}:
                results.update(validate_dispatch(work, wait(call(None)), ctx, trace))
            return results

        return Operation(untouched, call, validate, wait)

    if case.level == "L2" and op in {"serial_dispatch_gmm", "dispatch_gmm", "serial_gmm_combine", "gmm_combine"}:
        ref = CollectiveBackend(None, "alltoall")
        ref.initialize(ctx)
        layout = ref.dispatch(work)
        gate_up = gmm1(work, layout.expert_input, layout.expert_token_counts, ctx)
        counts = layout.expert_token_counts
        activation = activate(gate_up, work.limit).detach()
        if op in {"serial_dispatch_gmm", "dispatch_gmm"}:
            def serial(_):
                result = dispatch(work)
                return gmm1(work, result.expert_input, result.expert_token_counts, ctx)
            call = serial if op.startswith("serial") else lambda _: backend.dispatch_gmm(work)
            expected = gate_up
            # Adapters which expose their own dispatch may return a distinct local order.
            if op.startswith("serial"):
                native = dispatch(work)
                sorted_native = native.route_slots.argsort()
                sorted_ref = layout.route_slots.argsort()
                expected = gate_up[sorted_ref]
                check = lambda: call(None)[sorted_native]
            else:
                check = lambda: call(None)
        else:
            def serial(_):
                value = gmm2(work, native_activation, native.expert_token_counts, ctx)
                return combine(work, native, value)
            expected = ref.combine(work, layout, gmm2(work, activation, counts, ctx)).output
            if op.startswith("serial"):
                native = dispatch(work)
                native_activation = activation[layout.route_slots.argsort()][native.route_slots.argsort().argsort()]
                call = serial
            else:
                call = lambda _: backend.gmm_combine(work, activation)
            check = lambda: call(None)
        return Operation(untouched, call, lambda: {"relative_l2": error(check(), expected, threshold)})

    if op == "quantized_fusion":
        def call(_):
            with torch.profiler.record_function("ep_quantized_dispatch_gmm"):
                return backend.quantized_fusion(work)

        return Operation(untouched, call, lambda: backend.validate_quantized_fusion(work, call(None)))

    use_router = case.level == "L4"
    use_backward = case.training
    if op in {"planner", "redundant_experts", "rank_dedup_layer", "server_dedup_layer", "planner_training_steps",
              "concurrent", "tile_schedule", "comm_resources"}:
        raise BenchmarkUnavailable("not_implemented", f"{op} needs a dedicated native measurement contract")

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
        value = works[index]
        if op == "backward":
            return gradients(value, prepared)
        if op == "recompute":
            from torch.utils.checkpoint import checkpoint
            y = checkpoint(lambda x: forward(value.with_hidden(x)), value.hidden_states, use_reentrant=False)
        elif op == "multilayer":
            y = forward(value)
            return forward(value.with_hidden(y))
        else:
            y = forward(value)
        return gradients(value, y) if use_backward else y

    def setup(index):
        index %= len(works)
        return index, forward(works[index]) if op == "backward" else None

    def validate():
        checks = []
        for step, value in enumerate(works):
            y = forward(value)
            grads = gradients(value, y) if use_backward else None
            if hasattr(backend, "validate_forward") and not use_backward and not use_router:
                checks.append(backend.validate_forward(value, y))
            else:
                checks.append(validate_moe(ctx, value, y, grads, step, use_router))
            if op == "recompute":
                measured = call((step, None))
                checks[-1]["recompute_gradient_relative_l2"] = [
                    error(actual, reference, threshold) for actual, reference in zip(measured, grads)
                ]
            if op == "multilayer":
                ref = CollectiveBackend(None, "alltoall")
                ref.initialize(ctx)
                expected = ref.moe_forward(value.with_hidden(ref.moe_forward(value)))
                checks[-1]["two_layer_relative_l2"] = error(call((step, None)), expected, threshold)
        return {"steps": checks}

    return Operation(setup, call, validate)
