"""Actual timed operations and their untimed correctness preparations."""
from dataclasses import dataclass, replace

import torch

from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import activate, gmm1, gmm2
from common.validation import error, opaque_row_order, operation_gradients, validate_dispatch, validate_moe
from common.workload import router


@dataclass
class Operation:
    setup: object
    call: object
    validate: object
    wait: object = lambda value: value
    sample_metrics: object = lambda value: {}


PLANNER_TRAINING = {"planner", "redundant_experts", "planner_training_steps",
                    "rank_dedup_layer", "server_dedup_layer"}


def planned_training_case(ctx, case, backend, works, parameters):
    from operators.moonep_dynamic_planner.training import prepare_route, run_training_step

    cached = prepare_route(backend, works[0]) if parameters["variant"] == "cached_plan" else None
    collect_stages = case.operation == "redundant_experts"

    def call(_):
        return [run_training_step(backend, work, cached=cached, collect_stages=collect_stages)
                for work in works]

    def validate():
        checked = []
        for step, work in enumerate(works):
            value = run_training_step(backend, work, cached=cached)
            check = validate_moe(ctx, work, value.output, value.gradients, step, use_router=True)
            rank_counts = ctx.gather(value.received_rows)
            copies = int((value.replica_plan[ctx.rank] >= 0).sum().item())
            copied = sum(ctx.gather(copies))
            check.update(
                planned_receive_rows=rank_counts,
                planned_rank_max_mean=max(rank_counts) * ctx.size / sum(rank_counts),
                planned_expert_copies=copied,
                planned_redundant_weight_bytes=copied * sum(
                    tensor[0].numel() * tensor.element_size() for tensor in (work.w_gate, work.w_up, work.w_down)
                ),
            )
            checked.append(check)
        return {"steps": checked, "plan_mode": "cached_plan" if cached else "fresh_plan"}

    return Operation(lambda _: None, call, validate, sample_metrics=lambda values: values[0].timing_ns())


def uses_planner(operation, parameters):
    from common.scheduler import FIXED_OWNER

    return operation in PLANNER_TRAINING and parameters["variant"] != FIXED_OWNER


def operator_case(ctx, case, backend, works, traces, parameters):
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
                if native.route_slots is None:
                    sorted_native = opaque_row_order(native.expert_input, native.expert_token_counts)
                    sorted_ref = opaque_row_order(layout.expert_input, counts)
                else:
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
                if native.route_slots is None:
                    validate_dispatch(work, native, ctx, trace)
                    native_activation = activate(
                        gmm1(work, native.expert_input, native.expert_token_counts, ctx), work.limit
                    ).detach()
                else:
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

    if op in {"tile_schedule", "comm_resources"}:
        run = backend.tuned_dispatch(work, parameters["variant"])
        ref = CollectiveBackend(None, "alltoall")
        ref.initialize(ctx)
        native_layout = ref.dispatch(work)
        expected = gmm1(work, native_layout.expert_input, native_layout.expert_token_counts, ctx)

        def call(_):
            with torch.profiler.record_function("ep_dispatch_gmm"):
                return run(work)

        return Operation(untouched, call, lambda: {"relative_l2": error(call(None), expected, threshold)})

    if op == "chunk_pipeline":
        chunks = int(parameters["variant"].rsplit("_", 1)[1])

        def call(_):
            with torch.profiler.record_function("ep_chunk_forward"):
                return backend.chunk_forward(work, chunks)

        return Operation(untouched, call, lambda: validate_moe(ctx, work, call(None), None))

    use_router = case.level == "L4"
    use_backward = case.training
    if uses_planner(op, parameters):
        return planned_training_case(ctx, case, backend, works, parameters)
    if op == "concurrent":
        backend.prepare_concurrent()
        second = work.with_hidden(work.hidden_states.detach().clone())

        def call(_):
            with torch.no_grad(), torch.profiler.record_function("ep_concurrent_forward"):
                return backend.concurrent_forward(work, second)

        def validate():
            first_output, second_output = call(None)
            return {"requests": [validate_moe(ctx, work, first_output, None),
                                 validate_moe(ctx, second, second_output, None)]}

        return Operation(untouched, call, validate)

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
