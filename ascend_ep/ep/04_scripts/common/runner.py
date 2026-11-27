"""Execute a single plan item in an isolated rank worker.

One execution path serves every level.  The worker:

1. builds the untimed inputs (workload, routing trace and -- for a training
   backward case -- the forward graph),
2. builds a :class:`~common.prepared.PreparedCase` through
   ``benchmarks.cases.prepare`` and validates it *before* timing,
3. times ``run`` -> ``wait`` with :func:`common.timer.measure`, so the sample
   covers real completion rather than submission,
4. derives latency, bandwidth, compute and throughput from the measurement and the
   routing-trace model, keeping the two kinds of number apart,
5. records what actually happened (API, algorithm, transport, fallback, bytes and
   the completion guarantee) in an
   :class:`~common.observation.ExecutionObservation`.

The timing boundary is unchanged from the pre-refactor implementation: setup,
barrier, synchronisation and validation all happen outside the measured interval,
and ``wait`` is inside it.
"""
from dataclasses import replace
from statistics import mean
from time import perf_counter_ns
import traceback

from common.backend import BenchmarkUnavailable
from common.distributed import open_context
from common.environment import capture
from common.observation import from_traffic, unavailable as no_traffic
from common.registry import registry
from common.storage import fingerprint, load_json, write_json
from schemas import result


def status_for_error(exc):
    if isinstance(exc, BenchmarkUnavailable):
        return exc.status
    if type(exc).__name__ == "CorrectnessError":
        return "correctness_failed"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "dependency_missing"
    text = str(exc).lower()
    if "out of memory" in text or "outofmemory" in type(exc).__name__.lower():
        return "oom"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "runtime_failed"


def execute_worker(manifest_path, task_id, output, profile=False):
    from pathlib import Path
    import os

    output = Path(output)
    manifest = load_json(manifest_path)
    task = next(item for item in manifest["tasks"] if item["task_id"] == task_id)
    record = result(manifest, task)
    rank = int(os.getenv("RANK", "0"))
    try:
        with open_context(task["config"]) as context:
            run_task(context, task, record, output, profile)
            if rank == 0:
                write_json(output / "result.json", record)
    except Exception as exc:
        status = status_for_error(exc)
        write_json(output / f"rank-{rank}-error.json", {
            "rank": rank, "status": status, "reason": str(exc), "traceback": traceback.format_exc(),
        })
        print(f"rank {rank}: {status}: {exc}", flush=True)
        return 1
    return 0


# --------------------------------------------------------------- attribution --

def _attribute(backend, name):
    """Read an optional attribution hook from a backend, or return ``None``."""
    getter = getattr(backend, name, None)
    if getter is None:
        return None
    try:
        return getter() if callable(getter) else getter
    except (RuntimeError, AttributeError, TypeError):
        return None


def _transport(ctx):
    """The process-group backend actually in use, or ``None`` when unknown."""
    try:
        name = ctx.dist.get_backend(ctx.ep_group)
    except (RuntimeError, AttributeError, ValueError):
        return None
    return str(name) if name is not None else None


def _fallback_reason(ctx, task):
    """Detect a transport fallback the adapter did not intend.

    An adapter named for HCCL whose process group is not HCCL is running on a
    different transport; an operator case on a CPU process group is a functional
    stand-in.  Both are recorded instead of being reported as an Ascend transfer.
    """
    transport = _transport(ctx)
    if transport is None:
        return None
    if ctx.device.type == "npu" and transport != "hccl":
        return f"the process group backend is {transport}, not HCCL"
    if ctx.device.type == "cpu" and task["benchmark_level" if "benchmark_level" in task else "level"] != "L0":
        return f"CPU process group ({transport}) is a functional stand-in, not an Ascend transport"
    return None


def observe(ctx, task, backend, prepared, volume):
    """Build the execution observation for one case.

    A backend that reports its own traffic wins; otherwise the routing-trace model
    is used and labelled as a model.  A level that does not model cross-card
    traffic gets an explicit ``unavailable`` observation instead of a zero, and a
    wire-derived physical bandwidth is never invented.
    """
    supplied = prepared.observation
    if supplied is not None:
        observation = supplied
    elif volume is not None:
        operation = task["operation"]
        observation = from_traffic(
            volume,
            backend.actual_api() if backend is not None else task["backend_id"],
            algorithm=_attribute(backend, "actual_algorithm"),
            transport=_transport(ctx),
            dispatch=("dispatch",) if operation in {
                "dispatch", "cached_dispatch", "dispatch_combine", "training_tail_dispatch",
                "high_throughput_dispatch", "deduplicated_dispatch", "planned_dispatch",
                "quantized_dispatch"} else (),
            combine=("combine",) if operation in {
                "combine", "dispatch_combine", "async_combine", "combine_add_rmsnorm"} else (),
        )
    else:
        observation = no_traffic(
            "this level does not model cross-card traffic; see compute and throughput",
            backend.actual_api() if backend is not None else None)
    # The transport is observable for every case, not only for a level that models
    # wire traffic: leaving it unknown would hide a gloo run behind an HCCL label.
    observation.actual_transport = observation.actual_transport or _transport(ctx)
    if observation.fallback_reason is None:
        reason = _fallback_reason(ctx, task)
        if reason:
            observation.fallback_used = True
            observation.fallback_reason = reason
    observation.evidence.append({
        "source": "completion", "claim": {},
        "detail": f"{prepared.completion or 'none'}: {prepared.completion_detail or 'unknown'}",
    })
    return observation


# ----------------------------------------------------------------- execution --

def run_task(ctx, task, record, output, profile):
    from benchmarks.cases import CaseRequest, prepare as prepare_case
    from benchmarks.cases.common import uses_planner
    from common import memory
    from common.cases import catalog
    from common.routing import Routing, summarize, summarize_local
    from common.timer import measure
    from common.workload import make_workload, router
    from metrics.summary import bandwidth, compute, latency, throughput, traffic

    cfg = ctx.config
    operation = task["operation"]
    backend = None
    prepared = None
    planner_used = False
    rank_avg = None
    volume = None
    mapping = ctx.gather(ctx.mapping())
    record["environment"] = {**capture(), "device": cfg["device"]}
    if ctx.device.type == "npu":
        record["environment"]["npu_models"] = ctx.gather(ctx.torch.npu.get_device_name(ctx.device))
    record["topology"] = {**cfg["topology"], "rank_mapping": mapping}
    is_hw = task["backend_id"] == "hardware"
    case = catalog()[task["case_id"]]
    settings = cfg["benchmark"]
    # An overlap case drives communication and compute against a real workload, so
    # it needs the same materialised weights as an operator case; without them
    # ``ctx.workload`` stayed None and the overlap baseline failed on
    # ``work.topk_ids``.
    needs_weights = (case.level in {"L2", "L3", "L4"}
                     or case.id in {"HW-010", "HW-011", "HW-012"}
                     or case.category == "overlap")
    steps = len(cfg["workload"].get("token_steps", [0])) if operation == "dynamic_tokens" else 1
    if operation == "trace_replay":
        steps = len(load_json(cfg["workload"]["trace_path"])["steps"])
    if operation == "planner_training_steps":
        steps = settings.get("planner_steps", 4)
    works, traces = [], []
    if not is_hw or needs_weights:
        for step in range(steps):
            work = make_workload(ctx, step, allocate_weights=needs_weights and step == 0)
            if step:
                work = replace(work, w_gate=works[0].w_gate, w_up=works[0].w_up,
                               w_down=works[0].w_down, router_weight=works[0].router_weight)
            route = work.route
            if case.level == "L4":
                with ctx.torch.no_grad():
                    routed = router(work)
                    route = Routing(routed.topk_ids.cpu().numpy(), routed.topk_weights.float().cpu().numpy(),
                                    route.placement, route.tokens, route.seed)
            elif cfg["model"]["dtype"] != "float32":
                route = replace(route, weights=work.topk_weights.detach().float().cpu().numpy())
            trace = summarize(route, ctx.gather(summarize_local(
                route, ctx.rank, cfg["topology"].get("server_ids")
            )), cfg)
            works.append(work)
            traces.append(trace)
            write_json(output / f"rank-{ctx.rank}-trace-{step}.json", {
                "topk_ids": route.ids.tolist(), "topk_weights": route.weights.tolist(),
                "placement": route.placement, "seed": route.seed, "tokens_per_rank": route.tokens,
            })
    try:
        ctx.barrier()
        ctx.synchronize()
        initialization_start = perf_counter_ns()
        if is_hw:
            # L0 baselines read the workload through the context.
            setattr(ctx, "workload", works[0] if works else None)
            request = CaseRequest(ctx=ctx, case=case, backend=None, works=works, traces=traces,
                                  parameters=task["parameters"], output=output)
        else:
            spec = registry()[task["backend_id"]]
            backend = spec.create()
            backend.initialize(ctx)
            request = CaseRequest(ctx=ctx, case=case, backend=backend, works=works,
                                  traces=traces, parameters=task["parameters"], output=output)
            if uses_planner(request):
                if any(work.route.tokens != works[0].route.tokens for work in works):
                    raise BenchmarkUnavailable(
                        "unsupported",
                        "MoonEP native planner buffer needs the same per-rank shape between steps")
                backend.prepare_training(works[0])
            else:
                backend.prepare(works[0])
            if case.level == "L1" or operation in {"serial_dispatch_gmm", "serial_gmm_combine"}:
                backend.prepare_communication(works[0])
        prepared = prepare_case(request, output)
        planner_used = (not is_hw) and uses_planner(request)
        ctx.synchronize()
        initialization = perf_counter_ns() - initialization_start
        correctness = prepared.validate()
        if profile:
            from profiling.collect_profile import collect
            collect(ctx, prepared, output / f"rank-{ctx.rank}")
            record["status"] = "passed"
            record["source"] = "profile_only"
            return

        direct_metrics = None
        controller = None
        memory.reset(ctx)
        if prepared.direct is not None:
            # The backend measured internally: run it once, keep its numbers.
            prepared.run(None)
            direct_metrics = prepared.direct["extract"]()
            if direct_metrics is None:
                raise BenchmarkUnavailable("runtime_failed",
                                           "the backend produced no measurement")
            samples = direct_metrics.get("samples") or []
        elif settings["timing"] == "cold_start":
            samples = [{"iteration": 0, "latency_ns": initialization, "host_issue_ns": initialization,
                        "host_wait_ns": 0, "output_ready_ns": initialization, "operations": 1}]
        else:
            from common import straggler as straggler_module

            controller = straggler_module.StragglerController(ctx.config, ctx.rank, ctx.size)
            samples = measure(ctx, settings, prepared.setup, prepared.run, prepared.wait,
                              prepared.sample_metrics, controller)
        from common import parameter_evidence

        observed = (correctness.get("dimension_observed")
                    if isinstance(correctness, dict) else None)
        dimension_evidence = parameter_evidence.record(
            record.get("case_parameters") or {}, observed)
        if observed is not None:
            record["dimension_evidence"] = dimension_evidence
        local = {"rank": ctx.rank, "samples": samples, "correctness": correctness,
                 "memory": memory.snapshot(ctx), "initialization_ns": initialization,
                 "straggler": (controller.summary() if controller is not None
                               else {"enabled": False})}
        write_json(output / f"rank-{ctx.rank}-samples.json", local)
        ranks = sorted(ctx.gather(local), key=lambda item: item["rank"])
        injected = [dict(item.get("straggler") or {}, rank=item["rank"]) for item in ranks
                    if (item.get("straggler") or {}).get("enabled")]
        if injected:
            record["straggler"] = {
                "requested_delay_us": injected[0].get("requested_delay_us"),
                "method": injected[0].get("method"), "stage": injected[0].get("stage"),
                "ranks": injected,
                "injections": sum(item.get("injections") or 0 for item in injected),
                "calibration_failures": sum(item.get("calibration_failures") or 0
                                            for item in injected),
            }
        timings = latency(ranks)
        timing_is_performance = settings["timing"] not in {"host_api", "cold_start"}
        p50 = timings["p50"] if timing_is_performance else None
        record.update(status="passed", correctness={"passed": True, "ranks": [
            {"rank": item["rank"], **item["correctness"]} for item in ranks
        ]}, raw_samples=ranks, latency=timings,
            memory={"ranks": [item["memory"] for item in ranks], "unit": "bytes", "scope": "per rank"},
            overlap={"internal_overlap_ratio": None, "reason": "requires separate profiling and internal instrumentation"})
        if traces:
            tokens = works[0].route.tokens
            record["workload"] = {**cfg["model"], "tokens_per_rank": tokens,
                                  "global_source_tokens": sum(tokens), "steps": [list(w.route.tokens) for w in works],
                                  "trace_hashes": [t["trace_hash"] for t in traces],
                                  "numeric_contract": None if is_hw else backend.spec.constraints.get("numeric_contract")}
            record["routing_trace_hash"] = (traces[0]["trace_hash"] if len(traces) == 1
                                            else fingerprint([t["trace_hash"] for t in traces]))
            record["load_balance"] = {**traces[0], "source": "routing_trace", "scope": "ep_group",
                                      "planner_latency_ms": None, "weight_prefetch_latency_ms": None,
                                      "gradient_reduce_latency_ms": None, "redundant_weight_bytes": None,
                                      "planner_net_speedup": None}
            if ctx.rank == 0:
                write_json(output / "routing_trace.json", traces)
            if planner_used:
                from metrics.summary import stage_latency
                check = ranks[0]["correctness"]["steps"]
                for name in ("planned_receive_rows", "planned_rank_max_mean", "planned_expert_copies",
                             "planned_redundant_weight_bytes"):
                    record["load_balance"][name] = [step[name] for step in check]
                stages = stage_latency(ranks)
                record["load_balance"].update(
                    stage_latency=stages,
                    planner_dispatch_latency_ms=stages.get("planner_dispatch"),
                    weight_prefetch_latency_ms=stages.get("prefetch_weight"),
                    gradient_reduce_latency_ms=stages.get("reduce_grad"),
                    redundant_weight_bytes=max(step["planned_redundant_weight_bytes"] for step in check),
                )
                if len(traces) > 1:
                    record["compute"] = compute(cfg, sum(sum(t["expert_counts"]) for t in traces), operation, p50)
                    record["throughput"] = throughput(
                        sum(sum(work.route.tokens) for work in works), ctx.size, p50
                    )
            if operation in {"rank_dedup_layer", "server_dedup_layer"}:
                traffic_config = cfg if planner_used else {**cfg, "dedup": {"mode": "none"}}
                record["load_balance"]["dedup_dispatch_traffic"] = traffic(
                    traces[0], traffic_config, tokens, backend.spec.traffic_model
                )
            if not is_hw and len(traces) == 1:
                uses_bandwidth = case.level == "L1"
                if uses_bandwidth:
                    model = backend.spec.traffic_model
                    # Gather/reduce combine transfers K lanes of H values before local weighting.
                    direction = 2 if operation == "dispatch_combine" else 1
                    volume = traffic(traces[0], cfg, tokens, model, direction)
                    if model == "collective_alltoallv":
                        ep = ctx.size
                        volume["metadata_bytes"] = (
                            0 if operation in {"combine", "async_combine", "cached_dispatch"} else
                            16 * round(volume["remote_expanded_bytes"] / (
                                direction * cfg["model"]["hidden_size"] * works[0].hidden_states.element_size()
                            )) + 8 * ep * (ep - 1)
                        )
                    if model == "collective_allgather" and operation in {"combine", "dispatch_combine"}:
                        volume["payload_bytes"] = (cfg["model"]["top_k"] + int(direction == 2)) * (
                            volume["payload_bytes"] // direction
                        )
                        volume["padding_bytes"] = None
                    record["bandwidth"] = bandwidth(volume, p50)
                else:
                    record["compute"] = compute(cfg, sum(traces[0]["expert_counts"]), operation, p50)
                request_count = 2 if operation == "concurrent" else 1
                record["throughput"] = throughput(sum(tokens) * request_count, ctx.size, p50)
                if request_count > 1:
                    record["workload"]["requests_per_sample"] = request_count
        if is_hw and case.category == "compute":
            # Standalone compute: separate fixed-shape from workload-shape results.
            shape = task["parameters"].get("shape")
            if task["parameters"].get("shape_mode") == "fixed" and shape:
                model = {**cfg["model"], "hidden_size": shape["k"],
                         "intermediate_size": shape["n"]}
                routes = shape["m"]
                record["compute_shape"] = {"source": "fixed_shape_parameter",
                                           "mode": "fixed", **shape}
            else:
                model = cfg["model"]
                routes = sum(traces[0]["expert_counts"]) if traces else 0
                record["compute_shape"] = {"source": "workload_expert_rows", "mode": "workload",
                                           "rows": routes,
                                           "groups": len(works[0].local_experts) if works else 0,
                                           "k": cfg["model"]["hidden_size"],
                                           "n": cfg["model"]["intermediate_size"]}
            record["compute"] = compute({**cfg, "model": model}, routes, operation, p50)
            record["compute"]["shape_mode"] = task["parameters"].get("shape_mode", "workload")
        if direct_metrics:
            record["latency"] = direct_metrics.get("latency")
            record["bandwidth"] = direct_metrics.get("bandwidth")
            record["metrics_source"] = direct_metrics.get("provider")
            if direct_metrics.get("compute_shape") is not None:
                record["compute_shape"] = direct_metrics["compute_shape"]
            if direct_metrics.get("overlap") is not None:
                record["overlap"] = direct_metrics["overlap"]
        if prepared.payload is not None and not direct_metrics:
            payload = prepared.payload
            if isinstance(payload, dict) and "per_rank_bytes" in payload:
                described = {**payload, "payload_bytes": payload["per_rank_bytes"],
                             "source": "declared_payload_model"}
                record["bandwidth"] = bandwidth(described, p50)
                record["bandwidth"]["global_payload_bytes"] = payload.get("global_bytes")
                record["bandwidth"]["bandwidth_unit"] = "GB/s"
                record["bandwidth"]["scope"] = "per_rank"
            elif isinstance(payload, dict):
                record["bandwidth"] = bandwidth(payload, p50)
            else:
                record["bandwidth"] = bandwidth({"unit": "bytes", "scope": "EP group",
                                                  "payload_bytes": payload,
                                                  "source": "collective_payload_model"}, p50)
        if volume is None and not is_hw and traces and backend is not None:
            # A level that does not model wire traffic still has a derivable logical
            # and remote-expanded byte volume.  Report it as a model so an L2/L3 row
            # is not blank, and keep `measured_wire` unavailable rather than zero.
            volume = traffic(traces[0], cfg, works[0].route.tokens,
                             backend.spec.traffic_model)
        observation = observe(ctx, task, backend, prepared, volume)
        rank_avg = mean(timings["rank_mean"]) if timings.get("rank_mean") else None
        observation.with_latency(p50, rank_avg)
        if observation.actual_api is None and backend is not None:
            observation.actual_api = backend.actual_api()
        record["observation"] = observation.as_dict()
        record["completion"] = {"level": prepared.completion,
                                "detail": prepared.completion_detail,
                                "proves_transfer": prepared.transfers_distinct_from_submission()}
        record["handler"] = task.get("handler")
        if prepared.metadata:
            record["handler_metadata"] = {key: value
                                          for key, value in prepared.metadata.items()
                                          if key != "topology"}
        if is_hw:
            from baselines.l0.topology import rank_graph
            from baselines.l0.verification import hccl_log, path_verifier
            request_block = prepared.request or {}
            evidence = list(prepared.evidence)
            verification = ((cfg.get("benchmark") or {}).get("verification") or {})
            if verification.get("enabled", True):
                logs, inspected = hccl_log.collect(hccl_log.configured_paths(cfg),
                                                   [str(output)])
                evidence.extend(logs)
                record["verification_logs"] = {"inspected": inspected, "claims": len(logs)}
            path_verifier.annotate(record, request_block, evidence)
            levels = (prepared.metadata or {}).get("levels")
            if levels:
                from baselines.l0.verification.evidence import strongest_level
                record["verification_levels"] = dict(levels)
                record["verification_level"] = strongest_level(levels)
            # `category` (and `case_parameters`) are already on the record from
            # `schemas.result`, so a failed attempt keeps them too.  Restating the
            # category here keeps the value tied to the case that actually ran.
            record["category"] = case.category
            record["topology_graph"] = rank_graph.build(cfg, mapping)
    finally:
        ctx.synchronize()
        if prepared is not None:
            prepared.cleanup()
