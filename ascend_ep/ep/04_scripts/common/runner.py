"""Execute a single plan item in an isolated rank worker."""
from dataclasses import replace
from time import perf_counter_ns
import traceback

from common.backend import BenchmarkUnavailable
from common.distributed import open_context
from common.environment import capture
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


def run_task(ctx, task, record, output, profile):
    from benchmarks.l0_hardware import l0_case
    from benchmarks.operator_cases import Operation, operator_case, uses_planner
    from common import memory
    from common.cases import catalog
    from common.routing import Routing, summarize, summarize_local
    from common.timer import measure
    from common.workload import make_workload, router
    from metrics.summary import bandwidth, compute, latency, throughput, traffic

    cfg = ctx.config
    operation = task["operation"]
    backend = None
    mapping = ctx.gather(ctx.mapping())
    record["environment"] = {**capture(), "device": cfg["device"]}
    if ctx.device.type == "npu":
        record["environment"]["npu_models"] = ctx.gather(ctx.torch.npu.get_device_name(ctx.device))
    record["topology"] = {**cfg["topology"], "rank_mapping": mapping}
    is_hw = task["backend_id"] == "hardware"
    case = catalog()[task["case_id"]]
    settings = cfg["benchmark"]
    needs_weights = case.level in {"L2", "L3", "L4"} or case.id in {"HW-010", "HW-011", "HW-012"}
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
    cleanup = lambda: None
    payload = None
    try:
        ctx.barrier()
        ctx.synchronize()
        initialization_start = perf_counter_ns()
        l0_prepared = None
        direct_metrics = None
        if is_hw:
            params = task["parameters"]
            setattr(ctx, "workload", works[0] if works else None)
            l0_prepared = l0_case(ctx, case, params, output)
            cleanup = l0_prepared.cleanup
            payload = l0_prepared.payload
            status_evidence = list(l0_prepared.evidence)
            # The timer calls operation(setup_output) and wait(result); the L0
            # baselines are zero-argument closures, so adapt them here instead of
            # forcing every baseline to accept an unused argument.
            prepared = Operation(lambda _: None, lambda _: l0_prepared.run(),
                                 l0_prepared.validate, lambda _: l0_prepared.wait(None))
        else:
            spec = registry()[task["backend_id"]]
            backend = spec.create()
            cleanup = backend.finalize
            backend.initialize(ctx)
            if uses_planner(operation, task["parameters"]):
                if any(work.route.tokens != works[0].route.tokens for work in works):
                    raise BenchmarkUnavailable("unsupported", "MoonEP native planner buffer needs the same per-rank shape between steps")
                backend.prepare_training(works[0])
            else:
                backend.prepare(works[0])
            if case.level == "L1" or operation in {"serial_dispatch_gmm", "serial_gmm_combine"}:
                backend.prepare_communication(works[0])
            prepared = operator_case(ctx, case, backend, works, traces, task["parameters"])
        ctx.synchronize()
        initialization = perf_counter_ns() - initialization_start
        correctness = prepared.validate()
        if profile:
            from profiling.collect_profile import collect
            collect(ctx, prepared, output / f"rank-{ctx.rank}")
            record["status"] = "passed"
            record["source"] = "profile_only"
            return

        memory.reset(ctx)
        if is_hw and l0_prepared is not None and l0_prepared.direct:
            # The backend measured internally: run it once, keep its numbers.
            prepared.call(None)
            direct_metrics = l0_prepared.direct["extract"]()
            if direct_metrics is None:
                raise BenchmarkUnavailable("runtime_failed",
                                           "the backend produced no measurement")
            samples = direct_metrics.get("samples") or []
        elif settings["timing"] == "cold_start":
            samples = [{"iteration": 0, "latency_ns": initialization, "host_issue_ns": initialization,
                        "host_wait_ns": 0, "output_ready_ns": initialization, "operations": 1}]
        else:
            samples = measure(ctx, settings, prepared.setup, prepared.call, prepared.wait, prepared.sample_metrics)
        local = {"rank": ctx.rank, "samples": samples, "correctness": correctness,
                 "memory": memory.snapshot(ctx), "initialization_ns": initialization}
        write_json(output / f"rank-{ctx.rank}-samples.json", local)
        ranks = sorted(ctx.gather(local), key=lambda item: item["rank"])
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
            if uses_planner(operation, task["parameters"]):
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
                traffic_config = cfg if uses_planner(operation, task["parameters"]) else {
                    **cfg, "dedup": {"mode": "none"},
                }
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
        if direct_metrics:
            record["latency"] = direct_metrics.get("latency")
            record["bandwidth"] = direct_metrics.get("bandwidth")
            record["metrics_source"] = direct_metrics.get("provider")
        if payload is not None and not direct_metrics:
            if isinstance(payload, dict) and "per_rank_bytes" in payload:
                volume = {**payload, "payload_bytes": payload["per_rank_bytes"],
                          "source": "declared_payload_model"}
                record["bandwidth"] = bandwidth(volume, p50)
                record["bandwidth"]["global_payload_bytes"] = payload.get("global_bytes")
                record["bandwidth"]["bandwidth_unit"] = "GB/s"
                record["bandwidth"]["scope"] = "per_rank"
            elif isinstance(payload, dict):
                record["bandwidth"] = bandwidth(payload, p50)
            else:
                record["bandwidth"] = bandwidth({"unit": "bytes", "scope": "EP group",
                                                  "payload_bytes": payload,
                                                  "source": "collective_payload_model"}, p50)
        if is_hw and l0_prepared is not None:
            from baselines.l0.topology import rank_graph
            from baselines.l0.verification import hccl_log, path_verifier
            request = l0_prepared.request or {}
            evidence = list(l0_prepared.evidence)
            verification = ((cfg.get("benchmark") or {}).get("verification") or {})
            if verification.get("enabled", True):
                logs, inspected = hccl_log.collect(hccl_log.configured_paths(cfg),
                                                   [str(output)])
                evidence.extend(logs)
                record["verification_logs"] = {"inspected": inspected, "claims": len(logs)}
            path_verifier.annotate(record, request, evidence)
            record["category"] = case.category
            record["parameters"] = dict(task["parameters"])
            record["topology_graph"] = rank_graph.build(cfg, mapping)
    finally:
        ctx.synchronize()
        cleanup()
