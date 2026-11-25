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
    from benchmarks.l0_hardware import hardware_case
    from benchmarks.operator_cases import Operation, operator_case
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
        if is_hw:
            params = task["parameters"]
            call, validate, cleanup, payload = hardware_case(ctx, case, works[0] if works else None,
                                                             params["message_bytes"], params["variant"])
            prepared = Operation(lambda _: None, lambda _: call(), validate)
        else:
            spec = registry()[task["backend_id"]]
            backend = spec.create()
            cleanup = backend.finalize
            backend.initialize(ctx)
            backend.prepare(works[0])
            if case.level == "L1" or operation in {"serial_dispatch_gmm", "serial_gmm_combine"}:
                backend.prepare_communication(works[0])
            prepared = operator_case(ctx, case, backend, works, traces)
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
        if settings["timing"] == "cold_start":
            samples = [{"iteration": 0, "latency_ns": initialization, "host_issue_ns": initialization,
                        "host_wait_ns": 0, "output_ready_ns": initialization, "operations": 1}]
        else:
            samples = measure(ctx, settings, prepared.setup, prepared.call, prepared.wait)
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
                record["throughput"] = throughput(sum(tokens), ctx.size, p50)
        if payload is not None:
            record["bandwidth"] = bandwidth({"unit": "bytes", "scope": "EP group",
                                              "payload_bytes": payload, "source": "collective_payload_model"}, p50)
    finally:
        ctx.synchronize()
        cleanup()
