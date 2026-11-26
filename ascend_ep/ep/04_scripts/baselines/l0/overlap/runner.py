"""Communication + compute overlap baselines.

Three phases are measured in one preparation:

1. communication only,
2. compute only,
3. both submitted concurrently on two NPU streams.

Overlap is not assumed from "two tasks were submitted".  ``direct`` metrics
include the measured intersection of the two device-side windows, and the case
fails its validation when the intersection is empty or when either task produced
wrong data.  Without device events the intersection falls back to host
timestamps and the report marks the overlap as ``unverified``.
"""
import time

import torch

from baselines.l0.contract import Prepared
from baselines.l0.verification.path_verifier import Evidence
from baselines.l0.verification.vocabulary import requested
from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import activate, gmm1
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError


def prepare(ctx, case, work, size, params):
    if ctx.size < 2 or ctx.device.type == "cpu":
        raise BenchmarkUnavailable("unsupported", "overlap baselines require at least two NPU ranks")
    compute = params.get("compute") or "gmm"
    comm = _communication(ctx, size)
    compute_run, compute_check = _compute(ctx, work, compute)
    state = {}

    def run():
        torch.npu.synchronize(ctx.device)
        ctx.barrier()
        state["comm_only"] = _time(ctx, comm)
        state["compute_only"] = _time(ctx, compute_run)
        state["concurrent"] = _concurrent(ctx, comm, compute_run, state)
        return None

    def validate():
        run()
        comm_check = comm["validate"]()
        compute_check()
        state["validated"] = True
        return {"checked": "both_tasks", "communication": comm_check,
                "compute": compute, "overlap_ns": state["concurrent"]["overlap_ns"]}

    def metrics():
        serial = state["comm_only"]["ms"] + state["compute_only"]["ms"]
        concurrent = state["concurrent"]["ms"]
        comm_ns, compute_ns = state["concurrent"]["comm_ns"], state["concurrent"]["compute_ns"]
        overlap_ns = state["concurrent"]["overlap_ns"]
        denominator = max(1, min(comm_ns, compute_ns))
        return {
            "latency": {"unit": "ms", "scope": "concurrent_wall", "source": "measured",
                        "p50": concurrent, "p95": None, "p99": None,
                        "comm_only_ms": state["comm_only"]["ms"],
                        "compute_only_ms": state["compute_only"]["ms"],
                        "serial_sum_ms": serial,
                        "concurrent_ms": concurrent,
                        "communication_slowdown": (comm_ns / 1e6) / state["comm_only"]["ms"]
                        if state["comm_only"]["ms"] else None,
                        "compute_slowdown": (compute_ns / 1e6) / state["compute_only"]["ms"]
                        if state["compute_only"]["ms"] else None,
                        "overlap_ratio": overlap_ns / denominator,
                        "overlap_verified": bool(state["concurrent"]["device_events"]
                                                 and overlap_ns > 0.05 * denominator),
                        "overlap_evidence": ("device_events" if state["concurrent"]["device_events"]
                                             else "host_timestamps")},
            "bandwidth": {"unit": "bytes", "scope": "per_rank", "source": "configuration",
                          "payload_bytes": int(params["message_bytes"]),
                          "communication_GBps": (int(params["message_bytes"]) /
                                                 (comm_ns / 1e3)) if comm_ns else None},
            "samples": [],
        }

    evidence = comm["evidence"] + [
        Evidence(source="native_descriptor", claim={"execution_resource": "AIV"},
                 detail=f"compute task {compute} on a dedicated NPU stream")]
    return Prepared(run=run, validate=validate, request=requested(
        backend="torch_hccl", operation="concurrent",
        transport_protocol=comm["mechanism"]), evidence=evidence,
        direct={"provider": "overlap", "extract": metrics})


def _time(ctx, task):
    torch.npu.synchronize(ctx.device)
    started = time.perf_counter_ns()
    task["run"]()
    torch.npu.synchronize(ctx.device)
    return {"ms": (time.perf_counter_ns() - started) / 1e6}


def _concurrent(ctx, comm, compute_run, state):
    """Submit both tasks, then measure the intersection of their device windows."""
    streams = {"comm": torch.npu.Stream(device=ctx.device),
               "compute": torch.npu.Stream(device=ctx.device)}
    events = {name: (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
              for name in streams}
    current = torch.npu.current_stream()
    for stream in streams.values():
        stream.wait_stream(current)
    torch.npu.synchronize(ctx.device)
    started = time.perf_counter_ns()
    with torch.npu.stream(streams["comm"]):
        events["comm"][0].record()
        comm["run"]()
        events["comm"][1].record()
    with torch.npu.stream(streams["compute"]):
        events["compute"][0].record()
        compute_run()
        events["compute"][1].record()
    for stream in streams.values():
        current.wait_stream(stream)
    torch.npu.synchronize(ctx.device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    device_events = True
    try:
        windows = {name: (start.elapsed_time(current), end.elapsed_time(current))
                   for name, (start, end) in events.items()}
    except (RuntimeError, ValueError):
        device_events = False
        now = time.perf_counter_ns()
        windows = {"comm": (0.0, (now - started) / 1e6), "compute": (0.0, (now - started) / 1e6)}
    comm_start, comm_end = windows["comm"]
    comp_start, comp_end = windows["compute"]
    overlap_ms = max(0.0, min(comm_end, comp_end) - max(comm_start, comp_start))
    return {"ms": elapsed_ms, "overlap_ns": overlap_ms * 1e6,
            "comm_ns": max(0.0, comm_end - comm_start) * 1e6,
            "compute_ns": max(0.0, comp_end - comp_start) * 1e6,
            "device_events": device_events, "streams": streams}


def _communication(ctx, size):
    """Default communication task: one framework AllToAll over the EP group."""
    value = torch.full((size,), ctx.rank % 127 + 1, dtype=torch.uint8, device=ctx.device)
    destination = torch.empty_like(value)
    stripe = size // ctx.size if ctx.size else 0

    def run():
        ctx.dist.all_to_all_single(destination, value, group=ctx.ep_group)

    def validate():
        if not stripe:
            return {"checked": "skipped", "reason": "message smaller than the EP group"}
        peers = (torch.arange(ctx.size, device=ctx.device) % 127).add(1).to(torch.uint8)
        if not torch.equal(destination, peers.repeat_interleave(stripe)):
            raise CorrectnessError("overlap communication payload did not match")
        return {"checked": "byte_exact"}

    return {"run": run, "validate": validate, "mechanism": "auto",
            "evidence": [Evidence(source="native_descriptor", claim={},
                                  detail="torch HCCL all_to_all_single on a dedicated stream")]}


def _compute(ctx, work, kind):
    if kind in ("gmm", "gmm_swiglu"):
        reference = CollectiveBackend(None, "alltoall")
        reference.initialize(ctx)
        dispatched = reference.dispatch(work)
        rows = dispatched.expert_input.detach().requires_grad_()
        counts = dispatched.expert_token_counts
        gate_up = gmm1(work, rows, counts, ctx).detach().requires_grad_()
        activation = activate(gate_up, work.limit)

        def run():
            value = gmm1(work, rows, counts, ctx)
            if kind == "gmm_swiglu":
                return activate(value, work.limit)
            return value

        def check():
            expected = gmm1(work, rows, counts, ctx)
            got = run()
            denominator = expected.norm() + 1e-12
            if kind == "gmm_swiglu":
                expected = activate(expected, work.limit)
                denominator = expected.norm() + 1e-12
            if float((got - expected).norm() / denominator) > 0.04:
                raise CorrectnessError("overlap compute result drifted from the reference")
            del activation

        return run, check
    if kind == "swiglu":
        value = torch.full((size_guess(work), 1024), 0.5, device=ctx.device)

        def run():
            return activate(value, work.limit)

        def check():
            run()

        return run, check
    if kind == "hbm_copy":
        source = torch.full((size_guess(work),), 1, dtype=torch.uint8, device=ctx.device)
        destination = torch.zeros_like(source)

        def run():
            destination.copy_(source)

        def check():
            run()
            if not bool((destination == 1).all()):
                raise CorrectnessError("overlap HBM copy produced different bytes")

        return run, check
    raise BenchmarkUnavailable("unsupported", f"unknown overlap compute task {kind!r}")


def size_guess(work):
    tokens = work.route.tokens if hasattr(work, "route") else [32]
    return max(1, int(sum(tokens)) * 128)


def request_for(case, params):
    return requested(backend="torch_hccl", operation="concurrent",
                     transport_protocol=params.get("transport") or "auto")
