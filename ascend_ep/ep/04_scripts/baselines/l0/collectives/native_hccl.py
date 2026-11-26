"""Native HCCL test tool backend.

The tool measures internally and prints one row per message size:

    data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:

so this backend does not use the generic timing loop.  It runs the tool once per
case, keeps the tool's own averages, and reports them with
``metrics.source = "native_hccl_tool"``.  Only the average is available: the
tool exposes no percentiles, so ``p95``/``p99`` stay ``None`` instead of being
invented from the average.

The accelerator is passed with ``-a <mode>``.  That selects the *requested*
accelerator; the tool prints nothing about the path it took, so the observed
accelerator stays ``unverified`` unless an HCCL log or profiler artifact says
otherwise.
"""
import os
import re
import subprocess

from baselines.l0.collectives import accelerator
from baselines.l0.contract import Prepared
from baselines.l0.verification.path_verifier import Evidence
from baselines.l0.verification.vocabulary import ACCELERATOR_RESOURCES, requested
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError

ROW = re.compile(r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(NULL|success|failed)\s*$")
HEADER_HINT = "data_size(Bytes)"
DEFAULT_TIMEOUT = 1800


def prepare(ctx, case, params, output=None):
    operation = case.operation
    binary, reason = accelerator.binary_path(ctx.config, operation)
    if binary is None:
        raise BenchmarkUnavailable("dependency_missing", reason)
    modes = {entry["mode"] for entry in
             (accelerator.probe(ctx.config).detail or {}).get("modes", ())}
    mode = params.get("accelerator") or "default"
    if modes and mode not in modes:
        raise BenchmarkUnavailable(
            "unsupported",
            f"the installed hccl_test does not list accelerator mode {mode!r}; "
            f"available: {sorted(modes)}")
    if not modes and mode != "default":
        raise BenchmarkUnavailable("unsupported",
                                   "the installed hccl_test has no accelerator option")
    size = int(params["message_bytes"])
    iterations = int(params.get("iterations") or ctx.config["benchmark"].get("iterations", 5))
    warmup = int(params.get("warmup") or ctx.config["benchmark"].get("warmup", 2))
    command = [binary, "-b", str(size), "-e", str(size), "-i", "1", "-n", str(iterations),
               "-w", str(warmup), "-c", "1", "-p", str(ctx.size)]
    if mode != "default":
        command.extend(["-a", mode])
    launcher = _launcher(ctx.config, ctx.size)
    environment = {**os.environ, **_extra_env(ctx.config)}
    timeout = int(ctx.config["benchmark"].get("task_timeout_seconds", DEFAULT_TIMEOUT))

    state = {}

    def run():
        """Rank 0 runs the whole tool; the other ranks wait at the barrier."""
        if ctx.rank == 0:
            state["result"] = _invoke(ctx, launcher + command, environment, timeout, output)
        ctx.barrier()
        return state.get("result")

    def validate():
        result = state.get("result") or {}
        if ctx.rank == 0 and not result:
            raise CorrectnessError("the native HCCL test produced no result row")
        rank0 = ctx.gather(result)[0] if ctx.size > 1 else result
        if not rank0:
            raise CorrectnessError("the native HCCL test produced no result row")
        if rank0.get("check_result") != "success":
            raise CorrectnessError(
                f"native HCCL test check_result={rank0.get('check_result')}")
        return {"checked": "tool_verification_column", "tool": os.path.basename(binary),
                "accelerator": mode, "data_size_bytes": rank0["data_size"]}

    evidence = [Evidence(
        source="native_descriptor", claim={"execution_resource": ACCELERATOR_RESOURCES.get(mode)},
        path=state.get("log"),
        detail=(f"hccl_test {' '.join(command)}; the tool reports no accelerator field, so "
                "the execution resource is requested, not observed"))] if mode != "default" else []

    direct = _direct(ctx, case, params, mode, binary, command, state)
    return Prepared(run=run, validate=validate, direct=direct, evidence=evidence,
                    request=requested(backend="native_hccl", operation=operation,
                                      execution_resource=ACCELERATOR_RESOURCES.get(mode),
                                      accelerator=mode, transport_protocol="auto"))


def _direct(ctx, case, params, mode, binary, command, state):
    """Build the metrics block once the tool has run."""
    return {
        "provider": "native_hccl_tool",
        "extract": lambda: _metrics(ctx, case, params, mode, binary, command, state),
    }


def _metrics(ctx, case, params, mode, binary, command, state):
    result = state.get("result") or {}
    data_size = result.get("data_size") or int(params["message_bytes"])
    average_us = result.get("average_us")
    bandwidth = result.get("bandwidth_gbps")
    if average_us is None:
        return None
    latency = {"unit": "ms", "scope": "tool_reported_average", "source": "native_hccl_tool",
               "p50": average_us / 1000.0, "p95": None, "p99": None,
               "rank_min": [], "rank_mean": [], "rank_max": [],
               "stddev": None, "cv": None, "p95_p50_ratio": None,
               "note": "the tool reports one average per size; percentiles are unavailable"}
    payload = {
        "unit": "bytes", "scope": "per_rank", "source": "tool_configuration",
        "payload_bytes": data_size, "requested_bytes": int(params["message_bytes"]),
        "direction": "collective", "tool_bandwidth_GBps": bandwidth,
        "bandwidth_definition": ("the tool's alg_bandwidth column, for all_gather/"
                                 "reduce_scatter it multiplies by rank size"),
    }
    return {"latency": latency, "bandwidth": payload,
            "samples": [{"rank": ctx.rank, "samples": [{
                "iteration": 0, "latency_ns": average_us * 1000.0, "host_issue_ns": 0,
                "host_wait_ns": 0, "output_ready_ns": average_us * 1000.0,
                "source": "native_hccl_tool"}]}],
            "command": " ".join(command), "tool": os.path.basename(binary),
            "accelerator": mode, "data_size_bytes": data_size}


def _invoke(ctx, command, environment, timeout, output):
    log = None
    try:
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                   timeout=timeout, env=environment, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkUnavailable("environment_unavailable",
                                   f"cannot launch the native HCCL test: {exc}") from exc
    text = (completed.stdout or "") + (completed.stderr or "")
    if output is not None:
        log = output / "native_hccl_test.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
    row = _parse(text)
    if row is None:
        tail = "\n".join(text.strip().splitlines()[-8:])
        raise RuntimeError(f"native HCCL test produced no parsable row (exit {completed.returncode}):\n{tail}")
    row["exit_code"] = completed.returncode
    row["log"] = str(log) if log else None
    return row


def _parse(text):
    found = None
    for line in text.splitlines():
        match = ROW.match(line)
        if match:
            found = {"data_size": int(match.group(1)), "average_us": float(match.group(2)),
                     "bandwidth_gbps": float(match.group(3)), "check_result": match.group(4)}
    return found


def _launcher(config, world):
    section = ((config or {}).get("benchmark") or {}).get("native_hccl") or {}
    launcher = section.get("launcher") or os.getenv("EP_HCCL_TEST_LAUNCHER")
    if isinstance(launcher, str):
        launcher = launcher.split()
    if launcher:
        return [str(part).replace("{world}", str(world)) for part in launcher]
    if section.get("mpi", True) and subprocess.run(
            ["bash", "-lc", "command -v mpirun"], capture_output=True).returncode == 0:
        return ["mpirun", "-np", str(world)]
    return []


def _extra_env(config):
    section = ((config or {}).get("benchmark") or {}).get("native_hccl") or {}
    return {str(key): str(value) for key, value in (section.get("env") or {}).items()}
