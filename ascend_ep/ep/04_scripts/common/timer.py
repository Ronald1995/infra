"""Keep synchronization, input preparation and profiler runs outside performance samples."""
from time import perf_counter_ns

from common.backend import BenchmarkUnavailable


def measure(ctx, settings, setup, operation, wait=lambda value: value, sample_metrics=lambda value: {}):
    mode = settings["timing"]
    if mode == "device" and ctx.device.type == "cpu":
        raise BenchmarkUnavailable("unsupported", "device event timing requires an NPU")
    if mode == "async":
        mode = "output_ready"
    if mode == "cold_start":
        raise BenchmarkUnavailable("unsupported", "cold_start is reported in the worker initialization metrics")
    samples = []
    warmup, iterations = settings["warmup"], settings["iterations"]
    repeats = settings.get("throughput_window", 8) if mode == "throughput" else 1
    if type(repeats) is not int or repeats < 1:
        raise ValueError("throughput_window must be positive")
    for i in range(warmup + iterations):
        inputs = [setup(i * repeats + j) for j in range(repeats)]
        ctx.barrier()
        ctx.synchronize()
        first = last = None
        if mode == "device":
            first = ctx.torch.npu.Event(enable_timing=True)
            last = ctx.torch.npu.Event(enable_timing=True)
            first.record()
        started = perf_counter_ns()
        results = [operation(arg) for arg in inputs]
        issued = perf_counter_ns()
        for result in results:
            wait(result)
        waited = perf_counter_ns()
        if last is not None:
            last.record()
        ctx.synchronize()
        completed = perf_counter_ns()
        if i >= warmup:
            issue = (issued - started) / repeats
            ready = (completed - started) / repeats
            elapsed = (first.elapsed_time(last) * 1e6 if first is not None
                       else issue if mode == "host_api" else ready)
            sample = {
                "iteration": i - warmup, "latency_ns": elapsed, "host_issue_ns": issue,
                "host_wait_ns": (waited - issued) / repeats, "output_ready_ns": ready,
                "operations": repeats,
            }
            values = [sample_metrics(result) for result in results]
            if values[0]:
                sample["stage_ns"] = {key: sum(value[key] for value in values) / repeats for key in values[0]}
            samples.append(sample)
    return samples
