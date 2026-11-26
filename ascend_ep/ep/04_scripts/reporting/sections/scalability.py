"""Section 6: EP Scalability.

A scaling series needs the *same* case, backend, variant, model, workload semantics and
timing boundary at two or more EP sizes.  The normalised workload size comes from the
suite's declared scaling policy (strong: global work constant; weak: per-rank work
constant), so a strong-scaling run and a weak-scaling run are never mixed.
"""
from reporting import empty
from reporting import formatting as F


TITLE = "EP Scalability"


def render(ctx):
    return [f"## 6. {TITLE}", "", *body(ctx)]


def body(ctx):
    lines = []
    series = ctx.analysis.get("scaling_groups") or {}
    if not series:
        return _no_scaling(ctx)
    lines += ["Scaling efficiency is `(rate_n / rate_0) / (EP_n / EP_0)` inside one series. "
              "The EP size and the topology derived from it are excluded from the grouping key "
              "because they *are* the scaling dimension; the workload size is normalised by the "
              "suite's `scaling.mode`.", ""]
    rows = []
    for key, points in sorted(series.items()):
        base = points[0]
        for point in points:
            rows.append([F.text(base["mode"]), F.text(point["task_id"]),
                         f"EP{point['ep']}", F.number(point["tokens_per_second"]),
                         F.number(point["speedup_vs_base"]), F.percent(point["efficiency"])])
    lines += F.table(("Scaling mode", "Task", "EP", "Throughput tok/s", "Speedup vs base",
                      "Scaling efficiency"), rows)
    lines += ["### Series summary", ""]
    summary = []
    for key, points in sorted(series.items()):
        first, last = points[0], points[-1]
        summary.append([F.text(first["mode"]), f"EP{first['ep']} -> EP{last['ep']}",
                        F.number(first["tokens_per_second"]),
                        F.number(last["tokens_per_second"]),
                        F.percent(last["efficiency"]),
                        F.text(last["task_id"])])
    lines += F.table(("Mode", "Range", "Base throughput tok/s", "Last throughput tok/s",
                      "Last efficiency", "Last task"), summary)
    lines += ["An efficiency below 100% is a measured loss against linear scaling *for this "
              "workload*; it is not by itself a diagnosis. Section 7 holds the profiler evidence "
              "for where time went, when profiling was collected.", ""]
    return lines


def _no_scaling(ctx):
    sizes = ctx.ep_sizes()
    declared = ctx.environment.get("declared_ep_sizes") or []
    causes = []
    if len(sizes) < 2:
        causes.append(
            "Only the following EP size produced results: "
            + F.join([f"EP{size}" for size in sizes] or ["none"])
            + f" (declared: {F.join([f'EP{size}' for size in declared]) or 'none'}).")
        causes.append("A scaling trend needs at least two EP sizes with valid measurements.")
    else:
        causes.append("EP sizes " + F.join([f"EP{size}" for size in sizes])
                      + " were measured, but no case has compatible conditions at two of them.")
        causes.append("A compatible pair needs the same case, backend, variant, model, workload "
                      "semantics and timing boundary, and `benchmark.scaling` must declare the "
                      "scaling mode so the workload size can be normalised.")
    blocked = [size for size in declared if size not in sizes]
    if blocked and len(blocked) < len(declared):
        causes.append("Declared but not measured: "
                      + F.join([f"EP{size}" for size in blocked]) + ".")
    required = ["Run at least two EP sizes from the same plan point "
                "(for example `--plan configs/benchmark_plan.yaml` at EP8 and EP64).",
                "Select a suite that declares `scaling: {mode: strong|weak, ...}` so the workload "
                "is normalised instead of changing unbounded with the EP size.",
                "Keep the timing boundary, dtype and model shape identical across the series."]
    alternatives = ["Single-EP measurements are in section 4 and section 5."]
    if ctx.analysis.get("profiles"):
        alternatives.append("Profiler evidence for the measured EP sizes is in section 7.")
    return empty.section(
        "Why EP Scalability Cannot Be Analysed",
        level=3,
        findings="This run measured "
                 + F.join([f"EP{size}" for size in sizes] or ["no EP size"])
                 + ", so it supports no scalability conclusion.",
        causes=causes, required=required, alternatives=alternatives,
        conclusion="No strong-scaling or weak-scaling statement is made: the data can describe "
                   "a measurement point, not a trend.")
