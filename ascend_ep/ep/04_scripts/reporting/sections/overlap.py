"""Section 7: Communication and Compute Overlap.

Two different kinds of evidence are never merged here:

1. what a parsed profiler artifact says about the device (visible communication time,
   visible compute time, their intersection, and idle time);
2. the latency ratio between a serial composition and a fused entry point.

The second is a benchmark-level gain.  It is **not** a measured overlap percentage, and
the report says so next to the number.
"""
from reporting import empty
from reporting import formatting as F

#: A profiler-derived overlap needs at least this much visible communication time.
MIN_VISIBLE_NS = 1


TITLE = "Communication and Compute Overlap"

INTRO = ("Profiler evidence and latency-derived fusion gains are reported separately. "
         "`internal_overlap_ratio` stays `-` unless instrumentation measures it; the visible "
         "overlap is an interval intersection on the parsed device timeline, which is evidence "
         "only when the profiler actually recorded concurrent streams.")


def render(ctx):
    return [f"## 7. {TITLE}", "", *body(ctx)]


def body(ctx):
    lines = [INTRO, ""]
    lines += ["### Profiler evidence", ""]
    lines += _profiler(ctx)
    lines += ["### Serial vs fused benchmark", ""]
    lines += _fusion(ctx)
    return lines


def _profiler(ctx):
    profiles = ctx.analysis.get("profiles") or {}
    rows = []
    for task, entries in sorted(profiles.items()):
        for entry in entries:
            for rank, profile in sorted(entry["ranks"].items()):
                rows.append([
                    task, F.text(rank), F.text(entry["manifest"]),
                    _ns(profile.get("visible_communication_ns")),
                    _ns(profile.get("visible_compute_ns")),
                    _ns(profile.get("visible_overlap_ns")),
                    F.percent(profile.get("visible_overlap_ratio")),
                    _ns(profile.get("device_idle_ns")),
                    F.percent(profile.get("device_idle_ratio")),
                    _ns(profile.get("unoverlapped_communication_ns")),
                    F.text(profile.get("internal_overlap_ratio")),
                ])
    if not rows:
        return _no_profiler(ctx)
    lines = F.table(("Task", "Rank", "Artifact", "Visible comm ms", "Visible compute ms",
                     "Visible overlap ms", "Visible overlap ratio", "Device idle ms",
                     "Device idle ratio", "Unoverlapped comm ms", "Internal overlap ratio"),
                    rows)
    parse_errors = ctx.analysis.get("profile_parse_errors") or []
    if parse_errors:
        lines += [f"{len(parse_errors)} profiler artifact(s) could not be parsed: "
                  + F.join([f"{item['artifact']} ({item['reason']})" for item in parse_errors[:5]])
                  + ".", ""]
    lines += ["`Visible overlap ratio` = intersection(communication, compute) / visible "
              "communication time on that rank. `Device idle ratio` = idle time / observed span. "
              "Both are per rank: an aggregate would hide the rank that gates the step.", ""]
    return lines


def _no_profiler(ctx):
    profiling_enabled = any(
        (((row.get("config") or {}).get("profiling") or {}).get("enabled"))
        for row in ctx.rows)
    artifacts = [artifact for row in ctx.rows for artifact in (row.get("profiling_artifacts") or ())]
    causes = ["No successful profiling artifact was parsed for this run."]
    causes.append("Profiling was " + ("enabled" if profiling_enabled else "not enabled")
                  + " in the task configuration"
                  + (" and the `profile` stage did not produce a parsable artifact."
                     if profiling_enabled else "; the profile stage was not executed."))
    if artifacts:
        failed = [item for item in artifacts if item.get("status") != "passed"]
        causes.append(f"{len(artifacts)} artifact record(s) exist, {len(failed)} of them from a "
                      f"non-passing attempt.")
    required = ["Enable profiling in the configuration (`benchmark.profiling.enabled: true`).",
                "Execute `python run_benchmark.py profile --run-id <RUN_ID> --cases <CASES>` after "
                "a passing measurement exists.",
                "Make sure the collected trace can be parsed (`kernel_details.csv` or the "
                "supported export formats)."]
    alternatives = []
    fusion = ctx.analysis.get("fusion") or []
    if fusion:
        alternatives.append(f"{len(fusion)} serial/fused latency comparison(s) are available and "
                            f"are reported in section 7.2.")
    else:
        alternatives.append("No serial/fused pair was measured either.")
    latencies = [row for row in ctx.rows if (row.get("latency") or {}).get("p50")]
    if latencies:
        alternatives.append(f"{len(latencies)} task(s) carry measured P50/P95 latencies; they "
                            f"describe duration, not concurrency.")
    return empty.section(
        "Why Device-Level Overlap Is Not Reported",
        level=4,
        findings="No profiler artifact was collected or parsed, so no claim is made about "
                 "device-level communication/compute overlap.",
        causes=causes, required=required, alternatives=alternatives,
        conclusion="The report cannot state that communication and compute overlap in this run. "
                   "A fused-latency gain, where present, is reported separately and is not the "
                   "same quantity.")


def _fusion(ctx):
    fusion = ctx.analysis.get("fusion") or []
    if not fusion:
        return ["_No serial/fused pair was measured under the same conditions, so no fusion gain "
                "is reported._", ""]
    rows = []
    for point in fusion:
        rows.append([F.text(point["fused_task"]), F.text(point["serial_task"]),
                     F.text(point["baseline_type"]),
                     F.number(point["serial_latency_ms"]), F.number(point["fused_latency_ms"]),
                     F.number(point["fusion_speedup"]),
                     F.percent(point["total_time_saving_ratio"]),
                     F.text(point.get("internal_overlap_ratio"))])
    lines = F.table(("Fused task", "Serial reference", "Baseline type", "Serial P50 ms",
                     "Fused P50 ms", "Fusion speedup", "Total time saving",
                     "Internal overlap ratio"), rows)
    lines += ["`Fusion speedup` = serial P50 / fused P50. It is a latency-derived benchmark "
              "result: a 1.5x speedup does **not** mean 33% of device time overlapped, because "
              "the serial reference and the fused point are two separate measurements.", ""]
    return lines


def _ns(value):
    if value is None:
        return F.MISSING
    return F.number(value / 1e6)
