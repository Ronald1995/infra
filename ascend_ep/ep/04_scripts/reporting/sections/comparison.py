"""Section 5: Backend Comparison.

A comparison needs two different backends measured under *identical* conditions.  The
verdict comes from :mod:`analysis.comparability`, so this section can only report what
that module established:

* ``actual_comparison`` -- ranked, with speedup and difference percentages;
* ``single_measurement`` -- shown as a measured value, explicitly not an advantage;
* ``repeated_measurement`` -- repeat samples of one backend, never a comparison.
"""
from reporting import formatting as F
from reporting import empty
from reporting.sections import common

#: At most this many comparison groups are shown inline.
MAX_GROUPS = 20


TITLE = "Backend Comparison"

INTRO = ("The comparison rule is stated once, in `analysis/comparability.py`: two results "
         "are comparable only when case, benchmark level, operation, case parameters, "
         "device/NPU/CANN environment, topology, EP/TP size, model, per-rank workload, "
         "routing trace, timing boundary, dedup/load-balance state and **completion "
         "semantics** all match. Speedups are computed only inside such a group.")


def render(ctx):
    return [f"## 5. {TITLE}", "", *body(ctx)]


def body(ctx):
    lines = [INTRO, ""]
    comparisons = ctx.comparison_groups("actual_comparison")
    singles = ctx.comparison_groups("single_measurement")
    repeats = ctx.comparison_groups("repeated_measurement")
    lines += ["### Comparison groups", ""]
    if comparisons:
        for index, group in enumerate(sorted(comparisons,
                                             key=lambda item: (item["level"] or "",
                                                               item["case_id"] or "",
                                                               item["ep_size"] or 0))):
            if index >= MAX_GROUPS:
                break
            lines += _group_block(group, index)
        if len(comparisons) > MAX_GROUPS:
            lines += [f"_{len(comparisons) - MAX_GROUPS} further comparison group(s) are listed "
                      f"in `operator_comparison.md`._", ""]
    else:
        lines += _no_comparison(ctx, singles, repeats)
    lines += ["### Single measurements", ""]
    lines += _single_table(singles)
    lines += ["### Repeat measurements of one backend", ""]
    lines += _repeat_table(repeats)
    return lines


def _conditions(group):
    workload = group["workload"]
    tokens = workload.get("tokens_per_rank")
    topology = group["topology"]
    environment = group["environment"]
    servers = topology.get("server_ids") or ()
    topology_text = (f"{F.count(len(servers))} server(s) (source `{F.text(topology.get('source'))}`)"
                     if servers else "unavailable (the plan declared no server map)")
    parts = [
        f"workload tokens/rank {F.join([' x '.join(str(v) for v in tokens)] if tokens else [])}",
        f"routing `{F.text(workload.get('routing'))}`",
        f"seed {F.text(workload.get('seed'))}",
        f"routing trace `{F.text(workload.get('routing_trace'))}`",
        f"completion `{group['members'][0]['completion'] or 'unavailable'}`",
        f"topology {topology_text}",
        f"environment {F.text(environment.get('device'))}/"
        f"{F.join(environment.get('npu_models')) if environment.get('npu_models') else 'no NPU'}"
        f"/CANN {F.text(environment.get('cann_version'))}",
        f"samples per backend {_samples_text(group)}",
    ]
    return "; ".join(parts)


def _samples_text(group):
    return ", ".join(f"{backend}={count}" for backend, count in group["samples"].items())


def _group_block(group, index):
    title = (f"#### Group {index + 1}: `{group['case_id']}` · variant `{group['variant']}` · "
             f"EP{group['ep_size']} · {group['timing']} timing")
    lines = [title, "", "Conditions: " + _conditions(group) + ".", ""]
    rows = []
    for member in group["members"]:
        rows.append([member["backend"], F.number(member["p50_ms"]), F.number(member["p95_ms"]),
                     F.text(member["tokens_per_second"]) if member["has_throughput"]
                     else F.na("no throughput model for this case"),
                     _payload(member), _tflops(member), F.count(member["iterations"]),
                     F.number(member.get("speedup_vs_slowest")),
                     F.percent((member.get("difference_percent") or 0) / 100),
                     F.number(member.get("latency_delta_ms")),
                     F.text(member["completion"])])
    lines += F.table(("Backend", "P50 ms", "P95 ms", "Throughput tok/s", "Payload GB/s",
                      "Effective TFLOPS", "Iterations", "Speedup vs slowest", "Slower than "
                      "fastest by", "Latency delta ms", "Completion"), rows)
    lines += [f"Spread between fastest and slowest backend: {F.number(group['spread_ratio'])}x. "
              f"Backend IDs of this group: {F.join(group['backends'])}.", ""]
    return lines


def _payload(member):
    if member["payload_GBps"] is not None:
        return F.number(member["payload_GBps"])
    return F.na(common.NOT_COMMUNICATION) if member["has_bandwidth"] else F.na(
        common.NO_BANDWIDTH_MODEL)


def _tflops(member):
    if member["effective_tflops"] is not None:
        return F.number(member["effective_tflops"])
    return F.na(common.NOT_COMPUTE)


def _no_comparison(ctx, singles, repeats):
    causes = []
    if not ctx.rows:
        causes.append("The run produced no result record at all.")
    if singles:
        causes.append(f"{len(singles)} measurement group(s) contain exactly one backend, so "
                      f"there is nothing to compare.")
    if repeats:
        causes.append(f"{len(repeats)} group(s) contain repeat measurements of a single backend; "
                      f"repeats are samples of one implementation, not a comparison.")
    if not singles and not repeats:
        causes.append("No result passed the eligibility gate (status `passed`, source "
                      "`measured`, positive P50 latency, correctness passed, and for L0 a "
                      "`verified` execution path).")
    causes.append("Every comparison group needs at least two different backends under identical "
                  "conditions -- including the same completion semantics.")
    required = ["Run the same case with at least two adapters in one plan "
                "(`--operators a,b` or `operators: [a, b]`).",
                "Keep case parameters, EP size, workload, timing boundary and dedup/load-balance "
                "state identical across the adapters.",
                "For L0, collect verification evidence so the execution path is `verified`."]
    alternatives = []
    singles_rows = [group for group in singles]
    if singles_rows:
        alternatives.append(f"Single measurements exist for {len(singles_rows)} group(s) and are "
                            f"listed in section 5.2 with their measured value.")
    if ctx.analysis.get("scaling_groups"):
        alternatives.append("EP scaling is analysed in section 6.")
    return empty.section(
        "Why No Backend Can Be Ranked",
        level=3,
        findings="No pair of backends was measured under identical conditions, so this run "
                 "supports no backend performance ranking.",
        causes=causes, required=required, alternatives=alternatives,
        conclusion="Reporting a backend order here would compare different measurements. The "
                   "measured values are kept as single measurements instead.")


def _single_table(groups):
    rows = []
    for group in groups:
        member = group["members"][0]
        rows.append([group["case_id"], F.text(group["variant"]), f"EP{group['ep_size']}",
                     F.text(group["timing"]), member["backend"], F.number(member["p50_ms"]),
                     F.number(member["tokens_per_second"]),
                     F.number(member["payload_GBps"]),
                     "Single Measurement"])
    lines = F.table(("Case", "Variant", "EP", "Timing", "Backend", "P50 ms",
                     "Throughput tok/s", "Payload GB/s", "Verdict"), rows,
                    empty="No single measurement is available either.")
    if rows:
        lines += ["A `Single Measurement` is the measured value of that configuration. It is not "
                  "evidence of a performance advantage over any other backend.", ""]
    return lines


def _repeat_table(groups):
    rows = []
    for group in groups:
        rows.append([group["case_id"], F.text(group["variant"]), f"EP{group['ep_size']}",
                     F.text(group["timing"]), F.join(group["backends"]),
                     F.count(len(group["members"])),
                     F.number(common.median([member["p50_ms"] for member in group["members"]])),
                     "Repeated Measurement"])
    lines = F.table(("Case", "Variant", "EP", "Timing", "Backend", "Samples", "Median P50 ms",
                     "Verdict"), rows, empty="No backend was measured more than once.")
    if rows:
        lines += ["Repeat samples are not a comparison and are never presented as one; the median "
                  "is shown for reference only.", ""]
    return lines
