"""Section 11: Recommendations and Next Steps.

Every recommendation names the observation it follows from, so the reader can check it
against section 2, 5, 6 or 7.  A recommendation with no data behind it is not generated:
"run more tests" is only suggested when the coverage counts show a gap.
"""
from analysis import comparability
from reporting import formatting as F
from reporting.sections import common
from schemas import status_group

#: A single-iteration measurement cannot support a tail-latency conclusion.
MIN_ITERATIONS_FOR_TAIL = 20
#: Performance anomalies worth investigating, by threshold.
TAIL_RATIO_LIMIT = 1.2
CV_LIMIT = .1


def render(ctx):
    lines = ["## 11. Recommendations and Next Steps", ""]
    items = _recommendations(ctx)
    if not items:
        lines += ["_No follow-up work is derived from this run: every planned task reached a "
                  "terminal status and no measured anomaly exceeded the reporting thresholds._", ""]
        return lines
    for index, (title, entries) in enumerate(items):
        lines += [f"### 11.{index + 1} {title}", ""]
        lines += [f"- **{entry['action']}** — basis: {entry['basis']}" for entry in entries]
        lines += [""]
    return lines


def _recommendations(ctx):
    groups = []
    groups.append(("Unfinished tests", _unfinished(ctx)))
    groups.append(("EP sizes to add", _ep_sizes(ctx)))
    groups.append(("Backend comparisons to add", _backend_gaps(ctx)))
    groups.append(("Missing profiler data", _profiler(ctx)))
    groups.append(("Unverified communication mechanisms", _verification(ctx)))
    groups.append(("Performance anomalies to investigate", _anomalies(ctx)))
    return [(title, entries) for title, entries in groups if entries]


def _unfinished(ctx):
    status = ctx.status
    entries = []
    not_run = [task for task in ctx.manifest.get("tasks") or ()
               if task.get("task_id") in set(status.missing)]
    pending = [row for row in ctx.rows if row.get("status") == "not_run"]
    if pending:
        cases = sorted({row.get("case_id") for row in pending})
        entries.append({"action": f"Execute the {len(pending)} task(s) that never ran "
                                  f"({F.join(cases)}), for example with "
                                  f"`run_benchmark.py resume --run-id {ctx.run_id}`.",
                        "basis": "section 2 counts them as `Not Run`."})
    if not_run:
        entries.append({"action": f"Re-run the {len(not_run)} planned task(s) that have no result "
                                  f"record; the run is not complete without them.",
                        "basis": "section 9 lists them as `missing`."})
    blocked = [row for row in ctx.rows
               if status_group(row.get("status")) == "Dependency Missing"]
    if blocked:
        entries.append({
            "action": "Install the missing dependency or fix the import path for "
                      f"{len(blocked)} task(s): "
                      + F.join(sorted({row.get('case_id') for row in blocked})[:8]) + ".",
            "basis": "section 9 shows their `dependency_missing` reason."})
    return entries


def _ep_sizes(ctx):
    declared = set(ctx.environment.get("declared_ep_sizes") or ())
    observed = set(ctx.ep_sizes())
    missing = sorted(declared - observed)
    entries = []
    if missing:
        entries.append({"action": "Measure the declared EP size(s) that produced no result: "
                                  + F.join([f"EP{size}" for size in missing]) + ".",
                        "basis": "section 2.3 and the `ep_sizes` list of the plan."})
    if len(observed) < 2:
        entries.append({"action": "Add a second EP size to enable any scalability analysis.",
                        "basis": "section 6 states the same: a trend needs two EP sizes."})
    return entries


def _backend_gaps(ctx):
    singles = ctx.comparison_groups("single_measurement")
    entries = []
    if singles:
        cases = sorted({group["case_id"] for group in singles})
        entries.append({
            "action": "Measure a second backend on "
                      f"{F.join(cases[:10])} so the single measurement becomes a comparison.",
            "basis": f"section 5.2 lists {len(singles)} group(s) with one backend only."})
    return entries


def _profiler(ctx):
    if ctx.analysis.get("profiles"):
        return []
    enabled = any(((row.get("config") or {}).get("profiling") or {}).get("enabled")
                  for row in ctx.rows)
    action = ("Enable profiling (`benchmark.profiling.enabled: true`) and run the `profile` stage "
              "so device-level overlap and idle time can be analysed."
              if not enabled else
              "The profile stage ran but no artifact could be parsed; check the trace export "
              "format and the profiler output path.")
    return [{"action": action, "basis": "section 7 reports no profiler evidence."}]


def _verification(ctx):
    unverified = [entry for entry in ctx.l0["mechanisms"] if not entry["verified"]]
    if not unverified:
        return []
    mechanisms = sorted({entry["mechanism_label"] for entry in unverified})
    return [{"action": "Collect verification evidence (native channel descriptor, runtime query "
                       "API or a structured HCCL/HCOMM log) for the unverified mechanism(s) "
                       + F.join(mechanisms) + " so their numbers can be attributed.",
             "basis": "section 3.4 shows `unverified` for them and section 0.6 says no mechanism "
                      "claim is made."}]


def _anomalies(ctx):
    entries = []
    eligible = [row for row in ctx.rows if comparability.eligibility(row)]
    tails = [(row, (row.get("latency") or {}).get("p95_p50_ratio"),
              common.median((row.get("latency") or {}).get("rank_max") or ()))
            for row in eligible]
    risky = [item for item in tails if item[1] is not None and item[1] > TAIL_RATIO_LIMIT]
    if risky:
        entries.append({"action": f"Investigate the tail latency of {len(risky)} task(s); the "
                                  f"highest P95/P50 is "
                                  f"{F.number(max(item[1] for item in risky))}.",
                        "basis": "section 0.5 and the latency tables in section 4."})
    few = [(row, common.iterations(row)) for row in eligible]
    thin = [row for row, value in few if value == "1" or value == "2"]
    if thin:
        entries.append({
            "action": f"Increase the iteration count for {len(thin)} task(s) that ran only one or "
                      f"two iterations before quoting P95/P99 as a tail-latency result.",
            "basis": "the sample counts in section 4.1-4.4."})
    imbalanced = [row for row in eligible
                  if ((row.get("latency") or {}).get("cv") or 0) > CV_LIMIT]
    if imbalanced:
        entries.append({"action": f"Investigate the rank imbalance of {len(imbalanced)} task(s) "
                                  f"(CV above {CV_LIMIT}).",
                        "basis": "section 0.5 reports the worst coefficient of variation."})
    slow_fusion = [point for point in (ctx.analysis.get("fusion") or [])
                   if point["fusion_speedup"] < 1]
    if slow_fusion:
        entries.append({"action": "Investigate the fused point(s) slower than their serial "
                                  "reference: "
                                  + F.join([point["fused_task"] for point in slow_fusion[:5]]) + ".",
                        "basis": "section 7.2 lists the fusion speedup per pair."})
    lost_bandwidth = [row for row in eligible
                      if (row.get("bandwidth") and
                          (row.get("bandwidth") or {}).get("payload_bytes") is not None and
                          (row.get("bandwidth") or {}).get("payload_bandwidth_GBps") is None)]
    if lost_bandwidth:
        entries.append({"action": f"Check the latency denominator for {len(lost_bandwidth)} "
                                  f"task(s) that have payload bytes but no bandwidth value.",
                        "basis": "section 4.1-4.4 shows `-` in the payload bandwidth column."})
    return entries
