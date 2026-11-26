"""Section 0: Executive Summary.

The summary states what the run covered, what it proved and what it did not.  Every
performance sentence is generated from a gate in :mod:`analysis.comparability` or
:mod:`analysis.l0`, so the report cannot claim a backend ranking from one
measurement, a scaling trend from one EP size, or an overlap from a latency
comparison.
"""
from analysis import comparability
from reporting import formatting as F
from reporting.sections import common
from schemas import STATUS_GROUPS

#: A p95/p50 ratio above this is reported as a tail-latency risk worth investigating.
TAIL_RATIO_LIMIT = 1.2
#: A coefficient of variation above this is reported as rank imbalance.
CV_LIMIT = .1
#: An expert max/mean ratio above this is reported as load imbalance.
LOAD_LIMIT = 1.5
#: A device idle ratio above this is reported as idle time worth explaining.
IDLE_LIMIT = .2


def render(ctx):
    lines = ["## 0. Executive Summary", "", "### 0.1 Run scope", ""]
    lines += _scope_table(ctx)
    lines += ["### 0.2 Task outcomes", ""]
    lines += _outcome_table(ctx)
    lines += [f"_{_outcome_note(ctx)}_", ""]
    lines += ["### 0.3 Correctness", ""]
    lines += _correctness(ctx)
    lines += ["### 0.4 Main performance findings", ""]
    lines += F.bullet_list(_findings(ctx))
    lines += ["### 0.5 Current performance bottlenecks", ""]
    lines += F.bullet_list(_bottlenecks(ctx))
    lines += ["### 0.6 Capabilities this run did not verify", ""]
    lines += F.bullet_list(_unverified(ctx))
    return lines


def _scope_table(ctx):
    environment = ctx.environment
    status = ctx.status
    backends = sorted({row.get("backend_id") for row in ctx.rows if row.get("backend_id")})
    ep_sizes = ctx.ep_sizes()
    levels = ctx.levels()
    npu_models = environment.get("npu_models")
    rows = [
        ("Device", ctx.device_wording),
        ("NPU model", F.join(npu_models) if npu_models
         else F.na("the run did not record a device name")),
        ("EP size", F.join([f"EP{size}" for size in ep_sizes], separator=", ")),
        ("TP size", F.join([f"TP{size}" for size in environment.get("tp_sizes") or ()],
                           separator=", ")),
        ("Benchmark level", F.join(levels, separator=", ")),
        ("Backend", _backend_summary(backends)),
        ("Planned tasks", F.count(status.planned)),
        ("Result records", F.count(status.records)),
        ("CANN", F.text(environment.get("cann_version"))),
        ("Git commit", F.text(environment.get("git_commit"))),
    ]
    return F.table(("Item", "Value"), rows)


def _backend_summary(backends, limit=8):
    if len(backends) <= limit:
        return F.join(backends, separator=", ")
    return (F.join(backends[:limit], separator=", ")
            + f" and {len(backends) - limit} more (see section 2.4)")


def _outcome_table(ctx):
    status = ctx.status
    rows = [(label, F.count(status.groups.get(label, 0)), F.percent(
        status.groups.get(label, 0) / status.planned if status.planned else None))
        for label, _statuses in STATUS_GROUPS]
    return F.table(("Status category", "Tasks", "Share of planned"), rows)


def _outcome_note(ctx):
    status = ctx.status
    note = (f"{status.passed} of {status.planned} planned tasks produced a measurement; "
            f"{status.failed} failed, {status.unsupported} unsupported, {status.blocked} blocked "
            f"by a missing dependency or environment, {status.not_run} were never run and "
            f"{status.lost} have no result record at all.")
    if status.orphans:
        note += f" {len(status.orphans)} result record(s) are not in the manifest."
    return note


def _correctness(ctx):
    status = ctx.status
    rate = status.correctness_rate
    rows = [
        ("Tasks that ran", F.count(status.executed)),
        ("Correctness checked", F.count(status.correctness_checked)),
        ("Correctness passed", F.count(status.correctness_passed)),
        ("Correctness pass rate", F.percent(rate) if rate is not None
         else F.na("no task passed, so no correctness result exists")),
    ]
    return F.table(("Item", "Value"), rows)


# ------------------------------------------------------------------ findings --

def _findings(ctx):
    findings = []
    findings.extend(_backend_findings(ctx))
    findings.extend(_scale_findings(ctx))
    findings.extend(_l0_findings(ctx))
    findings.extend(_overlap_findings(ctx))
    findings.extend(_throughput_findings(ctx))
    return findings


def _backend_findings(ctx):
    comparisons = ctx.comparison_groups("actual_comparison")
    singles = ctx.comparison_groups("single_measurement")
    repeats = ctx.comparison_groups("repeated_measurement")
    findings = []
    for group in comparisons[:6]:
        fastest, slowest = group["members"][0], group["members"][-1]
        findings.append(
            f"{group['case_id']} · variant `{group['variant']}` · EP{group['ep_size']} · "
            f"{group['timing']} timing: {fastest['backend']} measured "
            f"{F.number(fastest['p50_ms'])} ms P50 against {slowest['backend']} "
            f"{F.number(slowest['p50_ms'])} ms ({F.number(group['spread_ratio'])}x spread) "
            f"under identical conditions.")
    if comparisons:
        findings.append(f"{len(comparisons)} group(s) satisfy the comparison rule and are "
                        f"ranked in section 5.")
    else:
        findings.append(
            "No backend can be ranked: this run produced no pair of backends under identical "
            "conditions (same case, variant, EP size, workload, timing boundary and completion "
            "semantics).")
    if singles:
        findings.append(
            f"{len(singles)} group(s) contain a single measurement. A single measurement is "
            f"reported as a measured value, never as a performance advantage.")
    if repeats:
        findings.append(
            f"{len(repeats)} group(s) contain repeat measurements of one backend; repeats are "
            f"not turned into a cross-backend comparison.")
    return findings


def _scale_findings(ctx):
    groups = ctx.analysis.get("scaling_groups") or {}
    findings = []
    for _key, points in list(sorted(groups.items()))[:4]:
        if len(points) < 2:
            continue
        mode = points[0]["mode"]
        first, last = points[0], points[-1]
        findings.append(
            f"{mode.capitalize()} scaling: EP{first['ep']} -> EP{last['ep']} measured "
            f"{F.number(first['tokens_per_second'])} -> {F.number(last['tokens_per_second'])} "
            f"source tokens/s at {F.percent(last['efficiency'])} scaling efficiency.")
    if groups:
        findings.append(f"{len(groups)} EP-scaling series satisfy the scaling rule and are "
                        f"analysed in section 6.")
    else:
        sizes = ctx.ep_sizes()
        if len(sizes) < 2:
            findings.append(
                f"No scalability conclusion is possible: the run measured EP size "
                f"{F.join([f'EP{size}' for size in sizes])} only, and a scaling trend needs at "
                f"least two EP sizes with the same case, backend, variant, model, workload "
                f"semantics and timing boundary.")
        else:
            findings.append(
                f"No scalability conclusion is possible: EP sizes "
                f"{F.join([f'EP{size}' for size in sizes])} were measured, but no case has "
                f"compatible conditions at two of them.")
    return findings


def _l0_findings(ctx):
    summary = ctx.l0["summary"]
    if not summary["records"]:
        return ["L0 hardware baseline: no L0 task produced a result, so no hardware capability "
                "is claimed in this report."]
    verification = summary["verification"]
    verified = verification.get("verified", 0)
    total = summary["records"]
    findings = [f"L0 hardware baseline: {total} result(s) over "
                f"{len(summary['ep_sizes'])} EP size(s); {verified} verified, "
                f"{total - verified} without a verified execution path."]
    mechanisms = [entry for entry in ctx.l0["mechanisms"] if entry["verified"]]
    if mechanisms:
        best = max(mechanisms, key=lambda entry: entry["max_GBps"] or 0)
        if best["max_GBps"] is not None:
            findings.append(
                f"Highest verified L0 payload bandwidth: {best['mechanism_label']} on "
                f"{best['execution_resource_label']} at {F.number(best['max_GBps'])} GB/s "
                f"({best['records']} record(s), payload model).")
    else:
        findings.append(
            "No L0 mechanism reached `verified`, so no number in this report is attributed to a "
            "specific communication mechanism. The measured latency/bandwidth values are real "
            "measurements of *some* path, not of the mechanism the case requested.")
    return findings


def _overlap_findings(ctx):
    profiles = ctx.analysis.get("profiles") or {}
    fusion = ctx.analysis.get("fusion") or []
    findings = []
    if profiles:
        idles = [profile.get("device_idle_ratio")
                 for entries in profiles.values() for entry in entries
                 for profile in entry["ranks"].values()
                 if profile.get("device_idle_ratio") is not None]
        if idles:
            findings.append(
                f"Profiler evidence exists for {len(profiles)} task(s); device idle ratio ranges "
                f"{F.percent(min(idles))} to {F.percent(max(idles))} over the observed span.")
    else:
        findings.append(
            "Communication/compute overlap is not verified: no parsed profiler artifact exists, "
            "so the report makes no device-level overlap claim.")
    if fusion:
        best = max(fusion, key=lambda point: point["fusion_speedup"])
        findings.append(
            f"Serial/fused benchmark: {best['fused_task']} is "
            f"{F.number(best['fusion_speedup'])}x the latency of {best['serial_task']}. "
            f"This is a latency-derived fusion gain, not a measured device overlap ratio.")
    return findings


def _throughput_findings(ctx):
    rows = [row for row in ctx.rows
            if comparability.eligibility(row)
            and (row.get("throughput") or {}).get("global_source_tokens_per_second")]
    finds = []
    if not rows:
        return ["No throughput measurement is available (no task reported a token rate)."]
    best = max(rows, key=lambda row: row["throughput"]["global_source_tokens_per_second"])
    rate = best["throughput"]["global_source_tokens_per_second"]
    finds.append(
        f"Highest measured throughput: {F.number(rate)} source tokens/s on "
        f"{best['case_id']} ({best['backend_id']}, EP{comparability.ep_size(best)}) — a "
        f"single-point measurement of this configuration.")
    return finds


# --------------------------------------------------------------- bottlenecks --

def _bottlenecks(ctx):
    findings = []
    eligible = [row for row in ctx.rows if comparability.eligibility(row)]
    tails = [(row, (row.get("latency") or {}).get("p95_p50_ratio")) for row in eligible]
    worst_tail = max((item for item in tails if item[1] is not None), key=lambda item: item[1],
                     default=None)
    if worst_tail and worst_tail[1] > TAIL_RATIO_LIMIT:
        row, ratio = worst_tail
        findings.append(
            f"Tail latency: {row['case_id']}/{row['backend_id']} has P95/P50 = "
            f"{F.number(ratio)} on {common.iterations(row)} iteration(s). With few iterations a "
            f"P95/P99 is a weak estimate and is reported as such, not as a reliable tail SLO.")
    spreads = [(row, (row.get("latency") or {}).get("cv")) for row in eligible]
    worst_cv = max((item for item in spreads if item[1] is not None), key=lambda item: item[1],
                   default=None)
    if worst_cv and worst_cv[1] > CV_LIMIT:
        row, value = worst_cv
        findings.append(
            f"Rank imbalance: {row['case_id']}/{row['backend_id']} has a coefficient of "
            f"variation of {F.number(value)} across rank maxima, so the slowest rank gates the "
            f"step.")
    host = [(row, (row.get("latency") or {}).get("host_issue_p50_ms"),
             (row.get("latency") or {}).get("p50")) for row in eligible]
    worst_host = max((item for item in host
                      if item[1] is not None and item[2]), key=lambda item: item[1] / item[2],
                     default=None)
    if worst_host and worst_host[1] / worst_host[2] > .5:
        row, issue, p50 = worst_host
        findings.append(
            f"Host submission overhead: on {row['case_id']}/{row['backend_id']} the host issue "
            f"time is {F.percent(issue / p50)} of the measured P50. Host issue and host wait are "
            f"reported separately; their sum is not device execution time.")
    load = [(row, (row.get("load_balance") or {}).get("expert_max_mean")) for row in ctx.rows]
    worst_load = max((item for item in load if item[1] is not None), key=lambda item: item[1],
                     default=None)
    if worst_load and worst_load[1] > LOAD_LIMIT:
        row, value = worst_load
        findings.append(
            f"Expert load imbalance: {row['case_id']}/{row['backend_id']} routes "
            f"{F.number(value)}x the mean row count to its hottest expert. The workload is a "
            f"synthetic routing trace, not production traffic.")
    idles = [profile.get("device_idle_ratio")
             for entries in (ctx.analysis.get("profiles") or {}).values()
             for entry in entries for profile in entry["ranks"].values()
             if profile.get("device_idle_ratio") is not None]
    if idles and max(idles) > IDLE_LIMIT:
        findings.append(
            f"Device idle time reaches {F.percent(max(idles))} of the observed span in the "
            f"profiled tasks; the per-rank intervals are in section 7.")
    slow_fusion = [point for point in (ctx.analysis.get("fusion") or [])
                   if point["fusion_speedup"] < 1]
    if slow_fusion:
        findings.append(
            f"{len(slow_fusion)} fused point(s) are slower than their serial reference "
            f"(lowest {F.number(min(p['fusion_speedup'] for p in slow_fusion))}x); see section 7.")
    if not findings:
        findings.append(
            "No bottleneck is claimed: with the measurements available, no metric exceeded the "
            "reporting thresholds for tail latency, rank imbalance, host overhead, load "
            "imbalance or idle time.")
    return findings


# ---------------------------------------------------------------- unverified --

def _unverified(ctx):
    findings = []
    verified_mechanisms = {entry["mechanism"] for entry in ctx.l0["mechanisms"]
                           if entry["verified"] and entry["mechanism_source"] == "observed"}
    unverified = [entry for entry in ctx.l0["mechanisms"] if not entry["verified"]]
    if unverified:
        names = sorted({entry["mechanism_label"] for entry in unverified})
        findings.append(
            f"Communication mechanisms with no verified result: {F.join(names)}. Their numbers "
            f"stay in the report but are not attributed to a mechanism.")
    if verified_mechanisms:
        findings.append("Mechanisms with at least one verified result: "
                        + F.join(sorted(verified_mechanisms)) + ".")
    levels = ctx.levels()
    passed_levels = {row.get("benchmark_level") for row in ctx.rows if row.get("status") == "passed"}
    for level in levels:
        if level not in passed_levels:
            findings.append(f"{common.level_name(level)}: no task passed, so this level has no "
                            f"verified performance result.")
    declared = set(ctx.environment.get("declared_ep_sizes") or ())
    observed = set(ctx.ep_sizes())
    missing = sorted(declared - observed)
    if missing:
        findings.append("Declared but not measured EP size(s): "
                        + F.join([f"EP{size}" for size in missing]) + ".")
    if not (ctx.analysis.get("profiles") or {}):
        findings.append("No profiler artifact was parsed, so device-level overlap and device "
                        "idle behaviour are unverified.")
    if not (ctx.analysis.get("dedup_comparisons") or []):
        findings.append("No rank/server dedup comparison exists (no dedup pair was measured).")
    if not (ctx.analysis.get("planner_comparisons") or []):
        findings.append("No Planner Suite comparison exists (no planner case was measured).")
    return findings
