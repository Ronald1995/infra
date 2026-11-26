"""Compare only compatible, validated measurements and matching scaling points.

All grouping decisions live in :mod:`analysis.comparability`; this module only
*collects* rows into those groups and derives the comparison statistics.  Keeping
the split means the report layer never has to re-decide what "comparable" means.
"""
from collections import defaultdict
from statistics import mean

from analysis import comparability
from common.storage import fingerprint

#: The comparison verdicts a group can reach.  Re-exported so report code imports a
#: single vocabulary.
GROUP_KINDS = comparability.GROUP_KINDS


def _verification_status(record):
    return comparability.verification_status(record)


def comparable(record):
    """True when a record may enter a cross-backend comparison."""
    return comparability.comparable(record)


def group_key(row, case=True):
    """Condition fingerprint, kept as the historic helper for the joins below.

    ``case=False`` drops the case identity so the *same* work measured through two
    case IDs (a fused point and its serial reference) can be joined.
    """
    return fingerprint({
        "case": comparability.case_identity(row) if case else None,
        "parameters": comparability.semantic_parameters(row),
        "environment": comparability.environment_dimensions(row),
        "topology": comparability.topology_dimensions(row),
        "model": comparability.model_dimensions(row),
        "workload": comparability.workload_dimensions(row),
        "timing": comparability.timing_dimensions(row),
        "mode": comparability.mode_dimensions(row),
        "scale": comparability.scale_dimensions(row),
    })


# ------------------------------------------------------------------ members --

def _iterations(record):
    samples = (record.get("latency") or {}).get("rank_max")
    return len(samples) if isinstance(samples, list) else None


def _member(record):
    latency = record.get("latency") or {}
    throughput = record.get("throughput") or {}
    bandwidth = record.get("bandwidth") or {}
    compute = record.get("compute") or {}
    return {
        "backend": record.get("backend_id"),
        "task_id": record.get("task_id"),
        "case": record.get("case_id"),
        "status": record.get("status"),
        "p50_ms": latency.get("p50"),
        "p95_ms": latency.get("p95"),
        "p99_ms": latency.get("p99"),
        "rank_max_ms": _last(latency.get("rank_max")),
        "cv": latency.get("cv"),
        "iterations": _iterations(record),
        "tokens_per_second": throughput.get("global_source_tokens_per_second"),
        "payload_GBps": bandwidth.get("payload_bandwidth_GBps"),
        # Presence flags: a section needs them to tell "no bandwidth model for this
        # case" apart from "the model exists but produced no payload".
        "has_bandwidth": bool(bandwidth),
        "has_compute": bool(compute),
        "has_throughput": bool(throughput),
        "effective_tflops": compute.get("effective_tflops"),
        "verification": _verification_status(record),
        "completion": (record.get("completion") or {}).get("level"),
        "observation_source": comparability.observation_source(record),
        "device": ((record.get("config") or {}).get("device")),
        "ep_size": comparability.ep_size(record),
    }


def _last(values):
    return values[-1] if isinstance(values, list) and values else values


def describe_group(key, rows):
    """Summarise one comparison group, with the statistics it is allowed to claim.

    ``speedup`` and the percentage differences are computed only for an
    ``actual_comparison``: a single measurement has no baseline, and repeating one
    backend is not a comparison.
    """
    eligible = [row for row in rows if comparability.eligibility(row)]
    members = sorted((_member(row) for row in eligible),
                     key=lambda item: (item["p50_ms"] is None, item["p50_ms"] or 0,
                                       str(item["backend"])))
    kind = comparability.group_kind(rows)
    first = eligible[0] if eligible else rows[0]
    samples = defaultdict(int)
    for row in rows:
        samples[row.get("backend_id")] += 1
    group = {
        "key": key,
        "kind": kind,
        "comparison": kind == "actual_comparison",
        "case_id": first.get("case_id"),
        "level": first.get("benchmark_level"),
        "operation": first.get("operation"),
        "variant": comparability.case_parameters(first).get("variant"),
        "parameters": comparability.semantic_parameters(first),
        "ep_size": comparability.ep_size(first),
        "tp_size": comparability.tp_size(first),
        "device": (first.get("config") or {}).get("device"),
        "timing": (first.get("config") or {}).get("benchmark", {}).get("timing"),
        "timing_scope": (first.get("latency") or {}).get("scope"),
        "workload": comparability.workload_dimensions(first),
        "topology": comparability.topology_dimensions(first),
        "environment": comparability.environment_dimensions(first),
        "backends": sorted({row.get("backend_id") for row in rows}),
        "samples": dict(sorted(samples.items())),
        "members": members,
        "spread_ratio": None,
        "reason": None,
    }
    if kind == "actual_comparison":
        slowest = max(member["p50_ms"] for member in members)
        fastest = min(member["p50_ms"] for member in members)
        for member in members:
            member["speedup_vs_slowest"] = slowest / member["p50_ms"]
            member["difference_percent"] = (member["p50_ms"] / fastest - 1) * 100
            member["latency_delta_ms"] = member["p50_ms"] - fastest
        group["spread_ratio"] = slowest / fastest
    elif kind == "repeated_measurement":
        group["reason"] = (f"{len(rows)} measurements from one backend "
                           f"({group['backends'][0]}): repeat samples, not a comparison")
    elif kind == "single_measurement":
        group["reason"] = "only one valid backend measurement, so there is nothing to compare"
    else:
        group["reason"] = "no valid measurement in this group"
    if kind in {"actual_comparison", "repeated_measurement", "single_measurement"}:
        totals = defaultdict(list)
        for member in members:
            if member["payload_GBps"] is not None:
                totals["payload_GBps"].append(member["payload_GBps"])
        group["mean_payload_GBps"] = mean(totals["payload_GBps"]) if totals["payload_GBps"] else None
    return group


def comparison_groups(rows):
    """Group comparable records by test condition and classify each group."""
    groups = defaultdict(list)
    excluded = []
    for row in rows:
        if not comparability.eligibility(row):
            excluded.append({"task_id": row.get("task_id"), "case_id": row.get("case_id"),
                             "backend_id": row.get("backend_id"),
                             "status": row.get("status"),
                             "reason": comparability.eligibility(row).reason})
            continue
        groups[comparability.backend_comparison_key(row)].append(row)
    described = [describe_group(key, group) for key, group in sorted(groups.items())]
    described.sort(key=lambda entry: (entry["level"] or "", entry["case_id"] or "",
                                      entry["variant"] or "", entry["key"]))
    return {"groups": described, "excluded": excluded}


def comparisons(rows):
    """``{group key: [member, ...]}`` -- the historic shape of the comparison set."""
    return {entry["key"]: entry["members"] for entry in comparison_groups(rows)["groups"]}


# ------------------------------------------------------------------ scaling --

def scaling(rows):
    """Join EP sizes that measured the same work under the same conditions.

    A group needs at least two *distinct* EP sizes: one point is not a scaling
    series, and the same point measured twice is a repeat, not a trend.
    """
    groups = defaultdict(list)
    for row in rows:
        cfg = row.get("config") or {}
        scale = cfg.get("scaling")
        if not scale or not comparability.comparable(row):
            continue
        workload = row.get("workload")
        throughput = row.get("throughput") or {}
        if not workload or not throughput:
            continue
        speed = throughput.get("global_source_tokens_per_second")
        if not speed:
            continue
        ep = comparability.ep_size(row)
        tokens = workload.get("global_source_tokens")
        experts = (cfg.get("model") or {}).get("num_experts")
        size = [tokens, experts]
        if scale["mode"] == "weak":
            counts = workload.get("tokens_per_rank")
            if not isinstance(counts, (list, tuple)) or len(set(counts)) != 1:
                continue
            if tokens % ep or experts % ep:
                continue
            size = [tokens // ep, experts // ep]
        groups[comparability.scaling_key(row, size)].append(
            (ep, speed, row.get("task_id"), scale["mode"], row.get("run_id")))
    report = {}
    for key, group in groups.items():
        group = sorted(group)
        if len(group) < 2 or len({item[0] for item in group}) != len(group):
            continue
        base_ep, base_rate, *_ = group[0]
        report[key] = [
            {"ep": ep, "tokens_per_second": rate, "efficiency": rate / base_rate / (ep / base_ep),
             "task_id": task_id, "mode": mode, "run_id": run_id,
             "speedup_vs_base": rate / base_rate}
            for ep, rate, task_id, mode, run_id in group
        ]
    return report


# --------------------------------------------------------------- joins -------

def fusion(rows):
    """Join measured serial and fused stages by workload and timing scope."""
    groups = defaultdict(list)
    for row in rows:
        if row["case_id"] in {"FUSE-001", "FUSE-002", "FUSE-003", "FUSE-004"} and comparable(row):
            groups[group_key(row, case=False)].append(row)
    output = []
    for group in groups.values():
        for serial_id, fused_id in (("FUSE-001", "FUSE-002"), ("FUSE-003", "FUSE-004")):
            staged = [row for row in group if row["case_id"] == serial_id]
            for fused in (row for row in group if row["case_id"] == fused_id):
                references = [r for r in staged if r["backend_id"] in {fused["backend_id"], "alltoall_hccl"}]
                for serial in references:
                    baseline = serial["latency"]["p50"]
                    measured = fused["latency"]["p50"]
                    output.append({
                        "fused_task": fused["task_id"], "serial_task": serial["task_id"],
                        "fused_backend": fused["backend_id"], "serial_backend": serial["backend_id"],
                        "baseline_type": "native_serial" if serial["backend_id"] == fused["backend_id"] else "common_serial",
                        "serial_latency_ms": baseline, "fused_latency_ms": measured,
                        "fusion_speedup": baseline / measured,
                        "total_time_saving_ratio": 1 - measured / baseline,
                        "internal_overlap_ratio": None,
                    })
    return output


def pareto(rows):
    fronts = {}
    groups = defaultdict(list)
    for row in rows:
        if comparable(row) and row.get("throughput") and row.get("memory"):
            speed = row["throughput"]["global_source_tokens_per_second"]
            memory = [v["peak_allocated"] for v in row["memory"]["ranks"] if v["peak_allocated"] is not None]
            if speed and memory:
                groups[group_key(row)].append((speed, max(memory), row["task_id"]))
    for key, points in groups.items():
        fronts[key] = [{"task_id": task_id, "tokens_per_second": speed, "peak_allocator_bytes": memory}
                       for speed, memory, task_id in points
                       if not any(s >= speed and m <= memory and (s > speed or m < memory) for s, m, _ in points)]
    return fronts


def planner_comparisons(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["case_id"] != "LAYER-003" or not comparable(row):
            continue
        # The cached plan keeps the same replicas and prefetch semantics.
        key = group_key({**row, "case_parameters": {}}, case=False)
        groups[(row["run_id"], row["backend_id"], key)][comparability.case_parameters(row)["variant"]] = row
    points = []
    for variants in groups.values():
        if not {"fresh_plan", "cached_plan"} <= variants.keys():
            continue
        fresh, cached = variants["fresh_plan"], variants["cached_plan"]
        fresh_ms, cached_ms = fresh["latency"]["p50"], cached["latency"]["p50"]
        points.append({
            "fresh_task": fresh["task_id"], "cached_task": cached["task_id"],
            "fresh_ms": fresh_ms, "cached_ms": cached_ms,
            "cached_plan_speedup": fresh_ms / cached_ms,
            "source": "matched native training cycle measurements",
        })
    return points


def fixed_owner_comparisons(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["case_id"] == "LAYER-003" and comparable(row):
            key = group_key({**row, "case_parameters": {}}, case=False)
            groups[(row["run_id"], key)][comparability.case_parameters(row)["variant"]] = row
    points = []
    for variants in groups.values():
        if {"fixed_owner", "fresh_plan"} <= variants.keys():
            native, reference = variants["fresh_plan"], variants["fixed_owner"]
            points.append({
                "planner_task": native["task_id"], "fixed_owner_task": reference["task_id"],
                "planner_ms": native["latency"]["p50"], "fixed_owner_ms": reference["latency"]["p50"],
                "speedup_vs_fixed_owner_hccl": reference["latency"]["p50"] / native["latency"]["p50"],
                "planner_backend": native["backend_id"], "fixed_owner_backend": reference["backend_id"],
            })
    return points


def dedup_comparisons(rows):
    groups = defaultdict(dict)
    for row in rows:
        mode = ((row.get("config") or {}).get("dedup") or {}).get("mode")
        if row["case_id"] not in {"COMM-010", "LAYER-005", "LAYER-006"} or mode in (None, "none"):
            continue
        if not comparable(row):
            continue
        config = {**row["config"], "dedup": {}}
        key = group_key({**row, "config": config, "case_parameters": {}}, case=False)
        if comparability.case_parameters(row).get("variant") == "fixed_owner":
            groups[(row["run_id"], key)][f"fixed_owner_{mode}"] = row
        else:
            groups[(row["run_id"], key)][mode] = row
    points = []
    for pair in groups.values():
        for baseline_mode, mode in (("rank", "server"), ("fixed_owner_rank", "rank"),
                                   ("fixed_owner_server", "server")):
            if not {baseline_mode, mode} <= pair.keys():
                continue
            baseline, native = pair[baseline_mode], pair[mode]
            baseline_model = baseline["bandwidth"] or baseline["load_balance"].get("dedup_dispatch_traffic", {})
            native_model = native["bandwidth"] or native["load_balance"].get("dedup_dispatch_traffic", {})
            baseline_ms, native_ms = baseline["latency"]["p50"], native["latency"]["p50"]
            baseline_bytes = baseline_model.get("cross_server_payload_bytes")
            native_bytes = native_model.get("cross_server_payload_bytes")
            points.append({
                "level": native["benchmark_level"], "baseline_mode": baseline_mode, "native_mode": mode,
                "baseline_task": baseline["task_id"], "native_task": native["task_id"],
                "baseline_ms": baseline_ms, "native_ms": native_ms, "speedup": baseline_ms / native_ms,
                "baseline_cross_server_bytes": baseline_bytes, "native_cross_server_bytes": native_bytes,
                "modeled_payload_saving": 1 - native_bytes / baseline_bytes
                if baseline_bytes and native_bytes is not None else None,
                "payload_source": "routing_trace_model_of_dispatch",
            })
    return points
