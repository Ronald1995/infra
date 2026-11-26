"""Compare only compatible, validated measurements and matching scaling points."""
from collections import defaultdict

from baselines.l0.verification.path_verifier import status_allows_comparison
from common.storage import fingerprint


def _verification_status(record):
    """The path-verification status a record carries, or ``None``."""
    return ((record.get("verification") or {}).get("status")
            or (record.get("observed") or {}).get("verification_status"))


def comparable(record):
    """True when a record may enter a cross-backend comparison.

    An L0 hardware baseline is only comparable when its *path* was verified: a
    number measured over an unverified transport is not a measurement of the
    mechanism it names, so pairing it with another mechanism would compare two
    different things.  Operator cases are gated on their own correctness instead,
    because their transport is reported separately in `observation`.
    """
    if record["benchmark_level"] == "L0" and not status_allows_comparison(
            _verification_status(record)):
        return False
    return (record["status"] == "passed" and record.get("source") == "measured"
            and record["correctness"] and record["correctness"]["passed"]
            and record.get("latency") and record["latency"]["p50"] > 0)


def group_key(row, case=True):
    cfg = row["config"]
    env = row["environment"]
    return fingerprint([
        cfg["device"], env.get("npu_models"), env.get("cann_version"), env.get("packages"),
        row["topology"], cfg["distributed"]["ep_size"], cfg["distributed"]["tp_size"],
        row["workload"], row["routing_trace_hash"], cfg["model"], cfg["benchmark"]["timing"],
        cfg["dedup"], cfg["load_balance"], row["benchmark_level"],
        row["operation"] if case else None, row["case_parameters"],
    ])


def comparisons(rows):
    groups = defaultdict(list)
    for row in rows:
        if comparable(row):
            groups[group_key(row)].append(row)
    return {
        key: [{"backend": row["backend_id"], "case": row["case_id"], "p50_ms": row["latency"]["p50"],
               "task_id": row["task_id"], "device": row["config"]["device"]}
              for row in sorted(group, key=lambda r: r["latency"]["p50"])]
        for key, group in groups.items()
    }


def scaling(rows):
    groups = defaultdict(list)
    for row in rows:
        cfg = row["config"]
        scale = cfg.get("scaling")
        if not scale or not comparable(row) or not row["workload"] or not row["throughput"]:
            continue
        speed = row["throughput"]["global_source_tokens_per_second"]
        if not speed:
            continue
        ep = cfg["distributed"]["ep_size"]
        workload = row["workload"]
        tokens, experts = workload["global_source_tokens"], cfg["model"]["num_experts"]
        size = [tokens, experts]
        if scale["mode"] == "weak":
            counts = workload["tokens_per_rank"]
            if len(set(counts)) != 1 or tokens % ep or experts % ep:
                continue
            size = [tokens // ep, experts // ep]
        model = {key: value for key, value in cfg["model"].items() if key != "num_experts"}
        env = row["environment"]
        key = fingerprint([row["backend_id"], row["source_versions"], row["case_id"], scale, size,
                           cfg["device"], model, cfg["dedup"], cfg["load_balance"],
                           cfg["distributed"]["tp_size"], cfg["benchmark"]["timing"],
                           cfg["workload"]["routing"], cfg["workload"]["seed"],
                           sorted(set(env.get("npu_models", []))), env.get("cann_version"), env.get("packages")])
        groups[key].append((ep, speed, row["task_id"], scale["mode"], row["run_id"]))
    report = {}
    for key, group in groups.items():
        group = sorted(group)
        if len(group) < 2 or len({item[0] for item in group}) != len(group):
            continue
        base_ep, base_rate, *_ = group[0]
        report[key] = [
            {"ep": ep, "tokens_per_second": rate, "efficiency": rate / base_rate / (ep / base_ep),
             "task_id": task_id, "mode": mode, "run_id": run_id}
            for ep, rate, task_id, mode, run_id in group
        ]
    return report


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
        groups[(row["run_id"], row["backend_id"], key)][row["case_parameters"]["variant"]] = row
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
            groups[(row["run_id"], key)][row["case_parameters"]["variant"]] = row
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
        mode = row["config"]["dedup"]["mode"]
        if row["case_id"] not in {"COMM-010", "LAYER-005", "LAYER-006"} or mode == "none" or not comparable(row):
            continue
        config = {**row["config"], "dedup": {}}
        key = group_key({**row, "config": config, "case_parameters": {}}, case=False)
        if row["case_parameters"]["variant"] == "fixed_owner":
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
