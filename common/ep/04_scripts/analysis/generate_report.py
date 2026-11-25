"""Analyze persisted performance and rank profiler artifacts."""
from analysis.comparison import comparisons, fusion, pareto, scaling
from common.storage import write_json
from profiling.analyze_overlap import analyze
from profiling.parse_profile import parse_directory


def analyze_run(root, rows, extra_rows=()):
    profiles, hypotheses, parse_errors = {}, [], []
    for row in rows:
        task = row["task_id"]
        for artifact in row["profiling_artifacts"]:
            if artifact["status"] != "passed":
                continue
            path = root / artifact["manifest"]
            try:
                events = parse_directory(path.parent)
            except (OSError, TypeError, ValueError) as exc:
                parse_errors.append({"artifact": artifact["manifest"], "reason": str(exc)})
                continue
            profile = analyze(events if row["config"]["device"] == "npu"
                              else [event for event in events if event.kernel_category == "host"])
            write_json(root / "analysis" / f"{task}-{path.parent.name}-events.json",
                       [event.as_dict() for event in events])
            profiles.setdefault(task, []).append({"manifest": artifact["manifest"], "ranks": profile})
            for rank, data in profile.items():
                idle = data["device_idle_ratio"]
                if idle is not None and idle > .2:
                    hypotheses.append({"task_id": task, "rank": rank, "device_idle_ratio": idle,
                                       "candidate": "Host scheduling or synchronization",
                                       "evidence": artifact["manifest"]})
    combined = [*rows, *extra_rows]
    measured_fusion = fusion(combined)
    for item in measured_fusion:
        if item["fusion_speedup"] < 1:
            hypotheses.append({
                "task_id": item["fused_task"], "candidate": "Fused schedule or resource contention",
                "fusion_speedup": item["fusion_speedup"], "serial_latency_ms": item["serial_latency_ms"],
                "fused_latency_ms": item["fused_latency_ms"], "evidence": item["serial_task"],
            })
    output = {"comparison_groups": comparisons(rows), "fusion": measured_fusion,
              "scaling_groups": scaling(combined), "pareto": pareto(rows), "profiles": profiles,
              "hypotheses": hypotheses, "profile_parse_errors": parse_errors}
    write_json(root / "analysis" / "results.json", output)
    return output
