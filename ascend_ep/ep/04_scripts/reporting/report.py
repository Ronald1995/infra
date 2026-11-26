"""Write Markdown, JSON and CSV from the same versioned result records."""
from collections import Counter
import csv
import json

from common.storage import write_json


def write_table(path, title, columns, rows, empty="No matching measurements."):
    text = [f"# {title}", ""]
    rows = list(rows)
    if rows:
        def line(values):
            return "| " + " | ".join(str(value if value is not None else "unavailable").replace(
                "|", "\\|"
            ).replace("\n", " ") for value in values) + " |"
        text.extend((line(columns), line(["---"] * len(columns))))
        text.extend(line(row) for row in rows)
    else:
        text.append(empty)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def report(root, manifest, rows, analysis):
    destination = root / "reports"
    destination.mkdir(exist_ok=True)
    stats = Counter(row["status"] for row in rows)
    write_json(destination / "benchmark_summary.json", {
        "schema_version": 1, "run_id": manifest["run_id"], "status_counts": dict(stats),
        "results": rows, "analysis": analysis,
    })
    columns = ("task_id", "backend_id", "case_id", "benchmark_level", "device", "status",
               "p50_ms", "p95_ms", "p99_ms", "tokens_per_second", "payload_GBps", "error_reason")

    def summarize(row):
        delay = row["latency"] or {}
        rate = row["throughput"] or {}
        traffic = row["bandwidth"] or {}
        return (row["task_id"], row["backend_id"], row["case_id"], row["benchmark_level"],
                row["config"]["device"], row["status"], delay.get("p50"), delay.get("p95"),
                delay.get("p99"), rate.get("global_source_tokens_per_second"), traffic.get("payload_bandwidth_GBps"),
                row["error_reason"])

    with (destination / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(map(summarize, rows))
    write_table(destination / "benchmark_report.md", f"EP benchmark {manifest['run_id']}",
                columns, map(summarize, rows))
    with (destination / "benchmark_report.md").open("a", encoding="utf-8") as file:
        file.write("\nCPU results validate framework behavior. NPU results require an Ascend run.\n")
        file.write("\nOutcome counts: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())) + "\n")
        files = ("operator_comparison.md", "bandwidth_analysis.md", "overlap_analysis.md",
                 "scalability_analysis.md", "load_balance_analysis.md", "planner_training.md", "dedup_comparison.md",
                 "failure_summary.md")
        file.write("\n" + "\n".join(f"- [{name}]({name})" for name in files) + "\n")
        file.write("\n[Environment and versions](../environment.json) · [Run manifest](../manifest.json)\n")
    write_table(destination / "operator_comparison.md", "Comparable operator measurements",
                ("Group", "Device", "Case", "Backend", "P50 ms"),
                ((key[:12], row["device"], row["case"], row["backend"], row["p50_ms"])
                 for key, group in analysis["comparison_groups"].items() for row in group))
    write_table(destination / "bandwidth_analysis.md", "Modelled payload bandwidth",
                ("Case", "Backend", "Bytes", "GB/s", "Source"),
                ((row["case_id"], row["backend_id"], row["bandwidth"].get("payload_bytes"),
                  row["bandwidth"].get("payload_bandwidth_GBps"), row["bandwidth"]["source"])
                 for row in rows if row["bandwidth"]))
    write_table(destination / "overlap_analysis.md", "Visible overlap and idle intervals",
                ("Task", "Rank", "Overlap ns", "Idle ns", "Artifact"),
                ((task, rank, metrics["visible_overlap_ns"], metrics["device_idle_ns"], attempt["manifest"])
                 for task, attempts in analysis["profiles"].items() for attempt in attempts
                 for rank, metrics in attempt["ranks"].items()))
    with (destination / "overlap_analysis.md").open("a", encoding="utf-8") as file:
        file.write("\n## Measured serial/fused comparisons\n\n")
        for point in analysis["fusion"]:
            file.write(f"- {point['fused_task']} / {point['serial_task']}: "
                       f"{point['fusion_speedup']:.3f}× ({point['baseline_type']})\n")
    write_table(destination / "scalability_analysis.md", "Comparable scaling points",
                ("Group", "Mode", "EP", "tokens/s", "Efficiency"),
                ((key[:12], row["mode"], row["ep"], row["tokens_per_second"], row["efficiency"])
                 for key, values in analysis["scaling_groups"].items() for row in values))
    write_table(destination / "load_balance_analysis.md", "Routing load and dedup opportunities",
                ("Task", "Expert max/mean", "Rank max/mean", "Cross-server ratio", "Rank dedup opportunity"),
                ((row["task_id"], row["load_balance"]["expert_max_mean"], row["load_balance"]["rank_max_mean"],
                  row["load_balance"]["cross_server_ratio"], (row["bandwidth"] or {}).get("dedup_opportunity_ratio"))
                 for row in rows if row["load_balance"]))
    planner_path = destination / "planner_training.md"
    write_table(planner_path, "Native replica training stages",
                ("Task", "Planned rank max/mean", "Weight prefetch P50 ms",
                 "Gradient reduction P50 ms", "Replicated weight bytes"),
                ((row["task_id"], row["load_balance"].get("planned_rank_max_mean"),
                  row["load_balance"].get("weight_prefetch_latency_ms"),
                  row["load_balance"].get("gradient_reduce_latency_ms"),
                  row["load_balance"].get("redundant_weight_bytes"))
                 for row in rows if (row.get("load_balance") or {}).get("planned_expert_copies") is not None))
    with planner_path.open("a", encoding="utf-8") as file:
        file.write("\n## Fresh and cached routing plans\n\n")
        for point in analysis["planner_comparisons"]:
            file.write(f"- {point['fresh_task']} and {point['cached_task']} · "
                       f"{point['fresh_ms']:.3f} ms and {point['cached_ms']:.3f} ms · "
                       f"{point['cached_plan_speedup']:.3f}× cached-plan speedup\n")
        file.write("\n## MoonEP planner and fixed-owner HCCL reference\n\n")
        for point in analysis["fixed_owner_comparisons"]:
            file.write(f"- {point['planner_task']} and {point['fixed_owner_task']} · "
                       f"{point['planner_ms']:.3f} ms and {point['fixed_owner_ms']:.3f} ms · "
                       f"{point['speedup_vs_fixed_owner_hccl']:.3f}× vs. fixed-owner HCCL\n")
    write_table(destination / "dedup_comparison.md", "Native dedup and fixed-owner HCCL measurements",
                ("Level", "Baseline mode", "Native mode", "Baseline task", "Native task",
                 "Baseline P50 ms", "Native P50 ms", "Speedup", "Baseline dispatch cross-server bytes (model)",
                 "Native dispatch cross-server bytes (model)", "Modelled payload saving"),
                ((p["level"], p["baseline_mode"], p["native_mode"], p["baseline_task"], p["native_task"],
                  p["baseline_ms"], p["native_ms"], p["speedup"], p["baseline_cross_server_bytes"], p["native_cross_server_bytes"],
                  p["modeled_payload_saving"]) for p in analysis["dedup_comparisons"]))
    write_table(destination / "failure_summary.md", "Failed, unsupported and skipped cases",
                ("Task", "Backend", "Status", "Evidence"),
                ((row["task_id"], row["backend_id"], row["status"], row["error_reason"] or "No run record")
                 for row in rows if row["status"] != "passed"))
    from reporting.coverage import write as write_coverage
    coverage = write_coverage(root, manifest, rows)
    with (destination / "benchmark_report.md").open("a", encoding="utf-8") as file:
        file.write("\n## Coverage and EP comparison\n\n")
        for label, key in (("Case coverage matrix", "coverage"),
                           ("Case x operator coverage", "case_operator"),
                           ("Case x operator x variant coverage", "variants"),
                           ("Operator coverage", "operators"),
                           ("EP size comparison", "comparison"),
                           ("L0 scale-out", "l0_scaleout")):
            if key in coverage:
                file.write(f"- [{label}]({coverage[key].name})\n")
    from reporting.l0_report import write as write_l0
    capabilities = None
    capability_path = root / "capabilities.json"
    if capability_path.is_file():
        capabilities = __import__("json").loads(capability_path.read_text(encoding="utf-8"))
    if any(row.get("benchmark_level") == "L0" for row in rows):
        write_l0(root, manifest, rows, capabilities)
        with (destination / "benchmark_report.md").open("a", encoding="utf-8") as file:
            file.write("\n- [l0_report.md](l0_report.md)\n")
    from reporting.charts import create_charts
    create_charts(destination, rows, analysis)
    charts = sorted((destination / "charts").glob("*.png"))
    if charts:
        with (destination / "benchmark_report.md").open("a", encoding="utf-8") as file:
            file.write("\n## Charts\n\n" + "\n".join(
                f"- [{path.stem}](charts/{path.name})" for path in charts
            ) + "\n")
    return destination
