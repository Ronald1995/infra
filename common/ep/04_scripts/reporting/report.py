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
                 "scalability_analysis.md", "load_balance_analysis.md", "failure_summary.md")
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
    write_table(destination / "failure_summary.md", "Failed, unsupported and skipped cases",
                ("Task", "Backend", "Status", "Evidence"),
                ((row["task_id"], row["backend_id"], row["status"], row["error_reason"] or "No run record")
                 for row in rows if row["status"] != "passed"))
    from reporting.charts import create_charts
    create_charts(destination, rows, analysis)
    charts = sorted((destination / "charts").glob("*.png"))
    if charts:
        with (destination / "benchmark_report.md").open("a", encoding="utf-8") as file:
            file.write("\n## Charts\n\n" + "\n".join(
                f"- [{path.stem}](charts/{path.name})" for path in charts
            ) + "\n")
    return destination
