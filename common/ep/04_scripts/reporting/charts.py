"""Generate standalone charts when the plotting extra is installed."""
from collections import defaultdict

from analysis.comparison import comparable
from common.storage import fingerprint


def create_charts(destination, rows, analysis):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return

    charts = destination / "charts"
    charts.mkdir(exist_ok=True)

    def plot(name, title, render):
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        ax.set_title(title)
        render(ax)
        fig.savefig(charts / (name + ".png"), dpi=130)
        plt.close(fig)

    validated = {r["task_id"]: r for r in rows if comparable(r)}
    for group, members in analysis["comparison_groups"].items():
        first = validated[members[0]["task_id"]]
        case = first["case_id"]
        if case not in {"COMM-001", "COMM-003", "MOE-001", "MOE-002", "MOE-003"}:
            continue
        plot(f"latency-{case}-{group[:10]}", f"{case} · {first['config']['device']} · P50 ms",
             lambda ax: ax.barh([m["backend"] for m in members], [m["p50_ms"] for m in members]))
    for group, samples in analysis["scaling_groups"].items():
        plot(f"scaling-{group[:10]}", f"{samples[0]['mode'].capitalize()} scaling efficiency",
             lambda ax: ax.plot([p["ep"] for p in samples], [p["efficiency"] for p in samples], marker="o"))
    for group, points in analysis["pareto"].items():
        plot(f"pareto-{group[:10]}", "Source tokens/s vs peak allocator MiB",
             lambda ax: ax.scatter([p["peak_allocator_bytes"] / (1 << 20) for p in points],
                                   [p["tokens_per_second"] for p in points]))
    if analysis["fusion"]:
        staged = analysis["fusion"]
        plot("fusion-speedup", "Measured serial latency / fused latency",
             lambda ax: ax.barh([p["fused_task"] for p in staged], [p["fusion_speedup"] for p in staged]))
    rates = defaultdict(list)
    profiled = []
    for task, profiles in analysis["profiles"].items():
        for entry in profiles:
            for rank, profile in entry["ranks"].items():
                idle = profile["device_idle_ratio"]
                if idle is not None:
                    profiled.append((f"{task} · rank {rank}", idle))
    if profiled:
        plot("device-idle", "Idle fraction of observed device span",
             lambda ax: ax.barh([task for task, _ in profiled], [idle for _, idle in profiled]))
    plotted_routes = set()
    for row in validated.values():
        load = row.get("load_balance")
        traffic = row.get("bandwidth") or {}
        amount = traffic.get("payload_bandwidth_GBps")
        if amount is not None and row["workload"]:
            cfg = row["config"]
            key = fingerprint([row["case_id"], row["backend_id"], cfg["model"], row["topology"],
                               cfg["benchmark"]["timing"], cfg["workload"]["routing"], cfg["dedup"]])
            rates[key].append((row["workload"]["global_source_tokens"], amount))
        if load is None:
            continue
        name = row["task_id"]
        if row["routing_trace_hash"] in plotted_routes:
            continue
        plotted_routes.add(row["routing_trace_hash"])
        plot(f"traffic-{name}", "Source rank × destination rank · routed rows",
             lambda ax: ax.imshow(load["source_destination_matrix"], aspect="auto"))
        plot(f"expert-load-{name}", "Routes per expert",
             lambda ax: ax.bar(range(len(load["expert_counts"])), load["expert_counts"]))
        plot(f"rank-load-{name}", "Routes per receiving rank",
             lambda ax: ax.bar(range(len(load["rank_counts"])), load["rank_counts"]))
    for key, points in rates.items():
        if len({tokens for tokens, _ in points}) > 1:
            x, y = zip(*sorted(points))
            plot(f"bandwidth-{key[:10]}", "Source tokens · modeled payload GB/s",
                 lambda ax: ax.plot(x, y, marker="o"))
    memory = [(r["task_id"], max(available)) for r in validated.values() if (available := [
        item["peak_allocated"] for item in (r["memory"] or {}).get("ranks", [])
        if item["peak_allocated"] is not None
    ])]
    if memory:
        plot("peak-allocator", "Peak framework allocator MiB",
             lambda ax: ax.barh([task for task, _ in memory], [size / (1 << 20) for _, size in memory]))
