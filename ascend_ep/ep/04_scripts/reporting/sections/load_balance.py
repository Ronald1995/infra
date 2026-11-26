"""Section 8: Load Balance and Planner.

The load-balance numbers come from the routing trace, not from a device counter, and for
a synthetic routing strategy they describe *that synthetic workload only*.  The section
states which workload produced them instead of implying production traffic.
"""
from analysis import comparability
from reporting import empty
from reporting import formatting as F

#: Routing strategies that are synthetic by construction.
SYNTHETIC_STRATEGIES = {"balanced", "uniform_random", "expert_hotspot", "rank_hotspot",
                        "server_hotspot", "high_locality", "cross_server_heavy",
                        "dedup_controlled", "imbalanced"}


TITLE = "Load Balance and Planner"


def render(ctx):
    return [f"## 8. {TITLE}", "", *body(ctx)]


def body(ctx):
    lines = []
    rows = [row for row in ctx.rows if row.get("load_balance")]
    lines += ["### Expert and rank distribution", ""]
    lines += _distribution(ctx, rows)
    lines += ["### Dedup", ""]
    lines += _dedup(ctx)
    lines += ["### Planner and training stages", ""]
    lines += _planner(ctx, rows)
    return lines


def distribution_body(ctx):
    """The distribution tables alone, for ``load_balance_analysis.md``."""
    return _distribution(ctx, [row for row in ctx.rows if row.get("load_balance")])


def dedup_body(ctx):
    """The dedup tables alone, for ``dedup_comparison.md``."""
    return _dedup(ctx)


def planner_body(ctx):
    """The planner tables alone, for ``planner_training.md``."""
    return _planner(ctx, [row for row in ctx.rows if row.get("load_balance")])


def _distribution(ctx, rows):
    if not rows:
        return empty.section(
            "Why No Load-Balance Distribution Is Reported",
            level=4,
            findings="No result carries a routing-trace load-balance summary, so no expert or "
                     "rank distribution is reported.",
            causes=["The run produced no passing L1-L4 task whose routing trace was summarised.",
                    "`load_balance` is written by the worker from the routing trace; a failed or "
                    "unplanned task has none."],
            required=["Run at least one operator case on a device (or the CPU stand-in) so the "
                      "routing trace is generated.",
                      "Keep `workload.routing` configured: the summary is derived from the trace, "
                      "not from the model shape."],
            alternatives=["Section 2 reports the coverage of the operator levels.",
                          "Section 9 lists every task that did not produce a measurement."],
            conclusion="No expert-load, rank-load or traffic-distribution statement is made for "
                       "this run.")
    table_rows = []
    for row in sorted(rows, key=lambda item: str(item["task_id"])):
        load = row["load_balance"]
        bandwidth = row.get("bandwidth") or {}
        size = comparability.ep_size(row)
        table_rows.append([
            row.get("case_id"), row.get("backend_id"),
            f"EP{size}" if size else F.MISSING, F.text(load.get("source")),
            F.number(load.get("expert_max_mean")), F.number(load.get("rank_max_mean")),
            F.number(load.get("rank_tx_max_mean")), F.number(load.get("rank_rx_max_mean")),
            F.percent(load.get("cross_server_ratio")),
            F.percent(load.get("local_route_ratio")),
            F.percent(bandwidth.get("dedup_opportunity_ratio")),
        ])
    lines = F.table(("Case", "Backend", "EP", "Source", "Expert max/mean", "Rank max/mean",
                     "Rank TX max/mean", "Rank RX max/mean", "Cross-server routes",
                     "Local route ratio", "Rank dedup opportunity"), table_rows)
    lines += _synthetic_warning(ctx)
    lines += ["`Expert max/mean` and `Rank max/mean` are `max * count / sum` over the routing "
              "trace; the source-to-destination traffic matrix is in `load_balance_analysis.md` and "
              "in the `.png` chart for each trace.", ""]
    return lines


def _synthetic_warning(ctx):
    strategies = set()
    for row in ctx.rows:
        routing = (((row.get("config") or {}).get("workload") or {}).get("routing"))
        if routing:
            strategies.add(routing)
    synthetic = sorted(strategies & SYNTHETIC_STRATEGIES)
    if not synthetic:
        return []
    return [f"> **Synthetic workload**: routing strategies {F.join(synthetic)} generate a "
            f"balanced/controlled synthetic routing trace. These numbers describe the load "
            f"balance reachable under that synthetic distribution and do **not** represent a real "
            f"production traffic mix.", ""]


def _dedup(ctx):
    points = ctx.analysis.get("dedup_comparisons") or []
    if not points:
        return empty.section(
            "Why No Dedup Result Is Reported",
            level=4,
            findings="No rank-dedup or server-dedup comparison exists in this run.",
            causes=["No dedup case (COMM-010, LAYER-005, LAYER-006) produced a passing result "
                    "with a non-`none` dedup mode.",
                    "A dedup comparison needs two modes of the *same* case at the same EP size: "
                    "`rank` vs `server`, or `fixed_owner` vs the native mode."],
            required=["Set `dedup.mode` to `rank` and to `server` in two configurations of the "
                      "same dedup case, or use `configs/dedup.yaml`.",
                      "Run both so the routing trace matches (same seed, same placement)."],
            alternatives=["Section 8.1 reports the rank-dedup opportunity derived from the routing "
                          "trace where a trace exists."],
            conclusion="No dedup benefit is claimed; the payload saving of a dedup mode was not "
                       "measured in this run.")
    rows = []
    for point in points:
        rows.append([F.text(point["level"]), F.text(point["baseline_mode"]),
                     F.text(point["native_mode"]), F.text(point["baseline_task"]),
                     F.text(point["native_task"]), F.number(point["baseline_ms"]),
                     F.number(point["native_ms"]), F.number(point["speedup"]),
                     F.binary(point["baseline_cross_server_bytes"]),
                     F.binary(point["native_cross_server_bytes"]),
                     F.percent(point["modeled_payload_saving"])])
    lines = F.table(("Level", "Baseline mode", "Native mode", "Baseline task", "Native task",
                     "Baseline P50 ms", "Native P50 ms", "Speedup",
                     "Baseline cross-server bytes (model)", "Native cross-server bytes (model)",
                     "Modelled payload saving"), rows)
    lines += ["Cross-server bytes come from the routing-trace model (`payload_source: "
              "routing_trace_model_of_dispatch`), not from a link counter.", ""]
    return lines


def _planner(ctx, rows):
    staged = [row for row in rows
              if (row["load_balance"] or {}).get("planned_expert_copies") is not None]
    planner_comparisons = ctx.analysis.get("planner_comparisons") or []
    fixed_owner = ctx.analysis.get("fixed_owner_comparisons") or []
    lines = []
    if staged:
        table_rows = []
        for row in sorted(staged, key=lambda item: str(item["task_id"])):
            load = row["load_balance"]
            table_rows.append([
                row.get("case_id"), row.get("backend_id"),
                F.text(load.get("planned_rank_max_mean")),
                F.number(load.get("planner_dispatch_latency_ms")),
                F.number(load.get("planner_latency_ms")),
                F.number(load.get("weight_prefetch_latency_ms")),
                F.number(load.get("gradient_reduce_latency_ms")),
                F.binary(load.get("redundant_weight_bytes")),
                F.number(load.get("planner_net_speedup")),
            ])
        lines += F.table(("Case", "Backend", "Planned rank max/mean", "Planner dispatch ms",
                          "Planner latency ms", "Weight prefetch ms", "Gradient reduce ms",
                          "Redundant weight bytes", "Net speedup"), table_rows)
    else:
        lines += ["_No planner stage measurement exists: no result carries "
                  "`planned_expert_copies` or the per-stage planner latencies._", ""]
    if planner_comparisons:
        lines += ["#### Fresh vs cached plan", ""]
        lines += F.table(("Fresh task", "Cached task", "Fresh P50 ms", "Cached P50 ms",
                          "Cached-plan speedup"),
                         ([point["fresh_task"], point["cached_task"],
                           F.number(point["fresh_ms"]), F.number(point["cached_ms"]),
                           F.number(point["cached_plan_speedup"])]
                          for point in planner_comparisons))
    else:
        lines += ["_No fresh/cached plan pair exists, so no cached-plan benefit is reported._", ""]
    if fixed_owner:
        lines += ["#### Native planner vs fixed-owner HCCL", ""]
        lines += F.table(("Planner task", "Fixed-owner task", "Planner P50 ms",
                          "Fixed-owner P50 ms", "Speedup vs fixed-owner HCCL"),
                         ([point["planner_task"], point["fixed_owner_task"],
                           F.number(point["planner_ms"]), F.number(point["fixed_owner_ms"]),
                           F.number(point["speedup_vs_fixed_owner_hccl"])]
                          for point in fixed_owner))
    return lines
