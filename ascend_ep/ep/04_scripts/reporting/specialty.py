"""The specialty reports that back up the main report.

Each one answers a narrower question than ``benchmark_report.md`` and must explain
itself when the data is missing: an empty specialty file is a defect, not a shortcut.
Reports whose content is also a main-report section reuse that section's ``body()`` so
the two can never disagree.
"""
from reporting import empty
from reporting import formatting as F
from reporting.sections import comparison, failures, load_balance, overlap, scalability


def _document(title, lines):
    return [f"# {title}", "", *lines]


def operator_comparison(ctx):
    """Backend comparison in full: every group, plus the excluded measurements."""
    lines = _document("Backend Comparison", comparison.body(ctx))
    excluded = (ctx.analysis.get("comparison_details") or {}).get("excluded") or []
    lines += ["## Measurements excluded from comparison", "",
              "A record is excluded when it is not an accepted measurement, or when an L0 "
              "execution path is not `verified`.", ""]
    rows = [(item.get("task_id"), item.get("case_id"), item.get("backend_id"),
             item.get("status"), item.get("reason")) for item in excluded]
    lines += F.table(("Task", "Case", "Backend", "Status", "Reason"), rows,
                     empty="Every result record passed the eligibility gate.")
    return lines


def bandwidth_analysis(ctx):
    """Payload bandwidth with scope, source and the four reasons a cell is empty."""
    lines = _document("Payload Bandwidth Analysis", [])
    lines += ["`payload_bandwidth_GBps` is **byte-model / latency**, not a physical link counter. "
              "The scope and source columns say which bytes and which model were used; "
              "`physical_bandwidth_GBps` stays unavailable in every row because no link counter "
              "is collected.", ""]
    buckets = {"measured": [], "no_payload": [], "no_model": [], "failed": [], "unavailable": []}
    for row in ctx.rows:
        bucket, _reason = _bandwidth_bucket(row)
        buckets[bucket].append(row)
    lines += ["## Why a bandwidth cell is empty", ""]
    lines += F.table(("Reason", "Tasks", "Meaning"), [
        ("Not a communication payload", F.count(len(buckets["no_payload"])),
         "compute/copy cases move no communication payload; the metric does not apply"),
        ("No bandwidth model", F.count(len(buckets["no_model"])),
         "the level does not model wire traffic for this case, so no payload bytes exist"),
        ("Measurement failed", F.count(len(buckets["failed"])),
         "the task did not pass, so there is no latency to divide by"),
        ("Metric unavailable", F.count(len(buckets["unavailable"])),
         "payload bytes exist but no bandwidth value was derived"),
    ])
    lines += ["## Measured payload bandwidth", ""]
    rows = []
    for row in buckets["measured"]:
        bandwidth = row.get("bandwidth") or {}
        rows.append([row.get("case_id"), row.get("backend_id"), _ep(row),
                     F.text(row.get("status")),
                     F.number((row.get("latency") or {}).get("p50")),
                     F.count(bandwidth.get("payload_bytes")),
                     F.count(bandwidth.get("logical_bytes")),
                     F.number(bandwidth.get("payload_bandwidth_GBps")),
                     F.number(bandwidth.get("logical_bandwidth_GBps")),
                     F.text(bandwidth.get("scope")), F.text(bandwidth.get("source")),
                     F.text(bandwidth.get("traffic_model"))])
    lines += F.table(("Case", "Backend", "EP", "Status", "P50 ms", "Payload bytes",
                      "Logical bytes", "Payload GB/s", "Logical GB/s", "Scope", "Source",
                      "Traffic model"), rows,
                     empty="No task derived a payload bandwidth in this run.")
    lines += ["## Cases without a communication payload", ""]
    lines += F.table(("Case", "Backend", "EP", "Status", "Reason"),
                     ([row.get("case_id"), row.get("backend_id"), _ep(row),
                       F.text(row.get("status")), _bandwidth_bucket(row)[1]]
                      for row in buckets["no_payload"]),
                     empty="Every measured case moves a communication payload.")
    return lines


def _bandwidth_bucket(row):
    """``(bucket, reason)`` for one record's bandwidth cell."""
    if row.get("status") != "passed":
        return "failed", "the task did not pass, so no latency exists to derive a bandwidth from"
    bandwidth = row.get("bandwidth")
    if not bandwidth:
        return "no_model", "no bandwidth model applies to this case"
    if bandwidth.get("payload_bytes") is None:
        if ((row.get("config") or {}).get("model") and
                row.get("benchmark_level") == "L0"):
            return "no_payload", F.REASON_NOT_COMMUNICATION
        return "no_model", "the bandwidth model produced no payload bytes for this case"
    if bandwidth.get("payload_bandwidth_GBps") is None:
        return "unavailable", "payload bytes exist but the latency denominator was zero or absent"
    return "measured", "payload bandwidth derived from the byte model and the measured latency"


def overlap_analysis(ctx):
    return _document("Overlap Analysis", overlap.body(ctx))


def scalability_analysis(ctx):
    lines = _document("EP Scalability Analysis", scalability.body(ctx))
    lines += ["## Per-operator EP comparison", "",
              "Grouped by `analysis.comparability.scaling_key`; only groups with at least two EP "
              "sizes are shown as comparable.", ""]
    comparison_data = ctx.coverage_data.get("ep_size_comparison") or {}
    groups = comparison_data.get("groups") or []
    comparable = [group for group in groups if group.get("comparable")]
    rows = []
    for group in comparable:
        points = {point["ep_size"]: point for point in group["points"]}
        sizes = sorted(points)
        rows.append([group["case_id"], F.text(group["operator"]),
                     F.text(group.get("variant")), F.text(group.get("timing")),
                     F.join([f"EP{size}" for size in sizes], separator=", "),
                     F.join([F.number(points[size]["p50"]) + " ms" for size in sizes],
                            separator=" -> ")])
    lines += F.table(("Case", "Operator", "Variant", "Timing", "EP sizes", "P50 sequence"), rows,
                     empty="No per-operator group is comparable across EP sizes.")
    excluded = [group for group in groups if not group.get("comparable")]
    if excluded:
        lines += ["### Not comparable", ""]
        lines += F.table(("Case", "Operator", "Reason"),
                         ([group["case_id"], F.text(group["operator"]), F.text(group["reason"])]
                          for group in excluded))
    return lines


def load_balance_analysis(ctx):
    return _document("Load Balance Analysis", load_balance.distribution_body(ctx))


def planner_training(ctx):
    lines = _document("Planner and Training Stages", [])
    rows = [row for row in ctx.rows if (row.get("load_balance") or {}).get(
        "planned_expert_copies") is not None]
    if not rows and not (ctx.analysis.get("planner_comparisons")
                         or ctx.analysis.get("fixed_owner_comparisons")):
        lines += empty.section(
            "Why No Planner Result Is Reported",
            level=2,
            findings="The Planner Suite was not executed in this run: no task carries a "
                     "planner stage latency or a planned replica count.",
            causes=["No `LAYER-003` / `LAYER-004` / `LAYER-008` (planner) case produced a "
                    "passing result.",
                    "A planner suite needs `load_balance.enabled: true` and a planner-capable "
                    "adapter (`moonep`, `moonep_dynamic_planner`)."],
            required=["Run a planner suite, for example `configs/load_balance.yaml` with `--cases "
                      "LAYER-003`.",
                      "Run both `fresh_plan` and `cached_plan` variants to get the cached-plan "
                      "comparison."],
            alternatives=["Section 8.1 of the main report shows the routing-trace load balance "
                          "for the tasks that did run."],
            conclusion="No planner latency, weight prefetch, gradient reduction or redundant "
                       "weight saving is claimed for this run.")
        return lines
    lines += load_balance.planner_body(ctx)
    return lines


def dedup_comparison(ctx):
    lines = _document("Dedup Comparison", [])
    if not (ctx.analysis.get("dedup_comparisons") or []):
        lines += empty.section(
            "Why No Dedup Result Is Reported",
            level=2,
            findings="No rank/server dedup comparison exists in this run.",
            causes=["No dedup case (`COMM-010`, `LAYER-005`, `LAYER-006`) produced a passing "
                    "result with a dedup mode other than `none`.",
                    "A comparison needs two modes of the same case at the same EP size."],
            required=["Configure `dedup.mode: rank` and `dedup.mode: server` for the same dedup "
                      "case, or use `configs/dedup.yaml`.",
                      "Keep the routing seed and expert placement identical so the trace matches."],
            alternatives=["Section 8.1 of the main report reports the rank-dedup opportunity "
                          "derived from the routing trace."],
            conclusion="No dedup payload saving or dedup speedup is claimed for this run.")
        return lines
    lines += load_balance.dedup_body(ctx)
    return lines


def failure_summary(ctx):
    """The complete failure list: this file is the authority on what did not run."""
    return _document("Failure Summary", failures.body(ctx, limit=None))


def _ep(row):
    size = (row.get("config") or {}).get("distributed", {}).get("ep_size") or row.get("ep_size")
    return f"EP{size}" if size else F.MISSING


#: File name -> renderer, in the order the main report links them.
REPORTS = (
    ("operator_comparison.md", operator_comparison),
    ("bandwidth_analysis.md", bandwidth_analysis),
    ("overlap_analysis.md", overlap_analysis),
    ("scalability_analysis.md", scalability_analysis),
    ("load_balance_analysis.md", load_balance_analysis),
    ("planner_training.md", planner_training),
    ("dedup_comparison.md", dedup_comparison),
    ("failure_summary.md", failure_summary),
)
