"""Section 3: L0 Hardware Baseline.

L0 answers hardware questions, so the section keeps requested and observed apart, shows
the verification status of every mechanism, and refuses to rank mechanisms that were
never verified.  Copy and compute cases legitimately have no payload GB/s: that cell
says ``N/A (not a communication payload)`` instead of staying blank.
"""
from baselines.l0.verification.vocabulary import MECHANISMS, mechanism_label
from reporting import formatting as F
from reporting.sections import common

#: At most this many scale-out rows are shown in the main report.
MAX_SCALEOUT_ROWS = 30


def render(ctx):
    l0 = ctx.l0
    lines = ["## 3. L0 Hardware Baseline", ""]
    if not l0["summary"]["records"]:
        lines += _no_l0(ctx)
        return lines
    summary = l0["summary"]
    lines += [f"{summary['records']} L0 result(s) over "
              f"{len(summary['ep_sizes'])} EP size(s); cases: {F.join(summary['cases'])}.", ""]
    lines += ["### 3.1 Communication baseline", ""]
    lines += _baseline_table(l0["communication"], "communication baseline")
    lines += ["### 3.2 Communication mechanism", ""]
    lines += _mechanism_table(l0["mechanisms"])
    lines += ["### 3.3 Compute and copy baseline", ""]
    lines += _compute_copy(ctx, l0)
    lines += ["### 3.4 Verification", ""]
    lines += _verification_table(l0["verification"])
    lines += ["### 3.5 L0 EP scale-out", ""]
    lines += _scale_out(ctx, l0["scale_out"])
    lines += ["The complete communication curves (every message size, outstanding count and peer "
              "count) are in `l0_report.md` and `l0_scaleout.md`.", ""]
    return lines


def _no_l0(ctx):
    from reporting import empty
    return empty.section(
        "Why L0 Is Not Available",
        level=3,
        findings="This run contains no L0 result record.",
        causes=["No L0 task was planned for this run, or every planned L0 task has no result "
                "record (see section 2 for `Missing`).",
                "L0 is a shared hardware baseline: it needs `benchmark_plan` levels to include "
                "`L0` and a device that exposes a communication mechanism."],
        required=["Plan at least one L0 case (LINK-*, EXEC-*, MEM-*, HW-010/HW-011, OVERLAP-*).",
                  "Run the plan on an Ascend device with the requested mechanism available.",
                  "Collect verification evidence (native descriptor, runtime API or HCCL log) so "
                  "the result can be attributed to a mechanism."],
        alternatives=["The L1-L4 operator performance is analysed in section 4.",
                      "The capability matrix in section 1.3 still lists which mechanisms the "
                      "probe considers supported."],
        conclusion="No hardware capability conclusion is drawn, because this run measured no L0 "
                   "baseline.")


def _baseline_table(entries, what):
    if not entries:
        return [f"_No {what} was measured in this run._", ""]
    rows = []
    for entry in entries:
        rows.append([entry["baseline"], entry["case_id"],
                     F.join([f"EP{size}" for size in entry["ep_sizes"]], separator=", "),
                     F.count(entry["records"]), F.count(entry["passed"]),
                     F.count(entry["verified"]),
                     _range(entry["min_p50_ms"], entry["max_p50_ms"]),
                     common.l0_bandwidth_cell(entry),
                     F.text(F.status_counts(entry["statuses"]))])
    return F.table(("Baseline", "Case", "EP sizes", "Records", "Passed", "Verified",
                    "P50 range ms", "Max payload GB/s", "Statuses"), rows)


def _mechanism_table(entries):
    mechanisms = [entry for entry in entries if entry["mechanism"] in MECHANISMS]
    if not mechanisms:
        return ["_No L0 result names a communication mechanism, so no mechanism can be "
                "characterised._", ""]
    rows = []
    for entry in sorted(mechanisms, key=lambda item: (item["mechanism"],
                                                      str(item["execution_resource"]))):
        rows.append([entry["mechanism_label"], entry["execution_resource_label"],
                     F.join(entry["cases"]),
                     F.join([f"EP{size}" for size in entry["ep_sizes"]], separator=", "),
                     F.count(entry["records"]), F.count(entry["passed"]),
                     F.count(entry["verified"]), F.count(entry["fallback"]),
                     F.number(entry["median_GBps"]) if entry["median_GBps"] is not None
                     else F.na(common.NOT_COMMUNICATION),
                     F.number(entry["max_GBps"]) if entry["max_GBps"] is not None
                     else F.na(common.NOT_COMMUNICATION),
                     F.text(entry["mechanism_source"])])
    lines = F.table(("Mechanism", "Execution resource", "Cases", "EP sizes", "Records",
                     "Passed", "Verified", "Fallback", "Median GB/s", "Max GB/s",
                     "Mechanism source"), rows)
    lines += ["`Mechanism source` says whether the mechanism was **observed** at runtime or only "
              "**requested** by the plan. A requested-only row must not be read as a measurement "
              "of that mechanism.", ""]
    return lines


def _compute_copy(ctx, l0):
    groups = (("HBM copy", "memory", l0["memory"]), ("Compute", "compute", l0["compute"]),
              ("Communication + compute", "overlap", l0["overlap"]))
    rows = []
    for label, family, entries in groups:
        for entry in entries:
            rows.append([label, entry["baseline"], entry["case_id"],
                         F.count(entry["records"]), F.count(entry["passed"]),
                         F.count(entry["verified"]),
                         _range(entry["min_p50_ms"], entry["max_p50_ms"]),
                         common.l0_tflops_cell(entry, family),
                         _payload_cell(family, entry), _local_rate_cell(family, entry)])
    lines = F.table(("Baseline", "Operation", "Case", "Records", "Passed", "Verified",
                     "P50 range ms", "Effective TFLOPS", "Payload GB/s", "Local copy GB/s"),
                    rows,
                    empty="No copy, compute or concurrent L0 case produced a result in this run.")
    if any(entry.get("passed") for _label, _family, entries in groups for entry in entries):
        lines += ["Copy and compute cases move no communication payload, so their payload "
                  "bandwidth is `N/A (not a communication payload)` by construction -- not a "
                  "missing value. A copy rate, when measured, is shown in its own column because "
                  "it is a device-local quantity, not a link bandwidth.", ""]
    return lines


def _payload_cell(family, entry):
    if family in {"memory", "compute"}:
        return F.MISSING if not entry.get("passed") else F.na(common.NOT_COMMUNICATION)
    return common.l0_bandwidth_cell(entry, family)


def _local_rate_cell(family, entry):
    if family != "memory" or entry["max_GBps"] is None:
        return F.MISSING
    return F.number(entry["max_GBps"])


def _verification_table(entries):
    if not entries:
        return ["_No L0 verification record exists in this run._", ""]
    rows = []
    for entry in entries:
        resource = entry["requested_resource"]
        observed = entry["observed_resource"]
        if observed and observed != resource:
            resource = f"{F.text(resource)} -> {observed}"
        status = entry["verification"]
        marker = "" if entry["verification_source"] == "recorded" else " (from status)"
        rows.append([entry["case_id"], F.text(entry["requested_mechanism"]),
                     F.text(entry["observed_mechanism"]), F.text(resource),
                     f"{status}{marker}",
                     F.join(entry["evidence_sources"]) if entry["evidence_sources"]
                     else F.na("no evidence item was attached"),
                     F.text(entry["completion"]), F.boolean(entry["fallback"]),
                     F.count(entry["records"]),
                     F.join([f"EP{size}" for size in entry["ep_sizes"]], separator=", ")])
    lines = F.table(("Case", "Requested mechanism", "Observed mechanism", "Execution resource",
                     "Verification", "Evidence source", "Completion", "Fallback", "Records",
                     "EP sizes"), rows)
    lines += ["`(from status)` marks a row with no verification block: the label then comes from "
              "the task status, never from a claim. Requested values come from the plan; only the "
              "observed column is evidence.", ""]
    return lines


def _range(low, high):
    if low is None and high is None:
        return F.MISSING
    if low == high:
        return F.number(low)
    return f"{F.number(low)} .. {F.number(high)}"


def _scale_out(ctx, entries):
    if not entries:
        return ["_No L0 scale-out group exists (no L0 result carries the parameters that define "
                "one). _", ""]
    sizes = sorted({point["ep_size"] for entry in entries for point in entry["points"]
                    if point["ep_size"] is not None})
    rows = []
    for entry in entries[:MAX_SCALEOUT_ROWS]:
        points = {point["ep_size"]: point for point in entry["points"]}
        ambiguous = {item["ep_size"] for item in entry["ambiguous"]}
        cells = []
        for size in sizes:
            if size in ambiguous:
                cells.append("ambiguous")
            else:
                cells.append(_scaleout_cell(points.get(size)))
        rows.append([entry["case_id"], F.text(entry.get("variant")),
                     mechanism_label(entry["mechanism"]) if entry["mechanism"] in MECHANISMS
                     else F.text(entry["mechanism"]),
                     F.text(entry.get("execution_resource")),
                     F.count(entry.get("message_bytes")), F.count(entry.get("outstanding")),
                     F.count(entry.get("peer_count")), *cells])
    lines = F.table(("Case", "Variant", "Mechanism", "Resource", "Message bytes", "Outstanding",
                     "Peers", *(f"EP{size}" for size in sizes)), rows)
    comparable = [entry for entry in entries if entry["comparable"]]
    lines += [f"{len(comparable)} of {len(entries)} scale-out variant(s) are comparable across EP "
              f"sizes (identical parameters and a `verified` path at two or more EP sizes). "
              f"`-` means the variant was not planned at that EP size.", ""]
    if len(entries) > MAX_SCALEOUT_ROWS:
        lines += [f"_{len(entries) - MAX_SCALEOUT_ROWS} further variant(s) are listed in "
                  f"`l0_scaleout.md`._", ""]
    return lines


def _scaleout_cell(point):
    if point is None:
        return F.MISSING
    if point["status"] != "passed":
        reason = point.get("reason")
        return f"{point['status']}" + (f" ({reason})" if reason else "")
    rate = point.get("bandwidth_GBps")
    return (f"{F.number(rate)} GB/s ({point['verification']})" if rate is not None
            else f"passed ({point['verification']})")
