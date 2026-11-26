"""Independent L0 hardware report.

The report is deliberately separate from the operator report: L0 answers hardware
questions (which mechanism, which execution resource, which topology) and must never mix
verified measurements with fallback or unverified ones.

It shares :mod:`analysis.l0` with the main report, so the two documents cannot disagree
about a verification status.  Failed tasks are part of the report: they keep their
category and their planned parameters, so a failed LINK case stays in the transport
table with its requested mechanism instead of disappearing.
"""
import csv

from analysis import comparability, l0 as l0_analysis
from baselines.l0.verification.vocabulary import mechanism_label, resource_label
from common.storage import write_json
from reporting import formatting as F
from reporting.sections import common

CURVE_FIELDS = ["case_id", "ep_size", "family", "mechanism", "execution_resource",
                "requested_mechanism", "observed_mechanism", "verification_source",
                "message_bytes", "outstanding", "peer_count", "peer_policy", "accelerator",
                "variant", "status", "verification", "completion", "fallback",
                "payload_bytes", "latency_ms", "bandwidth_GBps", "rank_max_GBps"]


def _rows(records):
    return l0_analysis.rows(records)


def _device(manifest, rows):
    environment = manifest.get("environment") or {}
    npu = next(((row.get("environment") or {}) for row in rows
                if (row.get("environment") or {}).get("npu_models")), {})
    return {
        "host": environment.get("host"),
        "platform": environment.get("platform"),
        "cann_version": npu.get("cann_version") or environment.get("cann_version"),
        "cann_version_source": environment.get("cann_version_source"),
        "cann_version_error": environment.get("cann_version_error"),
        "driver_version": environment.get("driver_version"),
        "packages": environment.get("packages"),
        "npu_models": npu.get("npu_models") or environment.get("npu_models"),
        "git_commit": environment.get("git_commit"),
    }


def curves(rows):
    """One row per L0 record, keeping the full measurement point.

    Reads the standard ``case_parameters`` field and falls back to the legacy
    ``parameters`` key for results written before the contract fix, so an old run still
    reports its message size and peer count.
    """
    output = []
    for row in rows:
        measurement = l0_analysis.measurement(row)
        output.append({
            "case_id": measurement["case_id"],
            "ep_size": measurement["ep_size"],
            "family": measurement["family"],
            "mechanism": measurement["observed"]["mechanism"]
            or measurement["requested"]["mechanism"],
            "execution_resource": measurement["observed"]["execution_resource"]
            or measurement["requested"]["execution_resource"],
            "requested_mechanism": measurement["requested"]["mechanism"],
            "observed_mechanism": measurement["observed"]["mechanism"],
            "message_bytes": measurement["parameters"].get("message_bytes"),
            "outstanding": measurement["parameters"].get("outstanding"),
            "peer_count": measurement["parameters"].get("peer_count"),
            "accelerator": measurement["parameters"].get("accelerator"),
            "status": measurement["status"],
            "verification": measurement["verification"],
            "verification_source": measurement["verification_source"],
            "payload_bytes": measurement["payload_bytes"],
            "latency_ms": measurement["latency_ms"],
            "bandwidth_GBps": measurement["bandwidth_GBps"],
            "completion": measurement["completion"]["level"],
            "fallback": measurement["fallback"]["used"],
            **{name: measurement["parameters"].get(name) for name in l0_analysis.POINT_PARAMETERS},
            "rank_max_GBps": None,
        })
    return output


def write(root, manifest, rows, capabilities=None, analysis=None, charts=None):
    """Write ``l0_report.md``, ``l0_metrics.json`` and ``l0_curves.csv``."""
    destination = root / "reports"
    destination.mkdir(exist_ok=True)
    l0 = _rows(rows)
    data = analysis or l0_analysis.analyse(rows)
    measured = curves(l0)
    write_json(destination / "l0_metrics.json", {
        "schema_version": 1, "run_id": manifest["run_id"],
        "device": _device(manifest, l0), "capabilities": capabilities,
        "summary": data["summary"], "measurements": measured,
    })
    with (destination / "l0_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measured)

    lines = [f"# L0 hardware baseline {manifest['run_id']}", ""]
    device = _device(manifest, l0)
    if not l0:
        from reporting import empty
        lines += empty.unavailable(
            "L0 hardware baseline",
            findings="This run contains no L0 result record.",
            causes=["No L0 task was planned, or no planned L0 task produced a result."],
            required=["Include `L0` in the plan levels and run it on a device that exposes the "
                      "requested mechanisms."],
            alternatives=["The operator performance is reported in `benchmark_report.md`."],
            conclusion="No hardware capability conclusion is drawn.")
        (destination / "l0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    lines += ["## 1. Device and software", "",
              f"- host: {F.text(device['host'])}",
              f"- platform: {F.text(device['platform'])}",
              f"- CANN: {F.text(device['cann_version'])}"
              + (f" (source: {device['cann_version_source']})"
                 if device["cann_version"] and device["cann_version_source"] else ""),
              f"- driver: {F.text(device['driver_version'])}",
              f"- NPU models: {F.join(device['npu_models'])}",
              f"- git commit: {F.text(device['git_commit'])}",
              f"- packages: {F.text(device['packages'])}", ""]
    if not device["cann_version"] and device["cann_version_error"]:
        lines += [f"CANN version unavailable: {device['cann_version_error']}.", ""]

    lines += ["## 2. Coverage and EP size", ""]
    lines += _ep_coverage(manifest, l0, data)
    lines += ["## 3. Capability matrix", ""]
    lines += _capability_lines(capabilities)
    lines += ["## 4. Communication baselines", ""]
    lines += _baseline_lines(data["communication"])
    lines += ["## 5. Transport curves", ""]
    lines += _transport_lines(measured)
    lines += ["## 6. Execution resource sweep", ""]
    lines += _execution_lines(data["execution"])
    lines += ["## 7. Compute and copy", ""]
    lines += _compute_lines(data)
    lines += ["## 8. Communication + compute concurrent", ""]
    lines += _overlap_lines(data["overlap"])
    lines += ["## 9. Verification evidence", ""]
    lines += _verification_lines(data["verification"])
    lines += ["## 10. Failures and unverified points", ""]
    lines += _failure_lines(l0)
    counts = data["summary"]["verification"]
    lines += ["", "## 11. Verification, fallback and unsupported counts", "",
              ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
              or "no L0 records",
              "", f"- fallback rows: {F.count(sum(1 for row in l0 if (row.get('observed') or {}).get('fallback')))}",
              f"- records with a recorded verification block: "
              f"{F.count(sum(1 for item in data['measurements'] if item['verification_source'] == 'recorded'))}",
              ""]
    lines += ["## 12. Notes", "",
              "- Only `verified` rows may be compared across mechanisms.",
              "- `unverified` means no evidence established the execution path; the number is "
              "still a valid measurement of *something*, but not attributable to a mechanism.",
              "- `fallback` means the observed path differed from the requested one.",
              "- Physical link bytes are reported only when a counter is available; logical "
              "payload bytes are not physical wire bytes.",
              "- A copy or compute case has no communication payload: its bandwidth cell is "
              "`N/A (not a communication payload)` rather than blank.",
              ""]
    (destination / "l0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _ep_coverage(manifest, rows, data):
    sizes = data["summary"]["ep_sizes"]
    topologies = (manifest.get("matrix") or {}).get("topology") or {}
    lines = ["L0 is a shared hardware baseline, not an operator case, so every EP size runs the "
             "same case catalog once. The topology source is recorded explicitly: a map derived "
             "from a `server_size` pattern is `verified=false` and must not be read as a "
             "discovered physical layout.", ""]
    table = []
    for size in sizes:
        subset = [row for row in rows
                  if comparability.ep_size(row) == size]
        points = [l0_analysis.measurement(row) for row in subset]
        topology = topologies.get(str(size)) or {}
        table.append([f"EP{size}", F.count(len(points)),
                      F.count(sum(1 for point in points if point["status"] == "passed")),
                      F.count(sum(1 for point in points if point["verification"] == "verified")),
                      F.text(topology.get("source", "unavailable")),
                      F.boolean(topology.get("verified", False)),
                      F.count(topology.get("servers")), F.count(topology.get("supernodes"))])
    lines += F.table(("EP", "Tasks", "Passed", "Verified", "Topology source",
                      "Topology verified", "Servers", "Supernodes"), table,
                     empty="No L0 result carries an EP size.")
    lines += [f"EP scale-out per mechanism and message size is in `l0_scaleout.md`; the analysis "
              f"covers {F.count(data['summary']['points'])} distinct parameter point(s).", ""]
    return lines


def _capability_lines(capabilities):
    matrix = capabilities or {}
    if not matrix:
        return ["_No capability matrix is available._", ""]
    lines = [f"Source: `{F.text(matrix.get('source'))}`", ""]
    rows = []
    for entry in (matrix.get("channels") or ()) + (matrix.get("engines") or ()):
        rows.append([entry.get("channel") or entry.get("engine"),
                     F.text(entry.get("execution_resource")),
                     F.join(entry.get("mechanisms"), separator=", "),
                     F.boolean(entry.get("supported")), F.text(entry.get("confidence")),
                     F.text(entry.get("reason"))])
    for entry in matrix.get("accelerators") or ():
        rows.append([f"accelerator {F.text(entry.get('mode'))}",
                     F.text(entry.get("execution_resource")), F.MISSING,
                     F.boolean(entry.get("supported")), "probed", F.text(entry.get("reason"))])
    lines += F.table(("Channel / engine", "Execution resource", "Mechanism", "Supported",
                      "Confidence", "Reason"), rows,
                     empty="The capability matrix lists no channel or engine.")
    return lines


def _baseline_lines(entries):
    rows = []
    for entry in entries:
        rows.append([entry["baseline"], entry["case_id"],
                     F.join([f"EP{size}" for size in entry["ep_sizes"]], separator=", "),
                     F.count(entry["records"]), F.count(entry["passed"]),
                     F.count(entry["verified"]),
                     F.number(entry["min_p50_ms"]), F.number(entry["max_p50_ms"]),
                     common.l0_bandwidth_cell(entry),
                     F.text(F.status_counts(entry["statuses"]))])
    return F.table(("Baseline", "Case", "EP sizes", "Records", "Passed", "Verified",
                    "Min P50 ms", "Max P50 ms", "Max payload GB/s", "Statuses"), rows,
                   empty="No communication baseline was measured in this run.")


def _transport_lines(measured):
    transport = [point for point in measured if point["family"] in {"transport", "collective"}]
    rows = []
    for point in transport:
        rows.append([point["case_id"],
                     mechanism_label(point["mechanism"]) if point["mechanism"] else F.MISSING,
                     resource_label(point["execution_resource"]),
                     F.count(point["message_bytes"]), F.count(point["outstanding"]),
                     F.count(point["peer_count"]), F.text(point["status"]),
                     F.text(point["verification"]),
                     F.number(point["latency_ms"]), F.number(point["bandwidth_GBps"]),
                     F.text(point["requested_mechanism"]),
                     F.text(point["observed_mechanism"])])
    lines = F.table(("Case", "Mechanism", "Resource", "Bytes", "Outstanding", "Peers", "Status",
                     "Verification", "P50 ms", "GB/s", "Requested mechanism",
                     "Observed mechanism"), rows,
                    empty="No transport or collective L0 record exists in this run.")
    lines += ["Every row is kept, including failed and unsupported ones: a `runtime_failed` "
              "LINK point is evidence about the mechanism, not a reason to drop the line.", ""]
    return lines


def _execution_lines(entries):
    rows = [[entry["baseline"], entry["case_id"],
             F.join([f"EP{size}" for size in entry["ep_sizes"]], separator=", "),
             F.count(entry["records"]), F.count(entry["passed"]), F.count(entry["verified"]),
             F.number(entry["min_p50_ms"]), F.number(entry["max_p50_ms"]),
             F.number(entry["max_GBps"])] for entry in entries]
    return F.table(("Operation", "Case", "EP sizes", "Records", "Passed", "Verified",
                    "Min P50 ms", "Max P50 ms", "Max payload GB/s"), rows,
                   empty="No native execution-resource sweep exists in this run.")


def _compute_lines(data):
    rows = []
    for label, family, entries in (("HBM copy", "memory", data["memory"]),
                                   ("Compute", "compute", data["compute"])):
        for entry in entries:
            rows.append([label, entry["baseline"], entry["case_id"],
                         F.count(entry["records"]), F.count(entry["passed"]),
                         F.count(entry["verified"]),
                         F.number(entry["min_p50_ms"]), F.number(entry["max_p50_ms"]),
                         common.l0_tflops_cell(entry, family),
                         F.MISSING if not entry.get("passed")
                         else F.na(common.NOT_COMMUNICATION),
                         F.number(entry["max_GBps"]) if family == "memory"
                         and entry["max_GBps"] is not None else F.MISSING])
    lines = F.table(("Family", "Operation", "Case", "Records", "Passed", "Verified",
                     "Min P50 ms", "Max P50 ms", "Effective TFLOPS", "Payload GB/s",
                     "Local copy GB/s"), rows,
                    empty="No copy or compute L0 case produced a result in this run.")
    if any(entry.get("passed") for family in (data["memory"], data["compute"])
           for entry in family):
        lines += ["A copy or compute case has no communication payload, so its payload bandwidth "
                  "is `N/A (not a communication payload)`; a measured copy rate is reported "
                  "separately because it is device-local.", ""]
    return lines


def _overlap_lines(entries):
    rows = [[entry["baseline"], entry["case_id"],
             F.join([f"EP{size}" for size in entry["ep_sizes"]], separator=", "),
             F.count(entry["records"]), F.count(entry["passed"]), F.count(entry["verified"]),
             F.number(entry["min_p50_ms"]), F.number(entry["max_p50_ms"]),
             common.l0_bandwidth_cell(entry, "overlap")] for entry in entries]
    return F.table(("Baseline", "Case", "EP sizes", "Records", "Passed", "Verified",
                    "Min P50 ms", "Max P50 ms", "Max payload GB/s"), rows,
                   empty="No communication + compute concurrent case produced a result.")


def _verification_lines(entries):
    rows = [[entry["case_id"], F.text(entry["requested_mechanism"]),
             F.text(entry["observed_mechanism"]), F.text(entry["requested_resource"]),
             F.text(entry["observed_resource"]), entry["verification"],
             F.text(entry["verification_source"]), F.text(entry["completion"]),
             F.boolean(entry["fallback"]), F.join(entry["evidence_sources"]),
             F.count(entry["records"]),
             F.join([f"EP{size}" for size in entry["ep_sizes"]], separator=", ")]
            for entry in entries]
    return F.table(("Case", "Requested mechanism", "Observed mechanism", "Requested resource",
                    "Observed resource", "Verification", "Verification source", "Completion",
                    "Fallback", "Evidence", "Records", "EP sizes"), rows,
                   empty="No L0 verification record exists in this run.")


def _failure_lines(rows):
    failures = [row for row in rows if row.get("status") != "passed"]
    if not failures:
        return ["_Every L0 task passed in this run._", ""]
    table = []
    for row in sorted(failures, key=lambda item: (str(item.get("case_id")),
                                                  str(item.get("status")))):
        measurement = l0_analysis.measurement(row)
        table.append([measurement["case_id"], F.text(measurement["family"]),
                      F.text(measurement["ep_size"]), F.text(measurement["status"]),
                      F.join([f"{key}={value}" for key, value
                              in sorted(measurement["parameters"].items())
                              if value is not None], separator=", "),
                      F.text(measurement["requested"]["mechanism"]),
                      F.text(measurement["verification"]),
                      F.text(measurement["reason"])])
    return F.table(("Case", "Family", "EP", "Status", "Parameters", "Requested mechanism",
                    "Verification", "Reason"), table)
