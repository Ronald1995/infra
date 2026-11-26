"""Independent L0 hardware report.

The report is deliberately separate from the operator report: L0 answers hardware
questions (which mechanism, which execution resource, which topology) and must
never mix verified measurements with fallback or unverified ones.
"""
import csv

from baselines.l0.verification.vocabulary import mechanism_label, resource_label
from common.storage import write_json


def _rows(rows):
    return [row for row in rows if row.get("benchmark_level") == "L0"]


def _verification(row):
    return ((row.get("verification") or {}).get("status")
            or (row.get("observed") or {}).get("verification_status") or "unverified")


def _device(manifest, rows):
    environment = manifest.get("environment") or {}
    npu = (rows[0].get("environment") or {}) if rows else {}
    return {
        "host": environment.get("host"),
        "platform": environment.get("platform"),
        "cann_version": npu.get("cann_version") or environment.get("cann_version"),
        "packages": environment.get("packages"),
        "npu_models": npu.get("npu_models"),
        "git_commit": environment.get("git_commit"),
    }


def curves(rows):
    """One row per (case, mechanism, execution resource, size, outstanding)."""
    output = []
    for row in rows:
        parameters = row.get("parameters") or {}
        metric = (row.get("bandwidth") or {})
        latency = row.get("latency") or {}
        payload = metric.get("payload_bytes") or metric.get("per_rank_bytes")
        output.append({
            "case_id": row["case_id"],
            "family": row.get("category"),
            "mechanism": (row.get("observed") or {}).get("transport_protocol"),
            "execution_resource": (row.get("observed") or {}).get("execution_resource"),
            "message_bytes": parameters.get("message_bytes"),
            "outstanding": parameters.get("outstanding"),
            "peer_count": parameters.get("peer_count"),
            "accelerator": parameters.get("accelerator"),
            "status": row["status"],
            "verification": _verification(row),
            "payload_bytes": payload,
            "latency_ms": latency.get("p50"),
            "bandwidth_GBps": metric.get("payload_bandwidth_GBps"),
            "rank_max_GBps": None,
        })
    return output


def write(root, manifest, rows, capabilities=None):
    destination = root / "reports"
    destination.mkdir(exist_ok=True)
    l0 = _rows(rows)
    measured = curves(l0)
    write_json(destination / "l0_metrics.json", {
        "schema_version": 1, "run_id": manifest["run_id"],
        "device": _device(manifest, l0), "capabilities": capabilities, "measurements": measured,
    })
    with (destination / "l0_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured[0]) if measured else [
            "case_id", "family", "mechanism", "execution_resource", "message_bytes",
            "outstanding", "peer_count", "accelerator", "status", "verification",
            "payload_bytes", "latency_ms", "bandwidth_GBps", "rank_max_GBps"])
        writer.writeheader()
        writer.writerows(measured)

    lines = [f"# L0 hardware baseline {manifest['run_id']}", ""]
    device = _device(manifest, l0)
    lines += ["## 1. Device and software", "",
              f"- host: {device['host']}",
              f"- platform: {device['platform']}",
              f"- CANN: {device['cann_version']}",
              f"- NPU models: {device['npu_models']}",
              f"- packages: {device['packages']}", ""]
    matrix = capabilities or {}
    if matrix:
        lines += ["## 3. Capability matrix", "",
                  f"Source: `{matrix.get('source')}`", "",
                  "| channel / engine | resource | mechanism | supported | confidence | reason |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for entry in matrix.get("channels", ()) + matrix.get("engines", ()):
            name = entry.get("channel") or entry.get("engine")
            lines.append("| {name} | {resource} | {mechanism} | {supported} | {confidence} | {reason} |".format(
                name=name, resource=entry.get("execution_resource"),
                mechanism=", ".join(entry.get("mechanisms", ())),
                supported=entry.get("supported"), confidence=entry.get("confidence"),
                reason=entry.get("reason") or ""))
        for entry in matrix.get("accelerators", ()):
            lines.append("| accelerator {mode} | {resource} | - | {supported} | probed | {reason} |".format(
                mode=entry.get("mode"), resource=entry.get("execution_resource"),
                supported=entry.get("supported"), reason=entry.get("reason") or ""))
        lines.append("")
    lines += ["## 4. Transport curves", "",
              "| case | mechanism | resource | bytes | outstanding | status | verification | GB/s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for point in measured:
        if point["family"] != "transport":
            continue
        lines.append("| {case_id} | {mechanism} | {resource} | {size} | {outstanding} | {status} | "
                     "{verification} | {rate} |".format(
                         case_id=point["case_id"],
                         mechanism=mechanism_label(point["mechanism"]),
                         resource=resource_label(point["execution_resource"]),
                         size=point["message_bytes"], outstanding=point["outstanding"],
                         status=point["status"], verification=point["verification"],
                         rate=point["bandwidth_GBps"]))
    lines += ["", "## 5. Accelerator sweep (native HCCL)", "",
              "| case | accelerator | bytes | status | verification | average ms | GB/s |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for point in measured:
        if point["family"] != "execution":
            continue
        lines.append(f"| {point['case_id']} | {point['accelerator']} | {point['message_bytes']} | "
                     f"{point['status']} | {point['verification']} | {point['latency_ms']} | "
                     f"{point['bandwidth_GBps']} |")
    counts = {}
    for row in l0:
        counts[_verification(row)] = counts.get(_verification(row), 0) + 1
    lines += ["", "## 12. Verification, fallback and unsupported counts", "",
              ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "no L0 records",
              "", "## 13. Notes", "",
              "- Only `verified` rows may be compared across mechanisms.",
              "- `unverified` means no evidence established the execution path; the number is "
              "still a valid measurement of *something*, but not attributable to a mechanism.",
              "- `fallback` means the observed path differed from the requested one.",
              "- Physical link bytes are reported only when a counter is available; logical "
              "payload bytes are not physical wire bytes.", ""]
    (destination / "l0_report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination
