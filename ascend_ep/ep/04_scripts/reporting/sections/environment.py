"""Section 1: Environment and Hardware Capability.

The environment table is the provenance of every number in the report.  A value that
could not be read says ``unavailable`` **and** why: the CANN version, for example,
is resolved from ``ASCEND_TOOLKIT_VERSION`` and then from the toolkit's own version
files, and the source (or the list of locations that were tried) is recorded next to
the result.
"""
from reporting import formatting as F


def render(ctx):
    lines = ["## 1. Environment and Hardware Capability", "",
             "### 1.1 Run environment", ""]
    lines += _environment_table(ctx)
    lines += _environment_notes(ctx)
    lines += ["### 1.2 Topology", ""]
    lines += _topology(ctx)
    lines += ["### 1.3 Capability matrix", ""]
    lines += _capability_matrix(ctx)
    return lines


def _value(value, source=None, missing_reason=None):
    if value in (None, "", []):
        reason = missing_reason or "not recorded by this run"
        return f"unavailable ({reason})"
    text = str(value)
    if source:
        return f"{text} (source: {source})"
    return text


def _environment_table(ctx):
    environment = ctx.environment
    driver_source = environment.get("driver_version_source")
    cann_source = environment.get("cann_version_source")
    if environment.get("cann_version"):
        cann_cell = _value(environment.get("cann_version"), source=cann_source)
    else:
        cann_cell = ("unavailable ("
                     f"{environment.get('cann_version_error') or 'ASCEND_TOOLKIT_VERSION is unset'}"
                     ")")
    driver_cell = _value(environment.get("driver_version"),
                         source=driver_source if environment.get("driver_version") else None,
                         missing_reason="npu-smi was not available or printed no driver version")
    rows = [
        ("Host", _value(environment.get("host"))),
        ("Device", ctx.device_wording),
        ("NPU model", _value(F.join(environment.get("npu_models")) if environment.get("npu_models")
                             else None, missing_reason="the run did not touch an NPU")),
        ("Driver", driver_cell),
        ("CANN", cann_cell),
        ("PyTorch", _value(environment.get("torch"))),
        ("torch_npu", _value(environment.get("torch_npu"))),
        ("Python", _value(environment.get("python"))),
        ("OS", _value(environment.get("os"))),
        ("Git", _value(environment.get("git_commit"))),
        ("EP size", F.join([f"EP{size}" for size in ctx.ep_sizes()], separator=", ")),
        ("TP size", F.join([f"TP{size}" for size in environment.get("tp_sizes") or ()],
                           separator=", ")),
        ("Topology", _topology_summary(environment)),
        ("Rank mapping", _rank_mapping_summary(environment)),
    ]
    return F.table(("Information", "Value"), rows)


def _environment_notes(ctx):
    environment = ctx.environment
    notes = []
    changes = environment.get("git_changes")
    if changes:
        collapsed = "; ".join(line.strip() for line in str(changes).splitlines() if line.strip())
        notes.append(f"The benchmark repository had uncommitted changes when the run started, so "
                     f"the commit hash alone does not identify the measured code: `{collapsed}`.")
    if environment.get("cann_version_error") and not environment.get("cann_version"):
        notes.append("CANN version resolution: " + environment["cann_version_error"] + ".")
    if environment.get("device_visibility"):
        notes.append(f"`ASCEND_RT_VISIBLE_DEVICES` = `{environment['device_visibility']}`.")
    if environment.get("capability_error"):
        notes.append(f"Capability discovery failed: {environment['capability_error']}.")
    if not notes:
        return []
    return [*notes, ""]


def _topology_summary(environment):
    topology = environment.get("topology") or {}
    if not topology:
        return "unavailable (the plan declared no topology)"
    parts = []
    for size, record in sorted(topology.items(), key=lambda item: str(item[0])):
        if "variants" in record:
            parts.append(f"EP{size}: conflicting maps ({len(record['variants'])} variants)")
            continue
        parts.append(
            f"EP{size}: {record.get('servers', '')} server(s), "
            f"{record.get('supernodes', '')} supernode(s), source `{record.get('source')}`, "
            f"verified={record.get('verified')}")
    return "; ".join(parts)


def _rank_mapping_summary(environment):
    mapping = environment.get("rank_mapping")
    if not mapping:
        return ("unavailable (no task recorded a rank-to-device mapping; the mapping is "
                "collected from a completed run)")
    hosts = sorted({str(item.get("host")) for item in mapping if item.get("host")})
    devices = sorted({str(item.get("physical_device")) for item in mapping
                      if item.get("physical_device")})
    return (f"{len(mapping)} rank(s) on {len(hosts)} host(s) "
            f"({F.join(hosts)}); visible devices: {F.join(devices)}")


def _topology(ctx):
    mapping = ctx.environment.get("rank_mapping")
    lines = []
    if mapping:
        rows = [(item.get("rank"), item.get("node_rank"), item.get("host"),
                 item.get("device"), item.get("physical_device"), item.get("server"),
                 item.get("supernode")) for item in
                sorted(mapping, key=lambda item: (item.get("rank") is None, item.get("rank")))]
        shown = rows[:64]
        lines += F.table(("Rank", "Node", "Host", "Device", "Physical device", "Server",
                          "Supernode"), shown)
        if len(rows) > len(shown):
            lines += [f"_{len(rows) - len(shown)} further rank(s) are recorded in "
                      f"`topology.json` and the per-task `metrics.json`._", ""]
    else:
        lines += ["_No rank mapping was recorded. A derived topology (`source: plan`) is a "
                  "synthetic layout and is never presented as a discovered physical one._", ""]
    return lines


def _capability_matrix(ctx):
    capabilities = ctx.capabilities or {}
    if not capabilities or capabilities.get("source") == "unavailable":
        reason = (capabilities or {}).get("error") or "capability discovery did not run"
        lines = ["_No capability matrix is available for this run._", "",
                 f"- Source: `{(capabilities or {}).get('source') or 'unavailable'}`",
                 f"- Reason: {reason}", ""]
        return lines
    rows = []
    for entry in capabilities.get("channels") or ():
        rows.append(("channel", entry.get("channel") or entry.get("name"),
                     F.join(entry.get("mechanisms"), separator=", "),
                     F.text(entry.get("execution_resource")),
                     F.boolean(entry.get("supported")),
                     F.text(entry.get("confidence")), F.text(entry.get("reason"))))
    for entry in capabilities.get("engines") or ():
        rows.append(("engine", entry.get("engine") or entry.get("name"),
                     F.join(entry.get("mechanisms"), separator=", "),
                     F.text(entry.get("execution_resource")),
                     F.boolean(entry.get("supported")),
                     F.text(entry.get("confidence")), F.text(entry.get("reason"))))
    for entry in capabilities.get("accelerators") or ():
        rows.append(("accelerator", entry.get("mode"),
                     F.MISSING, F.text(entry.get("execution_resource")),
                     F.boolean(entry.get("supported")), "probed",
                     F.text(entry.get("reason"))))
    lines = [f"Source: `{capabilities.get('source')}`"
             f" · probes: {F.count(len(capabilities.get('probes') or ()))}", ""]
    lines += F.table(("Kind", "Communication channel / engine", "Transport mechanism",
                      "Execution resource", "Supported", "Confidence", "Reason"), rows,
                     empty="The capability matrix was produced but lists no channel or engine.")
    lines += ["`Supported` is what the discovery probe answered; a supported mechanism still "
              "needs a `verified` L0 result before its measured number may be attributed to it "
              "(section 3.4).", ""]
    return lines
