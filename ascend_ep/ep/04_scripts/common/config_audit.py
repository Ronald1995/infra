"""Config field -> task parameter -> runtime consumer traceability.

The rule this module enforces: **a field a configuration declares must reach a
task parameter and a runtime consumer.**  A field that is declared but never
expanded (the old ``cores``/``quiet_modes`` bug, which silently ran the last
configured value) or expanded but never read by the runtime (a parameter the
backend ignores) is a broken promise, not a harmless extra key.

The audit is deliberately small and explicit: the fields below are the ones whose
"declared vs executed" gap has actually bitten this matrix.  Each entry names the
task parameter it must become and the runtime file/token that must consume it.
"""
from collections import Counter
from pathlib import Path

from common.registry import ROOT

#: ``config field -> (task parameter, consumer path, consumer token)``.
FIELDS = {
    "benchmark.message_bytes": ("message_bytes", "baselines/l0/overlap/runner.py",
                                "message_bytes"),
    "benchmark.outstanding": ("outstanding", "baselines/l0/transport/channel.py",
                              "outstanding"),
    "benchmark.peer_counts": ("peer_count", "baselines/l0/transport/peers.py", "peer_count"),
    "benchmark.cores": ("cores", "baselines/l0/shmem/udma.py", "cores"),
    "benchmark.quiet_modes": ("quiet_mode", "baselines/l0/shmem/udma.py", "quiet_mode"),
    "benchmark.execution.modes": ("accelerator", "baselines/l0/collectives/accelerator.py",
                                  "accelerator"),
    "benchmark.transport.enabled": ("mechanism", "baselines/l0/contract.py", "mechanism"),
}
#: Dimensions that a transport case may declare ``not_applicable``.  A field whose
#: dimension is not applicable anywhere is reported as such instead of as an error:
#: the configuration asked for a sweep that this hardware/API cannot express.
DIMENSION_FIELDS = {
    "benchmark.cores": "cores",
    "benchmark.quiet_modes": "quiet_mode",
}
#: Case families that consume each field.  A field whose family is not selected by
#: this plan is ``not_selected``, not "unconsumed": omitting the transport suite
#: cannot be an error about the core sweep.
FAMILIES = {
    "benchmark.message_bytes": {"transport", "overlap", "execution", "memory", "collective"},
    "benchmark.outstanding": {"transport"},
    "benchmark.peer_counts": {"transport"},
    "benchmark.cores": {"transport"},
    "benchmark.quiet_modes": {"transport"},
    "benchmark.execution.modes": {"execution"},
    "benchmark.transport.enabled": {"transport"},
}


def _dotted(config, field):
    node = config
    for part in field.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _configured(configs, field):
    values = []
    for config in configs:
        value = _dotted(config, field)
        if isinstance(value, list):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return list(dict.fromkeys(values))


def _consumer(path, token, root):
    file = Path(root) / path
    if not file.is_file():
        return False, f"consumer file {path} does not exist"
    if token not in file.read_text(encoding="utf-8", errors="replace"):
        return False, f"{path} never mentions {token!r}"
    return True, None


def audit(tasks, configs, dimensions=None, root=ROOT):
    """Trace every audited field from configuration to a runtime consumer.

    ``tasks`` is the expanded manifest, ``configs`` the effective configuration of
    every plan point, and ``dimensions`` the per-point dimension declarations
    (see :func:`common.variants.case_dimension_records`).  Returns a report with
    one record per field and an ``unconsumed`` list; an empty list is the only
    acceptable result for a full plan.
    """
    parameters = Counter()
    values = {}
    for task in tasks:
        for name, value in (task.get("parameters") or {}).items():
            parameters[name] += 1
            values.setdefault(name, set()).add(_hashable(value))
    applicability = {}
    for point in (dimensions or {}).values():
        for per_case in point.values():
            for variant, record in per_case.items():
                for name, entry in record.items():
                    applicability.setdefault(name, set()).add(entry.get("type"))
    rows, unconsumed = [], []
    for field, (parameter, path, token) in FIELDS.items():
        configured = _configured(configs, field)
        if not configured:
            rows.append({"field": field, "parameter": parameter, "status": "not_configured",
                         "configured": [], "observed": []})
            continue
        observed = sorted(values.get(parameter, ()), key=str)
        proven, problem = _consumer(path, token, root)
        dimension = DIMENSION_FIELDS.get(field)
        declared = applicability.get(dimension, set()) if dimension else set()
        families = FAMILIES.get(field, set())
        selected = any(task.get("case_family") in families for task in tasks)
        if not proven:
            status, reason = "consumer_missing", problem
        elif not selected:
            status, reason = "not_selected", (
                "this plan selects no case family that consumes the field")
        elif not observed and declared == {"not_applicable"}:
            status, reason = "declared_not_applicable", (
                "every case that could sweep this dimension declares it not_applicable")
        elif not observed and dimension and not declared:
            status, reason = "not_selected", "no case in this plan sweeps the dimension"
        elif not observed:
            status, reason = "unconsumed", "no task carries this parameter"
        elif len(observed) < len(configured):
            status, reason = "partial", (
                f"{len(observed)} of {len(configured)} configured values reached a task")
        else:
            status, reason = "swept", None
        row = {"field": field, "parameter": parameter, "status": status,
               "configured": configured, "observed": observed,
               "tasks": parameters.get(parameter, 0),
               "consumer": {"path": path, "token": token, "proven": proven},
               "dimension_status": sorted(declared), "reason": reason}
        rows.append(row)
        if status in ("consumer_missing", "unconsumed", "partial"):
            unconsumed.append(row)
    return {"schema_version": 1, "fields": rows, "unconsumed": unconsumed,
            "ok": not unconsumed}


def _hashable(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value
