"""Helpers shared by the report sections.

All of them answer "what may this cell say": a measured value, the explicit reason a
metric does not apply, or the missing marker.  No helper ever turns an absent value
into a zero.
"""
from analysis import comparability
from reporting import formatting as F

#: Metric applicability reasons, worded once so every section agrees.
NOT_COMMUNICATION = F.REASON_NOT_COMMUNICATION
NO_BANDWIDTH_MODEL = F.REASON_NO_MODEL
NOT_COMPUTE = "no compute model for this case"
NO_COMPUTE_MODEL = "no compute model recorded"
UNMEASURED = F.REASON_UNMEASURED

LEVEL_TITLES = {
    "L0": "hardware baseline",
    "L1": "communication operators",
    "L2": "communication / compute fusion",
    "L3": "full MoE execution",
    "L4": "router, planner and training",
}


def median(values):
    """Median of a list of per-iteration values, or ``None``."""
    values = sorted(value for value in (values or ()) if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def parameter_text(record):
    """The case parameters as one readable cell."""
    parameters = comparability.semantic_parameters(record)
    if not parameters:
        return F.MISSING
    return F.join([f"{key}={value}" for key, value in sorted(parameters.items())
                   if value is not None], separator=", ")


def level_name(level):
    return f"{level} ({LEVEL_TITLES.get(level, 'unknown level')})"


def payload_bandwidth(record):
    """Payload GB/s, or the explicit reason it does not apply.

    A task that did not run has no measurement at all, so it shows the missing marker
    rather than a reason: ``N/A`` is reserved for a metric that exists but does not
    apply to the case.
    """
    if record.get("status") != "passed":
        return F.MISSING
    bandwidth = record.get("bandwidth") or {}
    value = bandwidth.get("payload_bandwidth_GBps")
    if value is not None:
        return F.number(value)
    if not bandwidth:
        return F.na(NO_BANDWIDTH_MODEL)
    return F.na(NOT_COMMUNICATION)


def payload_scope(record):
    bandwidth = record.get("bandwidth") or {}
    if not bandwidth:
        return F.MISSING
    scope = bandwidth.get("scope") or F.MISSING
    source = bandwidth.get("source") or F.MISSING
    unit = bandwidth.get("bandwidth_unit") or bandwidth.get("unit") or F.MISSING
    return f"{scope} · {source} · {unit}"


def effective_tflops(record):
    if record.get("status") != "passed":
        return F.MISSING
    compute = record.get("compute") or {}
    value = compute.get("effective_tflops")
    if value is not None:
        return F.number(value)
    return F.na(NOT_COMPUTE) if not compute else F.MISSING


def iterations(record):
    latency = record.get("latency") or {}
    samples = latency.get("rank_max")
    if isinstance(samples, list) and samples:
        return F.count(len(samples))
    return F.MISSING


def verification_cell(record):
    status = comparability.verification_status(record)
    return F.text(status) if status else F.MISSING


def correctness_cell(record):
    correctness = record.get("correctness")
    if not correctness:
        return F.na(UNMEASURED)
    if correctness.get("passed"):
        return "passed"
    failures = [res for item in correctness.get("ranks") or ()
                for res in _failures(item)]
    detail = f" ({'; '.join(failures[:2])})" if failures else ""
    return "failed" + detail


def _failures(item):
    """The first failing field names of one rank's correctness record."""
    names = []
    for key, value in sorted((item or {}).items()):
        if isinstance(value, dict):
            if value.get("passed") is False:
                names.append(key)
        elif value is False:
            names.append(key)
    return names


def timing_mode(record):
    return F.text(((record.get("config") or {}).get("benchmark") or {}).get("timing"))


def workload_text(record):
    workload = comparability.workload_dimensions(record)
    tokens = workload.get("tokens_per_rank")
    shape = F.MISSING if tokens is None else F.join(
        [str(value) for value in tokens], separator=" x ")
    return (f"tokens/rank {shape} · routing {F.text(workload.get('routing'))}"
            f" · seed {F.text(workload.get('seed'))}")


def log_link(record, root):
    """A relative link to the task's log, or the missing marker."""
    logs = record.get("logs") or ()
    if not logs:
        attempt = record.get("attempt")
        return F.text(attempt) if attempt else F.MISSING
    return F.join(logs, separator=", ")


def path_link(label, path, destination):
    """A relative Markdown link when the target exists, else the plain label."""
    if path is None:
        return F.MISSING
    try:
        relative = path.relative_to(destination.parent)
    except ValueError:
        return str(path)
    return F.link(label, relative.as_posix())


def exists(path):
    try:
        return path is not None and path.is_file()
    except OSError:
        return False


def span(values):
    """``min .. max`` of the non-null values, or the missing marker."""
    values = [value for value in values if value is not None]
    if not values:
        return F.MISSING
    return f"{F.number(min(values))} .. {F.number(max(values))}"


# --------------------------------------------------------------- L0 cells --

def l0_bandwidth_cell(entry, family=None):
    """The payload-bandwidth cell of one L0 baseline row.

    A row that never produced a measurement has no bandwidth at all (``-``); a row that
    passed but moves no communication payload says ``N/A`` with the reason.  The two
    are never conflated.
    """
    if entry.get("max_GBps") is not None:
        return F.number(entry["max_GBps"])
    if not entry.get("passed"):
        return F.MISSING
    if family in {"memory", "compute"}:
        return F.na(NOT_COMMUNICATION)
    return F.na(NO_BANDWIDTH_MODEL)


def l0_tflops_cell(entry, family=None):
    """The effective-TFLOPS cell of one L0 baseline row, with the same split."""
    if entry.get("effective_tflops") is not None:
        return F.number(entry["effective_tflops"])
    if not entry.get("passed"):
        return F.MISSING
    if family == "compute":
        return F.na(NO_COMPUTE_MODEL)
    return F.MISSING
