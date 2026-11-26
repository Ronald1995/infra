"""The single definition of "these two measurements answer the same question".

Every report that ranks, compares or scales results asks this module instead of
re-deciding for itself.  That is the whole point: the previous code had three
different grouping rules (`analysis/comparison.py`, `reporting/coverage.py` and the
report templates) and the same two rows could be "comparable" in one report and not
in another.

## What a comparison key contains

Only conditions that change *what was measured*:

============================  ==================================================
Case identity                 ``case_id``, ``benchmark_level``, ``operation``
Parameter variant             the case parameters (``variant``, ``message_bytes``,
                              ``mechanism``, ``peer_policy``, ...)
Environment                   device type, NPU model, CANN version, package set
Topology                      declared server/supernode map and its source
Scale                         EP size, TP size (excluded for EP scaling)
Model                         model shape and dtype
Workload                      per-rank token shape, routing strategy, seed,
                              routing-trace hash
Timing boundary               timing mode plus latency/bandwidth unit and scope
Execution mode                dedup mode, load-balance state, completion level
============================  ==================================================

Deliberately **not** in the key:

* ``task_id`` / ``run_id`` -- identity of a run, not of a test condition.
* ``backend_id`` -- for a backend comparison it is the *dimension being compared*;
  for an EP-scaling group it is a required sibling and is added explicitly.
* latency, throughput, raw samples and every other measured result.
* the ``reason`` / ``notes`` parameter keys -- explanations, not test conditions.

``case_parameters`` participates in full by default.  A parameter is either a test
condition or it is named in :data:`NON_SEMANTIC_PARAMETERS`; there is no third
category, so a backend-specific knob that changes the work is never silently
dropped just because only one backend uses it.

## Completion level is part of the key

A latency measured until a transfer is *remote visible* and a latency measured until
the submission is *fenced* are different measurements, not a fast and a slow version
of the same one (see ``docs/observability.md``).  Two backends that promise different
completion levels therefore do not land in one comparison group; the report shows
them as separate single measurements with the completion level next to each row.

## Environment differences are never merged

CANN, driver and NPU model are in the key, so a run that spans two environments
produces two groups.  Cross-environment comparison is a separate, explicit group and
never happens implicitly.
"""
from dataclasses import dataclass

from common.storage import fingerprint

#: Case parameters that describe *why* a point did not run rather than *what* it
#: measures.  Only these are excluded from a comparison key.
NON_SEMANTIC_PARAMETERS = frozenset({"reason", "notes"})

#: Comparison outcomes, ordered from least to most evidence.
GROUP_KINDS = ("no_measurement", "single_measurement", "repeated_measurement",
               "actual_comparison")

#: Why a record was excluded from comparison, with a human-readable reason.
_ELIGIBILITY = (
    ("status", "the task status is {value}, not `passed`"),
    ("source", "the record carries no measurement (source is {value})"),
    ("latency", "no positive p50 latency was measured"),
    ("correctness", "correctness did not pass, so the measurement is not accepted"),
    ("verification", "the L0 execution path is {value}, not `verified`"),
)


@dataclass(frozen=True)
class Eligibility:
    """Whether one record may enter a comparison, and if not, why."""

    eligible: bool
    reason: str | None = None
    cause: str | None = None

    def __bool__(self):
        return self.eligible


# ------------------------------------------------------------------ readers --

def case_parameters(record):
    """The planned case parameters, from the standard field or the legacy alias.

    ``case_parameters`` is the one the code writes; ``parameters`` is only read so a
    result file produced before the contract fix still reports its variant.
    """
    parameters = record.get("case_parameters")
    if parameters is None:
        parameters = record.get("parameters")
    if isinstance(parameters, dict):
        return dict(parameters)
    return {}


def semantic_parameters(record):
    """Case parameters minus the explanatory keys."""
    return {key: value for key, value in case_parameters(record).items()
            if key not in NON_SEMANTIC_PARAMETERS}


_CATALOG_CACHE = {}


def _catalog_case(case_id):
    """The catalog entry for a case ID, loaded once and cached."""
    if not _CATALOG_CACHE:
        try:
            from common.cases import catalog
            _CATALOG_CACHE["catalog"] = catalog()
        except Exception:  # a broken catalog must not stop a report
            _CATALOG_CACHE["catalog"] = {}
    return _CATALOG_CACHE["catalog"].get(case_id)


def category(record):
    """The case category: the record's own field, else the catalog definition.

    A result written before the contract fix carries no category; reading it from the
    case catalog keeps those runs reportable without inventing anything.  The category
    is a property of the case, not of the measurement, so this fallback cannot
    mis-attribute a number.
    """
    value = record.get("category")
    if value:
        return value
    case = _catalog_case(record.get("case_id"))
    if case is not None:
        return getattr(case, "category", None)
    return ((record.get("config") or {}).get("case") or {}).get("category")


def level(record):
    return record.get("benchmark_level")


def ep_size(record):
    """The EP size that actually ran: the resolved configuration wins.

    ``record["ep_size"]`` is the planned value and the two agree in a real run; the
    configuration is authoritative because the distributed group was built from it.
    """
    size = (((record.get("config") or {}).get("distributed") or {}).get("ep_size"))
    return size or record.get("ep_size")


def tp_size(record):
    return (((record.get("config") or {}).get("distributed") or {}).get("tp_size"))


def verification_status(record):
    """The L0 path-verification status, or ``None`` for a non-L0 record."""
    return ((record.get("verification") or {}).get("status")
            or (record.get("observed") or {}).get("verification_status"))


def verification_evidence(record):
    """Evidence items that established the observed path."""
    evidence = (record.get("verification") or {}).get("evidence")
    if evidence:
        return list(evidence)
    return list((record.get("evidence") or {}).get("claims") or ())


def requested_mechanism(record):
    """The mechanism the plan asked for; never evidence of what ran."""
    requested = record.get("requested") or {}
    if requested.get("transport_protocol"):
        return requested["transport_protocol"]
    return case_parameters(record).get("mechanism")


def observed_mechanism(record):
    """The mechanism the runtime reported, or ``None`` when nothing observed it."""
    observed = record.get("observed") or {}
    return observed.get("transport_protocol")


def observation_source(record):
    return (record.get("observation") or {}).get("source")


# -------------------------------------------------------------- eligibility --

def eligibility(record):
    """Whether one record is an accepted measurement of its own case.

    This is the *local* test: it says nothing about whether the record may be
    compared across mechanisms or backends.
    """
    if record.get("status") != "passed":
        return Eligibility(False, f"the task status is {record.get('status')}, not `passed`",
                           "status")
    source = record.get("source")
    if source != "measured":
        return Eligibility(False, f"the record carries no measurement (source is {source!r})",
                           "source")
    latency = record.get("latency") or {}
    p50 = latency.get("p50")
    if not p50 or p50 <= 0:
        return Eligibility(False, "no positive p50 latency was measured", "latency")
    correctness = record.get("correctness")
    if not correctness or not correctness.get("passed"):
        return Eligibility(False, "correctness did not pass, so the measurement is not accepted",
                           "correctness")
    return Eligibility(True)


def comparable(record):
    """Whether one record may enter a cross-backend or cross-mechanism comparison.

    An L0 hardware baseline is only comparable when its *path* was verified: a
    number measured over an unverified transport is not a measurement of the
    mechanism it names, so pairing it with another mechanism would compare two
    different things.  Operator cases are gated on their own correctness instead,
    because their transport is reported separately in ``observation``.
    """
    local = eligibility(record)
    if not local:
        return False
    if level(record) == "L0" and verification_status(record) != "verified":
        return False
    return True


def exclusion_reason(record):
    """A human-readable reason why ``record`` is not comparable, or ``None``."""
    local = eligibility(record)
    if not local:
        return local.reason
    if level(record) == "L0" and verification_status(record) != "verified":
        status = verification_status(record) or "unverified"
        return (f"the L0 execution path is {status}, so the number cannot be attributed to "
                f"the requested mechanism")
    return None


# --------------------------------------------------------------- dimensions --

def _plain(value):
    """A hashable, JSON-stable view of a config value."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _shape(tokens):
    """Per-rank token shape, independent of how many ranks carry it.

    ``[128] * 8`` and ``[128] * 128`` describe the same per-rank work, so the EP
    size must not be implied by the length of the list.  A genuinely different
    distribution (``[16, 0, 5, 7]``) keeps its own shape.
    """
    if tokens is None:
        return None
    if isinstance(tokens, (list, tuple)):
        return sorted(_plain(item) for item in tokens)
    return [_plain(tokens)]


def case_identity(record):
    return {"case_id": record.get("case_id"), "level": level(record),
            "operation": record.get("operation")}


def environment_dimensions(record):
    env = record.get("environment") or {}
    config = record.get("config") or {}
    packages = env.get("packages")
    return {
        "device": config.get("device") or env.get("device"),
        "npu_models": sorted(env.get("npu_models") or ()),
        "cann_version": env.get("cann_version"),
        "packages": _plain(packages) if isinstance(packages, (dict, list)) else packages,
    }


def topology_dimensions(record):
    topology = (record.get("config") or {}).get("topology") or {}
    return {"source": topology.get("source"),
            "server_ids": _plain(topology.get("server_ids")),
            "supernode_ids": _plain(topology.get("supernode_ids"))}


def model_dimensions(record, include_expert_count=True):
    model = dict((record.get("config") or {}).get("model") or {})
    if not include_expert_count:
        # The global expert count changes with the EP size by design, so it is
        # reported per point instead of splitting an EP-scaling group.
        model.pop("num_experts", None)
    return _plain(model)


def workload_dimensions(record):
    config = record.get("config") or {}
    configured = config.get("workload") or {}
    measured = record.get("workload") or {}
    tokens = measured.get("tokens_per_rank")
    if tokens is None:
        tokens = configured.get("tokens_per_rank")
    return {
        "tokens_per_rank": _shape(tokens),
        "routing": configured.get("routing"),
        "seed": configured.get("seed"),
        "expert_placement": _plain(configured.get("expert_placement")),
        "steps": _plain(measured.get("steps")),
        "requests_per_sample": measured.get("requests_per_sample"),
        "routing_trace": record.get("routing_trace_hash"),
    }


def timing_dimensions(record):
    config = record.get("config") or {}
    latency = record.get("latency") or {}
    bandwidth = record.get("bandwidth") or {}
    return {
        "mode": (config.get("benchmark") or {}).get("timing"),
        "latency_unit": latency.get("unit"),
        "latency_scope": latency.get("scope"),
        "bandwidth_unit": bandwidth.get("bandwidth_unit") or bandwidth.get("unit"),
        "bandwidth_scope": bandwidth.get("scope"),
    }


def mode_dimensions(record):
    config = record.get("config") or {}
    return {
        "dedup": (config.get("dedup") or {}).get("mode"),
        "load_balance": boolean((config.get("load_balance") or {}).get("enabled")),
        "completion": (record.get("completion") or {}).get("level"),
    }


def boolean(value):
    return None if value is None else bool(value)


def scale_dimensions(record, include_ep_size=True):
    if not include_ep_size:
        return {}
    return {"ep_size": ep_size(record), "tp_size": tp_size(record)}


class _Auto:
    """Sentinel: derive the normalised workload size from the record itself."""

    def __repr__(self):
        return "AUTO"


AUTO = _Auto()


def workload_scale(record):
    """The EP-scaling workload size declared by the suite's scaling policy.

    ``strong`` scaling keeps the *global* work constant while the rank count grows;
    ``weak`` scaling keeps the *per-rank* work constant.  The two are different
    experiments, so the normalised size names the mode as well as the numbers.
    ``None`` means the run declared no scaling policy, and the per-rank token shape
    stays part of the comparison key instead.
    """
    config = record.get("config") or {}
    scaling = config.get("scaling") or {}
    mode = scaling.get("mode")
    model = config.get("model") or {}
    if mode == "strong":
        return ["strong", scaling.get("global_tokens"), model.get("num_experts")]
    if mode == "weak":
        return ["weak", scaling.get("tokens_per_rank"), scaling.get("experts_per_rank")]
    return None



# --------------------------------------------------------------------- keys --

def backend_comparison_key(record):
    """Group key for comparing different backends at the same point.

    The backend itself is not part of the key: it is the dimension being compared.
    """
    return fingerprint({
        "case": case_identity(record),
        "parameters": semantic_parameters(record),
        "environment": environment_dimensions(record),
        "topology": topology_dimensions(record),
        "model": model_dimensions(record),
        "workload": workload_dimensions(record),
        "timing": timing_dimensions(record),
        "mode": mode_dimensions(record),
        "scale": scale_dimensions(record),
    })


def scaling_key(record, size=AUTO):
    """Group key for an EP-scaling series: one backend, one case, several EP sizes.

    EP size, TP size and the derived topology are excluded because they *are* the
    scaling dimension; the normalised workload ``size`` is supplied by the caller (or
    derived from the suite's scaling policy) so strong and weak scaling stay separate
    comparisons.  The routing-trace hash is also excluded: it is derived from the EP
    size, so keeping it would make every series single-point.
    """
    if size is AUTO:
        size = workload_scale(record)
    workload = workload_dimensions(record)
    workload.pop("routing_trace", None)
    if size is not None:
        # `size` already fixes the workload, so the per-rank list (whose *length* is
        # the EP size) must not split the series.
        workload.pop("tokens_per_rank", None)
    return fingerprint({
        "case": case_identity(record),
        "backend": record.get("backend_id"),
        "parameters": semantic_parameters(record),
        "environment": environment_dimensions(record),
        "model": model_dimensions(record, include_expert_count=False),
        "workload": workload,
        "timing": timing_dimensions(record),
        "mode": mode_dimensions(record),
        "tp_size": tp_size(record),
        "size": _plain(size),
    })


# ----------------------------------------------------------- incompatibility --

def _describe_value(value, limit=3):
    if isinstance(value, list):
        shown = ", ".join(str(item) for item in value[:limit])
        return f"[{shown}{', ...' if len(value) > limit else ''}] ({len(value)} entries)"
    if isinstance(value, dict):
        keys = list(value)[:limit]
        return "{" + ", ".join(f"{key}: {value[key]}" for key in keys) + (
            ", ..." if len(value) > limit else "") + "}"
    return repr(value)


def _diff(label, left, right):
    if left == right:
        return None
    return f"{label} differs ({_describe_value(left)} vs {_describe_value(right)})"


def _parameter_diff(left, right):
    left_parameters = semantic_parameters(left)
    right_parameters = semantic_parameters(right)
    if left_parameters == right_parameters:
        return None
    keys = sorted(set(left_parameters) | set(right_parameters))
    changed = [f"{key} {left_parameters.get(key)!r} vs {right_parameters.get(key)!r}"
               for key in keys if left_parameters.get(key) != right_parameters.get(key)]
    return "case parameters differ: " + "; ".join(changed[:4]) + (
        f"; and {len(changed) - 4} more" if len(changed) > 4 else "")


_DIMENSIONS = (
    ("case", lambda r: r.get("case_id")),
    ("benchmark level", level),
    ("operation", lambda r: r.get("operation")),
    ("environment (device/NPU/CANN/packages)", environment_dimensions),
    ("topology", topology_dimensions),
    ("model", model_dimensions),
    ("workload", workload_dimensions),
    ("timing boundary", timing_dimensions),
    ("execution mode (dedup/load balance/completion)", mode_dimensions),
    ("EP size / TP size", scale_dimensions),
)


def incompatibility(left, right):
    """Every reason why two records are not comparable, in a stable order.

    An empty list means the two records describe the same test condition, so their
    measured values may be put next to each other.
    """
    if left is right:
        return []
    reasons = []
    for label, extract in _DIMENSIONS:
        reason = _diff(label, extract(left), extract(right))
        if reason:
            reasons.append(reason)
    parameters = _parameter_diff(left, right)
    if parameters:
        reasons.append(parameters)
    return reasons


def compatibility(left, right):
    """``(compatible, reasons)`` for two records."""
    reasons = incompatibility(left, right)
    return (not reasons), reasons


def group_kind(records):
    """Classify a group of records into one of :data:`GROUP_KINDS`.

    * ``single_measurement`` -- one record, so there is nothing to compare.
    * ``repeated_measurement`` -- several records from *one* backend.  These are
      repeat samples of one implementation, never a backend comparison.
    * ``actual_comparison`` -- at least two different backends with at least one
      accepted measurement each.
    """
    records = [record for record in records if eligibility(record)]
    if not records:
        return "no_measurement"
    backends = {record.get("backend_id") for record in records}
    if len(records) == 1:
        return "single_measurement"
    if len(backends) < 2:
        return "repeated_measurement"
    return "actual_comparison"
