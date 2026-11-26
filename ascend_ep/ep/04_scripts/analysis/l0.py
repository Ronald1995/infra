"""L0 hardware baseline analysis shared by the main report and ``l0_report.md``.

The main report and the specialty report must never disagree about what L0
measured, so the grouping, the verification reading and the scale-out join all
live here.  ``reporting/`` only formats what this module returns.

Three rules are enforced here rather than in a template:

* a *requested* mechanism is never promoted to an *observed* one.  When nothing
  observed the path the observed column says so;
* a status that never produced evidence is reported as ``unverified`` (or as the
  status-derived label), never as ``verified``;
* a failed task keeps its category and its parameters, so it stays visible in the
  mechanism and scale-out tables instead of disappearing with its latency.
"""
from collections import Counter, defaultdict

from analysis import comparability
from common.storage import fingerprint
from baselines.l0.verification.vocabulary import (
    MECHANISMS, mechanism_label, resource_label, result_status_for,
)

#: Human labels for the communication baselines of section 3.1.  The order is the
#: order the report shows them in; anything not listed is appended alphabetically.
COMMUNICATION_LABELS = {
    "p2p_unidirectional": "P2P one-way",
    "p2p_bidirectional": "P2P two-way",
    "multi_peer": "Multi-peer",
    "alltoall": "AllToAll",
    "alltoallv": "AllToAllV",
    "allgather": "AllGather",
    "reducescatter": "ReduceScatter",
    "gather_reduce": "AllGather / ReduceScatter",
    "allreduce": "AllReduce",
    "p2p": "P2P",
    "put": "Put (one-way)",
    "copy": "HBM copy",
    "copy_local": "Local copy",
    "copy_local_ub": "Local UB copy",
    "copy_strided": "Strided copy",
    "grouped_matmul": "Grouped matmul",
    "swiglu": "SwiGLU",
    "concurrent": "Communication + compute concurrent",
    "concurrent_gmm_communication": "Communication + GMM concurrent",
}

#: Families the L0 catalog declares, in report order.
FAMILIES = ("collective", "execution", "transport", "memory", "compute", "overlap")

#: Case parameters that identify one L0 measurement point.
POINT_PARAMETERS = ("variant", "message_bytes", "outstanding", "peer_count", "peer_policy",
                    "accelerator", "shape_mode", "compute", "mode", "cores", "quiet_mode")


def is_l0(record):
    return record.get("benchmark_level") == "L0"


def rows(records):
    return [record for record in records if is_l0(record)]


def family(record):
    """The L0 family, from the record's category (never guessed from the ID)."""
    value = comparability.category(record)
    return value or "unclassified"


def label(record):
    case = record.get("case_id") or "unknown"
    operation = record.get("operation")
    return COMMUNICATION_LABELS.get(operation, case)


def parameters(record):
    return comparability.case_parameters(record)


def point(record):
    """The L0 measurement point a record describes (its full parameter subset)."""
    values = parameters(record)
    return {name: values.get(name) for name in POINT_PARAMETERS if name in values}


# ---------------------------------------------------------------- verification --

def request_of(record):
    """What the plan asked for: the requested block, else the planned parameters."""
    requested = record.get("requested") or {}
    values = parameters(record)
    mechanism = requested.get("transport_protocol") or values.get("mechanism")
    resource = requested.get("execution_resource") or values.get("execution_resource")
    return {
        "mechanism": mechanism,
        "mechanism_source": ("recorded_request" if requested.get("transport_protocol")
                             else ("planned_parameters" if values.get("mechanism") else "unavailable")),
        "execution_resource": resource,
        "channel": requested.get("channel"),
        "accelerator": requested.get("accelerator") or values.get("accelerator"),
    }


def observation_of(record):
    """What ran: observed fields, or explicit unavailability."""
    observed = record.get("observed") or {}
    return {
        "mechanism": observed.get("transport_protocol"),
        "execution_resource": observed.get("execution_resource"),
        "channel": observed.get("channel"),
        "physical_link": observed.get("physical_link"),
        "available": bool(observed),
    }


def verification(record):
    """``(status, source)`` for one L0 record.

    ``source`` is ``recorded`` when a verification block exists (the record was
    annotated by the path verifier) and ``derived`` when the label comes from the
    task status because nothing was ever verified.
    """
    recorded = (record.get("verification") or {}).get("status")
    if recorded:
        return recorded, "recorded"
    observed = (record.get("observed") or {}).get("verification_status")
    if observed:
        return observed, "recorded"
    status = record.get("status")
    if status == "passed":
        return "unverified", "derived"
    return (result_status_for(status) or "unverified"), "derived"


def evidence(record):
    items = comparability.verification_evidence(record)
    sources = []
    for item in items:
        source = item.get("source")
        if source and source not in sources:
            sources.append(source)
    return {"count": len(items), "sources": sources}


def completion(record):
    block = record.get("completion") or {}
    return {"level": block.get("level") or "unavailable",
            "proves_transfer": block.get("proves_transfer")}


def fallback(record):
    observed = record.get("observed") or {}
    observation = record.get("observation") or {}
    used = bool(observed.get("fallback") or (observation.get("fallback") or {}).get("used"))
    reason = ((observation.get("fallback") or {}).get("reason")
              or (record.get("verification") or {}).get("reason") or None)
    return {"used": used, "reason": reason}


def measurement(record):
    """One display record for one L0 result, with no value invented."""
    latency = record.get("latency") or {}
    bandwidth = record.get("bandwidth") or {}
    compute = record.get("compute") or {}
    status, source = verification(record)
    return {
        "task_id": record.get("task_id"),
        "case_id": record.get("case_id"),
        "level": record.get("benchmark_level"),
        "category": comparability.category(record),
        "family": family(record),
        "label": label(record),
        "operation": record.get("operation"),
        "ep_size": comparability.ep_size(record),
        "status": record.get("status"),
        "status_group": None,  # filled by the report layer from schemas.STATUS_GROUPS
        "parameters": point(record),
        "requested": request_of(record),
        "observed": observation_of(record),
        "verification": status,
        "verification_source": source,
        "evidence": evidence(record),
        "completion": completion(record),
        "fallback": fallback(record),
        "latency_ms": latency.get("p50"),
        "p95_ms": latency.get("p95"),
        "payload_bytes": bandwidth.get("payload_bytes"),
        "global_payload_bytes": bandwidth.get("global_payload_bytes"),
        "bandwidth_GBps": bandwidth.get("payload_bandwidth_GBps"),
        "bandwidth_scope": bandwidth.get("scope"),
        "bandwidth_source": bandwidth.get("source"),
        "effective_tflops": compute.get("effective_tflops"),
        "payload_applicable": bandwidth.get("payload_bytes") is not None or family(record) in {
            "transport", "execution", "collective", "overlap"},
        "reason": record.get("error_reason"),
    }


def measurements(records):
    return [measurement(record) for record in rows(records)]


# ------------------------------------------------------------------- summaries --

def status_counts(records):
    return Counter(record.get("status") for record in rows(records))


def mechanism_rows(records):
    """One row per (mechanism, execution resource) with its coverage and results.

    Grouped by the *observed* mechanism when one exists, otherwise by the requested
    one -- and the source is reported so a requested-only row is never read as an
    observation.
    """
    groups = defaultdict(list)
    for record in rows(records):
        observed = observation_of(record)
        request = request_of(record)
        mechanism = observed["mechanism"] or request["mechanism"] or "unavailable"
        source = "observed" if observed["mechanism"] else request["mechanism_source"]
        resource = (observed["execution_resource"] or request["execution_resource"])
        groups[(mechanism, source, resource)].append(record)
    output = []
    for (mechanism, source, resource), group in sorted(groups.items(), key=lambda item: str(item[0])):
        points = [measurement(record) for record in group]
        output.append({
            "mechanism": mechanism,
            "mechanism_label": mechanism_label(mechanism) if mechanism in MECHANISMS else mechanism,
            "mechanism_source": source,
            "execution_resource": resource,
            "execution_resource_label": resource_label(resource),
            "cases": sorted({item["case_id"] for item in points}),
            "ep_sizes": sorted({item["ep_size"] for item in points if item["ep_size"]}),
            "records": len(points),
            "passed": sum(1 for item in points if item["status"] == "passed"),
            "verified": sum(1 for item in points if item["verification"] == "verified"),
            "fallback": sum(1 for item in points if item["fallback"]["used"]),
            "statuses": dict(sorted(Counter(item["status"] for item in points).items())),
            "median_GBps": _median([item["bandwidth_GBps"] for item in points
                                    if item["bandwidth_GBps"] is not None]),
            "max_GBps": max((item["bandwidth_GBps"] for item in points
                             if item["bandwidth_GBps"] is not None), default=None),
            "has_bandwidth": any(item["bandwidth_GBps"] is not None for item in points),
            "has_compute": any(item["effective_tflops"] is not None for item in points),
            "min_p50_ms": min((item["latency_ms"] for item in points
                               if item["latency_ms"] is not None), default=None),
            "max_p50_ms": max((item["latency_ms"] for item in points
                               if item["latency_ms"] is not None), default=None),
        })
    return output


def baseline_rows(records, families):
    """One row per (communication baseline label, case) in the requested families."""
    groups = defaultdict(list)
    for record in rows(records):
        if family(record) not in families:
            continue
        groups[(record.get("operation"), record.get("case_id"))].append(record)
    order = {name: index for index, name in enumerate(COMMUNICATION_LABELS)}
    output = []
    for (operation, case_id), group in sorted(
            groups.items(), key=lambda item: (order.get(item[0][0], 999), str(item[0][0]),
                                              item[0][1])):
        points = [measurement(record) for record in group]
        output.append({
            "baseline": COMMUNICATION_LABELS.get(operation, operation),
            "operation": operation,
            "case_id": case_id,
            "ep_sizes": sorted({item["ep_size"] for item in points if item["ep_size"]}),
            "records": len(points),
            "passed": sum(1 for item in points if item["status"] == "passed"),
            "verified": sum(1 for item in points if item["verification"] == "verified"),
            "statuses": dict(sorted(Counter(item["status"] for item in points).items())),
            "min_p50_ms": min((item["latency_ms"] for item in points
                               if item["latency_ms"] is not None), default=None),
            "max_p50_ms": max((item["latency_ms"] for item in points
                               if item["latency_ms"] is not None), default=None),
            "max_GBps": max((item["bandwidth_GBps"] for item in points
                             if item["bandwidth_GBps"] is not None), default=None),
            "effective_tflops": max((item["effective_tflops"] for item in points
                                     if item["effective_tflops"] is not None), default=None),
            "measurements": points,
        })
    return output


def verification_table(records):
    """Aggregate the requested/observed/verification evidence per point.

    Grouping by the whole identity keeps a fallback or an unverified point visible
    as its own line instead of being averaged into a verified one.
    """
    groups = defaultdict(list)
    for record in rows(records):
        item = measurement(record)
        key = (item["case_id"], item["requested"]["mechanism"], item["observed"]["mechanism"],
               item["requested"]["execution_resource"], item["observed"]["execution_resource"],
               item["verification"], item["verification_source"], item["completion"]["level"],
               item["fallback"]["used"], item["category"])
        groups[key].append(item)
    output = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        (case_id, requested, observed, requested_resource, observed_resource, status,
         source, level, fallback_used, category) = key
        evidence_sources = sorted({source_name for item in items
                                   for source_name in item["evidence"]["sources"]})
        output.append({
            "case_id": case_id, "category": category,
            "requested_mechanism": requested, "observed_mechanism": observed,
            "requested_resource": requested_resource, "observed_resource": observed_resource,
            "verification": status, "verification_source": source,
            "completion": level, "fallback": fallback_used,
            "evidence_sources": evidence_sources,
            "records": len(items),
            "ep_sizes": sorted({item["ep_size"] for item in items if item["ep_size"]}),
            "statuses": dict(sorted(Counter(item["status"] for item in items).items())),
        })
    return output


def verification_summary(records):
    counts = Counter()
    for record in rows(records):
        status, _source = verification(record)
        counts[status] += 1
    return dict(sorted(counts.items()))


def scale_out(records):
    """L0 hardware scale-out: one row per *complete* parameter variant.

    Grouping uses the whole case-parameter dictionary, plus the case, category and
    observed execution mechanism.  Nothing about a variant is inferred from a subset
    of keys: ``ring``/``same_server``/``cross_server``, ``device``/``host``,
    ``gmm``/``dgrad``/``wgrad`` and ``fixed``/``workload`` therefore never collapse
    into one line.  Two different tasks at the same EP size are reported as
    ambiguous instead of one silently overwriting the other.
    """
    groups = defaultdict(list)
    for record in rows(records):
        values = parameters(record)
        observed = observation_of(record)
        mechanism = observed["mechanism"] or request_of(record)["mechanism"]
        key = (record.get("case_id"), comparability.category(record), mechanism,
               observed["execution_resource"] or request_of(record)["execution_resource"],
               fingerprint(values))
        groups[key].append(record)
    output = []
    for (case_id, category, mechanism, resource, parameter_fingerprint), group in sorted(
            groups.items(), key=lambda item: str(item[0])):
        values = dict(parameters(group[0]))
        observed_mechanisms = sorted({observation_of(record)["mechanism"] for record in group
                                      if observation_of(record)["mechanism"]})
        mechanism_source = ("observed" if observed_mechanisms
                           else request_of(group[0])["mechanism_source"])
        by_ep_size = defaultdict(list)
        for record in group:
            by_ep_size[comparability.ep_size(record)].append(record)
        points, ambiguous = [], []
        for size, same in sorted(by_ep_size.items(), key=lambda item: (item[0] is None, item[0])):
            task_ids = sorted({record.get("task_id") for record in same})
            if len(task_ids) > 1:
                ambiguous.append({
                    "ep_size": size, "task_ids": task_ids,
                    "reason": ("distinct tasks share the same case parameters but different "
                               "configurations; the point is not collapsed into one value")})
                continue
            item = measurement(same[0])
            points.append({
                "ep_size": size, "task_id": item["task_id"], "status": item["status"],
                "reason": item["reason"], "verification": item["verification"],
                "p50": item["latency_ms"], "bandwidth_GBps": item["bandwidth_GBps"],
                "payload_bytes": item["payload_bytes"],
            })
        conflict = len(observed_mechanisms) > 1
        verified_sizes = {point["ep_size"] for point in points
                          if point["verification"] == "verified" and point["status"] == "passed"}
        entry = {
            "case_id": case_id, "category": category,
            "mechanism": mechanism, "mechanism_source": mechanism_source,
            "observed_mechanisms": observed_mechanisms, "mechanism_conflict": conflict,
            "execution_resource": resource,
            "parameter_fingerprint": parameter_fingerprint,
            "parameters": values,
            "points": points, "ambiguous": ambiguous,
            "comparable": bool(not conflict and not ambiguous and len(verified_sizes) >= 2),
        }
        for name in POINT_PARAMETERS:
            entry[name] = values.get(name)
        if conflict:
            entry["reason"] = ("observed transport mechanism differs across EP sizes; the "
                               "rows are kept separate")
        elif ambiguous:
            entry["reason"] = "several tasks share these parameters at one EP size"
        else:
            entry["reason"] = None
        output.append(entry)
    output.sort(key=lambda item: (item["case_id"], str(item.get("variant")),
                                  str(item.get("peer_policy")), str(item.get("message_bytes")),
                                  str(item.get("accelerator")), item["parameter_fingerprint"]))
    return output


def _median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def summary(records):
    """Everything a summary section needs, computed once."""
    l0 = rows(records)
    sizes = sorted({comparability.ep_size(record) for record in l0
                    if comparability.ep_size(record)})
    return {
        "records": len(l0),
        "ep_sizes": sizes,
        "cases": sorted({record.get("case_id") for record in l0}),
        "statuses": dict(sorted(Counter(record.get("status") for record in l0).items())),
        "verification": verification_summary(l0),
        "families": dict(sorted(Counter(family(record) for record in l0).items())),
        "points": len({(record.get("case_id"), fingerprint(point(record)))
                       for record in l0}),
    }


def analyse(records):
    """All L0 tables, built once and shared by the main report and ``l0_report.md``."""
    l0 = rows(records)
    return {
        "summary": summary(l0),
        "measurements": measurements(l0),
        "mechanisms": mechanism_rows(l0),
        "collective": baseline_rows(l0, {"collective"}),
        "execution": baseline_rows(l0, {"execution"}),
        "communication": baseline_rows(l0, {"collective", "execution"}),
        "transport": baseline_rows(l0, {"transport"}),
        "memory": baseline_rows(l0, {"memory"}),
        "compute": baseline_rows(l0, {"compute"}),
        "overlap": baseline_rows(l0, {"overlap"}),
        "verification": verification_table(l0),
        "scale_out": scale_out(l0),
        "status_counts": dict(sorted(Counter(record.get("status") for record in l0).items())),
    }
