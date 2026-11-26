"""Decide what a result may claim, from evidence rather than configuration.

Rules, in order:

1. A field is ``verified`` only when a *strong* evidence source names it:
   a native channel descriptor, a runtime query API, or a structured log line
   that explicitly reports the field.
2. A *heuristic* source (free-text log keyword, kernel-name classifier) may
   report a field but leaves it ``unverified``; the claim is still attached so a
   human can audit it.
3. If strong evidence contradicts the request, the record is ``fallback``.
4. If a required field has no evidence at all, the record is ``unverified``.
5. Configuration values are never evidence.
"""
from dataclasses import dataclass, field

from baselines.l0.verification.vocabulary import (
    MECHANISMS, SEVERITY, VERIFICATION_STATUSES, observation, requested, worst,
)

STRONG_SOURCES = ("native_descriptor", "shmem_query", "runtime_api", "hccl_runtime_log",
                  "hcomm_log", "rank_graph",
                  # Runtime answers from the endpoint/channel/kernel/executor probes.
                  # Each is an observation the running system produced, not a
                  # configuration value: see baselines/l0/verification/evidence.py.
                  "endpoint", "channel_kind", "kernel", "msprof_kernel",
                  "runtime_execution")
HEURISTIC_SOURCES = ("hccl_log_keyword", "hcomm_log_keyword", "profiler_kernel_name",
                     "soc_name_heuristic")

FIELDS = ("execution_resource", "transport_protocol", "channel", "physical_link")

#: Claim keys that may accompany the four observed fields.
CLAIM_EXTRAS = ("mechanism", "accelerator", "net_layer", "endpoint", "channel_kind",
                "kernel_name", "payload_validated")

#: Fields a case must be able to observe before its numbers may be compared
#: against another mechanism's numbers.
DEFAULT_REQUIRED = ("execution_resource", "transport_protocol")


@dataclass
class Evidence:
    source: str
    claim: dict
    path: str | None = None
    detail: str = ""
    confidence: str = "strong"

    def __post_init__(self):
        if self.confidence not in ("strong", "heuristic"):
            raise ValueError("evidence confidence must be strong or heuristic")
        if self.confidence == "strong" and self.source not in STRONG_SOURCES:
            raise ValueError(f"{self.source!r} is not a strong evidence source")
        if self.confidence == "heuristic" and self.source not in HEURISTIC_SOURCES:
            raise ValueError(f"{self.source!r} is not a heuristic evidence source")
        unknown = set(self.claim) - set(FIELDS) - set(CLAIM_EXTRAS)
        if unknown:
            raise ValueError(f"unsupported evidence claim fields {sorted(unknown)}")

    def as_dict(self):
        return {"source": self.source, "confidence": self.confidence, "claim": dict(self.claim),
                "path": self.path, "detail": self.detail}


@dataclass
class Verification:
    status: str
    observed: dict
    required: tuple[str, ...]
    missing: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    reason: str = ""

    def as_dict(self):
        return {
            "status": self.status,
            "required_fields": list(self.required),
            "missing_evidence": list(self.missing),
            "conflicts": list(self.conflicts),
            "reason": self.reason,
            "evidence": [item.as_dict() for item in self.evidence],
        }


def _promote(block, field_name, value, status="verified"):
    if value is None:
        return
    if block.get(field_name) is None:
        block[field_name] = value
    elif block[field_name] != value:
        return
    if status == "verified":
        block[f"{field_name}_verification"] = "verified"
        if field_name not in block["verified_fields"]:
            block["verified_fields"].append(field_name)
        if field_name in block["unverified_fields"]:
            block["unverified_fields"].remove(field_name)


def verify(request, evidence=(), required=DEFAULT_REQUIRED, initial=None):
    """Fold evidence into an ``observed`` block and an aggregate status."""
    if not isinstance(request, dict):
        raise TypeError("verify requires the requested block")
    observed = initial or observation(
        execution_resource=None, transport_protocol=None,
    )
    evidence = list(evidence)
    for item in evidence:
        if not isinstance(item, Evidence):
            raise TypeError("evidence items must be Evidence instances")
        claim = item.claim
        if item.confidence == "strong":
            _promote(observed, "execution_resource", claim.get("execution_resource"))
            _promote(observed, "transport_protocol", claim.get("transport_protocol"))
            _promote(observed, "channel", claim.get("channel"))
            _promote(observed, "physical_link", claim.get("physical_link"))
        else:
            # Heuristic claims are recorded but never promoted.
            for name in FIELDS:
                if claim.get(name) is not None and observed.get(name) is None:
                    observed[name] = claim[name]

    conflicts = []
    if observed["transport_protocol"] and request.get("transport_protocol") not in (None, "auto"):
        if observed["transport_protocol"] != request["transport_protocol"]:
            conflicts.append({"field": "transport_protocol",
                              "requested": request["transport_protocol"],
                              "observed": observed["transport_protocol"]})
    if observed["execution_resource"] and request.get("execution_resource"):
        if observed["execution_resource"] != request["execution_resource"]:
            conflicts.append({"field": "execution_resource",
                              "requested": request["execution_resource"],
                              "observed": observed["execution_resource"]})
    requested_channel = request.get("channel")
    if observed["channel"] and requested_channel and observed["channel"] != requested_channel:
        conflicts.append({"field": "channel", "requested": requested_channel,
                          "observed": observed["channel"]})
    if conflicts:
        observed["fallback"] = True

    missing = [name for name in required if observed.get(f"{name}_verification") != "verified"]
    statuses = [observed.get(f"{name}_verification", "unverified") for name in required]
    status = worst(statuses)
    if conflicts:
        status = "fallback"
    observed["verification_status"] = status
    reason = ""
    if status == "fallback":
        reason = "observed path differs from the requested path"
    elif status == "unverified":
        reason = "no strong evidence for " + ", ".join(missing)
    return Verification(status=status, observed=observed, required=tuple(required),
                        missing=missing, conflicts=conflicts, evidence=evidence, reason=reason)


def annotate(record, request, evidence, required=DEFAULT_REQUIRED):
    """Attach ``requested``/``observed``/``evidence``/``verification`` to a record."""
    result = verify(request, evidence, required)
    record["requested"] = request
    record["observed"] = result.observed
    record["verification"] = result.as_dict()
    record["evidence"] = {
        "log_paths": sorted({item.path for item in evidence if item.path}),
        "profile_paths": [],
        "claims": [item.as_dict() for item in evidence],
    }
    return result


def status_allows_comparison(status):
    """Only verified measurements may enter cross-mechanism comparisons."""
    return status == "verified"


def audit(records):
    """Group records by verification status for the L0 report."""
    counts = {status: 0 for status in VERIFICATION_STATUSES}
    for record in records:
        status = ((record.get("verification") or {}).get("status")
                  or (record.get("observed") or {}).get("verification_status"))
        if status in counts:
            counts[status] += 1
    return counts


def requested_block(case, params, backend, mechanism, execution_resource=None):
    """Convenience wrapper used by the L0 dispatcher and by unit tests."""
    return requested(
        backend=backend, operation=case.operation,
        execution_resource=execution_resource,
        transport_protocol=mechanism or "auto",
        accelerator=params.get("accelerator"),
        channel=params.get("channel"),
        topology_mode=params.get("topology_mode"),
    )


__all__ = [
    "Evidence", "Verification", "DEFAULT_REQUIRED", "FIELDS", "STRONG_SOURCES",
    "verify", "annotate", "audit", "status_allows_comparison", "requested_block",
    "MECHANISMS", "SEVERITY",
]
