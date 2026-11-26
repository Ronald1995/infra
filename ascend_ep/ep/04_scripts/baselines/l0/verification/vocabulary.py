"""Controlled vocabulary shared by every L0 hardware baseline.

The L0 suite deliberately keeps four dimensions apart, because collapsing them is
how benchmark reports end up attributing a number to a transport that never ran:

``operation``
    What the user asked for at the communication-operation layer: ``put``,
    ``get``, ``alltoall``, ``allgather``, ``reducescatter``, ``allreduce``,
    ``p2p`` and friends.

``execution_resource``
    Which on-device agent issues and completes the transfer: ``AIV``,
    ``AICPU_TS``, ``CCU`` or the ``HOST_CPU``.

``mechanism``
    The communication mechanism / protocol family under test: ``UB Memory``,
    ``URMA``, ``UB CTP``, ``UB RTP``, ``RoCE``, ``HCCS``, ``UBoE`` or the SHMEM
    data-movement engines ``shmem_mte`` / ``shmem_udma``.

``physical topology``
    Where the bytes actually travel: HBM, local UB, UB interconnect, HCCS,
    Ethernet, or a cross-supernode network.

A requested configuration is never evidence of an observed path.  Every record
carries both, and :mod:`baselines.l0.verification.path_verifier` decides which
fields are actually verified.
"""

EXECUTION_RESOURCES = ("AIV", "AICPU_TS", "CCU", "HOST_CPU")

#: Accelerator modes accepted by the native HCCL test tool.  The mapping to an
#: execution resource is a *declaration*, not a measurement: only log or
#: descriptor evidence promotes it to an observation.  The accepted mode names
#: themselves are always re-read from the installed tool, see
#: ``baselines/l0/collectives/accelerator.py``.
ACCELERATOR_RESOURCES = {
    "default": None,
    "aiv": "AIV",
    "aiv_only": "AIV",
    "aicpu": "AICPU_TS",
    "aicpu_ts": "AICPU_TS",
    "ccu": "CCU",
    "ccu_sched": "CCU",
    "ccu_ms": "CCU",
}

MECHANISMS = (
    "ub_memory", "urma", "ub_ctp", "ub_rtp", "roce", "hccs", "uboe",
    "shmem_mte", "shmem_udma",
)

#: ``auto`` means "the library decides"; it is a valid request and always an
#: unverified observation unless the runtime reports the choice.
REQUESTED_MECHANISMS = (*MECHANISMS, "auto")

OPERATIONS = (
    "put", "get", "read", "write", "send", "recv", "p2p",
    "alltoall", "alltoallv", "allgather", "reducescatter", "allreduce",
    "copy", "compute", "concurrent",
)

BACKENDS = (
    "torch_hccl",        # PyTorch distributed HCCL process group
    "native_hccl",       # native HCCL test tool (oam-tools hccl_test)
    "hcomm",             # native HCOMM channel
    "asc_comm",          # native ASC-COMM channel
    "shmem",             # CANN SHMEM runtime (host stream API or device kernel)
    "torch_memory",      # torch device copy / compute
    "native_kernel",     # purpose-built Ascend C kernel
)

PHYSICAL_LINKS = ("HBM", "LOCAL_UB", "UB_INTERCONNECT", "HCCS", "ETHERNET", "SUPERNODE", "UNKNOWN")

VERIFICATION_STATUSES = ("verified", "fallback", "unverified", "unsupported",
                         "environment_unavailable", "failed")

#: Composite severity.  A record is only as trustworthy as its weakest required
#: field, except that a detected path switch outranks "unverified" because it is
#: an actionable finding.
SEVERITY = {"verified": 0, "unverified": 1, "fallback": 2,
            "unsupported": 3, "environment_unavailable": 3, "failed": 4}

_OBSERVATION_FIELDS = ("execution_resource", "transport_protocol", "channel", "physical_link")


class VocabularyError(ValueError):
    """Raised when a case or result uses a token outside the controlled set."""


def check(value, allowed, name):
    if value is None:
        return None
    if value not in allowed:
        raise VocabularyError(f"{name}={value!r} is outside {allowed}")
    return value


def execution_resource_for(accelerator):
    """Map an accelerator mode name to an execution resource, without claiming it ran."""
    if accelerator is None:
        return None
    return ACCELERATOR_RESOURCES.get(accelerator)


def worst(statuses):
    """Combine per-field verification statuses into one record status."""
    values = [status for status in statuses if status]
    if not values:
        return "unverified"
    return max(values, key=lambda status: SEVERITY[status])


def requested(backend, operation, execution_resource=None, transport_protocol="auto",
              accelerator=None, channel=None, topology_mode=None):
    """Build the ``requested`` block of a result record.

    This block records intent only.  It must never be copied into ``observed``.
    """
    if backend not in BACKENDS:
        raise VocabularyError(f"backend={backend!r} is outside {BACKENDS}")
    if operation not in OPERATIONS:
        raise VocabularyError(f"operation={operation!r} is outside {OPERATIONS}")
    check(execution_resource, EXECUTION_RESOURCES, "execution_resource")
    check(transport_protocol, REQUESTED_MECHANISMS, "transport_protocol")
    if accelerator is not None:
        check(accelerator, tuple(ACCELERATOR_RESOURCES), "accelerator")
    return {
        "backend": backend,
        "operation": operation,
        "execution_resource": execution_resource,
        "transport_protocol": transport_protocol,
        "accelerator": accelerator,
        "channel": channel,
        "topology_mode": topology_mode,
    }


def observation(execution_resource=None, transport_protocol=None, channel=None,
                physical_link=None, fallback=False):
    """Build the ``observed`` block with per-field verification defaults.

    Every field starts ``unverified`` and is promoted only by explicit evidence.
    """
    check(execution_resource, EXECUTION_RESOURCES, "execution_resource")
    check(transport_protocol, MECHANISMS, "transport_protocol")
    check(physical_link, PHYSICAL_LINKS, "physical_link")
    block = {
        "execution_resource": execution_resource,
        "transport_protocol": transport_protocol,
        "channel": channel,
        "physical_link": physical_link,
        "fallback": bool(fallback),
        "verification_status": "unverified",
        "verified_fields": [],
        "unverified_fields": list(_OBSERVATION_FIELDS),
    }
    for field in _OBSERVATION_FIELDS:
        block[f"{field}_verification"] = "unverified"
    return block


def result_status_for(status):
    """Translate a benchmark outcome into the L0 verification vocabulary."""
    if status == "passed":
        return None
    if status in ("unsupported", "not_implemented", "dependency_missing", "source_unavailable"):
        return "unsupported"
    if status in ("environment_unavailable", "not_run"):
        return "environment_unavailable"
    return "failed"


def mechanism_label(mechanism):
    """Human-readable label used by reports and READMEs."""
    return {
        "ub_memory": "UB Memory", "urma": "URMA", "ub_ctp": "UB CTP", "ub_rtp": "UB RTP",
        "roce": "RoCE", "hccs": "HCCS", "uboe": "UBoE",
        "shmem_mte": "SHMEM MTE", "shmem_udma": "SHMEM UDMA", "auto": "auto",
    }.get(mechanism, mechanism)


def resource_label(resource):
    return {"AIV": "AIV", "AICPU_TS": "AICPU + TS", "CCU": "CCU",
            "HOST_CPU": "Host CPU", None: "unspecified"}.get(resource, resource)
