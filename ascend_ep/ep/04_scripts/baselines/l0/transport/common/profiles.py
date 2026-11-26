"""Declarative request table for the native L0 transport cases.

Every entry states only what the benchmark *asks* the communication library for:

``engine``
    ``CommEngine`` value passed to ``HcommChannelCreate``.
``protocol``
    ``CommProtocol`` value carried in ``HcommChannelDesc.remoteEndpoint``.
``endpoint``
    ``ENDPOINT_LOC_TYPE_DEVICE`` or ``_HOST`` for ``HcommEndpointCreate``.
``mechanism``
    The mechanism token the case claims to measure.

The mapping below is derived from the HCOMM public headers and the engine /
protocol support tables in the HCOMM API reference, not from channel class
names:

* ``CommEngine`` and ``CommProtocol`` are separate enumerations
  (``hcomm_res_defs.h``).
* ``URMA`` is **not** a ``CommProtocol``.  The URMA family appears only in
  channel implementation names (``AivUrmaChannel``, ``AicpuTsUrmaChannel``,
  ``CcuUrmaChannel``, ``HostCpuUrmaChannel``).  ``UB_CTP`` and ``UB_RTP`` are
  siblings of ``UB_MEM`` inside ``CommProtocol``; they are not URMA subtypes.
* ``CcuUrmaChannel`` requires ``COMM_PROTOCOL_UB_CTP`` (or ``UBC_TP``) at
  channel creation, so a CCU run labelled "URMA" is a URMA-family channel over
  the UB CTP protocol.  The result records both facts separately.

Because ``URMA`` is an implementation family rather than a protocol, the URMA
cases and their protocol cases request the same engine/protocol pair and are
distinguished by what they verify: the URMA case verifies the channel family,
the CTP/RTP case verifies the protocol.  Two claims, one descriptor, no
invented enum value.
"""
from dataclasses import dataclass

ENDPOINT_DEVICE = "DEVICE"
ENDPOINT_HOST = "HOST"


@dataclass(frozen=True)
class TransportProfile:
    case_id: str
    engine: str
    protocol: str
    endpoint: str
    mechanism: str
    execution_resource: str
    channel_hint: str | None
    note: str

    @property
    def request(self):
        return {"engine": self.engine, "protocol": self.protocol,
                "endpoint": self.endpoint, "mechanism": self.mechanism}


PROFILES = {
    "LINK-001": TransportProfile(
        "LINK-001", "AIV", "UB_MEM", ENDPOINT_DEVICE, "ub_memory", "AIV",
        "AivUbMemChannel", "AIV engine, UB_MEM protocol (memory-semantics channel)"),
    "LINK-002": TransportProfile(
        "LINK-002", "AIV", "UB_CTP", ENDPOINT_DEVICE, "urma", "AIV",
        "AivUrmaChannel", "AIV engine, URMA-family channel over UB_CTP queue pairs"),
    "LINK-003": TransportProfile(
        "LINK-003", "AIV", "UB_CTP", ENDPOINT_DEVICE, "ub_ctp", "AIV",
        "AivUrmaChannel", "AIV engine, UB_CTP protocol (same descriptor as LINK-002)"),
    "LINK-004": TransportProfile(
        "LINK-004", "AIV", "UB_RTP", ENDPOINT_DEVICE, "ub_rtp", "AIV",
        None, "AIV engine, UB_RTP protocol"),
    "LINK-005": TransportProfile(
        "LINK-005", "AICPU_TS", "UB_CTP", ENDPOINT_DEVICE, "urma", "AICPU_TS",
        "AicpuTsUrmaChannel", "AICPU+TS engine, URMA-family channel over UB_CTP"),
    "LINK-006": TransportProfile(
        "LINK-006", "AICPU_TS", "UB_CTP", ENDPOINT_DEVICE, "ub_ctp", "AICPU_TS",
        "AicpuTsUrmaChannel", "AICPU+TS engine, UB_CTP protocol (same descriptor as LINK-005)"),
    "LINK-007": TransportProfile(
        "LINK-007", "AICPU_TS", "UB_RTP", ENDPOINT_DEVICE, "ub_rtp", "AICPU_TS",
        "AicpuTsUbRtpChannel", "AICPU+TS engine, UB_RTP protocol"),
    "LINK-008": TransportProfile(
        "LINK-008", "CCU", "UB_CTP", ENDPOINT_DEVICE, "urma", "CCU",
        "CcuUrmaChannel", "CCU engine; HCOMM requires UB_CTP for CCU channels"),
    "LINK-009": TransportProfile(
        "LINK-009", "AICPU_TS", "HCCS", ENDPOINT_DEVICE, "hccs", "AICPU_TS",
        "AicpuTsHccsChannel", "HCCS protocol; only an A2/A3 device endpoint supports HCCS"),
    "LINK-010": TransportProfile(
        "LINK-010", "AICPU_TS", "ROCE", ENDPOINT_HOST, "roce", "AICPU_TS",
        "AicpuTsRoceHostLocChannel", "RoCE over a host-located endpoint"),
    "LINK-011": TransportProfile(
        "LINK-011", "AICPU_TS", "UBOE", ENDPOINT_DEVICE, "uboe", "AICPU_TS",
        "AicpuTsUboeChannel", "UBoE protocol over a device-located endpoint"),
}

#: Documented engine -> protocol support (HCOMM API reference, ``HcommChannelCreate``)
#: for Ascend 950PR/950DT.  Used only to explain *why* a probe result was
#: expected; the probe result itself always wins.
ENGINE_PROTOCOLS_950 = {
    "CPU": ("ROCE", "UB_CTP"),
    "CPU_TS": ("ROCE", "UB_CTP"),
    "AICPU_TS": ("UBOE", "UB_CTP", "UB_RTP", "ROCE"),
    "AIV": ("UB_CTP", "UB_MEM", "ROCE"),
    "CCU": ("UB_CTP", "UBC_TP"),
}

#: Documented endpoint protocol support (``HcommEndpointCreate``).
ENDPOINT_PROTOCOLS = {
    "950": {"DEVICE": ("UB_CTP", "UB_MEM", "PCIE", "UBOE", "UB_RTP"),
            "HOST": ("ROCE", "UB_CTP")},
    "A3": {"DEVICE": ("ROCE", "HCCS"), "HOST": ()},
    "A2": {"DEVICE": ("ROCE", "HCCS"), "HOST": ()},
    "unknown": {"DEVICE": (), "HOST": ()},
}


def soc_family(soc):
    """Map a device string to the documented support-table family."""
    if not soc:
        return "unknown"
    text = str(soc).casefold()
    if "950" in text or "960" in text:
        return "950"
    if "a3" in text or "910_93" in text:
        return "A3"
    if "a2" in text or "910b" in text or "910_9" in text:
        return "A2"
    return "unknown"


def documented_support(case_id, family):
    """Return ``(supported, reason)`` from the documented tables, for explanation."""
    profile = PROFILES[case_id]
    endpoints = ENDPOINT_PROTOCOLS.get(family, ENDPOINT_PROTOCOLS["unknown"])
    if profile.protocol not in endpoints.get(profile.endpoint, ()):
        return False, (f"{profile.protocol} is not a documented {profile.endpoint}-side "
                       f"endpoint protocol for {family}")
    if family == "950":
        allowed = ENGINE_PROTOCOLS_950.get(profile.engine, ())
        if profile.protocol not in allowed:
            return False, (f"{profile.engine} does not list {profile.protocol} in the "
                           f"documented 950 engine/protocol table")
    elif profile.engine not in ("AICPU_TS", "AICPU"):
        return False, f"{family} only documents the AICPU_TS engine for channel creation"
    return True, None


def profile(case_id):
    try:
        return PROFILES[case_id]
    except KeyError:
        raise KeyError(f"{case_id} has no native transport profile") from None
