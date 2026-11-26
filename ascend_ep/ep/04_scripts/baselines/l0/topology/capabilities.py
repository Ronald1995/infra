"""Capability matrix that decides which L0 cases can exist on this machine.

Three kinds of support are distinguished and never mixed:

``declared``
    A case names a channel/engine in its source.  This is a hypothesis.

``probed``
    A local probe succeeded: the native bridge described the channel it was
    compiled with, the HCCL test binary listed an accelerator mode, the SHMEM
    package is importable, or a helper library was found.

``rejected``
    A probe ran and said no.  The reason is kept verbatim and is reported.

Only ``probed`` entries are marked ``supported``.  Everything else is reported as
unsupported with the reason the probe produced, so a run can never silently
substitute another channel.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

from baselines.l0.topology import device_info, rank_graph

SCHEMA_VERSION = 1

#: Channels named by the L0 methodology.  ``channel`` is ``None`` when the class
#: name must be resolved from the installed HCOMM/ASC-COMM build rather than
#: guessed here; the native bridge reports the real name it found.
CHANNEL_CATALOG = (
    {"case_id": "LINK-001", "channel": "AivUbMemChannel", "backend": "hcomm",
     "execution_resource": "AIV", "mechanisms": ("ub_memory",)},
    {"case_id": "LINK-002", "channel": "AivUrmaChannel", "backend": "hcomm",
     "execution_resource": "AIV", "mechanisms": ("urma",)},
    {"case_id": "LINK-003", "channel": None, "backend": "hcomm",
     "execution_resource": "AIV", "mechanisms": ("ub_ctp",)},
    {"case_id": "LINK-004", "channel": None, "backend": "hcomm",
     "execution_resource": "AIV", "mechanisms": ("ub_rtp",)},
    {"case_id": "LINK-005", "channel": "AicpuTsUrmaChannel", "backend": "hcomm",
     "execution_resource": "AICPU_TS", "mechanisms": ("urma",)},
    {"case_id": "LINK-006", "channel": None, "backend": "hcomm",
     "execution_resource": "AICPU_TS", "mechanisms": ("ub_ctp",)},
    {"case_id": "LINK-007", "channel": None, "backend": "hcomm",
     "execution_resource": "AICPU_TS", "mechanisms": ("ub_rtp",)},
    {"case_id": "LINK-008", "channel": "CcuUrmaChannel", "backend": "hcomm",
     "execution_resource": "CCU", "mechanisms": ("urma",)},
    {"case_id": "LINK-009", "channel": "AicpuTsHccsChannel", "backend": "hcomm",
     "execution_resource": "AICPU_TS", "mechanisms": ("hccs",)},
    {"case_id": "LINK-010", "channel": "AicpuTsRoceChannel", "backend": "hcomm",
     "execution_resource": "AICPU_TS", "mechanisms": ("roce",)},
    {"case_id": "LINK-011", "channel": None, "backend": "hcomm",
     "execution_resource": "AICPU_TS", "mechanisms": ("uboe",)},
    {"case_id": "LINK-012", "channel": "shmem:MTE", "backend": "shmem",
     "execution_resource": "HOST_CPU", "mechanisms": ("shmem_mte",)},
    {"case_id": "LINK-013", "channel": "shmem:UDMA", "backend": "shmem",
     "execution_resource": "AIV", "mechanisms": ("shmem_udma",)},
    # Named by the methodology but not yet bound to a case; reported so the
    # matrix shows the full channel surface HCOMM may expose.
    {"case_id": None, "channel": "HostCpuUrmaChannel", "backend": "hcomm",
     "execution_resource": "HOST_CPU", "mechanisms": ("urma",)},
)

SHMEM_ENGINES = (
    {"engine": "shmem_mte", "execution_resource": "HOST_CPU", "mechanisms": ("shmem_mte",),
     "requires": ("shmem_package",), "source": "aclshmem OpEngineType MTE"},
    {"engine": "shmem_mte_device", "execution_resource": "AIV", "mechanisms": ("shmem_mte",),
     "requires": ("shmem_package", "udma_library"), "source": "aclshmem MTE device path"},
    {"engine": "shmem_udma_host", "execution_resource": "HOST_CPU", "mechanisms": ("shmem_udma",),
     "requires": ("shmem_package",), "source": "aclshmem OpEngineType UDMA host stream"},
    {"engine": "shmem_udma_device", "execution_resource": "AIV", "mechanisms": ("shmem_udma",),
     "requires": ("shmem_package", "udma_library"), "source": "aclshmemx_udma_* device kernel"},
)

MEMORY_KERNELS = (
    {"case_id": "MEM-002", "engine": "mte_local", "library": "mem_library"},
    {"case_id": "MEM-003", "engine": "ub_local", "library": "mem_library"},
)


@dataclass
class Probe:
    name: str
    ok: bool
    reason: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        return {"name": self.name, "ok": self.ok, "reason": self.reason, "detail": self.detail}


def _bridge(config):
    from baselines.l0.transport import bridge
    return bridge


def probe_transport_bridge(config):
    """Load the native transport bridge and read its self-description."""
    try:
        module = _bridge(config)
    except ImportError as exc:  # pragma: no cover - packaging error path
        return Probe("transport_bridge", False, f"bridge module unavailable: {exc}")
    try:
        descriptor = module.describe(config)
    except module.BridgeUnavailable as exc:
        return Probe("transport_bridge", False, str(exc), {"status": exc.status})
    return Probe("transport_bridge", True, None, descriptor)


def probe_native_hccl(config):
    from baselines.l0.collectives import accelerator
    return accelerator.probe(config)


def probe_shmem(config):
    import importlib.util
    spec = importlib.util.find_spec("shmem")
    if spec is None:
        return Probe("shmem", False, "python package 'shmem' is not installed")
    library = config.get("benchmark", {}).get("shmem_udma_library")
    return Probe("shmem", True, None, {
        "module_path": getattr(spec, "origin", None),
        "udma_library": library or None,
        "udma_library_present": bool(library) and __import__("os").path.isfile(library),
        "mte_available": True,
        "udma_available": bool(library) and __import__("os").path.isfile(library),
    })


def probe_memory_library(config):
    import os
    library = config.get("benchmark", {}).get("memory_library")
    if not library:
        return Probe("memory_library", False,
                     "benchmark.memory_library is not configured; build baselines/l0/memory/native")
    if not os.path.isfile(library):
        return Probe("memory_library", False, f"configured memory library does not exist: {library}")
    return Probe("memory_library", True, None, {"library": library})


MECHANISM_FOR_PROTOCOL = {
    "UB_MEM": "ub_memory", "UB_CTP": "ub_ctp", "UBC_TP": "ub_ctp", "UB_RTP": "ub_rtp",
    "ROCE": "roce", "HCCS": "hccs", "HCCS_ONLY": "hccs", "UBOE": "uboe",
    "PCIE": "pcie", "SIO": "sio",
}


def mechanisms_for_protocols(protocols):
    """Translate the bridge's CommProtocol names into mechanism tokens."""
    return tuple(MECHANISM_FOR_PROTOCOL[name] for name in protocols
                 if name in MECHANISM_FOR_PROTOCOL)


def _channel_support(entry, descriptor, probes):
    """Resolve one catalog entry against the bridge/shared-library probes."""
    reasons, evidence = [], []
    supported = False
    channel = entry["channel"]
    if entry["backend"] == "hcomm":
        probe = probes.get("transport_bridge")
        if probe is None or not probe.ok:
            reason = probe.reason if probe else "transport bridge probe did not run"
            return {"supported": False, "confidence": "declared", "channel": channel,
                    "reason": reason, "evidence": ["transport_bridge"]}
        channels = {item.get("channel"): item for item in descriptor.get("channels", ())}
        match = None
        if channel in channels:
            match = channels[channel]
        else:
            for item in descriptor.get("channels", ()):
                observed = set(mechanisms_for_protocols(item.get("mechanisms", ())))
                if observed & set(entry["mechanisms"]) and (
                        item.get("execution_resource") == entry["execution_resource"]):
                    match = item
                    break
        if match is None:
            reason = (f"{entry['backend']} bridge did not report a channel for "
                      f"{', '.join(entry['mechanisms'])} on {entry['execution_resource']}")
            return {"supported": False, "confidence": "probed", "channel": channel,
                    "reason": reason, "evidence": ["transport_bridge"]}
        supported = True
        evidence.append("transport_bridge:l0_bridge_describe")
        if match.get("channel") and match["channel"] != channel:
            channel = match["channel"]
    elif entry["backend"] == "shmem":
        probe = probes.get("shmem")
        if probe is None or not probe.ok:
            return {"supported": False, "confidence": "declared", "channel": channel,
                    "reason": probe.reason if probe else "shmem probe did not run",
                    "evidence": ["shmem_package"]}
        engine = "MTE" if entry["mechanisms"] == ("shmem_mte",) else "UDMA"
        if engine == "UDMA" and not probe.detail.get("udma_available"):
            return {"supported": False, "confidence": "probed", "channel": channel,
                    "reason": ("SHMEM UDMA device kernel library not built; run "
                               "baselines/l0/shmem/native/build_udma.sh and set "
                               "benchmark.shmem_udma_library"),
                    "evidence": ["shmem_package"]}
        supported = True
        evidence.extend(["shmem_package", f"OpEngineType.{engine}"])
    else:
        reasons.append(f"unknown backend {entry['backend']!r}")
    return {"supported": supported, "confidence": "probed" if supported else "declared",
            "channel": channel, "reason": None if supported else "; ".join(reasons) or None,
            "evidence": evidence}


def _accelerator_entries(probe):
    entries = []
    modes = (probe.detail or {}).get("modes") if probe.ok else None
    for mode in (modes or ()):
        entries.append({
            "mode": mode, "execution_resource": mode.get("execution_resource"),
            "supported": bool(mode.get("supported", True)),
            "reason": mode.get("reason"),
            "evidence": mode.get("evidence", []),
        })
    return entries


def discover(config=None, override=None, probe_runtime=False):
    """Build the capability matrix.

    ``override`` replaces the probe result set.  It exists so unit tests and air
    gapped runs can inject a documented matrix instead of pretending a device was
    probed; the override is echoed in ``probes`` as ``source: override``.
    """
    config = config or {}
    if override is not None:
        matrix = dict(override)
        matrix.setdefault("schema_version", SCHEMA_VERSION)
        matrix["source"] = "override"
        return matrix

    probes = {
        "transport_bridge": probe_transport_bridge(config),
        "native_hccl": probe_native_hccl(config),
        "shmem": probe_shmem(config),
        "memory_library": probe_memory_library(config),
    }
    descriptor = probes["transport_bridge"].detail if probes["transport_bridge"].ok else {}
    devices = device_info.summary(probe_npu=probe_runtime)
    graph = rank_graph.build(config or {"distributed": {"ep_size": 1, "tp_size": 1},
                                        "topology": {}})
    channels = []
    for entry in CHANNEL_CATALOG:
        resolution = _channel_support(entry, descriptor, probes)
        channels.append({
            "case_id": entry["case_id"],
            "channel": resolution["channel"],
            "backend": entry["backend"],
            "execution_resource": entry["execution_resource"],
            "mechanisms": list(entry["mechanisms"]),
            "supported": resolution["supported"],
            "confidence": resolution["confidence"],
            "reason": resolution["reason"],
            "evidence": resolution["evidence"],
        })
    engines = []
    for entry in SHMEM_ENGINES:
        shmem = probes["shmem"]
        missing = [name for name in entry["requires"]
                   if (name == "shmem_package" and not shmem.ok)
                   or (name == "udma_library" and not (shmem.ok and shmem.detail.get("udma_available")))]
        engines.append({
            "engine": entry["engine"], "backend": "shmem",
            "execution_resource": entry["execution_resource"],
            "mechanisms": list(entry["mechanisms"]),
            "supported": not missing,
            "confidence": "probed" if shmem.ok else "declared",
            "reason": None if not missing else f"missing {', '.join(missing)}",
            "evidence": [entry["source"]],
        })
    memory = []
    for entry in MEMORY_KERNELS:
        probe = probes["memory_library"]
        memory.append({"case_id": entry["case_id"], "engine": entry["engine"],
                       "supported": probe.ok, "reason": probe.reason,
                       "confidence": "probed" if probe.ok else "declared",
                       "evidence": ["memory_library"]})
    native = probes["native_hccl"]
    matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "probe",
        "device": devices,
        "topology": graph,
        "channels": channels,
        "engines": engines,
        "accelerators": _accelerator_entries(native),
        "native_hccl": (native.detail if native.ok else {"available": False,
                                                         "reason": native.reason}),
        "memory_kernels": memory,
        "probes": [probe.as_dict() for probe in probes.values()],
        "message_limits": {
            "shmem_udma_single_request": 256 * 1024 * 1024,
            "shmem_udma_source": "aclshmemx_udma_get_nbi/put_nbi: 256 MB per request",
        },
    }
    return matrix


def index(matrix):
    """Index a matrix by case id, channel name, mechanism and accelerator."""
    by_case, by_channel, by_mechanism, by_accelerator = {}, {}, {}, {}
    for entry in matrix.get("channels", ()):
        if entry.get("case_id"):
            by_case[entry["case_id"]] = entry
        if entry.get("channel"):
            by_channel[entry["channel"]] = entry
        for mechanism in entry.get("mechanisms", ()):
            by_mechanism.setdefault(mechanism, []).append(entry)
    for entry in matrix.get("engines", ()):
        for mechanism in entry.get("mechanisms", ()):
            by_mechanism.setdefault(mechanism, []).append(entry)
    for entry in matrix.get("accelerators", ()):
        by_accelerator[entry["mode"]] = entry
    return {"case": by_case, "channel": by_channel, "mechanism": by_mechanism,
            "accelerator": by_accelerator}


def support(matrix, case_id=None, mechanism=None, execution_resource=None, channel=None):
    """Return ``(supported, reason)`` for one combination."""
    entries = index(matrix)
    if channel and channel in entries["channel"]:
        entry = entries["channel"][channel]
        return entry["supported"], entry["reason"]
    if case_id and case_id in entries["case"]:
        entry = entries["case"][case_id]
        return entry["supported"], entry["reason"]
    candidates = entries["mechanism"].get(mechanism, ())
    if execution_resource:
        narrowed = [item for item in candidates
                    if item.get("execution_resource") == execution_resource]
        if narrowed:
            candidates = narrowed
    if candidates:
        if any(item["supported"] for item in candidates):
            return True, None
        return False, candidates[0].get("reason")
    return False, f"no capability entry for mechanism={mechanism!r}"


def write(matrix, path):
    from common.storage import write_json
    write_json(path, matrix)
    return path
