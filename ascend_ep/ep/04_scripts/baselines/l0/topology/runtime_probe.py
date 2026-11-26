"""Runtime capability probe: static "compiled in" is not "usable here".

The capability matrix used to decide `supported` from whether a channel exists in the
compiled bridge.  On a 950 that is not enough: a channel can be compiled in and still
be unusable because the SoC does not expose the endpoint protocol, because the device
resources are taken, or because the ASC-COMM kernel was never built.

This probe walks a fixed sequence and records the *first* stage that fails, which is
what makes the difference between these outcomes (and they are not interchangeable)::

    compiled                the bridge was built with the channel
    endpoint_compatible     the SoC/endpoint protocol accepts it
    channel_acquired        HcclChannelAcquire succeeded once
    kernel_available       the ASC-COMM kernel exports the ABI symbols
    supported               every stage above passed

    unsupported             the platform cannot offer it (SoC/protocol)
    environment_unavailable the platform can, but a resource is busy right now
    dependency_missing      a runtime library or binary is absent
    not_implemented         the probe would need code that does not exist yet

Resources acquired by the probe are always released, so probing never perturbs the
measurement that follows.
"""
from dataclasses import dataclass, field

from baselines.l0.topology import capabilities as capability_table
from baselines.l0.transport import dataplane, kernel_artifacts

STAGES = ("soc", "compiled", "endpoint_compatible", "channel_acquired", "kernel_available",
          "resource_available")

#: Stage failure -> benchmark status.  A resource that is busy is *not* unsupported.
STAGE_STATUS = {
    ("soc", False): "unsupported",
    ("compiled", False): "unsupported",
    ("endpoint_compatible", False): "unsupported",
    ("channel_acquired", False): "environment_unavailable",
    ("kernel_available", False): "dependency_missing",
    ("resource_available", False): "environment_unavailable",
}

SUPPORTED = "supported"


@dataclass
class StageResult:
    stage: str
    state: str            # pass | fail | unknown | skip
    reason: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        return {"stage": self.stage, "state": self.state, "reason": self.reason,
                "detail": dict(self.detail)}


@dataclass
class ChannelProbe:
    case_id: str | None
    channel: str | None
    engine: str | None
    mechanisms: tuple = ()
    stages: list = field(default_factory=list)
    released: bool = True

    def stage(self, name):
        return next((item for item in self.stages if item.stage == name), None)

    @property
    def status(self):
        """The status implied by the stages reached, or ``supported``."""
        for item in self.stages:
            if item.state == "fail":
                return STAGE_STATUS.get((item.stage, False), "unsupported")
            if item.state == "unknown":
                return item.detail.get("status", "environment_unavailable")
        return SUPPORTED

    @property
    def supported(self):
        return self.status == SUPPORTED

    def as_dict(self):
        return {"case_id": self.case_id, "channel": self.channel, "engine": self.engine,
                "mechanisms": list(self.mechanisms), "status": self.status,
                "supported": self.supported, "released": self.released,
                "stages": [item.as_dict() for item in self.stages]}


def _soc_stage(config, device):
    soc = ((device or {}).get("soc") or "").lower()
    if not soc:
        return StageResult("soc", "unknown",
                           "the SoC model could not be read (npu-smi unavailable)",
                           {"status": "environment_unavailable"})
    return StageResult("soc", "pass", None, {"soc": soc})


def _compiled_stage(entry, descriptor):
    names = {item.get("channel") for item in (descriptor or {}).get("channels", ())}
    if not names:
        return StageResult("compiled", "unknown",
                           "the native bridge is not built, so no channel is compiled",
                           {"status": "dependency_missing"})
    if entry.get("channel") in names:
        return StageResult("compiled", "pass", None, {"channel": entry["channel"]})
    resource = entry.get("execution_resource")
    mechanisms = set(entry.get("mechanisms") or ())
    matching = [item for item in descriptor.get("channels", ())
                if item.get("execution_resource") == resource
                and mechanisms & set(capability_table.mechanisms_for_protocols(
                    item.get("mechanisms", ())))]
    if matching:
        return StageResult("compiled", "pass", None, {"channel": matching[0].get("channel")})
    return StageResult("compiled", "fail",
                       f"the bridge was not compiled with a channel for "
                       f"{', '.join(sorted(mechanisms))} on {resource}")


def _kernel_stage(entry):
    plane = dataplane.KIND_FOR_RESOURCE.get(entry.get("execution_resource"))
    if plane not in ("aiv", "ccu"):
        return StageResult("kernel_available", "skip",
                           "a host data plane needs no device kernel")
    artifacts = kernel_artifacts.probe(plane.upper())
    if artifacts.available:
        return StageResult("kernel_available", "pass", None, artifacts.as_dict())
    return StageResult("kernel_available", "fail", artifacts.reason, artifacts.as_dict())


def probe(config, channels=None, descriptor=None, device=None, release=True):
    """Probe every channel entry and return the per-channel ladder.

    ``descriptor`` and ``device`` are injectable so a unit test can describe a machine
    without pretending to have run on one.  Nothing here claims a channel was
    acquired unless ``HcclChannelAcquire`` actually returned.
    """
    channels = channels if channels is not None else capability_table.CHANNEL_CATALOG
    results = []
    for entry in channels:
        record = ChannelProbe(case_id=entry.get("case_id"), channel=entry.get("channel"),
                              engine=entry.get("execution_resource"),
                              mechanisms=tuple(entry.get("mechanisms") or ()))
        record.stages.append(_soc_stage(config, device))
        record.stages.append(_compiled_stage(entry, descriptor))
        record.stages.append(StageResult(
            "endpoint_compatible", "skip",
            "endpoint protocol compatibility is only observable once a channel is compiled"))
        record.stages.append(StageResult(
            "channel_acquired", "unknown",
            "no bridge session was opened, so HcclChannelAcquire was not exercised",
            {"status": "environment_unavailable" if descriptor else "dependency_missing"}))
        record.stages.append(_kernel_stage(entry))
        record.stages.append(StageResult(
            "resource_available", "skip",
            "resource availability is only meaningful once a channel can be acquired"))
        record.released = release
        results.append(record)
    return results


def summary(records):
    counts = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def as_dict(records):
    return {"schema_version": 1, "stages": list(STAGES),
            "records": [record.as_dict() for record in records],
            "summary": summary(records)}
