"""Completion mechanisms for L0 communication, and what each one proves.

The refactor exists because these four things were being treated as one:

1. the driver accepted the request (``submitted``);
2. a local fence or drain returned (``fenced``);
3. the local engine reported the operation finished (``local_complete``);
4. the peer can read the data (``remote_visible``).

Only (3) and (4) may sit inside a bandwidth denominator.  The old transport driver
called ``HcclChannelQuiet`` after a host-NBI transfer and treated it as the transfer
completion: ``quiet`` is a *fence* on the local channel, so a sample could end while
the remote write was still in flight.  That produces a latency that looks better than
it is and a bandwidth that is simply wrong.

Each mechanism therefore declares a :class:`CompletionContract`.  The runner records
it next to the number (``result["completion"]``), and a case whose contract does not
prove the transfer is reported as such instead of as a completed transfer.
"""
from dataclasses import dataclass

#: Ordered from weakest to strongest.
LEVELS = ("submitted", "fenced", "local_complete", "remote_visible")


@dataclass(frozen=True)
class CompletionContract:
    """What one completion mechanism proves, and how to check it."""

    name: str
    level: str
    #: True when ``wait`` returning implies the payload reached the peer.
    proves_transfer: bool
    detail: str
    #: True when the case itself must additionally read the peer's data (the
    #: remote visibility is established by validation, not by the primitive).
    requires_peer_validation: bool = True

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"completion level {self.level!r} is outside {LEVELS}")
        if self.proves_transfer and self.level in ("submitted", "fenced"):
            raise ValueError(
                f"{self.name}: a {self.level} primitive cannot prove the transfer landed")

    @property
    def usable_for_bandwidth(self):
        return self.proves_transfer

    def as_dict(self):
        return {"mechanism": self.name, "level": self.level,
                "proves_transfer": self.proves_transfer,
                "requires_peer_validation": self.requires_peer_validation,
                "usable_for_bandwidth": self.usable_for_bandwidth,
                "detail": self.detail}


#: The HCOMM host-NBI transfer completion: a channel fence.  Ordering only.
HOST_NBI_FENCE = CompletionContract(
    "host_nbi_fence", "fenced", False,
    "HcclChannelFence orders the local channel; it does not show that the remote "
    "write landed, so it cannot terminate a timed transfer")

#: An explicit drain of submitted NBI requests on the local engine.
HOST_NBI_DRAIN = CompletionContract(
    "host_nbi_drain", "fenced", False,
    "draining submitted NBI requests proves the local queue is empty, not that the "
    "peer observed the data")

#: A peer acknowledgement: rank A writes, rank B raises a flag it read afterwards,
#: A waits for the flag.  This is the only host-side path that proves visibility.
NOTIFY_ROUND_TRIP = CompletionContract(
    "notify_round_trip", "remote_visible", True,
    "the peer sets an acknowledgement only after observing the payload, so the "
    "local wait cannot return before the data is visible")

#: AIV device data plane: the kernel drains its own writes before it exits, so the
#: kernel epilogue is a real completion of the device-side transfer.
AIV_DRAIN = CompletionContract(
    "aiv_drain", "remote_visible", True,
    "the AIV kernel drains its write_nbi requests before signalling completion",
    requires_peer_validation=True)

#: CCU device data plane: completion is reported by the CCU scheduler.
CCU_COMPLETION = CompletionContract(
    "ccu_completion", "remote_visible", True,
    "the CCU kernel reports completion after its remote-memory writes retire",
    requires_peer_validation=True)

#: A CPU engine that returns as soon as the request is queued.
CPU_ENGINE_QUEUED = CompletionContract(
    "cpu_engine_queued", "submitted", False,
    "the CPU engine returns after queueing the request; an immediate return is not "
    "evidence that the transfer completed")

#: A torch.distributed collective: the call returns after the HCCL operation
#: completed on this rank, so the local buffer holds the peer data.
TORCH_COLLECTIVE = CompletionContract(
    "torch_collective", "remote_visible", True,
    "the collective returns after the HCCL operation completed on this rank")

#: Frame-level completion of a synchronous socket/protocol exchange.
SYNC_EXCHANGE = CompletionContract(
    "sync_exchange", "remote_visible", True,
    "the exchange is request/response: the reply cannot arrive before the peer "
    "processed the payload")

BY_NAME = {contract.name: contract for contract in (
    HOST_NBI_FENCE, HOST_NBI_DRAIN, NOTIFY_ROUND_TRIP, AIV_DRAIN, CCU_COMPLETION,
    CPU_ENGINE_QUEUED, TORCH_COLLECTIVE, SYNC_EXCHANGE)}


def contract(name):
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown completion mechanism {name!r}; "
                       f"expected one of {sorted(BY_NAME)}") from None


def for_engine(engine, *, device_plane=False):
    """The completion mechanism a given execution resource uses.

    ``device_plane`` says whether the transfer is issued by a compiled device
    kernel.  Without one, an ``AIV``/``CCU`` request degrades to the host NBI path,
    and the *fence* contract applies -- which is precisely why such a case may not
    be reported as device-plane completion.
    """
    if device_plane:
        return {"AIV": AIV_DRAIN, "CCU": CCU_COMPLETION}.get(engine, AIV_DRAIN)
    if engine in ("AIV", "CCU"):
        return HOST_NBI_FENCE
    if engine == "HOST_CPU":
        return CPU_ENGINE_QUEUED
    return HOST_NBI_DRAIN
