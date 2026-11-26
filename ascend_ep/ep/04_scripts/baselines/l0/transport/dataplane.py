"""Data-plane backends: who actually moves the bytes.

The transport driver used to take an ``engine`` parameter, bind the channel to that
engine, and then submit the transfer itself through the *host* NBI path.  A case
labelled ``AIV`` or ``CCU`` therefore measured the host submission path and reported
it as a device data plane, which is the single largest attribution error in L0.

This module makes the data plane an explicit object with its own lifecycle::

    DataPlaneBackend
        initialize(context)   # control plane: endpoint/channel/buffers
        prepare(workload)     # build the transfer description
        submit(workload)      # device or host submission
        wait(handle)          # completion, per the declared contract
        validate(workload)    # data check
        finalize()            # release

Three implementations exist:

``HostDataPlane``
    the HCOMM host NBI path.  Real, and correctly labelled: its completion is a
    fence (:data:`completion.HOST_NBI_FENCE`), so it may not claim device-plane
    completion.
``AIVDataPlane`` / ``CCUDataPlane``
    the ASC-COMM device kernels.  Their host side is complete (the transfer
    descriptor, the ctypes ABI and the completion contract), but the device program
    must be built by ``kernels/asc_comm/build.sh``.  When the artifacts are missing
    they raise ``dependency_missing``/``not_implemented`` with the exact blocker and
    **never** fall back to the host path.

The distinction is not cosmetic: an ``AIV`` case whose kernel is not built reports
``dependency_missing`` instead of producing a plausible-looking number from the host
NBI path.
"""
import abc
import ctypes
from dataclasses import dataclass, field

from common.backend import BenchmarkUnavailable

from . import completion as completions
from . import kernel_artifacts

#: Transfer flags understood by the ASC-COMM ABI (see kernels/asc_comm/common/abi.h).
FLAG_NOTIFY = 1 << 0
FLAG_DRAIN = 1 << 1


class AscCommTransfer(ctypes.Structure):
    """Mirror of ``asc_comm_transfer`` in ``kernels/asc_comm/common/abi.h``."""

    _fields_ = [
        ("src", ctypes.c_uint64),
        ("dst", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("peer", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


@dataclass
class Transfer:
    """One remote write the data plane has to perform."""

    peer: int
    source_offset: int
    destination_offset: int
    length: int
    notify: bool = True

    def as_struct(self, source_base, destination_base):
        return AscCommTransfer(
            src=source_base + self.source_offset,
            dst=destination_base + self.destination_offset,
            bytes=self.length,
            peer=self.peer,
            flags=FLAG_NOTIFY | (FLAG_DRAIN if self.notify else 0),
        )


@dataclass
class TransferBatch:
    """The prepared unit of work for one submission."""

    transfers: list
    source_base: int
    destination_base: int
    metadata: dict = field(default_factory=dict)

    def structures(self):
        array = (AscCommTransfer * len(self.transfers))()
        for index, transfer in enumerate(self.transfers):
            array[index] = transfer.as_struct(self.source_base, self.destination_base)
        return array

    @property
    def bytes(self):
        return sum(item.length for item in self.transfers)


class DataPlaneBackend(abc.ABC):
    """Interface every data plane implements."""

    engine = None
    kind = None

    def __init__(self, config=None):
        self.config = config or {}
        self.initialized = False

    # -- lifecycle ------------------------------------------------------
    @abc.abstractmethod
    def initialize(self, context):
        """Control plane: acquire the resources the data plane needs."""

    @abc.abstractmethod
    def prepare(self, workload):
        """Turn a workload into a :class:`TransferBatch`."""

    @abc.abstractmethod
    def submit(self, workload):
        """Start the transfer; return the completion handle."""

    @abc.abstractmethod
    def wait(self, handle):
        """Wait for the declared completion guarantee."""

    def validate(self, workload):
        """Read back the destination; the caller supplies the comparison."""
        return workload

    def finalize(self):
        self.initialized = False

    # -- description ----------------------------------------------------
    @property
    @abc.abstractmethod
    def completion(self):
        """The :class:`~.completion.CompletionContract` this plane provides."""

    def blocker(self):
        """``(status, reason)`` when this plane cannot run, else ``None``."""
        return None

    def evidence(self):
        return {"data_plane": self.kind, "engine": self.engine,
                "completion": self.completion.as_dict()}


class HostDataPlane(DataPlaneBackend):
    """The HCOMM host NBI path.

    Submission and completion are host calls.  ``wait`` drains the channel, which is
    a fence: the contract says so and the runner records it.
    """

    kind = "host"

    def __init__(self, config=None, session=None, engine="HOST_CPU"):
        super().__init__(config)
        self.session = session
        self.engine = engine

    @property
    def completion(self):
        return completions.HOST_NBI_DRAIN

    def initialize(self, context):
        if self.session is None:
            raise BenchmarkUnavailable(
                "environment_unavailable",
                "the host data plane needs an initialised native bridge session")
        self.initialized = True
        return self

    def prepare(self, workload):
        return workload

    def submit(self, workload):
        return workload

    def wait(self, handle):
        quiet = self.session.quiet()
        if not quiet.get("ok"):
            raise RuntimeError(f"host NBI drain failed: {quiet}")
        return quiet


class _DeviceDataPlane(DataPlaneBackend):
    """Shared host side of a compiled ASC-COMM device data plane."""

    library: ctypes.CDLL | None = None

    def __init__(self, config=None, engine_id=0):
        super().__init__(config)
        self.engine_id = engine_id
        self.artifacts = kernel_artifacts.probe(self.engine)
        self.functions = {}

    # -- gating ---------------------------------------------------------
    def blocker(self):
        if not self.artifacts.available:
            return (self.artifacts.status,
                    self.artifacts.reason or f"the {self.engine} ASC-COMM kernel is unavailable")
        return None

    def initialize(self, context):
        blocker = self.blocker()
        if blocker is not None:
            # No fallback: a device case whose kernel is missing is a blocked
            # dependency, not a host-path measurement.
            raise BenchmarkUnavailable(*blocker)
        try:
            self.library = ctypes.CDLL(str(self.artifacts.library))
        except OSError as exc:
            raise BenchmarkUnavailable(
                "dependency_missing",
                f"cannot load the {self.engine} kernel library: {exc}") from exc
        for name in kernel_artifacts.REQUIRED_SYMBOLS[self.engine]:
            self.functions[name] = getattr(self.library, name)
        self.initialized = True
        return self

    def prepare(self, workload):
        return workload

    def submit(self, workload):
        raise BenchmarkUnavailable(
            "not_implemented",
            f"the {self.engine} data-plane submission is not wired to the benchmark "
            f"buffers yet; the kernel artifact exists but the case-level buffer "
            f"registration has not been executed on hardware")

    def wait(self, handle):
        raise BenchmarkUnavailable(
            "not_implemented",
            f"the {self.engine} completion symbol is resolved but has never been "
            f"exercised on hardware")

    def validate(self, workload):
        return workload

    def evidence(self):
        data = super().evidence()
        data["kernel"] = kernel_artifacts.build_evidence(self.engine, self.artifacts)
        data["verified_on_device"] = False
        return data


class AIVDataPlane(_DeviceDataPlane):
    """The ASC-COMM AIV kernel: ``write_nbi`` plus a device-side drain."""

    engine = "AIV"
    kind = "aiv"

    @property
    def completion(self):
        return completions.AIV_DRAIN


class CCUDataPlane(_DeviceDataPlane):
    """The ASC-COMM CCU kernel: remote-memory writes driven by the CCU scheduler."""

    engine = "CCU"
    kind = "ccu"

    @property
    def completion(self):
        return completions.CCU_COMPLETION


PLANES = {"host": HostDataPlane, "aiv": AIVDataPlane, "ccu": CCUDataPlane}

#: Execution resource -> data-plane kind.  This mapping is the whole point: an
#: ``AIV``/``CCU`` execution resource may not be served by the host plane.
KIND_FOR_RESOURCE = {"HOST_CPU": "host", "AICPU_TS": "host", "AIV": "aiv", "CCU": "ccu"}


def kind_for(execution_resource):
    try:
        return KIND_FOR_RESOURCE[execution_resource]
    except KeyError:
        raise BenchmarkUnavailable(
            "unsupported",
            f"no data plane for execution resource {execution_resource!r}") from None


def for_resource(execution_resource, config=None, session=None):
    """Build the data plane that must serve ``execution_resource``."""
    kind = kind_for(execution_resource)
    if kind == "host":
        return HostDataPlane(config, session=session, engine=execution_resource)
    plane = PLANES[kind](config)
    plane.engine_id = 0
    return plane


def describe(execution_resource, config=None):
    """Non-initialising description, used by the capability probe and reports."""
    plane = for_resource(execution_resource, config)
    blocker = plane.blocker()
    return {"execution_resource": execution_resource, "kind": plane.kind,
            "completion": plane.completion.as_dict(),
            "available": blocker is None,
            "blocker": None if blocker is None else {"status": blocker[0], "reason": blocker[1]}}
