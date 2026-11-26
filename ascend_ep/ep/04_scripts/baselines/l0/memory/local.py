"""Local on-device memory movement baselines (``MEM-002`` MTE, ``MEM-003`` UB).

Both use a purpose-built Ascend C kernel:

``mte``
    GM -> UB -> GM copy issued through MTE2/MTE3, i.e. what a data-copy unit can
    do without any interconnect involved.  This is a *local* copy, not UB Memory
    remote access.
``ub``
    UB -> UB copy inside one core's unified buffer.

The kernel library is optional: without it the cases report
``dependency_missing`` and say how to build it.
"""
import ctypes

from baselines.l0.contract import Prepared
from baselines.l0.verification.path_verifier import Evidence
from baselines.l0.verification.vocabulary import requested
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError

PATTERN_MAX = 254


def prepare(ctx, case, size, params):
    if ctx.device.type == "cpu":
        raise BenchmarkUnavailable("unsupported", "local copy baselines need an Ascend device")
    library_path = ctx.config["benchmark"].get("memory_library")
    if not library_path:
        raise BenchmarkUnavailable(
            "dependency_missing",
            "benchmark.memory_library is not configured; build "
            "baselines/l0/memory/native/build_mem.sh")
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as exc:
        raise BenchmarkUnavailable("dependency_missing",
                                   f"cannot load benchmark.memory_library: {exc}") from exc
    copy = library.l0_mem_copy
    copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                     ctypes.c_void_p]
    copy.restype = ctypes.c_int

    torch = ctx.torch
    source = torch.full((size,), 1 + ctx.rank % PATTERN_MAX, dtype=torch.uint8,
                        device=ctx.device)
    destination = torch.zeros_like(source)
    mode = 0 if case.id == "MEM-002" else 1
    stream = torch.npu.current_stream().npu_stream if hasattr(torch.npu.current_stream(),
                                                              "npu_stream") else None
    state = {}

    def run():
        status = copy(destination.data_ptr(), source.data_ptr(), size, mode, stream)
        if status != 0:
            raise RuntimeError(f"l0_mem_copy returned {status}")
        return None

    def validate():
        run()
        torch.npu.synchronize(ctx.device)
        head = destination[:64].detach().cpu()
        tail = destination[-64:].detach().cpu()
        expected = 1 + ctx.rank % PATTERN_MAX
        if not (bool((head == expected).all()) and bool((tail == expected).all())):
            raise CorrectnessError(f"{case.id} local copy produced different bytes")
        state["checked"] = True
        return {"checked": "head_and_tail", "mode": "mte" if mode == 0 else "ub"}

    evidence = [Evidence(source="native_descriptor",
                         claim={"execution_resource": "AIV"},
                         detail=("l0_mem_copy mode=GM->UB->GM (MTE2/MTE3)" if mode == 0
                                 else "l0_mem_copy mode=UB->UB"))]
    payload = {"per_rank_bytes": size * 2, "global_bytes": size * 2 * ctx.size,
               "unit": "bytes", "scope": "per_rank_and_global", "direction": "bidirectional",
               "definition": "logical read + write bytes of one local copy"}
    request = requested(backend="native_kernel", operation="copy",
                        execution_resource="AIV", transport_protocol="auto")
    return Prepared(run=run, validate=validate, request=request, evidence=evidence,
                    payload=payload)
