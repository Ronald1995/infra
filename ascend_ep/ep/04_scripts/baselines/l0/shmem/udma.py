"""SHMEM UDMA baseline: device kernel with multi-peer, multi-core, multi-outstanding.

Two submission paths are supported and reported separately:

``device``
    The Ascend C kernel in ``native/udma_kernel.cpp`` issues
    ``aclshmemx_udma_qp_put_nbi`` per chunk with a per-core QP index and a
    configurable quiet policy.  The host only launches the kernel; submission
    and completion are timed separately.

``host``
    ``shmem.core.put`` with the UDMA engine selected at initialisation.  The
    host-stream API has no QP index or quiet control, so this variant cannot
    sweep outstanding requests; it exists to show the difference between the two
    submission paths for the same engine.

UDMA is a data-movement engine.  It is not URMA, and neither this module nor its
results claim URMA: the evidence records the SHMEM engine and the exact API.
"""
import ctypes

from baselines.l0.contract import Prepared
from common import parameter_evidence
from baselines.l0.shmem import runtime as shmem_runtime
from baselines.l0.verification.path_verifier import Evidence
from baselines.l0.verification.vocabulary import requested
from common.backend import BenchmarkUnavailable

MECHANISM = "shmem_udma"
MAX_REQUEST_BYTES = 256 * 1024 * 1024


def prepare(ctx, case, size, params):
    if ctx.size < 2:
        raise BenchmarkUnavailable("unsupported", "SHMEM UDMA baseline needs at least two ranks")
    variant = params.get("variant") or "device"
    if variant not in ("device", "host"):
        raise BenchmarkUnavailable("unsupported", f"unknown SHMEM UDMA variant {variant!r}")
    world = ctx.size
    outstanding = max(1, int(params.get("outstanding") or 1))
    cores = max(1, int(params.get("cores") or 1)) if variant == "device" else 1
    peer_count = max(1, min(int(params.get("peer_count") or (world - 1)), world - 1))
    quiet_mode = int(params.get("quiet_mode") or 0)
    is_get = 1 if params.get("mode") == "get" else 0
    slot = size
    total = world * slot

    session = shmem_runtime.Runtime(ctx, "udma", total + slot + (1 << 20),
                                    [slot, total], qp_num=cores if variant == "device" else 0)
    source, destination = session.buffers
    try:
        session.fill(source, shmem_runtime.pattern(ctx.rank))
        session.fill(destination, 0)
    except BaseException:
        session.close()
        raise

    if variant == "device":
        library = shmem_runtime.native_library(
            ctx, "shmem_udma_library", "EP_L0_SHMEM_UDMA_LIBRARY",
            "build baselines/l0/shmem/native/build_udma.sh")
        submit = library.ep_l0_udma_submit
        submit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p]
        submit.restype = None

        def run():
            submit(destination.addr, source.addr, slot, ctx.rank, world, peer_count,
                   outstanding, quiet_mode, cores, 1 if cores > 1 else 0, is_get,
                   session.stream)
            return None

        submission = "device_kernel"
        evidence_detail = (f"ep_l0_udma_submit -> aclshmemx_udma_qp_{'get' if is_get else 'put'}_nbi "
                           f"(cores={cores}, outstanding={outstanding}, quiet_mode={quiet_mode})")
    else:
        def run():
            for _ in range(outstanding):
                for index in range(peer_count):
                    peer = (ctx.rank + 1 + index) % world
                    session.core.put(destination, source, peer, session.stream)
            return None

        submission = "host_stream"
        evidence_detail = "aclshmem_init(engine=UDMA) + shmem.core.put -> aclshmemx_putmem_on_stream"

    def wait(_):
        session._checked(session.acl.rt.synchronize_stream(session.stream), "synchronize_stream")

    def validate():
        session._checked(session.acl.rt.synchronize_stream(session.stream), "synchronize_stream")
        ctx.barrier()
        if variant == "device":
            expectations = {peer: shmem_runtime.pattern(peer)
                            for peer in range(world) if peer != ctx.rank}
            result = session.validate_slots(destination, slot, expectations)
        else:
            result = {"checked": "sampled", "bytes_checked": 0, "slots": 0, "slot_bytes": slot}
            for index in range(peer_count):
                peer = (ctx.rank + 1 + index) % world
                block = session.read(destination, 0, 1)
                if block != bytes([shmem_runtime.pattern(peer)]):
                    raise RuntimeError(f"host-stream UDMA payload from rank {peer} is corrupt")
                result["bytes_checked"] += 1
                result["slots"] += 1
        result.update(submission=submission, outstanding=outstanding, cores=cores,
                      peers=peer_count, quiet_mode=quiet_mode,
                      source_commit=shmem_runtime.SOURCE_COMMIT,
                      # These are the values handed to the runtime API on this rank,
                      # not a copy of the configuration: the device path passes both
                      # into ``ep_l0_udma_submit`` and the host path fixes them.
                      dimension_observed=parameter_evidence.observed_from(
                          params, cores=cores, quiet_mode=quiet_mode,
                          outstanding=outstanding, message_bytes=slot))
        return result

    transport = MAX_REQUEST_BYTES
    payload = {"per_rank_bytes": size * peer_count * outstanding,
               "global_bytes": size * peer_count * outstanding * world,
               "unit": "bytes", "scope": "per_rank_and_global", "direction": "one_way",
               "single_request_limit_bytes": transport,
               "definition": "each outstanding pass sends one payload copy per target peer"}
    evidence = [
        Evidence(source="native_descriptor",
                 claim={"execution_resource": "AIV" if variant == "device" else "HOST_CPU",
                        "transport_protocol": MECHANISM},
                 detail=evidence_detail),
    ]
    return Prepared(run=run, wait=wait, validate=validate, cleanup=session.close,
                    request=request(case, params), evidence=evidence, payload=payload)


def request(case, params):
    variant = params.get("variant") or "device"
    return requested(backend="shmem", operation=case.operation,
                     execution_resource="AIV" if variant == "device" else "HOST_CPU",
                     transport_protocol=MECHANISM,
                     topology_mode=params.get("topology_mode"))
