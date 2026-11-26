"""SHMEM MTE baseline: host-stream ``shmem.core.put`` over the MTE engine.

``aclshmem OpEngineType.MTE`` moves data through the MTE engine and exposes only
a host-stream API.  Multiple outstanding requests are therefore expressed as
several ``put`` calls enqueued on one stream before the synchronisation point;
there is no per-QP control on this path (see the SHMEM source: the Python
binding wraps ``aclshmemx_putmem_on_stream`` only).
"""
from baselines.l0.contract import Prepared
from baselines.l0.shmem import runtime as shmem_runtime
from baselines.l0.verification.path_verifier import Evidence
from baselines.l0.verification.vocabulary import requested
from common.backend import BenchmarkUnavailable

MECHANISM = "shmem_mte"


def prepare(ctx, case, size, params):
    if ctx.size < 2:
        raise BenchmarkUnavailable("unsupported", "SHMEM MTE baseline needs at least two ranks")
    world = ctx.size
    peers = list(range(world))
    session = shmem_runtime.Runtime(ctx, "mte", size * 2 + (1 << 20),
                                    [size] + [size] * world)
    try:
        source = session.buffers[0]
        destinations = session.buffers[1:]
        session.fill(source, shmem_runtime.pattern(ctx.rank))
        for buffer in destinations:
            session.fill(buffer, 0)
    except BaseException:
        session.close()
        raise

    outstanding = max(1, int(params.get("outstanding") or 1))
    peer_count = max(1, min(int(params.get("peer_count") or (world - 1)), world - 1))
    targets = [(peer, destinations[peer]) for peer in peers if peer != ctx.rank][:peer_count]

    def run():
        for _ in range(outstanding):
            for peer, buffer in targets:
                session.core.put(buffer, source, peer, session.stream)
        return None

    def wait(_):
        session._checked(session.acl.rt.synchronize_stream(session.stream), "synchronize_stream")

    def validate():
        session._checked(session.acl.rt.synchronize_stream(session.stream), "synchronize_stream")
        ctx.barrier()
        expectations = {peer: shmem_runtime.pattern(peer) for peer, _ in targets}
        checked = 0
        for peer, buffer in targets:
            head = session.read(buffer, 0, 1)
            tail = session.read(buffer, size - 1, 1)
            if head != bytes([expectations[peer]]) or tail != bytes([expectations[peer]]):
                raise CorrectnessError(f"SHMEM MTE payload from rank {peer} is corrupt")
            checked += 2
        return {"checked": "first_and_last_byte", "bytes_checked": checked,
                "peers": len(targets), "outstanding": outstanding,
                "source_commit": shmem_runtime.SOURCE_COMMIT}

    evidence = [
        Evidence(source="native_descriptor",
                 claim={"execution_resource": "HOST_CPU", "transport_protocol": MECHANISM},
                 detail="aclshmem_init(engine=MTE) + shmem.core.put -> aclshmemx_putmem_on_stream"),
    ]
    payload = {"per_rank_bytes": size * len(targets) * outstanding,
               "global_bytes": size * len(targets) * outstanding * world,
               "unit": "bytes", "scope": "per_rank_and_global", "direction": "one_way",
               "definition": "each outstanding pass sends one copy to every target peer"}
    return Prepared(run=run, wait=wait, validate=validate, cleanup=session.close,
                    request=request(case, params), evidence=evidence, payload=payload)


def request(case, params):
    return requested(backend="shmem", operation=case.operation,
                     execution_resource="HOST_CPU", transport_protocol=MECHANISM,
                     topology_mode=params.get("topology_mode"))
