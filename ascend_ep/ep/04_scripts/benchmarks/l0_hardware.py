"""L0 case dispatch.

This module owns routing only: which baseline family a case belongs to, which
parameters it needs, and how its prepared object is returned.  Implementations
live under ``baselines/l0/`` so a new mechanism never grows this file.
"""
from baselines.l0.contract import Prepared
from baselines.l0.verification.vocabulary import requested
from common.cases import l0_family


def l0_case(ctx, case, params, output=None):
    """Return a :class:`~baselines.l0.contract.Prepared` for an L0 case."""
    family = l0_family(case)
    size = params.get("message_bytes") or 4096
    variant = params.get("variant") or "default"
    if family == "transport":
        if case.id in ("HW-007", "HW-008"):
            return _legacy_shmem(ctx, case, size, variant)
        if case.id in ("LINK-012", "LINK-013"):
            from baselines.l0.shmem import prepare_engine
            return prepare_engine(ctx, case, size, params)
        from baselines.l0.transport import channel
        return channel.prepare(ctx, case, size, params)
    if family == "execution":
        from baselines.l0.collectives import native_hccl
        return native_hccl.prepare(ctx, case, params, output)
    if family == "memory":
        if case.id in ("MEM-002", "MEM-003"):
            from baselines.l0.memory import local
            return local.prepare(ctx, case, size, params)
        from baselines.l0.memory import device
        return device.prepare(ctx, case, size, params)
    if family == "overlap":
        if case.id == "HW-012":
            from baselines.l0.overlap import legacy
            return legacy.prepare(ctx, ctx.workload, size)
        from baselines.l0.overlap import runner
        return runner.prepare(ctx, case, ctx.workload, size, params)
    if family == "compute":
        from baselines.l0.compute import torch_ops
        return torch_ops.prepare(ctx, case, ctx.workload, size, variant)
    if family == "collective":
        from baselines.l0.collectives import torch_hccl
        return torch_hccl.prepare(ctx, case, size, variant)
    raise ValueError(f"no L0 handler for case {case.id}")


def hardware_case(ctx, case, work, size, variant):
    """Legacy four-tuple entry point kept for existing callers and tests."""
    ctx.workload = work
    params = {"variant": variant, "message_bytes": size}
    if case.operation.startswith("shmem"):
        prepared = _legacy_shmem(ctx, case, size, variant)
        return prepared.run, prepared.validate, prepared.cleanup, size * ctx.size
    prepared = l0_case(ctx, case, params)
    payload = prepared.payload
    if isinstance(payload, dict):
        payload = payload.get("per_rank_bytes")
    if case.operation.startswith("concurrent"):
        payload = size * (ctx.size - 1)
    return prepared.run, prepared.validate, prepared.cleanup, payload


def _legacy_shmem(ctx, case, size, variant):
    from baselines.l0 import shmem
    run, validate, cleanup = shmem.prepare(ctx, size, variant if variant != "default" else "mte")
    engine = variant if variant in ("mte", "udma") else "mte"
    return Prepared(run=run, validate=validate, cleanup=cleanup,
                    request=requested(backend="shmem", operation="put",
                                      execution_resource="HOST_CPU" if engine == "mte" else "AIV",
                                      transport_protocol=f"shmem_{engine}"),
                    payload=size * ctx.size)
