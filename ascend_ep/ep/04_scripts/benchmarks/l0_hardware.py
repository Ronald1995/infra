"""L0 case dispatch.

This module owns routing only: which baseline family a case belongs to, which
parameters it needs, and how its prepared object is returned.  Implementations live
under ``baselines/l0/`` so a new mechanism never grows this file.

Every family exposes a handler named ``l0.<family>`` through ``HANDLERS``; the case
catalog names one of them per case.  The handler returns the same
:class:`~common.prepared.PreparedCase` the L1-L4 handlers return, so the runner has
one execution path -- but L0 keeps its own baseline implementations, its own
evidence and its own payload description, which are carried through unchanged.
"""
from common.prepared import PreparedCase
from baselines.l0.contract import Prepared
from baselines.l0.verification.vocabulary import requested
from common.case_spec import l0_family

#: Families that must be routed by an explicit handler.
FAMILIES = ("memory", "transport", "execution", "overlap", "compute", "collective")


def _baseline(request):
    """Return the raw ``Prepared`` for one L0 request."""
    case = request.case
    params = request.parameters
    size = params.get("message_bytes") or 4096
    variant = params.get("variant") or "default"
    family = l0_family(case)
    if family == "transport":
        if case.id in ("HW-007", "HW-008"):
            return _legacy_shmem(request.ctx, case, size, variant)
        if case.id in ("LINK-012", "LINK-013"):
            from baselines.l0.shmem import prepare_engine
            return prepare_engine(request.ctx, case, size, params)
        from baselines.l0.transport import channel
        return channel.prepare(request.ctx, case, size, params)
    if family == "execution":
        from baselines.l0.collectives import native_hccl
        return native_hccl.prepare(request.ctx, case, params, request.output)
    if family == "memory":
        if case.id in ("MEM-002", "MEM-003"):
            from baselines.l0.memory import local
            return local.prepare(request.ctx, case, size, params)
        from baselines.l0.memory import device
        return device.prepare(request.ctx, case, size, params)
    if family == "overlap":
        if case.id == "HW-012":
            from baselines.l0.overlap import legacy
            return legacy.prepare(request.ctx, request.ctx.workload, size)
        from baselines.l0.overlap import runner
        return runner.prepare(request.ctx, case, request.ctx.workload, size, params)
    if family == "compute":
        from baselines.l0.compute import torch_ops
        return torch_ops.prepare(request.ctx, case, request.ctx.workload, size, variant, params)
    if family == "collective":
        from baselines.l0.collectives import torch_hccl
        return torch_hccl.prepare(request.ctx, case, size, variant)
    raise ValueError(f"no L0 handler for case {case.id}")


def _prepared(request):
    baseline = _baseline(request)
    metadata = {
        "family": l0_family(request.case),
        "parameters": dict(request.parameters),
        "completion": getattr(baseline, "completion", "local_complete"),
        "completion_detail": getattr(baseline, "completion_detail",
                                     "baseline's own wait/fence"),
    }
    return PreparedCase.from_l0(baseline, request.case.id, metadata=metadata)


def memory(request):
    return _prepared(request)


def transport(request):
    return _prepared(request)


def execution(request):
    return _prepared(request)


def overlap(request):
    return _prepared(request)


def compute(request):
    return _prepared(request)


def collective(request):
    return _prepared(request)


HANDLERS = {f"l0.{family}": globals()[family] for family in FAMILIES}


def prepare(request, output=None):
    """Entry point used by ``benchmarks.cases.prepare`` for every L0 family."""
    if request.level != "L0":
        raise ValueError("l0_hardware only prepares L0 cases")
    request.output = output
    return _prepared(request)


def l0_case(ctx, case, params, output=None):
    """Return a :class:`~baselines.l0.contract.Prepared` for an L0 case.

    Kept for callers that need the raw baseline object (evidence, direct metrics,
    payload) instead of the adapted case; the runner uses :func:`prepare`.
    """
    from benchmarks.cases import CaseRequest

    request = CaseRequest(ctx=ctx, case=case, backend=None, works=[], traces=[],
                          parameters=params, output=output)
    return _baseline(request)


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
