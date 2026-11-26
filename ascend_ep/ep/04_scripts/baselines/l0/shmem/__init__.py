"""SHMEM baselines: MTE (host stream) and UDMA (device kernel or host stream).

``prepare_engine`` is the L0 entry point.  ``prepare`` is kept with its original
signature so the legacy ``HW-007``/``HW-008`` cases keep working; it returns the
older ``(run, validate, cleanup)`` triple.

URMA is deliberately rejected: the pinned SHMEM release exposes MTE, SDMA, ROCE
and UDMA data-operation engines and no URMA engine.  "UDMA is not URMA" is not a
naming preference here, it is what the enum contains
(``include/host_device/shmem_common_types.h``).
"""
from common.backend import BenchmarkUnavailable

from baselines.l0.shmem import mte, udma

ENGINES = {"mte": mte, "udma": udma}


def prepare_engine(ctx, case, size, params):
    """Return the :class:`~baselines.l0.contract.Prepared` for a SHMEM case."""
    engine = "udma" if case.id == "LINK-013" else "mte"
    if case.id == "LINK-013" and params.get("variant") == "host":
        engine = "udma"
    return ENGINES[engine].prepare(ctx, case, size, params)


def request(case, params):
    return udma.request(case, params) if case.id == "LINK-013" else mte.request(case, params)


def prepare(ctx, size, engine):
    """Legacy entry point used by ``HW-007`` (MTE) and ``HW-008`` (UDMA/URMA)."""
    if engine == "urma":
        raise BenchmarkUnavailable(
            "unsupported",
            "pinned CANN SHMEM exposes MTE/SDMA/ROCE/UDMA data-operation engines, "
            "with no distinct URMA engine")
    module = ENGINES.get(engine)
    if module is None:
        raise BenchmarkUnavailable("unsupported", f"unknown SHMEM engine {engine!r}")

    class _Case:
        id = "HW-008" if engine == "udma" else "HW-007"
        operation = "put"

    prepared = module.prepare(ctx, _Case, size, {})
    run, wait, validate, cleanup = (prepared.run, prepared.wait, prepared.validate,
                                    prepared.cleanup)

    def blocking_run():
        run()
        wait(None)

    return blocking_run, validate, cleanup
