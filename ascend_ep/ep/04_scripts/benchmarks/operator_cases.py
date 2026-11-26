"""Compatibility shim for the pre-refactor case dispatcher.

The implementations moved to :mod:`benchmarks.cases` (one module per level).  This
module keeps the historic names importable -- ``Operation``, ``operator_case``,
``uses_planner`` -- so existing adapters and result tooling keep working, but it
contains no case logic of its own: it builds a
:class:`~benchmarks.cases.common.CaseRequest` and delegates to the handler named by
the case catalog.
"""
from benchmarks.cases import CaseRequest, prepare
from benchmarks.cases.common import Operation, PLANNER_TRAINING, PreparedCase, uses_planner

__all__ = ["Operation", "PLANNER_TRAINING", "PreparedCase", "operator_case", "uses_planner"]


def operator_case(ctx, case, backend, works, traces, parameters, output=None):
    """Run the registered handler for ``case`` and return the legacy ``Operation``."""
    from benchmarks.cases.common import as_operation

    request = CaseRequest(ctx=ctx, case=case, backend=backend, works=works, traces=traces,
                          parameters=parameters, output=output)
    prepared = prepare(request, output)
    return as_operation(prepared)
