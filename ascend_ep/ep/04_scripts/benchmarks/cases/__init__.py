"""Handler registry for L0-L4 cases.

``CaseSpec.handler`` names one entry here.  The registry is the *only* place that
maps a name to code, so adding a case is: one row in :mod:`common.case_spec`, one
function in the matching module, one line in that module's ``HANDLERS``.

Resolution is lazy on purpose: planning imports :mod:`common.case_spec` and must
never import torch, a device library or a baseline implementation.
"""
from importlib import import_module

from common.prepared import Operation, PreparedCase, as_prepared

from .common import CaseRequest, PLANNER_TRAINING, uses_planner

#: Handler-name prefix -> module that owns the implementations.
_MODULES = {
    "l0.": "benchmarks.l0_hardware",
    "l1.": "benchmarks.cases.l1_communication",
    "l2.": "benchmarks.cases.l2_fusion",
    "l3.": "benchmarks.cases.l3_moe",
    "l4.": "benchmarks.cases.l4_router_planner",
}


def _module_for(name):
    for prefix, module in _MODULES.items():
        if name.startswith(prefix):
            return import_module(module)
    raise KeyError(f"handler {name!r} does not start with one of {sorted(_MODULES)}")


def handler(name):
    """Resolve a dotted handler name to its callable."""
    module = _module_for(name)
    try:
        return module.HANDLERS[name]
    except KeyError:
        raise KeyError(f"handler {name!r} is not registered by {module.__name__}") from None


def known_handlers():
    """Every registered handler name, with the module that provides it."""
    result = {}
    for module_name in dict.fromkeys(_MODULES.values()):
        module = import_module(module_name)
        result.update(module.HANDLERS)
    return result


def prepare(request, output=None):
    """Run the handler for one case and return a :class:`PreparedCase`.

    This is the single entry point the runner uses for every level, which is what
    keeps one timing boundary, one validation entry and one cleanup path.
    """
    if not isinstance(request, CaseRequest):
        raise TypeError("prepare() requires a CaseRequest")
    name = request.case.handler
    if not name:
        raise ValueError(f"case {request.case.id} has no execution handler")
    if request.level == "L0":
        return _module_for(name).prepare(request, output)
    prepared = as_prepared(handler(name)(request), request.case.id, request.level)
    if prepared.completion == "none" and request.backend is not None:
        # A handler that did not state its completion guarantee inherits the
        # adapter's declaration, so no result can silently claim "none".
        level, detail = request.backend.completion_semantics()
        prepared.completion = level
        prepared.completion_detail = detail
    return prepared


__all__ = [
    "CaseRequest", "Operation", "PLANNER_TRAINING", "PreparedCase", "handler",
    "known_handlers", "prepare", "uses_planner",
]
