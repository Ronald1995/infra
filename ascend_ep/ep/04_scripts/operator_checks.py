"""Shared contract assertions for the operator directories.

Each operator's ``tests/`` used to assert only that ``make_plan()`` produced a task,
which says nothing about whether the adapter can run or produce a correct tensor.
The contract every operator is held to now has four parts:

1. **registration** -- the adapter is in the registry with an API and a source;
2. **capability correctness** -- the backend reports exactly the capabilities the
   registry declares, and requires no more than it claims;
3. **native initialisation** -- ``initialize()`` either succeeds or fails with a
   *typed, honest* error (``BenchmarkUnavailable`` with a benchmark status, or a
   missing native module).  A bare ``AttributeError``/``TypeError`` would mean the
   wiring is broken, and is rejected;
4. **representative execution + tensor output** -- proven by
   ``tests/operators/test_native_execution.py`` for every adapter whose constraints
   allow the CPU device, and reported as *blocked* (never as passed) for the rest.

Part 4 cannot be proven per adapter inside a single-process unit test: it needs two
real ranks.  So it lives in one shared process-level test and this module enforces
parts 1-3 plus the rule that a case which cannot execute carries a recorded reason
instead of a silent pass.
"""
from pathlib import Path

from common.backend import EPBackend
from common.config import load_configs
from common.registry import ROOT, registry
from common.scheduler import make_plan

#: Statuses a backend may legitimately report when it cannot initialise here.
HONEST_INITIALISATION_STATUSES = {
    "dependency_missing", "environment_unavailable", "source_unavailable",
    "not_implemented", "unsupported",
}


def representative_config():
    """The configuration a representative case is planned with."""
    return load_configs(ROOT / "configs/smoke_cpu.yaml")[0]


class StubContext:
    """The minimum context a backend may legitimately read while initialising.

    Deliberately not a device context: an adapter that needs tensors, a process
    group or torch_npu must fail with a typed status here, because that is exactly
    the environment difference the benchmark has to report.
    """

    def __init__(self, config):
        from types import SimpleNamespace

        self.config = config
        self.size = config["distributed"]["ep_size"]
        self.rank = 0
        self.local_rank = 0
        self.device = SimpleNamespace(type="cpu")
        self.torch = None
        self.dist = None
        self.ep_group = None

    def gather(self, value):
        return [value] * self.size

    def mapping(self):
        """One device/server record per rank, as ``common.distributed`` reports it.

        A backend may legitimately read this while initialising to check its
        topology; the stub says "one process per rank, no server information", which
        is why a topology-dependent adapter must report ``unsupported`` rather than
        assume a layout.
        """
        return [{"rank": rank, "local_rank": rank, "device": None, "server": None,
                 "host": None} for rank in range(self.size)]

    def barrier(self):
        return None

    def synchronize(self):
        return None


def stub_context():
    return StubContext(representative_config())


def check_capabilities(operator_id):
    """Parts 1 and 2 of the operator contract."""
    spec = registry()[operator_id]
    assert spec.api and spec.source.startswith("https://")
    backend = spec.create()
    assert isinstance(backend, EPBackend)
    assert backend.capabilities() == spec.capabilities
    assert spec.capabilities or spec.reason
    return spec


def check_native_initialisation(operator_id, context=None):
    """Part 3: initialising the native backend is typed, never a bare crash.

    Returns ``("initialised", detail)`` or ``("blocked", status)``.  The difference
    matters: an adapter that initialises has passed its wiring check, and one that is
    blocked has named why -- neither is a measurement, and this test claims neither.
    """
    from common.backend import BenchmarkUnavailable

    spec = registry()[operator_id]
    backend = spec.create()
    try:
        backend.initialize(context)
    except BenchmarkUnavailable as exc:
        assert exc.status in HONEST_INITIALISATION_STATUSES, (
            f"{operator_id} reported status {exc.status!r} while initialising")
        assert str(exc), f"{operator_id} reported an empty reason"
        return "blocked", exc.status
    except (ImportError, ModuleNotFoundError) as exc:
        assert str(exc), f"{operator_id} raised {type(exc).__name__} with no message"
        return "blocked", "dependency_missing"
    except (AttributeError, TypeError, KeyError) as exc:
        raise AssertionError(
            f"{operator_id} failed to initialise with an untyped {type(exc).__name__}: {exc}. "
            f"Raise BenchmarkUnavailable with a status instead so the failure is reportable."
        ) from exc
    return "initialised", None


def check_correctness_entry(operator_id):
    """Parts 1-3 plus the 'no silent pass' rule for a representative case."""
    spec = check_capabilities(operator_id)
    outcome, status = check_native_initialisation(operator_id, stub_context())
    assert outcome in {"initialised", "blocked"}
    path = Path(__file__).parent / "operators" / operator_id / "config.yaml"
    plan = make_plan(load_configs(path), levels=["L3"], cases=["MOE-001"])
    assert len(plan) == 1 and plan[0]["backend_id"] == operator_id
    task = plan[0]
    if task["status"] == "passed":  # never produced at planning time
        raise AssertionError(f"{operator_id} claims a pass without executing")
    if task["status"] != "not_run":
        # Anything that cannot run must say why, so a missing dependency can never
        # be mistaken for a passing test.
        assert task["reason"], f"{operator_id} reported {task['status']} with no reason"
    if "cpu" in (spec.constraints.get("devices") or ()):
        # A CPU-capable adapter must be provable end to end; that proof lives in the
        # shared two-rank test, which fails if the adapter is missing from it.
        assert task["status"] in {"not_run", "unsupported", "not_applicable",
                                 "dependency_missing", "environment_unavailable"}
    return {"operator": operator_id, "initialisation": outcome, "status": status,
            "planned_status": task["status"]}
