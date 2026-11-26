"""The contract between an L0 baseline module and the benchmark runner.

A baseline returns a :class:`Prepared` describing how to run it, how to wait for
completion, how to check it and what it asked the runtime for.  The runner owns
timing, so submission and completion stay separate: ``run`` only submits,
``wait`` only waits.
"""
from dataclasses import dataclass, field


def _identity(value):
    return value


@dataclass
class Prepared:
    run: object
    validate: object
    cleanup: object = lambda: None
    wait: object = _identity
    request: dict = None
    evidence: list = field(default_factory=list)
    payload: object = None
    #: Pre-computed metrics from a backend that measures internally (the native
    #: HCCL test tool).  When set, the runner does not time the call itself.
    direct: dict = None
    #: Extra observed facts, e.g. the configured topology graph for this task.
    topology: dict = None
