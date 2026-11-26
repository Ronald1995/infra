"""The contract between an L0 baseline module and the benchmark runner.

A baseline returns a :class:`Prepared` describing how to run it, how to wait for
completion, how to check it and what it asked the runtime for.  The runner owns
timing, so submission and completion stay separate: ``run`` only submits, ``wait``
only waits.

``completion`` states what ``wait`` actually proves, because the L0 baselines use
four different primitives and they are not interchangeable:

``submitted``
    the driver accepted the request (no completion primitive at all);
``fenced``
    a local fence/drain returned -- ordering holds, peer visibility does not;
``local_complete``
    the local engine reported the operation finished;
``remote_visible``
    the peer acknowledged the data, so the receiver can read it.

A baseline that can only fence (the host NBI path) must not claim
``remote_visible``: reporting submit-only latency as transfer latency is exactly
the wrong-bandwidth bug this field exists to prevent.
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
    #: Strongest completion guarantee :attr:`wait` provides; see the module docstring.
    completion: str = "submitted"
    #: Human-readable description of the completion mechanism.
    completion_detail: str = ""
    #: What actually ran: an ``ExecutionObservation`` when the baseline can report it.
    observation: object = None
    #: Verification ladder reached by this case (see ``verification/evidence.py``):
    #: configured -> endpoint_verified -> channel_acquired -> kernel_verified ->
    #: execution_verified -> physical_link_verified.
    levels: dict = None
