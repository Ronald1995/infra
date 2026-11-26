"""The single execution contract shared by L0 baselines and L1-L4 operators.

Before this module there were two shapes:

``baselines.l0.contract.Prepared``
    ``run`` / ``wait`` / ``validate`` / ``cleanup`` / ``request`` / ``evidence`` /
    ``payload`` / ``direct``;
``benchmarks.operator_cases.Operation``
    ``setup`` / ``call`` / ``validate`` / ``wait`` / ``sample_metrics``.

The runner had to special-case both, and a field that only one shape had (L0
evidence, or the L1 setup phase) was easy to drop on the way to the result file.
:class:`PreparedCase` is the union, with the timing boundary made explicit:

::

    Timer Start
        |
        v
    run()      # submit only; must not block on completion
        |
        v
    wait()     # real completion: the data is visible on the receiver
        |
        v
    Timer Stop

``wait()`` therefore has to mean "the transfer finished", not "the submission was
accepted".  A backend whose completion primitive only fences the submission
(the HCOMM ``quiet``/drain path, a CPU engine that returns immediately) reports
that weaker meaning through :attr:`PreparedCase.completion` so a result can never
present submit-only latency as transfer latency.
"""
from dataclasses import dataclass, field
from typing import Callable

NO_COMPLETION = "none"
#: Completion semantics a backend can promise, weakest first.
COMPLETION_LEVELS = (
    # `run` returned: the request was accepted by the driver, nothing more.
    "submitted",
    # A local fence/drain returned.  Ordering is guaranteed, transfer visibility
    # to the peer is not.
    "fenced",
    # The local engine reported the operation complete.
    "local_complete",
    # The peer acknowledged the data, so it is visible on the receiver.
    "remote_visible",
)


def _identity(value):
    return value


def _noop(*_args, **_kwargs):
    return None


def _no_input(index):
    return index


def _no_metrics(_value):
    return {}


@dataclass
class PreparedCase:
    """A prepared, timed case.  Every level returns this type."""

    #: Untimed input preparation for iteration ``index``.
    setup: Callable = _no_input
    #: Submit the operation.  Must not block waiting for completion.
    run: Callable = _noop
    #: Wait for real completion, then return the value the validator inspects.
    wait: Callable = _identity
    #: Correctness check, run before the timed loop.
    validate: Callable = _noop
    #: Release resources.  Always called, including on failure.
    cleanup: Callable = _noop
    #: Optional per-sample stage timings.
    sample_metrics: Callable = _no_metrics
    #: Route request parameters.  *Requested* values only.
    request: dict | None = None
    #: Evidence items collected from the runtime.
    evidence: list = field(default_factory=list)
    #: Payload description consumed by ``metrics.summary.bandwidth``.
    payload: object = None
    #: Pre-computed metrics from a backend that measures internally.
    direct: dict | None = None
    #: Observation of what actually ran (bytes, API, algorithm, fallback).
    observation: object = None
    #: Free-form facts: case id, handler, correctness tokens, topology.
    metadata: dict = field(default_factory=dict)
    #: The strongest completion guarantee :meth:`wait` provides.
    completion: str = NO_COMPLETION
    #: Human-readable note about the completion mechanism.
    completion_detail: str = ""

    def __post_init__(self):
        if self.completion != NO_COMPLETION and self.completion not in COMPLETION_LEVELS:
            raise ValueError(f"completion={self.completion!r} is outside {COMPLETION_LEVELS}")
        for name in ("run", "wait", "validate", "cleanup", "setup", "sample_metrics"):
            if not callable(getattr(self, name)):
                raise TypeError(f"PreparedCase.{name} must be callable")
        if self.metadata is None:
            self.metadata = {}

    # ------------------------------------------------------------ adapters --

    @property
    def call(self):
        """Historic name for :attr:`run`; kept so old call sites keep working."""
        return self.run

    def transfers_distinct_from_submission(self):
        """True when ``wait`` proves more than "the request was accepted"."""
        return self.completion in ("local_complete", "remote_visible")

    # --------------------------------------------------------- construction --

    @classmethod
    def from_l0(cls, prepared, case_id, payload=None, metadata=None):
        """Adapt a :class:`baselines.l0.contract.Prepared`.

        The L0 baselines are zero-argument closures, so ``setup`` returns a
        placeholder and ``run`` ignores it.  Nothing else is changed: request,
        evidence, payload and internal metrics are carried through untouched.
        """
        payload = prepared.payload if payload is None else payload
        observation = getattr(prepared, "observation", None)
        return cls(
            setup=lambda index: None,
            run=lambda _value: prepared.run(),
            wait=lambda value: prepared.wait(None) if value is None else prepared.wait(value),
            validate=prepared.validate,
            cleanup=prepared.cleanup,
            request=prepared.request,
            evidence=list(prepared.evidence or ()),
            payload=payload,
            direct=prepared.direct,
            observation=observation,
            metadata={**(metadata or {}), "case_id": case_id, "level": "L0",
                      "topology": getattr(prepared, "topology", None),
                      "levels": getattr(prepared, "levels", None)},
            completion=getattr(prepared, "completion", NO_COMPLETION),
            completion_detail=getattr(prepared, "completion_detail", ""),
        )

    @classmethod
    def from_operation(cls, operation, case_id, level, metadata=None, completion=NO_COMPLETION,
                       completion_detail="", observation=None):
        """Adapt the historic ``Operation`` tuple used by L1-L4 handlers."""
        return cls(
            setup=operation.setup,
            run=operation.call,
            wait=operation.wait,
            validate=operation.validate,
            sample_metrics=operation.sample_metrics,
            observation=observation,
            metadata={**(metadata or {}), "case_id": case_id, "level": level},
            completion=completion,
            completion_detail=completion_detail,
        )


@dataclass
class Operation:
    """The pre-refactor L1-L4 shape, kept for handlers and third-party adapters.

    New code returns :class:`PreparedCase`; the runner converts an ``Operation``
    through :meth:`PreparedCase.from_operation`, so both work.
    """

    setup: Callable
    call: Callable
    validate: Callable
    wait: Callable = _identity
    sample_metrics: Callable = _no_metrics


def as_prepared(value, case_id, level, **kwargs):
    """Return ``value`` as a :class:`PreparedCase`, converting an ``Operation``."""
    if isinstance(value, PreparedCase):
        return value
    if isinstance(value, Operation):
        return PreparedCase.from_operation(value, case_id, level, **kwargs)
    raise TypeError(f"{type(value).__name__} is not a prepared case or an operation")
