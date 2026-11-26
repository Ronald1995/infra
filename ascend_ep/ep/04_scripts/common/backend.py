"""Native backend contract and explicit execution outcomes."""
from dataclasses import dataclass
from typing import Any


class BenchmarkUnavailable(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


@dataclass
class DispatchResult:
    expert_input: Any
    expert_token_counts: tuple[int, ...]
    routing_handle: Any
    layout: str
    expert_ids: Any = None
    route_slots: Any = None
    completion: Any = None


@dataclass
class CombineResult:
    output: Any
    completion: Any = None


class EPBackend:
    """Contract every EP operator adapter implements.

    ``completion_semantics`` is part of the contract, not documentation: the
    runner records the guarantee a measurement actually has, so a submit-only
    adapter can never be reported as a completed transfer.
    """

    #: Strongest completion guarantee this adapter provides after ``wait``.
    COMPLETION = ("local_complete", "the adapter's completion handle is waited")
    #: Native API names this adapter calls; recorded as the result's `actual_api`.
    ACTUAL_API = None

    def __init__(self, spec):
        self.spec = spec
        self.context = None

    def capabilities(self):
        return self.spec.capabilities

    def completion_semantics(self):
        """``(level, detail)`` describing what ``wait_until_ready`` proves."""
        return self.COMPLETION

    def actual_api(self):
        """The API identity a result may report, or ``None`` when it is unknown."""
        return self.ACTUAL_API or self.spec.api

    def initialize(self, context):
        self.context = context

    def prepare(self, workload):
        return None

    def prepare_communication(self, workload):
        return None

    def prepare_training(self, workload):
        return None

    def wait_until_ready(self, result):
        completion = getattr(result, "completion", None)
        if completion is not None:
            completion.wait()
        return result

    def finalize(self):
        pass


class UnavailableBackend(EPBackend):
    def initialize(self, context):
        raise BenchmarkUnavailable(self.spec.status, self.spec.reason)
