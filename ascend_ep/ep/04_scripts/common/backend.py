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
    def __init__(self, spec):
        self.spec = spec
        self.context = None

    def capabilities(self):
        return self.spec.capabilities

    def initialize(self, context):
        self.context = context

    def prepare(self, workload):
        return None

    def prepare_communication(self, workload):
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
