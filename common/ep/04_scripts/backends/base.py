"""Lifecycle and optional operations exposed by EP benchmark adapters."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal

BenchmarkLevel = Literal["L1", "L2", "L3", "L4"]


class EPBackend(ABC):
    """Expose native capabilities and manage per-run backend resources."""

    @abstractmethod
    def capabilities(self) -> frozenset[BenchmarkLevel]:
        """Return the benchmark levels supported by the adapter."""

    @abstractmethod
    def initialize(self, config: Mapping[str, object]) -> None:
        """Initialize the backend for the supplied configuration."""

    def dispatch(self, inputs: object) -> object:
        raise NotImplementedError("Dispatch is unsupported")

    def combine(self, inputs: object) -> object:
        raise NotImplementedError("Combine is unsupported")

    def dispatch_gmm(self, inputs: object) -> object:
        raise NotImplementedError("Fused dispatch and GMM is unsupported")

    def gmm_combine(self, inputs: object) -> object:
        raise NotImplementedError("Fused GMM and combine is unsupported")

    def moe_forward(self, inputs: object) -> object:
        raise NotImplementedError("MoE forward is unsupported")

    def moe_backward(self, inputs: object) -> object:
        raise NotImplementedError("MoE backward is unsupported")

    @abstractmethod
    def finalize(self) -> None:
        """Release resources initialized for this run."""
