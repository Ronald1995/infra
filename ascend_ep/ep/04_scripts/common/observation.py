"""One execution observation shape for every operator backend.

The old result record mixed four different kinds of number in one flat namespace,
which is how a local copy ended up inside a cross-card bandwidth:

``logical_payload_bytes``
    what the *model* asked to move (tokens x top_k x hidden x dtype), counted once
    per logical transfer even when the same bytes are replicated to several peers.
``remote_payload_bytes``
    the bytes every rank actually *sends to other ranks*, summed over the EP group
    but excluding local rows that never leave the card.
``measured_wire_bytes``
    bytes a counter or profiler reported on the wire.  Almost always unavailable;
    ``None`` means "not measured", never "zero".
``local_copy_bytes``
    bytes moved inside one card (reordering, packing, quantisation).  It is
    reported for completeness and is **never** part of a cross-card bandwidth.

:class:`ExecutionObservation` keeps them apart and refuses to synthesise the ones
it cannot observe: a physical bandwidth is only derived from
``measured_wire_bytes``.  Everything else is a model, is labelled ``source``
accordingly and is not comparable with a wire-derived number.

It also carries the four attribution fields a result needs to be readable:
the API actually called, the algorithm and transport actually used (or
``unknown``), and whether the adapter fell back to another implementation.
"""
from dataclasses import dataclass, field

#: Where a number came from.  Ordered by strength of evidence.
SOURCES = ("measured", "backend_reported", "routing_trace_model",
           "declared_payload_model", "unavailable")
CONFIDENCES = ("verified", "declared", "unknown")

#: Fields whose absence must be recorded rather than defaulted to zero.
BYTE_FIELDS = ("logical_payload_bytes", "remote_payload_bytes", "measured_wire_bytes",
               "local_copy_bytes", "dispatch_bytes_sent", "dispatch_bytes_received",
               "combine_bytes_sent", "combine_bytes_received")


@dataclass
class ExecutionObservation:
    """What one backend actually did for one case, with its evidence."""

    actual_api: str | None = None
    dispatch_bytes_sent: int | None = None
    dispatch_bytes_received: int | None = None
    combine_bytes_sent: int | None = None
    combine_bytes_received: int | None = None
    logical_payload_bytes: int | None = None
    remote_payload_bytes: int | None = None
    measured_wire_bytes: int | None = None
    local_copy_bytes: int | None = None
    rank_max_latency_ms: float | None = None
    rank_avg_latency_ms: float | None = None
    actual_algorithm: str | None = None
    actual_transport: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    evidence: list = field(default_factory=list)
    #: ``measured`` / ``backend_reported`` / ``routing_trace_model`` /
    #: ``declared_payload_model`` / ``unavailable``.
    source: str = "unavailable"
    #: ``verified`` / ``declared`` / ``unknown``.
    confidence: str = "unknown"
    #: Byte fields this observation cannot provide, with the reason.
    unavailable: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"observation source {self.source!r} is outside {SOURCES}")
        if self.confidence not in CONFIDENCES:
            raise ValueError(f"observation confidence {self.confidence!r} is outside "
                             f"{CONFIDENCES}")
        if self.fallback_used and not self.fallback_reason:
            raise ValueError("a fallback observation must say why it fell back")

    # ---------------------------------------------------------------- bytes --

    @property
    def cross_card_bytes(self) -> int | None:
        """The only number that may be divided by a link bandwidth.

        ``None`` when neither a measurement nor a routing-trace model is
        available: a physical bandwidth is never derived from logical bytes.
        """
        if self.measured_wire_bytes is not None:
            return self.measured_wire_bytes
        return self.remote_payload_bytes

    @property
    def cross_card_source(self) -> str:
        return "measured_wire" if self.measured_wire_bytes is not None else (
            "remote_payload_model" if self.remote_payload_bytes is not None else "unavailable")

    def cross_card_bandwidth(self, milliseconds) -> float | None:
        """GB/s over cross-card bytes, or ``None`` when they are unknown.

        Dividing by a local copy or by logical bytes is what produced the wrong
        bandwidth numbers this module exists to prevent.
        """
        volume = self.cross_card_bytes
        if volume is None or not milliseconds:
            return None
        return volume / (milliseconds * 1e6)

    def require(self, *names):
        """Mark byte fields that this observation cannot provide."""
        for name in names:
            if name not in BYTE_FIELDS:
                raise ValueError(f"{name!r} is not a byte field")
            if getattr(self, name) is None and name not in self.unavailable:
                self.unavailable[name] = "not reported by the backend"
        return self

    # ------------------------------------------------------------- latency --

    def with_latency(self, rank_max_ms, rank_avg_ms=None):
        """Attach the cross-rank latency the runner measured."""
        self.rank_max_latency_ms = rank_max_ms
        self.rank_avg_latency_ms = rank_avg_ms
        return self

    # -------------------------------------------------------------- output --

    def as_dict(self):
        return {
            "actual_api": self.actual_api,
            "actual_algorithm": self.actual_algorithm or "unknown",
            "actual_transport": self.actual_transport or "unknown",
            "fallback": {"used": self.fallback_used, "reason": self.fallback_reason},
            "bytes": {
                "dispatch": {"sent": self.dispatch_bytes_sent,
                             "received": self.dispatch_bytes_received},
                "combine": {"sent": self.combine_bytes_sent,
                            "received": self.combine_bytes_received},
                "logical_payload": self.logical_payload_bytes,
                "remote_payload": self.remote_payload_bytes,
                "measured_wire": self.measured_wire_bytes,
                "local_copy": self.local_copy_bytes,
                "unit": "bytes",
                "cross_card_source": self.cross_card_source,
            },
            "latency": {"rank_max_ms": self.rank_max_latency_ms,
                        "rank_avg_ms": self.rank_avg_latency_ms,
                        "unit": "ms"},
            "source": self.source,
            "confidence": self.confidence,
            "unavailable": dict(self.unavailable),
            "evidence": list(self.evidence),
        }


def from_traffic(volume, api, *, algorithm=None, transport=None, fallback_reason=None,
                 dispatch=(), combine=()):
    """Build an observation from a :func:`metrics.summary.traffic` volume.

    ``traffic`` already separates logical from remote-expanded bytes; this adapter
    keeps the names it uses but states explicitly that they are a *model* and that
    the wire volume was not measured.
    """
    if not isinstance(volume, dict):
        raise TypeError("traffic volume must be a mapping")
    observation = ExecutionObservation(
        actual_api=api,
        logical_payload_bytes=volume.get("logical_bytes"),
        remote_payload_bytes=volume.get("remote_expanded_bytes"),
        measured_wire_bytes=None,
        local_copy_bytes=None,
        actual_algorithm=algorithm,
        actual_transport=transport,
        fallback_used=fallback_reason is not None,
        fallback_reason=fallback_reason,
        source="routing_trace_model",
        confidence="declared",
        evidence=[{"source": "routing_trace", "claim": {"traffic_model": volume.get("traffic_model")}}],
        unavailable={"measured_wire_bytes": "no link counter or profiler counter was collected"},
    )
    if "dispatch" in dispatch:
        observation.dispatch_bytes_sent = volume.get("remote_expanded_bytes")
    if "combine" in combine:
        observation.combine_bytes_sent = volume.get("remote_expanded_bytes")
    return observation


def unavailable(reason, api=None):
    """An explicit 'this backend cannot report its traffic' observation."""
    observation = ExecutionObservation(
        actual_api=api, source="unavailable", confidence="unknown",
        unavailable={name: reason for name in BYTE_FIELDS},
        evidence=[{"source": "backend", "claim": {}, "detail": reason}],
    )
    return observation
