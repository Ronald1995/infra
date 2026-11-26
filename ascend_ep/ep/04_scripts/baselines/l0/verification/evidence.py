"""Evidence collection: what a transport result may actually claim.

The old LINK evidence was a descriptor, a rank graph and the endpoint configuration.
None of those show that UB, URMA, HCCS, RoCE or UBoE moved a byte: a descriptor says
which channel the runtime *accepted*, and a configuration says what was *requested*.
A report that turns either into "the physical link was verified" is wrong.

This module collects evidence items in named kinds and derives two outputs:

1. an observed block (``physical_link``, ``channel_kind``, ``engine``,
   ``kernel_name``, ``confidence``) that is only filled from evidence --
   configuration is never an input;
2. a verification **ladder** so a reader can see how far a case actually got::

       configured              the plan asked for this mechanism
       endpoint_verified       an HCOMM endpoint was created
       channel_acquired        HcclChannelAcquire succeeded for the channel
       kernel_verified         the expected kernel appears in a profiler record
       execution_verified      the peer's data matched the payload
       physical_link_verified  the link is named by runtime evidence, not inference

A case that acquires a channel but whose kernel never appears in a profiler record
stops at ``channel_acquired``.  It is not "verified".
"""
from dataclasses import dataclass, field

from .path_verifier import Evidence, verify

#: Evidence kinds.  Each maps to one :class:`path_verifier.Evidence` source.
KINDS = ("descriptor", "rank_graph", "endpoint", "channel_kind", "kernel", "msprof_kernel",
         "runtime_execution")

#: ``Evidence.source`` value for each kind.  Two kinds share a source when they are
#: the same runtime answer seen twice (the bridge reports the channel kind).
SOURCE_FOR_KIND = {
    "descriptor": "native_descriptor",
    "rank_graph": "rank_graph",
    "endpoint": "endpoint",
    "channel_kind": "channel_kind",
    "kernel": "kernel",
    "msprof_kernel": "msprof_kernel",
    "runtime_execution": "runtime_execution",
}

#: Ladder rungs, weakest first.  A rung is reached only when its evidence exists.
LADDER = ("configured", "endpoint_verified", "channel_acquired", "kernel_verified",
          "execution_verified", "physical_link_verified")

#: Link tokens this module may report.  ``unverified`` is a real answer.
PHYSICAL_LINKS = ("ub_memory", "urma", "hccs", "roce", "uboe", "shmem_mte", "shmem_udma",
                  "unverified")

#: Channel-kind family -> link family.  This is an inference *from runtime evidence*
#: and is therefore only used when the channel kind itself was observed.
_CHANNEL_LINK = (
    ("urma", "urma"), ("ubmem", "ub_memory"), ("ub_mem", "ub_memory"),
    ("hccs", "hccs"), ("roce", "roce"), ("uboe", "uboe"),
    ("shmem:udma", "shmem_udma"), ("shmem:mte", "shmem_mte"),
)

#: Rank-graph protocol -> link family.
_PROTOCOL_LINK = {
    "UB_MEM": "ub_memory", "UB_CTP": "urma", "UBC_TP": "ub_ctp", "UB_RTP": "ub_rtp",
    "ROCE": "roce", "HCCS": "hccs", "HCCS_ONLY": "hccs", "UBOE": "uboe",
}


def link_for_channel(channel_kind):
    """Link family implied by an observed channel class name, or ``None``."""
    if not channel_kind:
        return None
    lowered = str(channel_kind).lower()
    for token, link in _CHANNEL_LINK:
        if token in lowered:
            return link
    return None


def link_for_protocol(protocol):
    return _PROTOCOL_LINK.get(protocol)


@dataclass
class EvidenceCollector:
    """Accumulate evidence, then resolve what it supports."""

    requested: dict = field(default_factory=dict)
    items: list = field(default_factory=list)
    #: Facts used to reach ladder rungs.
    endpoint_created: bool = False
    channel_kind: str | None = None
    engine: str | None = None
    kernel_name: str | None = None
    profiled_kernel: str | None = None
    protocol: str | None = None
    execution_verified: bool = False

    # ------------------------------------------------------------ collection --
    def add(self, kind, claim, detail="", path=None, confidence="strong"):
        if kind not in KINDS:
            raise ValueError(f"unknown evidence kind {kind!r}; expected one of {KINDS}")
        item = Evidence(source=SOURCE_FOR_KIND[kind], claim=dict(claim), detail=detail,
                        path=path, confidence=confidence)
        self.items.append(item)
        return item

    def descriptor(self, detail, **claim):
        return self.add("descriptor", claim, detail)

    def rank_graph(self, peer, protocol):
        return self.add("rank_graph", {}, f"rank graph reports {protocol} towards rank {peer}")

    def endpoint(self, detail):
        self.endpoint_created = True
        return self.add("endpoint", {"endpoint": True}, detail)

    def channel(self, channel_kind, engine=None, detail=""):
        self.channel_kind = channel_kind
        self.engine = engine or self.engine
        claim = {"channel": channel_kind}
        if engine:
            claim["execution_resource"] = engine
        return self.add("channel_kind", claim,
                        detail or f"HcclChannelAcquire returned {channel_kind}")

    def kernel(self, name, detail=""):
        self.kernel_name = name
        return self.add("kernel", {"kernel_name": name}, detail or "kernel launch recorded")

    def msprof(self, name, path=None):
        self.profiled_kernel = name
        return self.add("msprof_kernel", {"kernel_name": name},
                        f"profiler record contains {name}", path=path)

    def execution(self, detail, path=None):
        self.execution_verified = True
        return self.add("runtime_execution", {"payload_validated": True}, detail, path=path)

    # -------------------------------------------------------------- resolving --
    def ladder(self):
        """Which rungs are reached, and the first one that is not."""
        reached = {"configured": bool(self.requested.get("transport_protocol")
                                      or self.requested.get("execution_resource")),
                   "endpoint_verified": self.endpoint_created,
                   "channel_acquired": bool(self.channel_kind),
                   "kernel_verified": bool(self.kernel_name or self.profiled_kernel),
                   "execution_verified": self.execution_verified,
                   "physical_link_verified": False}
        return reached

    def resolve_link(self):
        """``(physical_link, confidence, sources)`` from evidence only.

        ``verified`` needs the channel kind **and** an independent runtime source
        (rank-graph protocol or a profiler kernel record) that agrees with it.  The
        channel kind alone yields ``inferred``; nothing yields ``unverified``.
        """
        channel_link = link_for_channel(self.channel_kind)
        protocol_link = link_for_protocol(self.protocol)
        profiled = bool(self.profiled_kernel)
        sources = [name for name, present in
                   (("channel_kind", bool(self.channel_kind)),
                    ("rank_graph", bool(self.protocol)),
                    ("msprof_kernel", profiled)) if present]
        if channel_link and protocol_link and channel_link == protocol_link:
            return channel_link, "verified", sources
        if channel_link and profiled and channel_link == protocol_link:
            return channel_link, "verified", sources
        if channel_link:
            return channel_link, "inferred", sources
        if protocol_link:
            return protocol_link, "inferred", sources
        return "unverified", "unknown", sources

    def observation(self, required=("execution_resource", "transport_protocol")):
        """The ``observed`` block plus its verification status."""
        verified = verify(self.requested, self.items, required=required)
        link, confidence, sources = self.resolve_link()
        verified.observed["physical_link"] = None if link == "unverified" else link
        verified.observed["physical_link_confidence"] = confidence
        verified.observed["channel_kind"] = self.channel_kind
        verified.observed["engine"] = self.engine
        verified.observed["kernel_name"] = self.kernel_name or self.profiled_kernel
        verified.observed["evidence_sources"] = sources
        return verified

    def levels(self, required=("execution_resource", "transport_protocol")):
        """The verification ladder for this case.

        The ladder is monotone: a rung is only reported as reached when every rung
        below it was reached too.  Link resolution on its own must not be able to
        claim `physical_link_verified` while `kernel_verified` is false -- that would
        make the ladder readable in exactly one wrong direction.
        """
        reached = self.ladder()
        link, confidence, _sources = self.resolve_link()
        physical = confidence == "verified" and link != "unverified"
        monotone = True
        for rung in LADDER:
            value = physical if rung == "physical_link_verified" else bool(reached.get(rung))
            monotone = monotone and value
            reached[rung] = monotone
        observation = self.observation(required)
        reached["verification_status"] = observation.status
        return reached

    def as_dict(self, required=("execution_resource", "transport_protocol")):
        observation = self.observation(required)
        return {
            "requested": dict(self.requested),
            "observed": observation.observed,
            "verification": observation.as_dict(),
            "levels": self.levels(required),
            "evidence": [item.as_dict() for item in self.items],
        }


def strongest_level(levels):
    """The deepest ladder rung a case reached."""
    reached = [rung for rung in LADDER if levels.get(rung)]
    return reached[-1] if reached else "configured"


def annotate_record(record, collector, required=("execution_resource", "transport_protocol")):
    """Attach the collector's view to a result record."""
    resolved = collector.as_dict(required)
    record["requested"] = resolved["requested"]
    record["observed"] = resolved["observed"]
    record["verification"] = resolved["verification"]
    record["verification_levels"] = resolved["levels"]
    record["verification_level"] = strongest_level(resolved["levels"])
    record["evidence"] = {
        "log_paths": sorted({item.path for item in collector.items if item.path}),
        "profile_paths": [],
        "claims": resolved["evidence"],
    }
    return record
