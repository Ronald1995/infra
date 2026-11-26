"""Rank-graph view of the configured topology.

This module answers "which ranks are configured to share a node, a server domain
or a supernode" and nothing else.  It is *not* a hardware discovery: CANN exposes
the real rank-graph (link type, protocol, hop count) through HCOMM, and that
information is recorded separately as evidence when the native bridge is
available.  Conflating the two is how a configured intra-node run gets reported
as an HCCS measurement.
"""
from dataclasses import dataclass

CLASSES = ("intra_node", "intra_server", "intra_supernode", "cross_supernode", "unknown")


@dataclass(frozen=True)
class Link:
    source_rank: int
    destination_rank: int
    configured_class: str
    observed_link: str | None = None
    verification: str = "unverified"

    def as_dict(self):
        return {"source_rank": self.source_rank, "destination_rank": self.destination_rank,
                "configured_class": self.configured_class, "observed_link": self.observed_link,
                "verification": self.verification, "source": "configured_topology"}


def _domain(ids, rank, default):
    if not ids or rank >= len(ids):
        return default
    return ids[rank]


def classify(source, destination, nodes, servers, supernodes):
    """Classify a rank pair from configuration only."""
    if nodes and nodes[source] is not None and nodes[destination] is not None:
        if nodes[source] == nodes[destination]:
            return "intra_node"
    server_a = _domain(servers, source, None)
    server_b = _domain(servers, destination, None)
    if server_a is not None and server_b is not None:
        if server_a == server_b:
            return "intra_server"
    super_a = _domain(supernodes, source, None)
    super_b = _domain(supernodes, destination, None)
    if super_a is not None and super_b is not None:
        if super_a != super_b:
            return "cross_supernode"
        return "intra_supernode"
    if server_a is not None and server_b is not None and server_a != server_b:
        return "cross_supernode"
    return "unknown"


def build(config, mapping=None):
    """Build the configured rank graph for one task.

    ``mapping`` is the per-rank mapping recorded by the runner (host and
    node rank).  Without it every rank is placed on an unknown node and only the
    server/supernode domains remain meaningful.
    """
    distributed = config.get("distributed") or {"ep_size": 1, "tp_size": 1}
    size = distributed.get("ep_size", 1) * distributed.get("tp_size", 1)
    topology = config.get("topology") or {}
    servers = topology.get("server_ids")
    supernodes = topology.get("supernode_ids")
    nodes = None
    if mapping:
        nodes = [entry.get("host") if entry else None for entry in mapping]
        if nodes and len(set(nodes)) == 1:
            # A single hostname for every rank is still an intra-node layout;
            # keep it explicit instead of inferring one host == one node.
            nodes = [0] * len(nodes)
    links = [Link(source, destination, classify(source, destination, nodes, servers, supernodes))
             for source in range(size) for destination in range(size) if source != destination]
    classes = {}
    for link in links:
        classes[link.configured_class] = classes.get(link.configured_class, 0) + 1
    return {
        "source": "configured_topology",
        "verification": "unverified",
        "rank_size": size,
        "nodes": nodes,
        "server_ids": servers,
        "supernode_ids": supernodes,
        "link_classes": classes,
        "links": [link.as_dict() for link in links],
        "note": ("Configured domains only. Physical link type requires an HCOMM rank-graph "
                 "query or runtime evidence."),
    }


def topology_mode(graph):
    """Reduce a rank graph to the ``intra_node`` / ``inter_node`` mode used by configs."""
    classes = graph.get("link_classes") or {}
    if not classes:
        return "unknown"
    if set(classes) <= {"intra_node"}:
        return "intra_node"
    return "inter_node"


def peers(graph, rank, classes=None):
    """Peer ranks matching one of ``classes`` (all peers by default)."""
    result = []
    for link in graph.get("links", ()):
        if link["source_rank"] != rank:
            continue
        if classes and link["configured_class"] not in classes:
            continue
        result.append(link["destination_rank"])
    return result
