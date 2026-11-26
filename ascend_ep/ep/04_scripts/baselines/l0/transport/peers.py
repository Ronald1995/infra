"""Deterministic, topology-aware peer selection for the native transport cases.

Peer selection is pure policy: it reads the configured rank-to-server map and
nothing else, so it can be unit tested on a CPU host and shared by the planner
(same rule) and the executor.  Two invariants matter:

* a peer set is never widened silently -- ``cross_server`` on a single-server EP
  group raises :class:`~common.backend.BenchmarkUnavailable` instead of falling
  back to intra-server peers, which would mislabel an intra-server measurement as
  cross-server;
* a requested ``peer_count`` is never truncated -- if the policy cannot supply
  that many peers for this rank, the case is reported as not applicable and the
  reason names the shortfall.

``peer_count=0`` keeps its meaning: every peer that satisfies the policy.
"""
from common.backend import BenchmarkUnavailable

POLICIES = ("ring", "same_server", "cross_server")

_LABELS = {"ring": "ring", "same_server": "same-server", "cross_server": "cross-server"}


def server_ids(config, size):
    """Return one server domain per rank, or ``None`` when the map is unusable.

    ``None`` covers both "no topology configured" and "the map does not describe
    exactly this many ranks"; neither may be silently padded.
    """
    topology = (config or {}).get("topology") or {}
    domains = topology.get("server_ids")
    if isinstance(domains, (list, tuple)) and len(domains) == size:
        return list(domains)
    return None


def candidates(size, rank, policy="ring", domains=None):
    """Peer ranks allowed by ``policy`` for ``rank``, in ascending rank order."""
    if policy not in POLICIES:
        raise BenchmarkUnavailable("unsupported", f"unknown peer policy {policy!r}")
    others = [item for item in range(size) if item != rank]
    if policy == "ring":
        return others
    if domains is None or len(domains) != size:
        raise BenchmarkUnavailable(
            "not_applicable",
            f"{policy} peer selection requires an explicit rank-to-server topology")
    home = domains[rank]
    if policy == "same_server":
        return [item for item in others if domains[item] == home]
    return [item for item in others if domains[item] != home]


def select(size, rank, policy="ring", peer_count=0, domains=None, rank_offset=None):
    """Resolve the destination peers for ``rank``.

    The rotation offset is deterministic (the rank itself by default), so a fixed
    peer count selects the same *kind* of neighbour at EP8 and EP128 instead of
    always the lowest rank IDs.
    """
    peers = candidates(size, rank, policy, domains)
    label = _LABELS.get(policy, policy)
    if not peers:
        raise BenchmarkUnavailable("not_applicable", f"no {label} peer available")
    offset = int(rank if rank_offset is None else rank_offset) % len(peers)
    peers = peers[offset:] + peers[:offset]
    limit = int(peer_count or 0)
    if limit and limit > len(peers):
        raise BenchmarkUnavailable(
            "not_applicable",
            f"peer_count={limit} exceeds the {len(peers)} available {label} peers")
    return peers[:limit] if limit else peers
