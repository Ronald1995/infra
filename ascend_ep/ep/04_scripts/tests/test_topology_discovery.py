"""Topology discovery: an observed host map replaces an assumed one, never a given one.

The five-state vocabulary has a producer for `discovered` now: the workers gather the
host name of every rank and the result becomes `server_ids` with `verified: true`.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import topology  # noqa: E402


def config(ep=4, block=None):
    return {"distributed": {"ep_size": ep}, "topology": dict(block or {})}


def test_a_synthetic_map_is_replaced_by_the_observed_one():
    result = topology.apply_discovered_topology(
        config(block={"source": "synthetic", "server_size": 4}),
        ["a", "a", "b", "b"])
    block = result["topology"]
    assert block["source"] == "discovered"
    assert block["verified"] is True
    assert block["server_ids"] == [0, 0, 1, 1]
    assert block["server_size"] == 2
    assert block["host_ids"] == ["a", "b"]
    assert block["rank_to_host"] == ["a", "a", "b", "b"]


def test_an_unknown_or_absent_map_is_replaced():
    for block in ({}, {"source": "unknown"}, {"source": "inferred"}):
        result = topology.apply_discovered_topology(config(block=block), ["a", "b", "a", "b"])
        assert result["topology"]["source"] == "discovered"
        assert result["topology"]["server_ids"] == [0, 1, 0, 1]


def test_a_user_provided_map_always_wins():
    given = config(block={"source": "user_provided", "server_ids": [0, 0, 0, 0],
                          "verified": True})
    assert topology.apply_discovered_topology(given, ["a", "b", "a", "b"]) is given


def test_an_already_discovered_map_is_left_alone():
    given = config(block={"source": "discovered", "server_ids": [0, 1, 0, 1]})
    assert topology.apply_discovered_topology(given, ["x", "y", "x", "y"]) is given


def test_incomplete_or_invalid_host_lists_change_nothing():
    given = config()
    assert topology.apply_discovered_topology(given, None) is given
    assert topology.apply_discovered_topology(given, ["a", "b"]) is given
    assert topology.apply_discovered_topology(given, ["a", None, "b", "c"]) is given
    assert topology.apply_discovered_topology(given, []) is given


def test_one_host_is_a_discovery_not_an_absence():
    result = topology.apply_discovered_topology(config(block={"source": "synthetic"}), ["a"] * 4)
    assert result["topology"]["source"] == "discovered"
    assert result["topology"]["server_ids"] == [0, 0, 0, 0]
    assert result["topology"]["server_size"] == 4
    assert result["topology"]["verified"] is True


def test_the_input_configuration_is_not_mutated():
    given = config(block={"source": "synthetic", "server_size": 4})
    snapshot = dict(given["topology"])
    topology.apply_discovered_topology(given, ["a", "a", "b", "b"])
    assert given["topology"] == snapshot


def test_gather_hosts_with_one_rank_needs_no_group():
    assert topology.gather_hosts(None, 1, hostname="solo") == ["solo"]


def test_gather_hosts_returns_none_without_an_initialised_group():
    class Dist:
        def is_initialized(self):
            return False

    assert topology.gather_hosts(Dist(), 4) is None


def test_gather_hosts_reads_every_rank():
    class Dist:
        def is_initialized(self):
            return True

        def all_gather_object(self, output, value, group=None):
            output[:] = ["host-a", "host-a", "host-b", "host-b"]

    assert topology.gather_hosts(Dist(), 4, hostname="host-a") == [
        "host-a", "host-a", "host-b", "host-b"]


def test_gather_hosts_fails_closed():
    class Dist:
        def is_initialized(self):
            return True

        def all_gather_object(self, output, value, group=None):
            raise RuntimeError("collective failed")

    assert topology.gather_hosts(Dist(), 4) is None


def test_gather_hosts_rejects_a_partial_result():
    class Dist:
        def is_initialized(self):
            return True

        def all_gather_object(self, output, value, group=None):
            output[:] = ["host-a", None, "host-b", "host-b"]

    assert topology.gather_hosts(Dist(), 4) is None
