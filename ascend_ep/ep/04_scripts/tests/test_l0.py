"""L0 transport refactor: offline tests for discovery, planning, schema and report.

Everything here is mock/offline.  No NPU is touched, so nothing in this file is a
hardware performance result; the fixtures are explicitly marked as overrides.
"""
import json
from types import SimpleNamespace

import jsonschema
import pytest

from baselines.l0.topology import capabilities, discovery
from baselines.l0.transport import peers
from baselines.l0.verification import path_verifier, vocabulary
from common.backend import BenchmarkUnavailable
from common.cases import L0_PARAMETERS, catalog, l0_family
from common.config import DEFAULT, load_configs, merge, validate
from common.controller import create_manifest, execute, result_path, results
from common.registry import ROOT
from common.scheduler import make_plan
from schemas import result

OVERRIDE = {
    "schema_version": 1,
    "device": {"soc": "Ascend950", "cann_version": "test"},
    "channels": [
        {"case_id": "LINK-001", "channel": "AivUbMemChannel", "execution_resource": "AIV",
         "mechanisms": ["ub_memory"], "supported": True, "confidence": "probed",
         "reason": None, "evidence": ["bridge"]},
        {"case_id": "LINK-002", "channel": "AivUrmaChannel", "execution_resource": "AIV",
         "mechanisms": ["urma"], "supported": True, "confidence": "probed",
         "reason": None, "evidence": ["bridge"]},
        {"case_id": "LINK-003", "channel": None, "execution_resource": "AIV",
         "mechanisms": ["ub_ctp"], "supported": False, "confidence": "probed",
         "reason": "channel_not_reported", "evidence": ["bridge"]},
        {"case_id": "LINK-013", "channel": "shmem:UDMA", "execution_resource": "AIV",
         "mechanisms": ["shmem_udma"], "supported": True, "confidence": "probed",
         "reason": None, "evidence": ["shmem"]},
    ],
    "engines": [
        {"engine": "shmem_mte", "execution_resource": "HOST_CPU", "mechanisms": ["shmem_mte"],
         "supported": True, "confidence": "probed", "reason": None, "evidence": []},
    ],
    "accelerators": [
        {"mode": "default", "execution_resource": None, "supported": True, "reason": None},
        {"mode": "aiv", "execution_resource": "AIV", "supported": True, "reason": None},
    ],
    "probes": [{"name": "override", "ok": True, "reason": None, "detail": {}}],
    "message_limits": {},
}


@pytest.fixture(autouse=True)
def clear_cache():
    discovery.clear_cache()
    yield
    discovery.clear_cache()


def test_catalog_distinguishes_layers():
    cases = catalog()
    assert {case.category for case in cases.values() if case.level == "L0"} >= {
        "memory", "transport", "execution", "collective", "compute", "overlap"}
    assert cases["LINK-001"].category == "transport"
    assert cases["MEM-002"].category == "memory"
    assert cases["EXEC-001"].category == "execution"
    assert cases["OVERLAP-009"].category == "overlap"
    # Hardware operation, execution resource and protocol stay separate fields.
    assert L0_PARAMETERS["LINK-001"]["mechanism"] == "ub_memory"
    assert L0_PARAMETERS["LINK-001"]["execution_resource"] == "AIV"
    assert L0_PARAMETERS["LINK-008"]["execution_resource"] == "CCU"
    # URMA is not re-labelled as a protocol subtype of CTP/RTP.
    assert L0_PARAMETERS["LINK-002"]["mechanism"] == "urma"
    assert L0_PARAMETERS["LINK-002"]["protocol"] == "UB_CTP"
    assert l0_family(cases["HW-003"]) == "collective"


def test_capability_matrix_override_and_index():
    matrix = capabilities.discover({}, override=OVERRIDE)
    assert matrix["source"] == "override"
    assert capabilities.support(matrix, case_id="LINK-001")[0] is True
    supported, reason = capabilities.support(matrix, case_id="LINK-003")
    assert supported is False and reason == "channel_not_reported"
    assert capabilities.support(matrix, mechanism="shmem_udma", execution_resource="AIV")[0] is True
    assert capabilities.support(matrix, mechanism="uboe")[0] is False
    jsonschema.validate(matrix, json.loads(
        (ROOT / "schemas/l0-capabilities-v1.json").read_text(encoding="utf-8")))


def test_probe_without_dependencies_is_honest():
    matrix = capabilities.discover({"benchmark": {}})
    assert matrix["source"] == "probe"
    reasons = " ".join(str(probe.get("reason")) for probe in matrix["probes"])
    assert "SHMEM" in reasons or "hccl_test" in reasons
    assert all(entry["confidence"] in ("declared", "probed") for entry in matrix["channels"])


def test_plan_expands_dimensions_and_marks_unsupported(monkeypatch):
    config = load_configs(ROOT / "configs/transport.yaml")[0]
    config = merge(config, {"benchmark": {"message_bytes": [4096], "outstanding": [1, 4],
                                          "peer_counts": [0]}})
    monkeypatch.setenv("EP_L0_CAPABILITIES", "")
    monkeypatch.setattr(discovery, "load", lambda *args, **kwargs: capabilities.discover(
        {}, override=OVERRIDE))
    tasks = make_plan([config], cases=["LINK-001", "LINK-003", "LINK-013"])
    link1 = [task for task in tasks if task["case_id"] == "LINK-001"]
    assert {task["parameters"]["outstanding"] for task in link1} == {1, 4}
    # A plain configuration carries no rank-to-server map, so the topology-aware
    # policies are explicitly not applicable instead of silently degrading to
    # intra-server peers.
    ring = [task for task in link1 if task["parameters"]["peer_policy"] == "ring"]
    assert ring and all(task["status"] == "not_run" for task in ring)
    topology_aware = [task for task in link1
                      if task["parameters"]["peer_policy"] in ("same_server", "cross_server")]
    assert topology_aware
    assert all(task["status"] == "not_applicable" for task in topology_aware)
    assert all("topology" in task["reason"] for task in topology_aware)
    unsupported = [task for task in tasks if task["case_id"] == "LINK-003"]
    assert unsupported and all(task["status"] == "unsupported" for task in unsupported)


def test_plan_execution_sweeps_accelerator(monkeypatch):
    config = load_configs(ROOT / "configs/collectives.yaml")[0]
    config = merge(config, {"benchmark": {"message_bytes": [4096],
                                          "execution": {"modes": ["default", "aiv"]}}})
    monkeypatch.setattr(discovery, "load", lambda *args, **kwargs: capabilities.discover(
        {}, override=OVERRIDE))
    tasks = make_plan([config], cases=["EXEC-001"])
    assert {task["parameters"]["accelerator"] for task in tasks} == {"default", "aiv"}
    assert all(task["operation"] == "alltoall" for task in tasks)


def test_requested_and_observed_are_separate():
    request = vocabulary.requested("hcomm", "put", execution_resource="AIV",
                                   transport_protocol="ub_ctp", channel="AivUrmaChannel")
    evidence = [path_verifier.Evidence(
        source="rank_graph", claim={"transport_protocol": "roce"},
        detail="rank graph reports ROCE")]
    verification = path_verifier.verify(request, evidence)
    assert verification.status == "fallback"
    assert verification.observed["fallback"] is True
    assert verification.conflicts[0]["requested"] == "ub_ctp"
    # The requested block is untouched: configuration is not evidence.
    assert request["transport_protocol"] == "ub_ctp"


def test_unverified_and_verified_paths():
    request = vocabulary.requested("hcomm", "put", execution_resource="AIV",
                                   transport_protocol="ub_memory")
    unknown = path_verifier.verify(request, [])
    assert unknown.status == "unverified"
    assert set(unknown.missing) == {"execution_resource", "transport_protocol"}
    evidence = [path_verifier.Evidence(source="native_descriptor",
                                       claim={"execution_resource": "AIV",
                                              "transport_protocol": "ub_memory"})]
    verified = path_verifier.verify(request, evidence)
    assert verified.status == "verified"
    assert verified.observed["verified_fields"] == ["execution_resource", "transport_protocol"]


def test_heuristic_log_evidence_cannot_verify():
    request = vocabulary.requested("native_hccl", "alltoall", accelerator="aiv",
                                   transport_protocol="auto")
    evidence = [path_verifier.Evidence(source="hccl_log_keyword", confidence="heuristic",
                                       claim={"mechanism": "urma"},
                                       detail="keyword")]
    verification = path_verifier.verify(request, evidence)
    assert verification.status == "unverified"
    assert verification.observed["transport_protocol"] is None


def test_old_result_records_still_validate(tmp_path):
    config = load_configs(ROOT / "configs/smoke_cpu.yaml")[0]
    task = make_plan([config], cases=["COMM-001"])[0]
    manifest = {"run_id": "legacy", "environment": {}}
    record = result(manifest, task, "passed")
    schema = json.loads((ROOT / "schemas/result-v1.json").read_text(encoding="utf-8"))
    jsonschema.validate(record, schema)
    from common.controller import result_path, persist_case
    persist_case(result_path(tmp_path, task), record)
    assert list(results(tmp_path, {"tasks": [task]}))[0]["status"] == "passed"


def test_config_rejects_bad_l0_dimensions():
    with pytest.raises(ValueError, match="transport.enabled"):
        validate(merge(DEFAULT, {"benchmark": {"transport": {"enabled": ["infiniband"]}}}))
    with pytest.raises(ValueError, match="execution.modes"):
        validate(merge(DEFAULT, {"benchmark": {"execution": {"modes": ["gpu"]}}}))
    with pytest.raises(ValueError, match="message_bytes"):
        validate(merge(DEFAULT, {"benchmark": {"message_bytes": []}}))
    with pytest.raises(ValueError, match="outstanding"):
        validate(merge(DEFAULT, {"benchmark": {"outstanding": [0]}}))


def test_l0_report_separates_verification(tmp_path):
    from reporting.l0_report import write
    row = {"case_id": "LINK-001", "benchmark_level": "L0", "category": "transport",
           "status": "passed", "parameters": {"message_bytes": 4096, "outstanding": 1},
           "observed": {"transport_protocol": "ub_memory", "execution_resource": "AIV"},
           "verification": {"status": "verified"},
           "latency": {"p50": 0.01}, "bandwidth": {"payload_bytes": 4096,
                                                   "payload_bandwidth_GBps": 0.4},
           "environment": {}}
    unverified = json.loads(json.dumps(row))
    unverified.update(case_id="LINK-003", verification={"status": "unverified"})
    manifest = {"run_id": "r", "environment": {}}
    write(tmp_path, manifest, [row, unverified], capabilities=OVERRIDE)
    text = (tmp_path / "reports/l0_report.md").read_text(encoding="utf-8")
    assert "verified=1" in text and "unverified=1" in text
    assert "Only `verified` rows may be compared" in text

# ------------------------------------------------- transport mechanism gate ---

def _transport_config(enabled, ep_size=8, **benchmark):
    """A validated transport configuration with an explicit enabled-mechanism list.

    A rank-to-server map is supplied so the topology-aware peer policies are
    testable: EP8 is one server, EP64/EP128 span several.
    """
    config = load_configs(ROOT / "configs/transport.yaml")[0]
    server_size = 8
    topology = {"server_ids": [rank // server_size for rank in range(ep_size)]}
    config = merge(config, {"distributed": {"ep_size": ep_size},
                            "topology": topology,
                            "workload": {"tokens_per_rank": 128},
                            "benchmark": {"transport": {"enabled": enabled}, **benchmark}})
    return validate(config)


@pytest.fixture()
def capability_override(monkeypatch):
    monkeypatch.setenv("EP_L0_CAPABILITIES", "")
    monkeypatch.setattr(discovery, "load", lambda *args, **kwargs: capabilities.discover(
        {}, override=OVERRIDE))


def test_transport_enabled_filters_cases(capability_override):
    config = _transport_config(["ub_memory"])
    tasks = make_plan([config], cases=["LINK-001", "LINK-002"])
    link1 = [task for task in tasks if task["case_id"] == "LINK-001"]
    link2 = [task for task in tasks if task["case_id"] == "LINK-002"]
    # The enabled mechanism keeps its normal parameter variants.
    ring = [task for task in link1 if task["parameters"]["peer_policy"] == "ring"]
    assert ring and all(task["status"] == "not_run" for task in ring)
    assert len({task["parameters"]["peer_count"] for task in ring}) == 4
    # The disabled mechanism produces no executable task at all.
    assert link2 and all(task["status"] != "not_run" for task in link2)


def test_disabled_transport_is_not_applicable(capability_override):
    config = _transport_config(["ub_memory"])
    tasks = make_plan([config], cases=["LINK-002"])
    assert len(tasks) == 1
    task = tasks[0]
    assert task["status"] == "not_applicable"
    assert "urma" in task["reason"]
    assert "benchmark.transport.enabled" in task["reason"]
    # The case is still in the manifest: it is reported, not deleted.
    assert task["case_id"] == "LINK-002"
    assert task["parameters"]["mechanism"] == "urma"


def test_disabled_transport_never_executes(tmp_path, monkeypatch, capability_override):
    config = _transport_config(["ub_memory"])
    tasks = make_plan([config], cases=["LINK-002"])
    launched = []

    def launch(*args, **kwargs):
        launched.append(args)
        raise AssertionError("a disabled transport case must not be launched")

    import common.controller
    monkeypatch.setattr(common.controller, "launch_worker", launch)
    root, manifest = create_manifest(tmp_path, tasks, "disabled-transport")
    assert execute(root, manifest, SimpleNamespace(nnodes=1, node_rank=0)) == 0
    assert launched == []
    record = json.loads(result_path(root, tasks[0]).read_text(encoding="utf-8"))
    assert record["status"] == "not_applicable"
    assert "benchmark.transport.enabled" in record["error_reason"]


def test_empty_transport_enabled_disables_all_transport(capability_override):
    config = _transport_config([])
    tasks = make_plan([config], cases=["LINK-001", "LINK-002", "LINK-013"])
    assert tasks
    assert all(task["status"] == "not_applicable" for task in tasks)
    assert all("not enabled" in task["reason"] for task in tasks)
    # An unset list keeps the legacy meaning: no subset was chosen, so the
    # mechanism is allowed.  The explicit empty list stays a real decision.
    legacy = validate(merge(config, {"benchmark": {"transport": {"enabled": None}}}))
    allowed = make_plan([legacy], cases=["LINK-001"])
    assert allowed and any(task["status"] == "not_run" for task in allowed)


def test_non_transport_l0_cases_are_unaffected(capability_override):
    config = _transport_config([])
    tasks = make_plan([config], cases=["MEM-001", "HW-001", "HW-010", "EXEC-001", "OVERLAP-001"])
    assert tasks
    assert not any("transport mechanism" in (task["reason"] or "") for task in tasks)
    assert {task["case_id"] for task in tasks} == {
        "MEM-001", "HW-001", "HW-010", "EXEC-001", "OVERLAP-001"}
    # Their own dimension sweeps are untouched by the transport gate.
    memory = [task for task in tasks if task["case_id"] == "MEM-001"]
    assert {task["parameters"]["message_bytes"] for task in memory} == {
        4096, 65536, 1048576, 16777216, 268435456, 1073741824}
    assert {task["parameters"]["shape_mode"] for task in tasks
            if task["case_id"] == "HW-010"} == {"fixed", "workload"}


# ------------------------------------------------------- transport peers ------

def _domains(ep_size, server_size=8):
    return [rank // server_size for rank in range(ep_size)]


def test_cross_server_without_remote_peer_is_not_applicable():
    with pytest.raises(BenchmarkUnavailable) as failure:
        peers.select(8, 0, policy="cross_server", domains=_domains(8))
    assert failure.value.status == "not_applicable"
    assert "cross-server" in str(failure.value)
    # Every rank of a single-server group is refused, at any peer count.
    for rank in range(8):
        with pytest.raises(BenchmarkUnavailable):
            peers.select(8, rank, policy="cross_server", domains=_domains(8), peer_count=0)


def test_same_server_without_topology_is_not_applicable():
    for policy in ("same_server", "cross_server"):
        with pytest.raises(BenchmarkUnavailable) as failure:
            peers.select(8, 0, policy=policy, domains=None)
        assert failure.value.status == "not_applicable"
        assert "topology" in str(failure.value)
    # An incomplete map is as unusable as a missing one.
    with pytest.raises(BenchmarkUnavailable):
        peers.select(8, 0, policy="same_server", domains=[0] * 4)


def test_peer_selection_does_not_fallback():
    domains = _domains(8)
    with pytest.raises(BenchmarkUnavailable):
        peers.select(8, 0, policy="cross_server", domains=domains, peer_count=0)
    # An unknown policy is refused, never treated as `ring`.
    with pytest.raises(BenchmarkUnavailable):
        peers.select(8, 0, policy="auto", domains=domains)
    # A topology-aware policy never widens to intra-server peers either: EP64
    # cross-server peers are all outside rank 0's server, EP8 has none.
    cross = peers.select(64, 0, policy="cross_server", domains=_domains(64), peer_count=0)
    assert len(cross) == 56 and all(rank >= 8 for rank in cross)


def test_peer_count_cannot_exceed_policy_candidates():
    domains = _domains(64)
    with pytest.raises(BenchmarkUnavailable) as failure:
        peers.select(64, 0, policy="same_server", domains=domains, peer_count=8)
    assert "exceeds" in str(failure.value)
    assert len(peers.select(64, 0, policy="same_server", domains=domains, peer_count=7)) == 7
    assert len(peers.select(64, 0, policy="same_server", domains=domains, peer_count=0)) == 7


def test_ring_peer_selection_is_deterministic():
    for ep_size in (8, 64, 128):
        domains = _domains(ep_size)
        for rank in (0, 1, ep_size - 1):
            first = peers.select(ep_size, rank, "ring", 4, domains)
            assert first == peers.select(ep_size, rank, "ring", 4, domains)
            assert len(first) == len(set(first)) == 4
            assert rank not in first
        assert len(peers.select(ep_size, 0, "ring", 0, domains)) == ep_size - 1


def test_peer_selection_matches_the_topology():
    expected = {
        8: {"same_server": 7, "cross_server": 0, "ring": 7},
        64: {"same_server": 7, "cross_server": 56, "ring": 63},
        128: {"same_server": 7, "cross_server": 120, "ring": 127},
    }
    for ep_size, counts in expected.items():
        domains = _domains(ep_size)
        for policy, count in counts.items():
            if policy == "cross_server" and count == 0:
                with pytest.raises(BenchmarkUnavailable):
                    peers.select(ep_size, 0, policy=policy, domains=domains)
                continue
            selected = peers.select(ep_size, 0, policy=policy, domains=domains, peer_count=0)
            assert len(selected) == count, (ep_size, policy)
            if policy == "same_server":
                assert all(rank < 8 for rank in selected)
            if policy == "cross_server":
                assert all(rank >= 8 for rank in selected)


def test_channel_peer_helper_delegates_to_the_policy():
    """The L0 driver uses the same policy; it never resolves peers on its own."""
    pytest.importorskip("torch")  # the native driver imports the tensor runtime
    from baselines.l0.transport import channel

    ctx = SimpleNamespace(size=16, rank=3,
                          config={"distributed": {"ep_size": 16},
                                  "topology": {"server_ids": _domains(16)}})
    ring = channel._peers(ctx, {"peer_count": 4})
    assert len(ring) == 4 and ctx.rank not in ring
    assert ring == peers.select(16, 3, "ring", 4, _domains(16))
    with pytest.raises(BenchmarkUnavailable):
        channel._peers(SimpleNamespace(size=8, rank=0, config={"topology": {}}),
                       {"peer_count": 0, "peer_policy": "cross_server"})
