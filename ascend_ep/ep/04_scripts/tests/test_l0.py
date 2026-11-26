"""L0 transport refactor: offline tests for discovery, planning, schema and report.

Everything here is mock/offline.  No NPU is touched, so nothing in this file is a
hardware performance result; the fixtures are explicitly marked as overrides.
"""
import json

import jsonschema
import pytest

from baselines.l0.topology import capabilities, discovery
from baselines.l0.verification import path_verifier, vocabulary
from common.cases import L0_PARAMETERS, catalog, l0_family
from common.config import DEFAULT, load_configs, merge, validate
from common.controller import results
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
    unsupported = [task for task in tasks if task["case_id"] == "LINK-003"]
    assert unsupported and all(task["status"] == "unsupported" for task in unsupported)
    assert all(task["status"] == "not_run" for task in link1)


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
