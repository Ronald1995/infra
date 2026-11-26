"""L0 reporting: failed cases keep their identity and stay in the tables."""
import json

import pytest

from analysis import l0 as l0_analysis
from common.cases import catalog
from reporting.l0_report import write as write_l0
from schemas import result

LINK_PARAMETERS = {
    "variant": "default", "message_bytes": 4096, "outstanding": 1, "peer_count": 1,
    "peer_policy": "ring", "mechanism": "ub_memory", "execution_resource": "AIV",
}
STATUSES = ("passed", "runtime_failed", "unsupported", "dependency_missing")


@pytest.fixture
def link_tasks(task_for):
    """One LINK-001 point per status, at a distinct message size.

    A distinct size keeps each attempt its own variant line, which is the shape a
    real run has: one task per (case, parameters, EP size).
    """
    return [task_for("LINK-001", ep_size=8, device="npu",
                     parameters={**LINK_PARAMETERS, "message_bytes": 4096 << index},
                     index=index) for index in range(len(STATUSES))]


@pytest.fixture
def link_rows(manifest_for, link_tasks):
    manifest = manifest_for(link_tasks, run_id="l0-failures")
    rows = []
    for task, status in zip(link_tasks, STATUSES):
        record = result(manifest, task, status,
                        None if status == "passed" else f"{status} in the fixture")
        record["correctness"] = {"passed": True} if status == "passed" else None
        if status == "passed":
            record["latency"] = {"unit": "ms", "scope": "per_iteration_rank_max", "p50": .01}
            record["bandwidth"] = {"unit": "bytes", "bandwidth_unit": "GB/s",
                                   "scope": "per_rank", "source": "declared_payload_model",
                                   "payload_bytes": 4096, "payload_bandwidth_GBps": .4}
            record["completion"] = {"level": "remote_visible", "detail": "fixture",
                                    "proves_transfer": True}
            record["observed"] = {"transport_protocol": "ub_memory", "execution_resource": "AIV",
                                  "verification_status": "verified", "fallback": False}
            record["verification"] = {"status": "verified", "evidence": [
                {"source": "native_descriptor", "claim": {"transport_protocol": "ub_memory"}}]}
            record["requested"] = {"backend": "native_hccl", "operation": "put",
                                   "transport_protocol": "ub_memory",
                                   "execution_resource": "AIV"}
        rows.append(record)
    return manifest, rows


def test_failed_l0_records_keep_category_and_parameters(link_rows):
    """The P0 contract: a failure keeps the identity of the point that was planned."""
    _manifest, rows = link_rows
    for record in rows:
        assert record["category"] == "transport"
        assert record["case_parameters"]["mechanism"] == LINK_PARAMETERS["mechanism"]
        assert record["case_parameters"]["message_bytes"] in {4096, 8192, 16384, 32768}
        assert record["benchmark_level"] == "L0"
        assert record["backend_id"] == "hardware"
        assert record["operation"] == "put"
        assert record["ep_size"] == 8


def test_failed_l0_record_matches_its_planned_task(link_rows, link_tasks):
    _manifest, rows = link_rows
    for record, task in zip(rows, link_tasks):
        assert record["task_id"] == task["task_id"]
        assert record["case_id"] == task["case_id"]
        # The category comes from the catalog, not from ID text guessing.
        assert record["category"] == catalog()[task["case_id"]].category


def test_l0_transport_report_still_shows_failed_cases(link_rows, tmp_path):
    manifest, rows = link_rows
    write_l0(tmp_path, manifest, rows, capabilities=None)
    text = (tmp_path / "reports/l0_report.md").read_text(encoding="utf-8")
    for status in STATUSES:
        assert status in text
    # The failed LINK point appears in the transport table with its parameters.
    transport = text[text.index("## 5. Transport curves"):text.index("## 6.")]
    for status in STATUSES:
        assert status in transport
    assert "4096" in transport
    assert "1" in transport
    # And in the failure section with its requested mechanism.
    failures = text[text.index("## 10. Failures"):text.index("## 11.")]
    assert "runtime_failed" in failures
    assert "ub_memory" in failures


def test_failed_l0_rows_survive_the_analysis(link_rows):
    _manifest, rows = link_rows
    data = l0_analysis.analyse(rows)
    assert data["summary"]["records"] == 4
    assert data["summary"]["statuses"] == {"dependency_missing": 1, "passed": 1,
                                           "runtime_failed": 1, "unsupported": 1}
    transport = data["transport"]
    assert len(transport) == 1
    entry = transport[0]
    assert entry["case_id"] == "LINK-001"
    assert entry["passed"] == 1 and entry["verified"] == 1
    assert entry["statuses"] == {"dependency_missing": 1, "passed": 1,
                                 "runtime_failed": 1, "unsupported": 1}
    # One point per distinct parameter set, and a failed point is kept with its status.
    scaleout = data["scale_out"]
    assert len(scaleout) == 4
    assert all(entry["ambiguous"] == [] for entry in scaleout)
    assert {entry["points"][0]["status"] for entry in scaleout} == set(STATUSES)
    assert {entry["message_bytes"] for entry in scaleout} == {4096, 8192, 16384, 32768}


def test_unverified_and_failed_are_not_reported_as_verified(link_rows):
    _manifest, rows = link_rows
    statuses = [l0_analysis.verification(record) for record in rows]
    assert ("verified", "recorded") in statuses
    # `unsupported` and `dependency_missing` both mean "the point could not be tested".
    assert statuses.count(("unsupported", "derived")) == 2
    assert ("failed", "derived") in statuses


def test_requested_values_come_from_the_plan_not_the_observation(link_rows):
    _manifest, rows = link_rows
    for record in rows:
        request = l0_analysis.request_of(record)
        assert request["mechanism"] == "ub_memory"
        assert request["mechanism_source"] in {"recorded_request", "planned_parameters"}
        observed = l0_analysis.observation_of(record)
        if record["status"] != "passed":
            assert observed["mechanism"] is None and observed["available"] is False


def test_l0_metrics_and_curves_are_written(link_rows, tmp_path):
    manifest, rows = link_rows
    write_l0(tmp_path, manifest, rows, capabilities={"source": "test", "channels": [],
                                                     "engines": [], "accelerators": []})
    metrics = json.loads((tmp_path / "reports/l0_metrics.json").read_text(encoding="utf-8"))
    assert metrics["summary"]["records"] == 4
    for measurement in metrics["measurements"]:
        assert measurement["requested_mechanism"] == "ub_memory"
        assert measurement["message_bytes"] in {4096, 8192, 16384, 32768}
    csv = (tmp_path / "reports/l0_curves.csv").read_text(encoding="utf-8")
    assert "requested_mechanism" in csv.splitlines()[0]
    assert csv.count("LINK-001") == 4


def test_l0_without_results_explains_itself(task_for, manifest_for, tmp_path):
    manifest = manifest_for([], run_id="no-l0")
    write_l0(tmp_path, manifest, [], capabilities=None)
    text = (tmp_path / "reports/l0_report.md").read_text(encoding="utf-8")
    assert "Why Analysis Is Unavailable" in text or "Why Analysis Is Unavailable" in text
    assert "This run contains no L0 result record." in text
    assert "## Required Conditions" in text


def test_a_passed_l0_row_without_evidence_is_unverified(link_rows):
    _manifest, rows = link_rows
    passed = rows[0]
    passed.pop("verification")
    passed["observed"] = {"transport_protocol": "ub_memory", "execution_resource": "AIV"}
    status, source = l0_analysis.verification(passed)
    assert status == "unverified"
    assert source == "derived"


def test_l0_scaleout_keeps_ring_and_cross_server_apart(task_for, manifest_for):
    """Two peer policies are two variants; they must not collapse into one line."""
    tasks = [task_for("LINK-001", ep_size=8, device="npu",
                      parameters={**LINK_PARAMETERS, "peer_policy": policy}, index=index)
             for index, policy in enumerate(("ring", "cross_server"))]
    manifest = manifest_for(tasks, run_id="scaleout")
    rows = [result(manifest, task, "unsupported", "no cross-server peer") for task in tasks]
    data = l0_analysis.analyse(rows)
    assert len(data["scale_out"]) == 2
    assert {entry["peer_policy"] for entry in data["scale_out"]} == {"ring", "cross_server"}
    assert all(not entry["comparable"] for entry in data["scale_out"])
