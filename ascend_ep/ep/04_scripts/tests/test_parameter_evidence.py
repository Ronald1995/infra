"""Requested vs observed parameter evidence.

A configured sweep is a claim about the runtime. These tests pin the rule that a
claim becomes evidence only when the handler reports the value it used -- copying the
configuration into `observed` is not verification.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import coverage, parameter_evidence  # noqa: E402


def test_a_fully_observed_task_is_verified():
    record = parameter_evidence.record({"cores": 2, "quiet_mode": 1, "outstanding": 4},
                                       {"cores": 2, "quiet_mode": 1, "outstanding": 4})
    assert record["verified"] is True
    assert record["unverified"] == []
    assert record["parameters"]["cores"] == {"requested": 2, "observed": 2,
                                             "verification": "verified"}


def test_a_missing_observation_is_unverified_with_a_reason():
    record = parameter_evidence.record({"cores": 2}, None)
    entry = record["parameters"]["cores"]
    assert entry["verification"] == "unverified"
    assert entry["observed"] is None
    assert "intention, not evidence" in entry["reason"]
    assert record["verified"] is False


def test_a_mismatching_observation_is_unverified():
    record = parameter_evidence.record({"outstanding": 4}, {"outstanding": 1})
    assert record["parameters"]["outstanding"] == {
        "requested": 4, "observed": 1, "verification": "unverified",
        "reason": "the runtime recorded a different value"}


def test_parameters_outside_the_audited_set_are_ignored():
    record = parameter_evidence.record({"variant": "device", "mode": "write"})
    assert record["parameters"] == {}
    assert record["verified"] is True


def test_a_none_value_is_not_a_claim():
    record = parameter_evidence.record({"message_bytes": None, "cores": 1},
                                       {"cores": 1})
    assert set(record["parameters"]) == {"cores"}
    assert record["verified"] is True


def test_observed_from_only_keeps_requested_dimensions():
    observed = parameter_evidence.observed_from(
        {"cores": 4, "quiet_mode": 0}, cores=4, quiet_mode=0, outstanding=2)
    assert observed == {"cores": 4, "quiet_mode": 0}


def test_the_shmem_udma_handler_reports_what_it_passed():
    """The device/host variants record the values handed to the runtime API."""
    source = (ROOT / "baselines" / "l0" / "shmem" / "udma.py").read_text(encoding="utf-8")
    assert "dimension_observed=parameter_evidence.observed_from(" in source
    assert "cores=cores" in source and "quiet_mode=quiet_mode" in source
    assert "outstanding=outstanding" in source


def test_coverage_excludes_an_unverified_dimension():
    task = {"case_id": "LINK-013", "level": "L0", "backend_id": "hardware", "ep_size": 8}
    manifest = {"tasks": [task]}
    verified_row = {"task_id": "t", "case_id": "LINK-013", "benchmark_level": "L0",
                    "backend_id": "hardware", "ep_size": 8, "status": "passed",
                    "dimension_evidence": {"verified": True, "unverified": []}}
    unverified_row = dict(verified_row, dimension_evidence={
        "verified": False, "unverified": ["quiet_mode"]})
    assert coverage.metrics(manifest, [verified_row])["verified"] == 1
    assert coverage.metrics(manifest, [unverified_row])["verified"] == 0
