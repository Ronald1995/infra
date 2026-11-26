"""The comparability rules: what may be compared with what, and why not."""
from copy import deepcopy

import pytest

from analysis import comparability
from analysis.comparison import comparison_groups
from schemas import result


def _row(manifest, task, status="passed", **overrides):
    record = result(manifest, task, status)
    record["correctness"] = {"passed": True} if status == "passed" else None
    record["latency"] = ({"unit": "ms", "scope": "per_iteration_rank_max",
                          "p50": overrides.pop("p50", 1.0)} if status == "passed" else None)
    record["source"] = "measured"
    record["completion"] = {"level": "remote_visible"}
    record.update(overrides)
    return record


@pytest.fixture
def run(manifest_for, task_for):
    two = [task_for("COMM-001", backend_id=backend, index=index)
           for index, backend in enumerate(("alltoall_hccl", "allgather_hccl"))]
    single = task_for("COMM-002", backend_id="alltoall_hccl", index=7)
    manifest = manifest_for(two + [single], run_id="comparability")
    return manifest


def test_two_backends_same_conditions_are_comparable(run):
    first, second, _ = run["tasks"]
    left = _row(run, first, p50=2.0)
    right = _row(run, second, p50=1.0)
    assert comparability.incompatibility(left, right) == []
    assert comparability.backend_comparison_key(left) == comparability.backend_comparison_key(right)
    groups = comparison_groups([left, right])["groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["kind"] == "actual_comparison"
    assert group["comparison"] is True
    assert group["backends"] == ["allgather_hccl", "alltoall_hccl"]
    # Sorted by P50, so the fastest backend is first and the spread is 2x.
    assert group["members"][0]["backend"] == "allgather_hccl"
    assert group["spread_ratio"] == pytest.approx(2.0)
    assert group["members"][0]["speedup_vs_slowest"] == pytest.approx(2.0)
    assert group["members"][1]["difference_percent"] == pytest.approx(100.0)


def test_one_backend_is_a_single_measurement(run):
    _, _, single = run["tasks"]
    groups = comparison_groups([_row(run, single)])["groups"]
    assert len(groups) == 1
    assert groups[0]["kind"] == "single_measurement"
    assert groups[0]["comparison"] is False
    assert "nothing to compare" in groups[0]["reason"]
    assert "speedup_vs_slowest" not in groups[0]["members"][0]


def test_repeats_of_one_backend_are_not_a_comparison(run):
    first, _, _ = run["tasks"]
    repeat = deepcopy(first)
    repeat["task_id"] = first["task_id"] + "-repeat"
    groups = comparison_groups([_row(run, first), _row(run, repeat)])["groups"]
    assert len(groups) == 1
    assert groups[0]["kind"] == "repeated_measurement"
    assert "not a comparison" in groups[0]["reason"]


def test_different_workload_is_not_comparable(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = _row(run, second)
    right["config"] = deepcopy(second["config"])
    right["config"]["workload"]["tokens_per_rank"] = 4096
    reasons = comparability.incompatibility(left, right)
    assert reasons and any("workload" in reason for reason in reasons)
    assert len(comparison_groups([left, right])["groups"]) == 2


def test_different_timing_boundary_is_not_comparable(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = _row(run, second)
    right["config"] = deepcopy(second["config"])
    right["config"]["benchmark"]["timing"] = "host_api"
    reasons = comparability.incompatibility(left, right)
    assert any("timing boundary" in reason for reason in reasons)


def test_different_variant_is_not_comparable(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = _row(run, second)
    right["case_parameters"] = {**right["case_parameters"], "variant": "chunks_4"}
    reasons = comparability.incompatibility(left, right)
    assert any("case parameters" in reason for reason in reasons)


def test_different_ep_size_lands_in_scaling_not_backend_comparison(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = deepcopy(left)
    right["task_id"] = second["task_id"]
    right["backend_id"] = second["backend_id"]
    right["config"] = deepcopy(first["config"])
    right["config"]["distributed"]["ep_size"] = 64
    reasons = comparability.incompatibility(left, right)
    assert any("EP size" in reason for reason in reasons)
    assert comparability.scaling_key(left) == comparability.scaling_key(
        deepcopy({**left, "config": right["config"]}))


def test_different_environment_is_never_merged(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = _row(run, second)
    right["environment"] = {**right["environment"], "cann_version": "9.0.RC1"}
    reasons = comparability.incompatibility(left, right)
    assert any("environment" in reason for reason in reasons)


def test_l0_unverified_path_is_not_comparable(l0_matrix):
    task = {"task_id": "LINK-001-x", "case_id": "LINK-001", "level": "L0",
            "operation": "put", "backend_id": "hardware",
            "config": {"suite_id": "test", "topology": {}, "device": "npu",
                       "distributed": {"ep_size": 8, "tp_size": 1},
                       "benchmark": {"timing": "output_ready"}, "model": {}, "workload": {}},
            "parameters": {"variant": "default", "message_bytes": 4096},
            "source_versions": {}, "baseline": True, "ep_size": 8,
            "case_category": "transport"}
    manifest = {"run_id": "l0", "environment": {"npu_models": ["Ascend910B4"]}}
    unverified = _row(manifest, task, verification={"status": "unverified"})
    verified = _row(manifest, task, verification={"status": "verified"})
    assert comparability.eligibility(unverified).eligible is True
    assert comparability.comparable(unverified) is False
    assert "cannot be attributed" in comparability.exclusion_reason(unverified)
    assert comparability.comparable(verified) is True
    assert comparability.exclusion_reason(verified) is None


def test_failed_rows_explain_why_they_are_excluded(run):
    first, _, _ = run["tasks"]
    for status in ("runtime_failed", "unsupported", "dependency_missing", "timeout"):
        record = _row(run, first, status)
        assert comparability.eligibility(record).eligible is False
        assert status in comparability.exclusion_reason(record)
    wrong = _row(run, first, correctness={"passed": False})
    assert "correctness" in comparability.exclusion_reason(wrong)
    no_latency = _row(run, first, latency={"unit": "ms", "p50": None})
    assert "p50" in comparability.exclusion_reason(no_latency)


def test_completion_semantics_are_part_of_the_key(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = _row(run, second)
    right["completion"] = {"level": "local_complete"}
    reasons = comparability.incompatibility(left, right)
    assert any("completion" in reason for reason in reasons)


def test_legacy_parameters_field_is_read(run):
    first, _, _ = run["tasks"]
    record = _row(run, first)
    record["parameters"] = record.pop("case_parameters")
    assert comparability.case_parameters(record)["variant"] == "default"
    assert comparability.semantic_parameters(record) == {"variant": "default"}


def test_non_semantic_parameters_do_not_split_a_group(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = _row(run, second)
    right["case_parameters"] = {**right["case_parameters"], "reason": "not applicable"}
    assert comparability.incompatibility(left, right) == []


def test_scaling_key_keeps_backend_and_drops_ep_size(run):
    first, second, _ = run["tasks"]
    left = _row(run, first)
    right = deepcopy(left)
    right["config"] = deepcopy(first["config"])
    right["config"]["distributed"] = {"ep_size": 64, "tp_size": 1}
    right["routing_trace_hash"] = "other-trace"
    assert comparability.scaling_key(left) == comparability.scaling_key(right)
    right["backend_id"] = second["backend_id"]
    assert comparability.scaling_key(left) != comparability.scaling_key(right)
