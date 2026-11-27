"""Coverage metrics and long-tail analysis.

Coverage: planning can be complete while verified coverage is not -- the four numbers
must stay separate.  Long tail: a ratio needs a baseline under the same condition; a
missing baseline is ``not_applicable``, never a made-up denominator.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import long_tail  # noqa: E402
from common import coverage  # noqa: E402

TASK = {"case_id": "COMM-001", "level": "L1", "backend_id": "x", "ep_size": 8}


def row(status="passed", **extra):
    base = {"task_id": "t", "case_id": "COMM-001", "benchmark_level": "L1",
            "backend_id": "x", "ep_size": 8, "status": status,
            "config": {"benchmark": {}, "workload": {"tokens_per_rank": 4096}}}
    base.update(extra)
    return base


def manifest(*tasks):
    return {"tasks": [{"case_id": t["case_id"], "level": t["level"],
                       "backend_id": t["backend_id"], "ep_size": t["ep_size"]}
                      for t in tasks]}


# ------------------------------------------------------------------ coverage ---

def test_planning_is_complete_even_when_nothing_ran():
    report = coverage.metrics(manifest(TASK), [row("not_run")])
    assert report["planning_coverage"] == 1.0
    assert report["execution_coverage"] == 0.0
    assert report["verified_performance_coverage"] == 0.0


def test_not_applicable_leaves_the_denominator():
    tasks = [TASK, {"case_id": "COMM-002", "level": "L1", "backend_id": "x", "ep_size": 8}]
    rows = [row(), row("not_applicable", case_id="COMM-002", task_id="t2")]
    report = coverage.metrics(manifest(*tasks), rows)
    assert report["applicable"] == 1
    assert report["execution_coverage"] == 1.0
    assert report["status_breakdown"]["not_applicable"] == 1


def test_unverified_is_executed_but_not_verified():
    report = coverage.metrics(manifest(TASK), [row("unverified")])
    assert report["executed"] == 1
    assert report["correct"] == 0
    assert report["verified"] == 0


def test_a_fallback_measurement_is_not_verified():
    rows = [row(observation={"fallback": True})]
    report = coverage.metrics(manifest(TASK), rows)
    assert report["correct"] == 1
    assert report["verified"] == 0


def test_a_failed_attempt_lowers_correctness():
    report = coverage.metrics(manifest(TASK), [row("correctness_failed")])
    assert report["executed"] == 1
    assert report["correct"] == 0
    assert report["correctness_pass_rate"] == 0.0
    assert report["status_breakdown"]["failed"] == 1


def test_a_missing_row_is_counted_and_named():
    tasks = [TASK, {"case_id": "MOE-001", "level": "L3", "backend_id": "x", "ep_size": 8}]
    report = coverage.metrics(manifest(*tasks), [row()])
    assert report["missing_count"] == 1
    assert report["execution_coverage"] == pytest.approx(0.5)


def test_blocked_topology_keeps_the_requirement_in_the_denominator():
    report = coverage.metrics(manifest(TASK), [row("blocked_topology")])
    assert report["executed"] == 0
    assert report["execution_coverage"] == 0.0
    assert "blocked_topology" in report["blocked"]


# ----------------------------------------------------------------- long tail ---

def _stressed(status="passed", p50=12.0, straggler=None):
    config = {"benchmark": {"straggler": {"enabled": True, "injection": {
        "method": "host_delay", "stage": "before_operation", "delay_us": 1000}}},
        "workload": {"tokens_per_rank": 4096}}
    return row(status, task_id="stressed", latency={"p50_ms": p50}, config=config,
               straggler=straggler)


def test_slowdown_and_retention_use_the_matching_baseline():
    rows = [row(latency={"p50_ms": 10.0}, task_id="base"), _stressed()]
    result = long_tail.analyze(rows)
    pair = result["pairs"][0]
    assert pair["slowdown_ratio"] == pytest.approx(1.2)
    assert pair["throughput_retention"] == pytest.approx(1 / 1.2)
    assert pair["amplification"] == pytest.approx(2.0)
    assert result["summary"]["slowdown_p50"] == pytest.approx(1.2)


def test_without_a_baseline_the_point_is_not_applicable():
    result = long_tail.analyze([_stressed()])
    assert result["pairs"] == []
    assert result["not_applicable"][0]["status"] == "not_applicable"
    assert "baseline" in result["not_applicable"][0]["reason"]


def test_a_different_condition_is_not_used_as_a_baseline():
    other = row(latency={"p50_ms": 10.0}, task_id="base",
                case_id="MOE-001")
    result = long_tail.analyze([other, _stressed()])
    assert result["pairs"] == []
    assert result["not_applicable"]


def test_calibration_error_is_reported():
    straggler = {"ranks": [{"rank": 0, "requested_delay_us": 1000,
                            "actual_delay_us_sum": 2500, "injections": 2}],
                 "calibration_failures": 1}
    rows = [row(latency={"p50_ms": 10.0}, task_id="base"), _stressed(straggler=straggler)]
    pair = long_tail.analyze(rows)["pairs"][0]
    assert pair["calibration"]["actual_us"] == pytest.approx(1250.0)
    assert pair["calibration"]["error"] == pytest.approx(0.25)
    assert pair["calibration"]["failures"] == 1


def test_a_routing_target_is_also_a_stress_kind():
    config = {"benchmark": {}, "workload": {
        "tokens_per_rank": 4096, "routing": "target_distribution",
        "routing_control": {"target_metric": "rank_max_mean", "target_value": 4.0}}}
    stressed = row(latency={"p50_ms": 15.0}, task_id="target", config=config)
    baseline_rows = [row(latency={"p50_ms": 10.0}, task_id="base")]
    baseline_rows[0]["config"] = {"benchmark": {}, "workload": {
        "tokens_per_rank": 4096, "routing": "target_distribution",
        "routing_control": {"target_metric": "rank_max_mean", "target_value": 4.0}}}
    # Same routing control on both sides means the "baseline" is not a control.
    assert long_tail.analyze([*baseline_rows, stressed])["pairs"] == []
    baseline_rows[0]["config"] = {"benchmark": {}, "workload": {
        "tokens_per_rank": 4096, "routing": "balanced"}}
    result = long_tail.analyze([*baseline_rows, stressed])
    assert len(result["pairs"]) == 1
    pair = result["pairs"][0]
    assert pair["slowdown_ratio"] == pytest.approx(1.5)
    assert pair["amplification"] is None  # no injected delay for a routing target


def test_a_recorded_non_fallback_is_not_a_fallback():
    """The runtime records `fallback: {used: false}` on the normal path."""
    passed = row(observation={"fallback": {"used": False, "reason": None}})
    assert coverage.metrics(manifest(TASK), [passed])["verified"] == 1
    fell_back = row(observation={"fallback": {"used": True, "reason": "cpu"}})
    assert coverage.metrics(manifest(TASK), [fell_back])["verified"] == 0


def test_a_single_configuration_manifest_carries_the_summary():
    from common.plan import matrix_metadata

    tasks = [dict(TASK, task_id="t1", status="not_run", config={"suite_id": "all"},
                  parameters={}, suite_ids=["all"], case_category="operator",
                  case_family="operator", case_tags=[], case_phase="inference",
                  handler="h", correctness_requirements=[], alias_of=None,
                  config_fingerprint="f", source_versions={}, baseline=False)]
    metadata = matrix_metadata(tasks)
    assert metadata["plan_summary"]["tasks"] == 1
    assert metadata["plan_summary"]["estimation"]["status"] == "unavailable"
