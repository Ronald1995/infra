"""The coverage section reports the four metrics and the capability gaps.

The report must not collapse coverage into one number, and it must name what could not
be exercised instead of leaving a green-looking summary.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting.sections import coverage  # noqa: E402

METRICS = {
    "planned": 10, "present": 10, "applicable": 8, "executed": 6, "correct": 5,
    "verified": 4,
    "planning_coverage": 1.0, "execution_coverage": 0.75,
    "correctness_pass_rate": 5 / 6, "verified_performance_coverage": 0.5,
    "status_breakdown": {"passed": 5, "unverified": 1, "not_applicable": 2,
                         "not_run": 1, "failed": 1},
    "missing_count": 0,
}


def context(metrics=METRICS, rows=()):
    return SimpleNamespace(analysis={"coverage_metrics": metrics}, rows=list(rows))


def joined(lines):
    return "\n".join(lines)


def test_the_four_metrics_are_reported_separately():
    text = joined(coverage.coverage_metrics_lines(context()))
    for label in ("Planning", "Execution", "Correctness", "Verified performance"):
        assert label in text
    assert "100.0%" in text
    assert "75.0%" in text
    assert "50.0%" in text
    assert "6 / 8 applicable" in text


def test_missing_metrics_are_explained_not_zeroed():
    text = joined(coverage.coverage_metrics_lines(context(metrics=None)))
    assert "No coverage metrics" in text
    assert "not reported" in text
    assert "0.0%" not in text


def test_a_zero_denominator_is_not_reported_as_zero_percent():
    text = joined(coverage.coverage_metrics_lines(context(metrics={
        "planned": 2, "present": 2, "applicable": 0, "executed": 0, "correct": 0,
        "verified": 0, "planning_coverage": 1.0, "execution_coverage": None,
        "correctness_pass_rate": None, "verified_performance_coverage": None,
        "status_breakdown": {"not_applicable": 2}, "missing_count": 0})))
    assert "| n/a |" in text
    # "100.0%" contains the substring "0.0%", so compare without it.
    assert "0.0%" not in text.replace("100.0%", "")


def test_gaps_are_grouped_with_counts_and_reasons():
    rows = [
        {"status": "not_implemented", "case_id": "LINK-012", "backend_id": "hardware",
         "error_reason": "MTE kernel missing"},
        {"status": "not_implemented", "case_id": "LINK-012", "backend_id": "hardware",
         "error_reason": "MTE kernel missing"},
        {"status": "unverified", "case_id": "LINK-013", "backend_id": "hardware",
         "error_reason": "quiet_mode unverified"},
        {"status": "passed", "case_id": "LINK-001", "backend_id": "hardware"},
    ]
    text = joined(coverage.gap_lines(context(rows=rows)))
    assert "`not_implemented` | 2" in text
    assert "MTE kernel missing" in text
    assert "`unverified` | 1" in text
    assert "quiet_mode unverified" in text
    assert "not count as verified coverage" in text
    assert "LINK-001" not in text


def test_no_gaps_is_stated_explicitly():
    text = joined(coverage.gap_lines(context(rows=[
        {"status": "passed", "case_id": "COMM-001", "backend_id": "alltoall_hccl"}])))
    assert "No capability gap" in text


def test_blocked_topology_is_reported_as_a_gap():
    text = joined(coverage.gap_lines(context(rows=[
        {"status": "blocked_topology", "case_id": "COMM-002", "backend_id": "alltoall_hccl",
         "error_reason": "no cross-server peer in the discovered topology"}])))
    assert "`blocked_topology`" in text
    assert "cross-server" in text


def test_the_section_renders_with_a_minimal_context():
    """The four new subsections must not require fields other subsections lack."""
    ctx = context(rows=[{"status": "not_implemented", "case_id": "X", "backend_id": "y",
                         "error_reason": "r"}])
    lines = coverage.coverage_metrics_lines(ctx) + coverage.gap_lines(ctx)
    assert any("Planning" in line for line in lines)
    assert any("not_implemented" in line for line in lines)
