"""Plan summary: requirement accounting and the runtime estimate policy.

"12,629 tasks" must never be read as "12,629 measurements". The summary separates the
executable from the unsupported/blocked/not-applicable ones, and it refuses to invent a
runtime: an estimate needs an explicit assumption or a pilot measurement.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import plan_summary  # noqa: E402
from common.plan import expand, load_plan  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return load_plan(ROOT / "configs" / "benchmark_plan.yaml")


def test_the_summary_counts_requirements_per_ep_size(plan):
    run = expand(plan, ep_sizes=[8, 16], profile="core")
    summary = plan_summary.summarize(run)
    assert summary["tasks"] == len(run.tasks)
    assert summary["unique_execution_tasks"] == len(run.tasks)
    assert summary["ep_sizes"] == [8, 16]
    assert summary["by_ep_size"]["8"] == summary["by_ep_size"]["16"]
    assert sum(summary["by_level"].values()) == summary["tasks"]
    assert sum(summary["by_suite"].values()) >= summary["tasks"]  # one task may serve suites


def test_the_classes_partition_the_tasks(plan):
    summary = plan_summary.summarize(expand(plan, ep_sizes=[8], profile="core"))
    classes = summary["classes"]
    assert sum(classes.values()) == summary["tasks"]
    assert summary["executable"] == classes["executable"]
    assert summary["not_executable"] == summary["tasks"] - classes["executable"]


def test_status_classification_covers_the_vocabulary():
    assert plan_summary.classify("not_run") == "executable"
    assert plan_summary.classify("passed") == "executable"
    assert plan_summary.classify("unverified") == "executable"
    assert plan_summary.classify("unsupported") == "unsupported"
    assert plan_summary.classify("not_implemented") == "unsupported"
    assert plan_summary.classify("blocked_topology") == "blocked"
    assert plan_summary.classify("dependency_missing") == "blocked"
    assert plan_summary.classify("not_applicable") == "not_applicable"
    assert plan_summary.classify("correctness_failed") == "failed"


def test_no_assumption_means_no_estimate(plan):
    estimation = plan_summary.summarize(expand(plan, ep_sizes=[8], profile="smoke"))["estimation"]
    assert estimation["status"] == "unavailable"
    assert "iterations x kernel latency" in estimation["reason"]


def test_an_explicit_assumption_produces_a_component_estimate(plan):
    run = expand(plan, ep_sizes=[8, 16], profile="smoke")
    estimation = plan_summary.summarize(run, estimation={
        "per_task_seconds": 2.0, "startup_seconds_per_ep": 30.0,
        "init_seconds_per_ep": 60.0, "profiling_seconds_per_task": 1.0})["estimation"]
    assert estimation["status"] == "estimated"
    assert estimation["basis"].startswith("explicit assumption")
    tasks = len(run.tasks)
    assert estimation["components"]["tasks_seconds"] == pytest.approx(2.0 * tasks)
    assert estimation["components"]["startup_seconds"] == pytest.approx(60.0)
    assert estimation["components"]["profiling_seconds"] == pytest.approx(1.0 * tasks)
    assert estimation["total_seconds"] == pytest.approx(2.0 * tasks + 60.0 + 120.0 + tasks)


def test_a_bad_assumption_is_explained_not_estimated():
    estimation = plan_summary.estimate(10, {8: 10}, {"per_task_seconds": -1})
    assert estimation["status"] == "unavailable"
    assert "non-negative" in estimation["reason"]
    estimation = plan_summary.estimate(10, {8: 10}, {"startup_seconds_per_ep": 5})
    assert estimation["status"] == "unavailable"
    assert "per_task_seconds" in estimation["reason"]


def test_the_plan_rejects_a_negative_assumption(tmp_path):
    path = tmp_path / "plan.yaml"
    path.write_text("\n".join([
        "plan:",
        "  name: bad-estimate",
        "  mode: full_matrix",
        "  ep_sizes: [8]",
        "  suites: [all]",
        "  estimation: {per_task_seconds: -2}",
        "  base_config: " + str(ROOT / "configs" / "base_benchmark.yaml"),
        "  ep_configs: {8: " + str(ROOT / "configs" / "ep8_bf16.yaml") + "}",
        "  suite_configs: {all: " + str(ROOT / "configs" / "hardware.yaml") + "}",
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="estimation.per_task_seconds"):
        load_plan(path)


def test_the_manifest_carries_the_summary(plan):
    metadata = expand(plan, ep_sizes=[8], profile="core").metadata()
    summary = metadata["plan_summary"]
    assert summary["tasks"] == len(expand(plan, ep_sizes=[8], profile="core").tasks)
    assert summary["estimation"]["status"] == "unavailable"
