"""Charts degrade to text: never a hard failure, never a dangling image link."""
import sys

import pytest

from reporting import charts as charting


def _analysis(rows, tmp_path):
    return {
        "comparison_details": {"groups": [], "excluded": []},
        "scaling_groups": {}, "pareto": {}, "fusion": [], "profiles": {},
        "planner_comparisons": [], "fixed_owner_comparisons": [], "dedup_comparisons": [],
        "profile_parse_errors": [],
    }


def test_missing_matplotlib_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    report = charting.create_charts(tmp_path, [], _analysis([], tmp_path))
    assert report.available is False
    assert report.missing_dependency == "matplotlib"
    assert "not installed" in report.reason or "unavailable" in report.reason
    assert report.created == []
    assert "Charts Not Generated" in report.status_line()


def test_no_data_skips_a_chart_with_a_reason(tmp_path):
    pytest.importorskip("matplotlib")
    report = charting.create_charts(tmp_path, [], _analysis([], tmp_path))
    assert report.available is True
    assert report.created == []
    names = {item.name for item in report.skipped}
    assert {"backend-comparison", "ep-scaling", "device-idle", "fusion-speedup",
            "expert-load"} <= names
    for item in report.skipped:
        assert item.reason
    assert "Charts Not Generated" in report.status_line()


def test_chart_links_point_at_real_files(tmp_path):
    pytest.importorskip("matplotlib")
    analysis = _analysis([], tmp_path)
    analysis["scaling_groups"] = {"k": [
        {"ep": 8, "tokens_per_second": 100.0, "efficiency": 1.0, "mode": "strong",
         "task_id": "t1", "run_id": "r"},
        {"ep": 16, "tokens_per_second": 150.0, "efficiency": .75, "mode": "strong",
         "task_id": "t2", "run_id": "r"},
    ]}
    report = charting.create_charts(tmp_path, [], analysis)
    assert report.created
    for label, path in report.links():
        assert (tmp_path / path).is_file(), path
        assert label == (tmp_path / path).stem


def test_report_without_matplotlib_still_writes_every_section(
        run_report, task_for, record_for, manifest_for, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    tasks = [task_for("COMM-001", backend_id=backend, index=index)
             for index, backend in enumerate(("alltoall_hccl", "allgather_hccl"))]
    manifest = manifest_for(tasks, run_id="no-charts")
    rows = [record_for(manifest, task, "passed", p50=1.0 + index)
            for index, task in enumerate(tasks)]
    run = run_report(tasks, rows, run_id="no-charts", environment=manifest["environment"])
    text = run.text
    assert "## 11. Recommendations and Next Steps" in text
    charts = text[text.index("### 10.6 Charts"):text.index("### 10.7")]
    assert "Charts Not Generated" in charts
    assert "matplotlib" in charts
    assert "charts/" not in charts
    assert not (run.destination / "charts").exists()


def test_chart_report_serialises(run_report, task_for, record_for, manifest_for):
    pytest.importorskip("matplotlib")
    tasks = [task_for("COMM-001", backend_id=backend, index=index)
             for index, backend in enumerate(("alltoall_hccl", "allgather_hccl"))]
    manifest = manifest_for(tasks, run_id="charts-json")
    rows = [record_for(manifest, task, "passed", p50=1.0 + index)
            for index, task in enumerate(tasks)]
    run = run_report(tasks, rows, run_id="charts-json", environment=manifest["environment"])
    details = run.analysis["comparison_details"]["groups"]
    assert details and details[0]["kind"] == "actual_comparison"
    assert (run.destination / "charts").is_dir()
    assert any((run.destination / "charts").glob("latency-*.png"))
