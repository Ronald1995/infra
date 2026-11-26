"""The main report: complete sections, honest verdicts and resolvable links."""
import json
import re

import pytest


SECTIONS = (
    "## 0. Executive Summary",
    "## 1. Environment and Hardware Capability",
    "## 2. Test Coverage Summary",
    "## 3. L0 Hardware Baseline",
    "## 4. L1-L4 Operator Performance",
    "## 5. Backend Comparison",
    "## 6. EP Scalability",
    "## 7. Communication and Compute Overlap",
    "## 8. Load Balance and Planner",
    "## 9. Failures and Limitations",
    "## 10. Evidence and Raw Data",
    "## 11. Recommendations and Next Steps",
)

LINK_PARAMETERS = {
    "variant": "default", "message_bytes": 4096, "outstanding": 1, "peer_count": 1,
    "peer_policy": "ring", "mechanism": "ub_memory", "execution_resource": "AIV",
}


def _l0_extras():
    return {
        "observed": {"transport_protocol": "ub_memory", "execution_resource": "AIV",
                     "verification_status": "verified", "fallback": False},
        "verification": {"status": "verified", "evidence": [
            {"source": "native_descriptor", "claim": {"transport_protocol": "ub_memory"}}]},
        "requested": {"backend": "native_hccl", "operation": "put",
                      "transport_protocol": "ub_memory", "execution_resource": "AIV"},
    }


def _links(text):
    return [target for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
            if "://" not in target]


def _assert_links_resolve(destination, text):
    missing = []
    for target in _links(text):
        path = (destination / target).resolve()
        if not path.exists():
            missing.append(target)
    assert not missing, f"dangling links: {missing}"


@pytest.fixture
def mixed_run(run_report, task_for, record_for, manifest_for):
    tasks = [
        task_for("LINK-001", ep_size=8, device="npu", parameters=LINK_PARAMETERS, index=0),
        task_for("COMM-001", backend_id="alltoall_hccl", ep_size=8, device="npu", index=1),
        task_for("COMM-001", backend_id="allgather_hccl", ep_size=8, device="npu", index=2),
        task_for("FUSE-001", backend_id="alltoall_hccl", ep_size=8, device="npu", index=3),
        task_for("MOE-001", backend_id="alltoall_hccl", ep_size=8, device="npu", index=4),
        task_for("LAYER-001", backend_id="alltoall_hccl", ep_size=8, device="npu", index=5),
        task_for("COMM-002", backend_id="alltoall_hccl", ep_size=8, device="npu", index=6),
        task_for("MEM-001", ep_size=8, device="npu", parameters={"variant": "default"},
                 index=7),
    ]
    manifest = manifest_for(tasks, run_id="mixed")
    rows = [
        record_for(manifest, tasks[0], "passed", p50=.5, extra=_l0_extras()),
        record_for(manifest, tasks[1], "passed", p50=1.0, compute=True, throughput=1000.0),
        record_for(manifest, tasks[2], "passed", p50=2.0, compute=True, throughput=500.0),
        record_for(manifest, tasks[3], "passed", p50=3.0, compute=True),
        record_for(manifest, tasks[4], "runtime_failed"),
        record_for(manifest, tasks[5], "unsupported"),
        # tasks[6] deliberately has no record: it is the `missing` case.
        record_for(manifest, tasks[7], "passed", p50=.02),
    ]
    return run_report(tasks, rows, run_id="mixed")


def test_main_report_has_every_section(mixed_run):
    for section in SECTIONS:
        assert section in mixed_run.text, section


def test_run_scope_and_outcome_counts(mixed_run):
    text = mixed_run.text
    assert "Ascend NPU" in text
    assert "| EP size | EP8 |" in text
    assert "| Benchmark level | L0, L1, L2, L3, L4 |" in text
    summary = text[text.index("### 0.2 Task outcomes"):text.index("### 0.3")]
    assert "| Passed | 5 |" in summary
    assert "| Failed | 1 |" in summary
    assert "| Unsupported | 1 |" in summary
    assert "| Missing | 1 |" in summary


def test_correctness_rate_is_reported(mixed_run):
    text = mixed_run.text
    assert "Correctness pass rate | 100%" in text


def test_backend_comparison_ranks_only_a_real_pair(mixed_run):
    text = mixed_run.text
    comparison = text[text.index("## 5. Backend Comparison"):text.index("## 6. EP Scalability")]
    assert "Group 1" in comparison
    assert "alltoall_hccl" in comparison and "allgather_hccl" in comparison
    assert "Spread between fastest and slowest backend: 2x" in comparison
    assert "completion `remote_visible`" in comparison
    assert "samples per backend allgather_hccl=1, alltoall_hccl=1" in comparison


def test_single_backend_is_not_called_a_comparison(mixed_run):
    text = mixed_run.text
    comparison = text[text.index("## 5. Backend Comparison"):text.index("## 6. EP Scalability")]
    assert "Single Measurement" in comparison
    assert "not evidence of a performance advantage" in comparison


def test_scalability_is_refused_for_one_ep_size(mixed_run):
    text = mixed_run.text
    scaling = text[text.index("## 6. EP Scalability"):text.index("## 7. Communication")]
    assert "Why EP Scalability Cannot Be Analysed" in scaling
    assert "Only the following EP size produced results: EP8" in scaling
    assert "No strong-scaling or weak-scaling statement is made" in scaling


def test_overlap_is_not_claimed_without_profiler(mixed_run):
    text = mixed_run.text
    overlap = text[text.index("## 7. Communication and Compute Overlap"):text.index(
        "## 8. Load Balance")]
    assert "Why Device-Level Overlap Is Not Reported" in overlap
    assert "no claim is made about device-level communication/compute overlap" in overlap


def test_failed_and_unsupported_tasks_keep_their_request(mixed_run):
    text = mixed_run.text
    failures = text[text.index("## 9. Failures and Limitations"):text.index(
        "## 10. Evidence")]
    assert "runtime_failed" in failures and "unsupported" in failures
    assert "MOE-001" in failures and "LAYER-001" in failures
    assert "variant=default" in failures
    assert "no result record for a planned task" in failures
    assert "COMM-002" in failures


def test_l0_verified_result_is_reported(mixed_run):
    text = mixed_run.text
    l0 = text[text.index("## 3. L0 Hardware Baseline"):text.index("## 4. L1-L4")]
    assert "LINK-001" in l0
    assert "UB Memory" in l0
    assert "verified" in l0
    assert "0.5" in l0 or "0.00819" in l0


def test_operator_tables_include_failed_rows(mixed_run):
    text = mixed_run.text
    section = text[text.index("### 4.3 L3 (full MoE execution)"):text.index("### 4.4")]
    assert "MOE-001" in section and "runtime_failed" in section
    # A failed task has no latency, and that is a dash, never a zero.
    assert "| - | - | - |" in section


def test_no_none_or_fake_zero_in_the_report(mixed_run):
    assert "None" not in mixed_run.text
    assert "error_reason=unavailable" not in mixed_run.text
    # A metric that does not apply says so; it is never rendered as 0.
    assert "N/A (not a communication payload)" in mixed_run.text


def test_every_link_resolves(mixed_run):
    _assert_links_resolve(mixed_run.destination, mixed_run.text)


def test_charts_section_states_what_was_generated(mixed_run):
    text = mixed_run.text
    assert "### 10.6 Charts" in text
    charts = text[text.index("### 10.6 Charts"):text.index("### 10.7")]
    assert "chart(s) generated" in charts or "Charts Not Generated" in charts
    _assert_links_resolve(mixed_run.destination, charts)


def test_specialty_reports_exist_and_explain_themselves(mixed_run):
    reports = mixed_run.destination
    for name in ("operator_comparison.md", "bandwidth_analysis.md", "overlap_analysis.md",
                 "scalability_analysis.md", "load_balance_analysis.md", "planner_training.md",
                 "dedup_comparison.md", "failure_summary.md"):
        path = reports / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) > 5, name
        if "Not Available" in text:
            assert "Why Analysis Is Unavailable" in text, name
            assert "Required Conditions" in text, name
            assert "Conclusion" in text, name


def test_coverage_products_are_generated_in_the_report_phase(mixed_run):
    for name in ("coverage_matrix.md", "coverage_matrix.json", "coverage_matrix.csv",
                 "operator_coverage.md", "variant_coverage.md", "case_operator_coverage.csv",
                 "variant_coverage.csv"):
        assert (mixed_run.destination / name).is_file(), name


def test_report_is_deterministic(run_report, task_for, record_for, manifest_for):
    tasks = [task_for("COMM-001", backend_id=backend, index=index)
             for index, backend in enumerate(("alltoall_hccl", "allgather_hccl"))]
    manifest = manifest_for(tasks, run_id="deterministic")
    rows = [record_for(manifest, task, "passed", p50=1.0 + index)
            for index, task in enumerate(tasks)]
    first = run_report(tasks, rows, run_id="deterministic",
                       environment=manifest["environment"])
    snapshot = {path.name: path.read_bytes() for path in first.destination.iterdir()
                if path.is_file() and path.suffix in {".md", ".csv", ".json"}}
    second = run_report(tasks, rows, run_id="deterministic",
                        environment=manifest["environment"])
    for name, content in snapshot.items():
        assert (second.destination / name).read_bytes() == content, name


def test_raw_metrics_are_not_modified(mixed_run):
    """The report layer reads the result records; it never rewrites them."""
    before = {}
    for path in mixed_run.root.rglob("metrics.json"):
        before[path] = path.read_bytes()
    assert before
    from analysis.generate_report import analyze_run
    from reporting.report import report
    analyze_run(mixed_run.root, mixed_run.rows)
    report(mixed_run.root, mixed_run.manifest, mixed_run.rows, mixed_run.analysis)
    for path, content in before.items():
        assert path.read_bytes() == content, path


def test_cpu_run_uses_cpu_wording(run_report, task_for, record_for, manifest_for):
    tasks = [task_for("COMM-001", backend_id="alltoall_hccl", index=0)]
    manifest = manifest_for(tasks, run_id="cpu", environment={
        **json.loads(json.dumps({
            "host": "cpu-host", "platform": "Windows-10", "packages": {"torch": "2.6.0+cpu"},
            "npu_models": None, "cann_version": None, "cann_version_source": "unavailable",
            "cann_version_error": "ASCEND_TOOLKIT_VERSION is unset", "git_commit": "abc"}))})
    rows = [record_for(manifest, tasks[0], "passed", p50=1.0)]
    run = run_report(tasks, rows, run_id="cpu", environment=manifest["environment"])
    assert "CPU (framework validation, not a device performance measurement)" in run.text
    assert "CPU results validate framework behavior" not in run.text
    assert "NPU results require an Ascend run" not in run.text


def test_l0_only_run_marks_the_operator_levels_out_of_scope(
        run_report, task_for, record_for, manifest_for):
    tasks = [task_for("LINK-001", ep_size=8, device="npu", parameters=LINK_PARAMETERS)]
    manifest = manifest_for(tasks, run_id="l0-only")
    rows = [record_for(manifest, tasks[0], "passed", p50=.5, extra=_l0_extras())]
    run = run_report(tasks, rows, run_id="l0-only")
    assert "No L1 task was planned for this run" in run.text
    assert "No L4 task was planned for this run" in run.text
    assert "l0_report.md" in run.text
    assert (run.destination / "l0_report.md").is_file()


def test_a_run_without_results_still_reports(task_for, manifest_for, run_report):
    tasks = [task_for("COMM-001", index=0)]
    run = run_report(tasks, [], run_id="empty")
    for section in SECTIONS:
        assert section in run.text, section
    assert "no pair of backends" in run.text.lower() or "Why No Backend Can Be Ranked" in run.text
    assert "missing" in run.text
