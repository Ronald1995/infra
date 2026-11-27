"""NPU smoke driver: NPU_ONLY gating and coverage-aware reporting.

No device is required: the driver's planning, policy and reporting are pure, and
the two places that would touch hardware (``device_probe`` and ``run_tasks``) are
patched here.
"""
from types import SimpleNamespace

import pytest

from common import execution_mode, smoke_profile, status_map
from common.case_spec import catalog
from scripts import npu_smoke


@pytest.fixture(scope="module")
def table():
    return smoke_profile.load()


def _probe(npu=True, devices=8):
    return {"npu_available": npu, "device_count": devices if npu else 0,
            "models": ["Ascend950DT_9582"] * (devices if npu else 0),
            "hccl_backend": npu, "torch": "2.10.0+cpu", "torch_npu": "2.10.0.post7",
            "hbm_bytes": [98_000_000_000] * (devices if npu else 0), "error": None}


def _rows_all(tasks, status="passed", reason=""):
    """Fabricate result records only for tasks that were actually scheduled."""
    return {task["task_id"]: {"task_id": task["task_id"], "status": status,
                              "error_reason": reason}
            for task in tasks if task.get("status") == "not_run"}


def _report(table, stage="all", probe=None, rows_status="passed"):
    # 64 devices by default so that the EP32 profile is not hardware-blocked; the
    # eight-card case is covered by its own test.
    probe = probe or _probe(devices=64)
    tasks, owners = npu_smoke.build_plan(table, stage)
    npu_smoke.apply_preflight(probe, tasks)
    return tasks, owners, npu_smoke.build_report(
        table, tasks, owners, _rows_all(tasks, rows_status), {}, probe, stage, "unit")


def test_plan_covers_every_operator_and_both_directions(table):
    tasks, owners = npu_smoke.build_plan(table, "operators")
    assert set(owners.values()) == {profile.label for profile in table}
    assert len(tasks) == sum(len(npu_smoke.profile_tasks(profile)) for profile in table)


def test_every_planned_task_is_npu(table):
    tasks, _ = npu_smoke.build_plan(table, "all")
    assert tasks
    for task in tasks:
        assert task["config"]["device"] == "npu"


def test_npu_only_refuses_a_cpu_task(table):
    tasks, _ = npu_smoke.build_plan(table, "operators")
    tasks[0]["config"] = dict(tasks[0]["config"], device="cpu")
    with pytest.raises(execution_mode.FormalModeViolation):
        npu_smoke.apply_npu_only("npu_only", tasks)


def test_preflight_blocks_ep32_on_an_eight_card_host(table):
    tasks, _ = npu_smoke.build_plan(table, "operators")
    kinds = npu_smoke.apply_preflight(_probe(devices=8), tasks)
    blocked = [task for task in tasks if task["status"] != "not_run"]
    assert blocked, "the EP32 profile must not be scheduled on eight cards"
    for task in blocked:
        assert task["config"]["distributed"]["ep_size"] > 8
        assert kinds[task["task_id"]] == "hardware_shortfall"
        assert "never replaced by CPU" in task["reason"]


def test_full_pass_reports_complete_coverage(table):
    tasks, owners, report = _report(table)
    assert report["conclusion"] == "PASS", report["gaps"]
    assert report["coverage"]["operator_coverage"]["passed"] == len(table)
    assert report["coverage"]["level_coverage"]["covered"] == 5
    assert report["coverage"]["path_coverage"]["covered"] == \
        report["coverage"]["path_coverage"]["total"]
    assert report["stage1"]["passed"] is True


def test_a_blocked_operator_makes_the_run_incomplete(table):
    tasks, owners = npu_smoke.build_plan(table, "all")
    probe = _probe()
    npu_smoke.apply_preflight(probe, tasks)
    rows = _rows_all(tasks, "dependency_missing", "No module named 'deep_ep'")
    report = npu_smoke.build_report(table, tasks, owners, rows, {}, probe, "all", "unit")
    assert report["conclusion"] == "INCOMPLETE"
    assert report["counts"][status_map.BLOCKED_DEPENDENCY] > 0
    assert report["counts"][status_map.FAILED] == 0


def test_hardware_blocked_is_reported_as_hardware_not_environment(table):
    tasks, owners = npu_smoke.build_plan(table, "operators")
    kinds = npu_smoke.apply_preflight(_probe(devices=8), tasks)
    report = npu_smoke.build_report(table, tasks, owners, _rows_all(tasks), kinds,
                                    _probe(devices=8), "operators", "unit")
    assert report["counts"][status_map.BLOCKED_HARDWARE] >= 1
    assert report["counts"][status_map.BLOCKED_ENVIRONMENT] == 0


def test_no_npu_environment_concludes_blocked(table):
    tasks, owners, report = _report(table, stage="all", probe=_probe(npu=False))
    assert report["conclusion"] == "BLOCKED"
    assert report["counts"][status_map.PASSED] == 0


def test_markdown_names_the_three_coverages_and_gaps(table):
    _, _, report = _report(table)
    text = npu_smoke.render_markdown(report)
    assert "execution mode: **npu_only**" in text
    assert "operators (NPU main path passed)" in text
    assert "levels L0-L4" in text
    assert "execution paths" in text
    assert "COVERAGE_GAP" in text
    assert "| `alltoall_hccl` |" in text


def test_stage1_case_selection_matches_the_catalog():
    ids = npu_smoke.stage1_cases()
    cases = catalog()
    assert ids
    for case_id in ids:
        assert cases[case_id].level == "L0"
    covered = {cases[case_id].category for case_id in ids}
    for required in npu_smoke.STAGE1_REQUIRED:
        assert required in covered, f"Stage 1 requires {required} but selects no such case"


def test_stage1_selection_refuses_a_renamed_case():
    cases = dict(catalog())
    cases.pop("HW-001")
    with pytest.raises(ValueError, match="unknown case HW-001"):
        npu_smoke.stage1_cases(cases)


def test_stage1_plan_stays_cheap(table):
    tasks = npu_smoke.environment_tasks()
    # A gate, not a measurement: the L0 matrix would be several hundred points.
    assert 0 < len(tasks) <= 40
    for task in tasks:
        assert task["level"] == "L0"
        assert task["config"]["device"] == "npu"


def test_operator_without_a_profile_is_a_coverage_gap(table):
    trimmed = smoke_profile.ProfileTable(
        profiles={k: v for k, v in table.profiles.items() if k != "moonep"},
        required_paths=table.required_paths, defaults=table.defaults, source=table.source)
    tasks, owners = npu_smoke.build_plan(trimmed, "operators")
    probe = _probe()
    npu_smoke.apply_preflight(probe, tasks)
    report = npu_smoke.build_report(trimmed, tasks, owners, _rows_all(tasks), {}, probe,
                                    "operators", "unit")
    operator_gaps = [gap for gap in report["gaps"] if gap["scope"] == "operator"]
    assert any(gap["path"] == "moonep" for gap in operator_gaps)
