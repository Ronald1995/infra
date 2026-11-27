"""Per-operator smoke profiles: the table must be complete and executable.

These tests read the committed ``operators/smoke_profiles.yaml``, so a declaration
that drifts away from the operator registry fails here instead of on a device.
"""
from pathlib import Path

import pytest
import yaml

from common import smoke_profile
from common.case_spec import catalog
from common.registry import registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def table():
    return smoke_profile.load()


def test_every_registry_adapter_has_a_profile(table):
    requirements = smoke_profile.coverage_requirements(table, None)
    assert requirements["missing"] == []
    assert requirements["declared"] == requirements["required"] == len(registry())


def test_profile_declares_its_own_case_not_the_planners_first(table):
    """Every operator must name its case; none may be left to the planner."""
    raw = yaml.safe_load(smoke_profile.DEFAULT_PROFILES.read_text(encoding="utf-8"))
    for operator, entry in raw["operators"].items():
        assert entry.get("case"), f"{operator} does not declare a case"
        assert entry.get("family"), f"{operator} does not declare a backend family"
        profile = table.get(operator)
        assert profile.case == entry["case"]
        assert profile.family == entry["family"]


def test_profiles_are_formal_npu_configurations(table):
    for profile in table:
        assert profile.config["device"] == "npu", profile.operator
        assert profile.config["distributed"]["ep_size"] == profile.ep_size


def test_case_matches_the_declared_level(table):
    cases = catalog()
    for profile in table:
        assert cases[profile.case].level == profile.level


def test_backward_is_required_exactly_where_the_adapter_declares_it(table):
    specs = registry()
    for profile in table:
        declared = smoke_profile.BACKWARD_CAPABILITY in specs[profile.operator].capabilities
        assert (profile.direction == "both") is declared, profile.operator
        assert bool(profile.backward_case) is declared, profile.operator
        if declared:
            assert profile.backward_case == "MOE-002"


def test_forward_and_backward_tasks_are_planned_and_executable(table):
    for profile in table:
        forward = profile.forward_tasks()
        assert forward, profile.operator
        for task in forward:
            assert task["status"] == "not_run" and task["reason"] is None
            assert task["backend_id"] == profile.operator
        if profile.backward_case:
            backward = profile.backward_tasks()
            assert backward
            for task in backward:
                assert task["status"] == "not_run"


def test_levels_l1_to_l4_are_targeted_by_profiles(table):
    levels = {profile.level for profile in table}
    assert levels == {"L1", "L2", "L3", "L4"}


def test_l0_is_covered_only_by_environment_evidence(table):
    outcomes = {profile.label: "PASSED" for profile in table}
    without = table.coverage(outcomes)
    assert not next(row for row in without["levels"] if row["level"] == "L0")["covered"]
    assert any(gap["path"] == "L0" and gap["scope"] == "level" for gap in without["gaps"])

    with_evidence = table.coverage(outcomes, level_evidence={
        "L0": {"passed": True, "source": "Stage 1 environment smoke"}})
    assert next(row for row in with_evidence["levels"] if row["level"] == "L0")["covered"]
    assert with_evidence["level_coverage"]["covered"] == 5
    assert with_evidence["gaps"] == []


def test_every_required_path_is_declared_by_some_profile(table):
    declared = set(table.paths())
    missing = [tag for tag in table.required_paths if tag not in declared]
    assert missing == [], f"required paths nobody declares: {missing}"


def test_a_blocked_profile_opens_a_coverage_gap(table):
    outcomes = {profile.label: "PASSED" for profile in table}
    # Only moonep declares dedup:server, so blocking it must surface as a gap.
    target = next(profile for profile in table if profile.operator == "moonep_server_dedup")
    outcomes[target.label] = "BLOCKED_DEPENDENCY"
    coverage = table.coverage(outcomes)
    assert coverage["operator_coverage"]["passed"] == len(table) - 1
    gaps = [gap for gap in coverage["gaps"] if gap["scope"] == "execution_path"]
    assert any(gap["path"] == "dedup:server" for gap in gaps)


def test_missing_outcome_is_a_gap_not_a_pass(table):
    coverage = table.coverage({})
    assert coverage["operator_coverage"]["passed"] == 0
    assert all(not row["passed"] for row in coverage["operators"])


# ---------------------------------------------------------------------------
# invalid tables must be refused
# ---------------------------------------------------------------------------

def _write(tmp_path, operators, required_paths=("backend:hccl",)):
    payload = {
        "schema_version": 1,
        "defaults": {"ep_size": 8, "benchmark": {"warmup": 1, "iterations": 1}},
        "required_paths": list(required_paths),
        "operators": operators,
    }
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_unknown_operator_is_refused(tmp_path):
    path = _write(tmp_path, {"not_an_operator": {"family": "x", "level": "L1", "case": "COMM-001"}})
    with pytest.raises(smoke_profile.ProfileError, match="unknown operator"):
        smoke_profile.load(path)


def test_unknown_case_is_refused(tmp_path):
    path = _write(tmp_path, {"alltoall_hccl": {"family": "hccl", "level": "L1", "case": "NOPE-999"}})
    with pytest.raises(smoke_profile.ProfileError, match="unknown case"):
        smoke_profile.load(path)


def test_level_must_match_the_case(tmp_path):
    path = _write(tmp_path, {"alltoall_hccl": {"family": "hccl", "level": "L3", "case": "COMM-001"}})
    with pytest.raises(smoke_profile.ProfileError, match="does not match case"):
        smoke_profile.load(path)


def test_case_requiring_a_missing_capability_is_refused(tmp_path):
    # quantized_fusion-only adapter asked to run a dispatch case.
    path = _write(tmp_path, {"alltoallv_quant_gmm_cann": {
        "family": "cann", "level": "L1", "case": "COMM-001"}})
    with pytest.raises(smoke_profile.ProfileError, match="lacks"):
        smoke_profile.load(path)


def test_profile_the_planner_cannot_execute_is_refused(tmp_path):
    """LAYER-004 needs an enabled planner; without it the profile is a lie."""
    path = _write(tmp_path, {"moonep_dynamic_planner": {
        "family": "moonep", "level": "L4", "case": "LAYER-004"}})
    with pytest.raises(smoke_profile.ProfileError, match="planner marks"):
        smoke_profile.load(path)


def test_unknown_keys_are_refused(tmp_path):
    path = _write(tmp_path, {"alltoall_hccl": {
        "family": "hccl", "level": "L1", "case": "COMM-001", "cases": ["COMM-001"]}})
    with pytest.raises(smoke_profile.ProfileError, match="unknown keys"):
        smoke_profile.load(path)
