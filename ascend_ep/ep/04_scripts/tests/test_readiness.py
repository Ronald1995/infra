"""Dependency readiness: a platform restriction is not a missing package."""
from pathlib import Path

import pytest

from common import readiness, status_map
from scripts import doctor

ROOT = Path(__file__).resolve().parents[1]


def test_soc_normalisation_covers_the_three_families():
    assert readiness.normalise_soc("Ascend950DT_9582") == "Ascend950"
    assert readiness.normalise_soc("Ascend910B4") == "Ascend910B"
    assert readiness.normalise_soc("Ascend910_93") == "Ascend910_93"
    assert readiness.normalise_soc(None) is None


def test_soc_supported_honours_wildcards_and_missing_declarations():
    assert readiness.soc_supported(["*"], "Ascend950")
    assert readiness.soc_supported(["Ascend910B", "Ascend950"], "Ascend950")
    assert not readiness.soc_supported(["Ascend910B", "Ascend910_93"], "Ascend950")
    # No declaration, or no host SoC: do not invent a blocker.
    assert readiness.soc_supported(None, "Ascend950")
    assert readiness.soc_supported(["Ascend910B"], None)


def test_a_missing_dependency_documented_for_other_socs_is_not_a_platform_block():
    """`supported_soc` is what the release is *documented* for, nothing stronger.

    fsdp-turbo lists only 910B/910_93, and megamoe_lueelu still ran on 950DT per its
    reference report.  So an undocumented pairing must stay a dependency blocker
    with a "confirm on this SoC" note, never "do not retry here".
    """
    rows = [{"operator": "megamoe_lueelu", "dependencies": ["deep-ep-experimental"],
             "states": {"deep-ep-experimental": "fail"}, "unmapped_modules": []}]
    entries = {"deep-ep-experimental": {"id": "deep-ep-experimental",
                                        "supported_soc": ["Ascend910B", "Ascend910_93"],
                                        "ownership": "external"}}
    result = readiness.classify(rows, entries, "Ascend950")["megamoe_lueelu"]
    assert result["state"] == readiness.UNDOCUMENTED_ON_SOC
    assert result["bucket"] == status_map.BLOCKED_DEPENDENCY
    assert result["undocumented"] == ["deep-ep-experimental"]
    # ... and it must not be rewritten into a platform verdict.
    assert readiness.refine_bucket(status_map.BLOCKED_DEPENDENCY, "megamoe_lueelu",
                                   {"megamoe_lueelu": result}) == \
        status_map.BLOCKED_DEPENDENCY


def test_an_installed_dependency_is_ready_even_when_undocumented():
    rows = [{"operator": "x", "dependencies": ["cann-ops-transformer"],
             "states": {"cann-ops-transformer": "pass"}, "unmapped_modules": []}]
    entries = {"cann-ops-transformer": {"id": "cann-ops-transformer",
                                        "supported_soc": ["Ascend910_93"]}}
    assert readiness.classify(rows, entries, "Ascend950")["x"]["state"] == readiness.READY


def test_only_an_operator_hardware_constraint_blocks_the_platform():
    rows = [{"operator": "dispatch_ffn_combine_vllm_ascend",
             "dependencies": ["torch-npu"], "states": {"torch-npu": "pass"},
             "unmapped_modules": []}]
    entries = {"torch-npu": {"id": "torch-npu", "supported_soc": ["*"]}}
    hardware = {"dispatch_ffn_combine_vllm_ascend": ["Atlas A3 (ascend910_93)"]}
    result = readiness.classify(rows, entries, "Ascend950", hardware)[
        "dispatch_ffn_combine_vllm_ascend"]
    assert result["state"] == readiness.UNSUPPORTED_PLATFORM
    assert result["bucket"] == status_map.BLOCKED_PLATFORM
    assert "operator:hardware" in result["unsupported"]
    # The same operator on A3 is not blocked by its hardware constraint.
    assert readiness.hardware_blocked(["Atlas A3 (ascend910_93)"], "Ascend910_93") is False
    # An operator that names 950 is not blocked on 950.
    assert readiness.hardware_blocked(["Ascend 950"], "Ascend950") is False


def test_a_supported_missing_package_stays_missing():
    rows = [{"operator": "op_b", "dependencies": ["deep-ep"], "states": {"deep-ep": "fail"},
             "unmapped_modules": []}]
    entries = {"deep-ep": {"id": "deep-ep", "kind": "native_build",
                           "supported_soc": ["Ascend950"], "ownership": "external"}}
    result = readiness.classify(rows, entries, "Ascend950")["op_b"]
    assert result["state"] == readiness.MISSING
    assert result["bucket"] == status_map.BLOCKED_DEPENDENCY


def test_ready_needs_every_dependency_present():
    rows = [{"operator": "op_c", "dependencies": ["a", "b"],
             "states": {"a": "pass", "b": "pass"}, "unmapped_modules": []}]
    entries = {"a": {"id": "a", "supported_soc": ["*"]},
               "b": {"id": "b", "supported_soc": ["*"]}}
    assert readiness.classify(rows, entries, "Ascend950")["op_c"]["state"] == readiness.READY


def test_version_warning_is_its_own_state():
    rows = [{"operator": "op_d", "dependencies": ["torch-npu"],
             "states": {"torch-npu": "warn"}, "unmapped_modules": []}]
    entries = {"torch-npu": {"id": "torch-npu", "supported_soc": ["*"],
                             "kind": "pip", "ownership": "external"}}
    assert readiness.classify(rows, entries, "Ascend950")["op_d"]["state"] == \
        readiness.VERSION_MISMATCH


def test_a_vendored_tree_that_is_absent_is_a_source_problem():
    rows = [{"operator": "op_e", "dependencies": ["vendored-ascendep"],
             "states": {"vendored-ascendep": "fail"}, "unmapped_modules": []}]
    entries = {"vendored-ascendep": {"id": "vendored-ascendep", "ownership": "vendored",
                                     "supported_soc": ["*"]}}
    assert readiness.classify(rows, entries, "Ascend950")["op_e"]["state"] == \
        readiness.SOURCE_UNAVAILABLE


def test_an_unmapped_module_is_a_missing_dependency():
    rows = [{"operator": "op_f", "dependencies": [], "states": {},
             "unmapped_modules": ["mystery_module"]}]
    result = readiness.classify(rows, {}, "Ascend950")["op_f"]
    assert result["state"] == readiness.MISSING


def test_refine_only_upgrades_a_dependency_bucket():
    readiness_map = {"op_a": {"state": readiness.UNSUPPORTED_PLATFORM}}
    assert readiness.refine_bucket(status_map.BLOCKED_DEPENDENCY, "op_a", readiness_map) == \
        status_map.BLOCKED_PLATFORM
    # A failure is a failure: the SoC verdict must not excuse it.
    assert readiness.refine_bucket(status_map.FAILED, "op_a", readiness_map) == \
        status_map.FAILED
    assert readiness.refine_bucket(status_map.BLOCKED_DEPENDENCY, "unknown", readiness_map) == \
        status_map.BLOCKED_DEPENDENCY


def test_real_manifest_declares_fsdp_turbo_unsupported_on_950():
    """The classification is only as good as the manifest; pin the claim we rely on."""
    data = doctor.load_manifest(ROOT / "dependencies" / "manifest.yaml")
    entries = {entry["id"]: entry for entry in data["dependencies"]}
    fsdp = entries["fsdp-turbo"]
    assert "Ascend950" not in fsdp["supported_soc"]
    # ... while the deps we still expect to be installable do list it.
    assert "Ascend950" in entries["megatron"]["supported_soc"]
    assert "Ascend950" in entries["deep-ep"]["supported_soc"]


def test_the_installer_only_receives_the_installable_subset(monkeypatch):
    """`deps --install` must never try to install a platform-unsupported package."""
    from scripts import npu_smoke

    recorded = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        recorded["command"] = command
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    args = type("Args", (), {"profile": "npu", "dry_run": True})()
    code = npu_smoke._run_installer(["moonep", "deepep_ascend"],
                                   ["alltoall_fsdpturbo"], args)
    assert code == 0
    command = recorded["command"]
    assert command[:2] == ["bash", str(ROOT / "scripts" / "install_deps.sh")]
    assert "--dry-run" in command and "--profile" in command
    # The unsupported operator is reported and not turned into an install action.
    assert "alltoall_fsdpturbo" not in command

