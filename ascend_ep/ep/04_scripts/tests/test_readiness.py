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


def test_platform_unsupported_outranks_a_missing_package():
    rows = [{"operator": "op_a", "dependencies": ["dep-ok", "dep-other-soc"],
             "states": {"dep-ok": "fail", "dep-other-soc": "fail"},
             "unmapped_modules": []}]
    entries = {
        "dep-ok": {"id": "dep-ok", "supported_soc": ["Ascend950"], "kind": "native_build"},
        "dep-other-soc": {"id": "dep-other-soc", "kind": "python_import",
                          "supported_soc": ["Ascend910B"], "ownership": "external"},
    }
    result = readiness.classify(rows, entries, "Ascend950")["op_a"]
    assert result["state"] == readiness.UNSUPPORTED_PLATFORM
    assert result["unsupported"] == ["dep-other-soc"]
    assert result["bucket"] == status_map.BLOCKED_PLATFORM


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
