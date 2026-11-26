"""Tests that the catalog, the requirement files and the system descriptors agree.

Why a separate file: the doctor can only be as honest as its inputs.  A pin that
lives in a requirement file but not in the catalog, a ``pip`` entry marked "not
installable", a host-provided dependency without the manual step, or a system
descriptor that names a dependency id nobody declares -- each of those makes the
doctor report the wrong thing with full confidence.  These tests are cheap,
CPU-only and fail loudly on the first drift.
"""
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `pytest tests/dependencies` from another cwd
    sys.path.insert(0, str(ROOT))

from scripts import doctor  # noqa: E402  (path bootstrap must run first)

DEPENDENCIES = ROOT / "dependencies"
SYSTEM_FILES = ("cann.yaml", "mpi.yaml", "hccl_test.yaml", "asc_comm.yaml")


@pytest.fixture(scope="module")
def manifest():
    return doctor.load_manifest(DEPENDENCIES / "manifest.yaml")


@pytest.fixture(scope="module")
def entries(manifest):
    return {entry["id"]: entry for entry in manifest["dependencies"]}


def distribution_index(entries):
    index = {}
    for identifier, entry in entries.items():
        check = entry.get("check") or {}
        if check.get("type") != "python_module":
            continue
        for name in (check.get("distribution"), check.get("module")):
            if name:
                index[doctor.normalize_name(name)] = identifier
    return index


# ---------------------------------------------------------------------------
# Requirement files <-> catalog
# ---------------------------------------------------------------------------
def test_requirement_files_exist():
    names = sorted(path.name for path in (DEPENDENCIES / "python").glob("*.txt"))
    assert names == ["base.txt", "deep_ep.txt", "npu.txt", "profiling.txt", "shmem.txt"]


def test_every_requirement_pin_has_a_matching_catalog_entry(manifest, entries):
    index = distribution_index(entries)
    pinned = 0
    for path in sorted((DEPENDENCIES / "python").glob("*.txt")):
        pins, unpinned = doctor.parse_requirements(path)
        assert unpinned == [], f"{path.name}: unpinned requirement lines {unpinned}"
        for name, version, raw in pins:
            pinned += 1
            identifier = index.get(doctor.normalize_name(name))
            assert identifier is not None, f"{path.name}: {raw} has no catalog entry"
            constraint = str(entries[identifier]["version"] or "")
            assert constraint.lstrip("=") == version, (
                f"{identifier}: manifest version {constraint!r} does not match the pin {raw!r}")
    assert pinned >= 7, "the pinned requirements disappeared from the catalog"


def test_base_requirements_are_the_cpu_minimum(entries):
    pins, _ = doctor.parse_requirements(DEPENDENCIES / "python" / "base.txt")
    names = {doctor.normalize_name(name) for name, _, _ in pins}
    assert {"torch", "numpy", "pyyaml", "jsonschema"} <= names
    # Nothing NPU-specific and nothing host-provided may leak into the offline CPU file.
    assert "torch-npu" not in names
    for name in names:
        identifier = distribution_index(entries).get(name)
        assert identifier is not None
        assert entries[identifier]["category"] == "A", identifier


def test_npu_requirements_only_contain_pip_entries(entries):
    index = distribution_index(entries)
    pins, _ = doctor.parse_requirements(DEPENDENCIES / "python" / "npu.txt")
    assert pins, "npu.txt must pin the torch_npu wheel"
    for name, _, _ in pins:
        identifier = index.get(doctor.normalize_name(name))
        assert identifier is not None
        assert entries[identifier]["kind"] == "pip"
        assert entries[identifier]["installable"] is True
        assert entries[identifier]["category"] == "B"


# ---------------------------------------------------------------------------
# Installability rules
# ---------------------------------------------------------------------------
def test_pip_entries_are_installable_and_host_entries_are_not(entries):
    for identifier, entry in entries.items():
        if entry["kind"] == "pip":
            assert entry["installable"] is True, f"{identifier}: a pip dependency must be installable"
        if entry["kind"] == "host_provided":
            assert entry["installable"] is False, f"{identifier}: a host component cannot be installed here"
        if entry["installable"] is True:
            assert entry["install"], f"{identifier}: installable without install commands"
            assert entry["kind"] != "host_provided", identifier
        else:
            assert entry["install_reason"], f"{identifier}: not installable without a reason"


def test_host_provided_entries_document_the_manual_step(entries):
    for identifier, entry in entries.items():
        if entry["kind"] != "host_provided":
            continue
        assert entry["manual_step"], f"{identifier}: no manual step"
        assert entry["manual_step_owner"] in ("host", "host image", "deployment"), identifier


def test_native_builds_are_pinned_to_a_source(entries):
    for identifier, entry in entries.items():
        if entry["kind"] != "native_build":
            continue
        assert entry["source_url"], identifier
        assert entry["version"], identifier
        if entry["commit"] is None and str(entry["source_url"]).startswith("http"):
            # An unpinned external build must say so, otherwise the report cannot
            # be reproduced.  In-repo builds (baselines/, kernels/) are pinned by
            # the commit of this repository and do not need the note.
            assert "commit is null" in entry["notes"] or "commit: null" in entry["notes"], identifier


def test_the_asc_comm_entry_records_the_blocker(entries):
    kernels = entries["asc-comm-kernels"]
    assert kernels["installable"] is False
    assert "BLOCKED" in kernels["install_reason"]
    assert "kernels/asc_comm" in kernels["manual_step"]
    toolchain = entries["asc-comm-build-toolchain"]
    assert "BLOCKED" in toolchain["notes"] or "blocker" in toolchain["notes"]


# ---------------------------------------------------------------------------
# Check types <-> probers
# ---------------------------------------------------------------------------
def test_every_check_type_has_a_prober(manifest):
    declared = list(manifest["vocabulary"]["check_types"])
    assert declared == list(doctor.CHECK_TYPES)
    assert set(declared) == set(doctor.PROBES)
    for entry in manifest["dependencies"]:
        assert entry["check"]["type"] in doctor.PROBES, entry["id"]


def test_every_profile_is_a_known_category_set(manifest):
    assert dict(manifest["profiles"]) == {name: list(categories)
                                          for name, categories in doctor.PROFILES.items()}
    for categories in manifest["profiles"].values():
        assert set(categories) <= set(doctor.CATEGORIES)


# ---------------------------------------------------------------------------
# Compatibility table
# ---------------------------------------------------------------------------
def test_compatibility_rows_are_concrete(manifest):
    rows = manifest["compatibility"]["rows"]
    assert len(rows) >= 2
    statuses = set()
    for row in rows:
        assert row["id"]
        assert row["status"] in ("documented", "probed", "unverified")
        assert row["source_url"]
        assert row["evidence"]
        for key in ("cann", "torch", "torch_npu"):
            if row.get(key) is not None:
                assert str(row[key]).strip().lower() != "latest", row["id"]
                assert doctor.numeric_version(row[key]) is not None, row["id"]
        statuses.add(row["status"])
    assert "documented" in statuses, "the catalog must contain a vendor-documented pairing"
    assert any(row["status"] == "probed" for row in rows), "record what was actually observed"


def test_compatibility_annotations_match_the_host(manifest):
    table = doctor.compatibility(manifest)
    assert set(table) == {"observed", "rows", "cann_home", "version_file"}
    assert set(table["observed"]) == {"cann", "torch", "torch_npu"}
    for row in table["rows"]:
        assert row["matches"] in (True, False, None)
        if row["matches"] is True:
            for key in ("cann", "torch", "torch_npu"):
                if row.get(key):
                    assert row["compared_fields"], row["id"]
                    assert doctor.numeric_version(table["observed"][key]) == doctor.numeric_version(row[key])


# ---------------------------------------------------------------------------
# System descriptors
# ---------------------------------------------------------------------------
def test_system_descriptors_parse_and_reference_real_dependencies(manifest, entries):
    declared_ids = set(entries)
    declared_groups = {group["id"] for group in manifest["capability_groups"]}
    seen = set()
    for name in SYSTEM_FILES:
        path = DEPENDENCIES / "system" / name
        assert path.is_file(), name
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1, name
        assert data["id"], name
        for key in ("detect", "install", "env", "evidence"):
            assert key in data, f"{name}: missing {key}"
        for identifier in data["manifest_ids"]:
            assert identifier in declared_ids, f"{name}: unknown dependency {identifier}"
            seen.add(identifier)
        blocks = data.get("blocks") or {}
        if isinstance(blocks, dict):
            blocked_groups = [group for value in blocks.values()
                              for group in (value if isinstance(value, list) else [])]
        else:
            blocked_groups = [group for group in blocks if isinstance(group, str)]
        for group in blocked_groups:
            assert group in declared_groups, f"{name}: unknown capability group {group}"
        for item in data["evidence"]:
            assert item.get("id"), name
            assert item.get("probe"), f"{name}: evidence {item.get('id')} has no probe"
    assert {"cann-toolkit", "mpi", "hccl-test", "asc-comm-kernels"} <= seen


def test_ld_library_path_is_documented_where_it_matters():
    cann = yaml.safe_load((DEPENDENCIES / "system" / "cann.yaml").read_text(encoding="utf-8"))
    assert "ASCEND_HOME_PATH" in cann["env"]["required"]
    assert any("lib64" in entry for entry in cann["env"]["path_entries"])
    text = " ".join(cann["env"]["notes"].split())
    assert "LD_LIBRARY_PATH" in text


def test_asc_comm_descriptor_is_explicitly_blocked():
    data = yaml.safe_load((DEPENDENCIES / "system" / "asc_comm.yaml").read_text(encoding="utf-8"))
    assert data["status"] == "blocked"
    blocker_ids = {item["id"] for item in data["blockers"]}
    assert {"device_sources_absent", "cann_asc_comm_toolchain"} <= blocker_ids
    for item in data["blockers"]:
        assert item["detail"], item["id"]
        assert item["clears_when"], item["id"]
    assert data["blocks"], "a blocked dependency must name the families it blocks"


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def test_readme_documents_the_matrix_and_the_host_boundary():
    text = (DEPENDENCIES / "README.md").read_text(encoding="utf-8")
    for needle in ("wheelhouse", "LD_LIBRARY_PATH", "host", "doctor.py", "install_deps.sh",
                   "BLOCKED", "available", "degraded", "blocked"):
        assert needle in text, f"dependencies/README.md does not mention {needle!r}"
