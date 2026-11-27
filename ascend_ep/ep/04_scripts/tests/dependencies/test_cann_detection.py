"""CANN detection and shared-library checks against a real Ascend layout.

Two bugs the liurong_vllm container exposed:

* CANN 9.0.0 declares its release in ``<arch>/ascend_toolkit_install.info`` (and some
  installs declare none at all). The probe only looked at ``version.cfg`` in the root
  and reported "no CANN installation" on a working toolkit.
* ``hccl``/``hcomm`` list ``libhccl.so`` *and* ``libhccl.dll`` as platform
  alternatives, but the probe required every name, so Linux always failed.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import doctor  # noqa: E402


@pytest.fixture(autouse=True)
def clear_cache():
    doctor.reset_cache()
    yield
    doctor.reset_cache()


def make_toolkit(root, version_file=None, content="version=9.0.0\n", landmark=None):
    root.mkdir(parents=True, exist_ok=True)
    if version_file:
        path = root / version_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if landmark:
        path = root / landmark
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return root


def test_the_arch_subdirectory_install_info_is_read(tmp_path, monkeypatch):
    root = make_toolkit(tmp_path / "cann", "x86_64-linux/ascend_toolkit_install.info")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    found, version, version_file, evidence = doctor._cann_installation()
    assert found == str(root)
    assert version == "9.0.0"
    assert version_file.endswith("ascend_toolkit_install.info")
    assert evidence["root"] == str(root)


def test_a_root_version_cfg_still_wins(tmp_path, monkeypatch):
    root = make_toolkit(tmp_path / "cann", "version.cfg", "version=8.2.RC1\n")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    _, version, version_file, _ = doctor._cann_installation()
    assert version == "8.2.RC1"
    assert version_file.endswith("version.cfg")


def test_a_toolkit_without_a_version_file_is_found_by_its_landmark(tmp_path, monkeypatch):
    root = make_toolkit(tmp_path / "cann", landmark="lib64/libascendcl.so")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    found, version, version_file, evidence = doctor._cann_installation()
    assert found == str(root)
    assert version is None and version_file is None
    assert evidence["landmark"] == "lib64/libascendcl.so"


def test_the_cann_probe_warns_instead_of_failing_on_a_versionless_toolkit(
        tmp_path, monkeypatch):
    root = make_toolkit(tmp_path / "cann", landmark="bin/ccec")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    entry = {"id": "cann-toolkit", "category": "B", "version": "8.2.RC1",
             "required_by": ["l0_memory"], "manual_step": "install CANN"}
    check = {"type": "cann_home", "version_file": "version.cfg"}
    result = doctor.probe_cann_home(entry, check, type("C", (), {"root": tmp_path})())
    assert result.status == "warn"
    assert "declares no version" in result.detail


def test_an_empty_candidate_root_still_fails(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(empty))
    monkeypatch.setattr(doctor, "_cann_installation", lambda: (None, None, None, {}))
    entry = {"id": "cann-toolkit", "category": "B", "version": "8.2.RC1",
             "required_by": ["l0_memory"], "manual_step": "install CANN"}
    result = doctor.probe_cann_home(entry, {"type": "cann_home"},
                                    type("C", (), {"root": tmp_path})())
    assert result.status == "fail"


def test_platform_library_alternatives_only_need_one_match(tmp_path):
    (tmp_path / "libhccl.so").write_text("", encoding="utf-8")
    entry = {"id": "hccl", "category": "B", "required_by": ["l0_hardware"],
             "version": "*", "manual_step": "install CANN"}
    check = {"type": "shared_library", "names": ["libhccl.so", "libhccl.dll"],
             "require_all": False, "search": [str(tmp_path)]}
    result = doctor.probe_shared_library(entry, check,
                                         type("C", (), {"root": tmp_path})())
    assert result.status == "pass", result.detail
    assert any(name.endswith("libhccl.so") for name in result.evidence["resolved"])


def test_a_missing_platform_library_still_fails(tmp_path):
    entry = {"id": "hccl", "category": "B", "required_by": ["l0_hardware"],
             "version": "*", "manual_step": "install CANN"}
    check = {"type": "shared_library", "names": ["libhccl.so", "libhccl.dll"],
             "require_all": False, "search": [str(tmp_path)]}
    result = doctor.probe_shared_library(entry, check,
                                         type("C", (), {"root": tmp_path})())
    assert result.status == "fail"


def test_the_manifest_does_not_require_both_platform_names():
    manifest = doctor.load_manifest(ROOT / "dependencies" / "manifest.yaml")
    entries = {entry["id"]: entry for entry in manifest["dependencies"]}
    for identifier in ("hccl", "hcomm"):
        names = entries[identifier]["check"]["names"]
        assert len(names) > 1
        assert entries[identifier]["check"].get("require_all") is False, identifier
