"""Data plane and completion honesty.

The rule these tests protect: a case whose data plane is an AIV or CCU kernel may not
produce a number unless that kernel exists, and no completion primitive that only
fences the submission may be reported as a completed transfer.
"""
import json
import os
from pathlib import Path

import pytest

from baselines.l0.transport import completion, dataplane, kernel_artifacts
from common.backend import BenchmarkUnavailable
from common.prepared import COMPLETION_LEVELS, PreparedCase


# ------------------------------------------------------------- completion ---

def test_a_fence_cannot_prove_a_transfer():
    with pytest.raises(ValueError, match="cannot prove"):
        completion.CompletionContract("bad", "fenced", True, "claims too much")


def test_unknown_completion_level_is_refused():
    with pytest.raises(ValueError, match="outside"):
        completion.CompletionContract("bad", "maybe", False, "unknown level")


def test_the_host_nbi_drain_is_a_fence_and_says_so():
    assert completion.HOST_NBI_DRAIN.level == "fenced"
    assert completion.HOST_NBI_DRAIN.proves_transfer is False
    assert completion.HOST_NBI_DRAIN.usable_for_bandwidth is False
    assert "peer" in completion.HOST_NBI_DRAIN.detail


def test_device_planes_declare_remote_visibility():
    assert completion.AIV_DRAIN.level == "remote_visible"
    assert completion.CCU_COMPLETION.level == "remote_visible"
    assert completion.AIV_DRAIN.proves_transfer


def test_a_cpu_engine_returning_immediately_is_not_completion():
    assert completion.CPU_ENGINE_QUEUED.level == "submitted"
    assert completion.CPU_ENGINE_QUEUED.proves_transfer is False


def test_completion_is_chosen_by_engine_and_device_plane():
    assert completion.for_engine("AIV").name == "host_nbi_fence"
    assert completion.for_engine("AIV", device_plane=True).name == "aiv_drain"
    assert completion.for_engine("CCU", device_plane=True).name == "ccu_completion"
    assert completion.for_engine("HOST_CPU").name == "cpu_engine_queued"


def test_every_contract_serialises():
    for name in completion.BY_NAME:
        data = completion.contract(name).as_dict()
        assert set(data) == {"mechanism", "level", "proves_transfer",
                             "requires_peer_validation", "usable_for_bandwidth", "detail"}


# -------------------------------------------------------------- data plane ---

def test_execution_resource_selects_an_independent_data_plane():
    assert dataplane.kind_for("AIV") == "aiv"
    assert dataplane.kind_for("CCU") == "ccu"
    assert dataplane.kind_for("AICPU_TS") == "host"
    assert dataplane.kind_for("HOST_CPU") == "host"
    with pytest.raises(BenchmarkUnavailable):
        dataplane.kind_for("TPU")


def test_aiv_and_ccu_planes_are_not_the_host_plane():
    aiv = dataplane.for_resource("AIV")
    ccu = dataplane.for_resource("CCU")
    host = dataplane.for_resource("AICPU_TS")
    assert aiv.kind == "aiv" and ccu.kind == "ccu" and host.kind == "host"
    assert not isinstance(aiv, dataplane.HostDataPlane)
    assert not isinstance(ccu, dataplane.HostDataPlane)


def test_a_device_plane_without_a_kernel_reports_its_blocker_and_never_hosts():
    plane = dataplane.for_resource("AIV")
    blocker = plane.blocker()
    assert blocker is not None
    status, reason = blocker
    assert status in {"dependency_missing", "not_implemented"}
    assert reason
    with pytest.raises(BenchmarkUnavailable) as caught:
        plane.initialize(object())
    assert caught.value.status == status
    # The host path is never silently substituted.
    assert not plane.initialized


def test_kernel_probe_is_honest_about_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_artifacts, "MANIFEST", tmp_path / "missing.json")
    artifacts = kernel_artifacts.probe("AIV")
    assert artifacts.available is False
    assert artifacts.status in {"dependency_missing", "not_implemented"}
    assert artifacts.reason
    data = artifacts.as_dict()
    assert data["built"] is False


def test_kernel_probe_rejects_a_library_without_the_abi(tmp_path, monkeypatch):
    library = tmp_path / "libasc_comm_kernels.so"
    library.write_bytes(b"not a shared object")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"engines": {
        "AIV": {"built": True, "library": str(library)}}}), encoding="utf-8")
    monkeypatch.setattr(kernel_artifacts, "MANIFEST", manifest)
    artifacts = kernel_artifacts.probe("AIV")
    assert artifacts.available is False
    assert any(name in artifacts.missing for name in
               kernel_artifacts.REQUIRED_SYMBOLS["AIV"])


def test_kernel_probe_refuses_a_library_that_does_not_export_the_symbols(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"engines": {
        "CCU": {"built": True, "library": str(tmp_path / "gone.so")}}}), encoding="utf-8")
    monkeypatch.setattr(kernel_artifacts, "MANIFEST", manifest)
    artifacts = kernel_artifacts.probe("CCU")
    assert artifacts.available is False
    assert "ccu_library" in artifacts.missing


def test_the_python_abi_matches_the_c_header():
    header = (kernel_artifacts.PROJECT / "common" / "abi.h").read_text(encoding="utf-8")
    fields = [name for name, _type in dataplane.AscCommTransfer._fields_]
    positions = [header.index(f" {name};") for name in fields]
    assert positions == sorted(positions), "ASCommTransfer field order drifted from abi.h"
    for name in kernel_artifacts.REQUIRED_SYMBOLS["AIV"] + kernel_artifacts.REQUIRED_SYMBOLS["CCU"]:
        assert f"{name}(" in header, f"{name} is not declared in abi.h"


def test_the_kernel_build_script_blocks_without_the_toolchain(tmp_path):
    import shutil
    import subprocess

    project = kernel_artifacts.PROJECT
    script = project / "build.sh"
    if not script.is_file():
        pytest.skip("build.sh is not present")
    if not shutil.which("bash"):
        pytest.skip("bash is not available on this host")
    env = {key: value for key, value in os.environ.items() if key != "ASCEND_HOME_PATH"}
    result = subprocess.run(["bash", str(script), "--strict"], cwd=str(project), env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "BLOCKED" in (result.stderr + result.stdout)
    assert not (project / "build" / "manifest.json").exists()


# ------------------------------------------------------------ prepared case ---

def test_prepared_case_validates_its_contract():
    prepared = PreparedCase(run=lambda value: value, validate=lambda: {})
    assert prepared.completion == "none"
    assert prepared.transfers_distinct_from_submission() is False
    prepared = PreparedCase(run=lambda value: value, validate=lambda: {},
                            completion="remote_visible", completion_detail="peer ack")
    assert prepared.transfers_distinct_from_submission() is True
    with pytest.raises(ValueError, match="outside"):
        PreparedCase(run=lambda value: value, validate=lambda: {}, completion="maybe")


def test_prepared_case_rejects_a_non_callable_step():
    with pytest.raises(TypeError, match="must be callable"):
        PreparedCase(run=None, validate=lambda: {})


def test_completion_levels_are_ordered():
    assert COMPLETION_LEVELS.index("submitted") < COMPLETION_LEVELS.index("fenced")
    assert COMPLETION_LEVELS.index("fenced") < COMPLETION_LEVELS.index("local_complete")
    assert COMPLETION_LEVELS.index("local_complete") < COMPLETION_LEVELS.index("remote_visible")
