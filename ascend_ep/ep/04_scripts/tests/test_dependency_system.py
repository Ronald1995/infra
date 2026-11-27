"""Dependency-system modules: platform evidence, distributions, sources, envs, builds.

These tests are read-only against the repository and never touch a device.  The
only subprocess they run is `git`, to build a synthetic checkout for the source
verifier.
"""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from common import distributions, platform_support, readiness, status_map
from scripts import env_manager, fetch_third_party, native_build

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# platform support: documentation is not a verdict
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def support():
    return platform_support.load()


def test_the_four_undocumented_releases_need_validation(support):
    for dep in ("fsdp-turbo", "deep-ep-experimental", "mega-moe", "moonep-runtime"):
        assert support[dep]["Ascend950"]["status"] == platform_support.UNDOCUMENTED
        assert support[dep]["Ascend950"]["validation_required"] is True


def test_undocumented_does_not_block_and_needs_validation(support):
    assert platform_support.blocks(platform_support.UNDOCUMENTED) is False
    assert platform_support.blocks(platform_support.REJECTED) is True
    queue = platform_support.validation_queue(support, "Ascend950")
    assert {item["dependency"] for item in queue} == {
        "fsdp-turbo", "deep-ep-experimental", "mega-moe", "moonep-runtime"}


def test_a_rejection_without_evidence_is_refused(support):
    with pytest.raises(platform_support.PlatformSupportError, match="evidence"):
        platform_support.promote(support, "fsdp-turbo", "Ascend950",
                                 platform_support.REJECTED, evidence=None)


def test_evidence_promotes_an_undocumented_dependency(support):
    updated = platform_support.promote(support, "fsdp-turbo", "Ascend950",
                                       platform_support.REJECTED,
                                       "run smoke-a5: fsdp_turbo import raised "
                                       "UnsupportedSoc (a3-only build)")
    assert updated["fsdp-turbo"]["Ascend950"]["status"] == platform_support.REJECTED
    # the original table is untouched
    assert support["fsdp-turbo"]["Ascend950"]["status"] == platform_support.UNDOCUMENTED


def test_status_falls_back_to_the_manifest(support):
    entry = {"supported_soc": ["Ascend910B", "Ascend910_93"]}
    assert platform_support.status_for(support, "unknown-dep", "Ascend950", entry) == \
        platform_support.UNDOCUMENTED
    assert platform_support.status_for(support, "unknown-dep", "Ascend910_93", entry) == \
        platform_support.DOCUMENTED
    assert platform_support.status_for(support, "numpy", "Ascend950",
                                       {"supported_soc": ["*"]}) == \
        platform_support.DOCUMENTED


def test_a_rejected_dependency_becomes_a_platform_block_in_readiness():
    rows = [{"operator": "x", "dependencies": ["fsdp-turbo"],
             "states": {"fsdp-turbo": "fail"}, "unmapped_modules": []}]
    entries = {"fsdp-turbo": {"id": "fsdp-turbo", "supported_soc": ["Ascend910B"]}}
    table = {"fsdp-turbo": {"Ascend950": {"status": platform_support.REJECTED,
                                          "validation_required": False,
                                          "note": "", "evidence": "run 1 failed on soc"}}}
    result = readiness.classify(rows, entries, "Ascend950", support_table=table)["x"]
    assert result["state"] == readiness.UNSUPPORTED_PLATFORM
    assert result["bucket"] == status_map.BLOCKED_PLATFORM


def test_undocumented_stays_a_dependency_block_in_readiness():
    rows = [{"operator": "y", "dependencies": ["fsdp-turbo"],
             "states": {"fsdp-turbo": "fail"}, "unmapped_modules": []}]
    entries = {"fsdp-turbo": {"id": "fsdp-turbo", "supported_soc": ["Ascend910B"]}}
    table = {"fsdp-turbo": {"Ascend950": {"status": platform_support.UNDOCUMENTED,
                                          "validation_required": True,
                                          "note": "", "evidence": None}}}
    result = readiness.classify(rows, entries, "Ascend950", support_table=table)["y"]
    assert result["state"] == readiness.UNDOCUMENTED_ON_SOC
    assert result["bucket"] == status_map.BLOCKED_DEPENDENCY


# ---------------------------------------------------------------------------
# distributions: a lookalike name is not a substitute
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dists():
    return distributions.load()


def test_the_four_lookalikes_are_flagged(dists):
    for dep, lookalike in (("megatron", "megatron-core"), ("mindspeed", "mindspeed"),
                           ("deep-ep", "deep-ep"), ("deepgemm-ascend", "deep-gemm")):
        hazard = distributions.substitution_hazard(dists, dep)
        assert hazard and lookalike in hazard, dep
        assert "must not be substituted" in hazard


def test_genuine_pypi_dependencies_are_substitutable(dists):
    for dep in ("numpy", "pyyaml", "jsonschema", "pandas", "matplotlib"):
        assert distributions.substitution_hazard(dists, dep) is None
        assert distributions.package_for(dists, dep) is not None


def test_pinned_checkouts_expose_modules_but_no_distribution(dists):
    assert distributions.package_for(dists, "fsdp-turbo") is None
    assert distributions.modules_for(dists, "fsdp-turbo") == ("fsdp_turbo",)
    # the two DeepEP forks share a module name, which is why they need separate envs
    assert distributions.modules_for(dists, "deep-ep") == \
        distributions.modules_for(dists, "deep-ep-v2") == ("deep_ep",)
    assert distributions.modules_for(dists, "npu-patch") == \
        distributions.modules_for(dists, "npu-patch-dedup") == ("npu_patch",)


def test_module_origin_check_detects_an_outside_import(tmp_path):
    import sys

    package_root = tmp_path / "checkout"
    module_dir = package_root / "ep_probe_module"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    sys.path.insert(0, str(package_root))
    try:
        table = {"x": {"import_modules": ["ep_probe_module"]}}
        inside = distributions.check_module_origin(table, "x", checkout=package_root)
        assert inside["ok"] is True and inside["outside"] == []

        # The same module, claimed to come from a different checkout: that is the
        # signal that a lookalike package won the import race.
        elsewhere = tmp_path / "other"
        elsewhere.mkdir()
        outside = distributions.check_module_origin(table, "x", checkout=elsewhere)
        assert outside["ok"] is False
        assert outside["outside"] == ["ep_probe_module"]
    finally:
        sys.path.remove(str(package_root))


# ---------------------------------------------------------------------------
# third-party source catalogue
# ---------------------------------------------------------------------------

@pytest.fixture()
def git_repo(tmp_path):
    """A real one-commit repository, so the verifier is tested against git."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    import os
    run = lambda *args: subprocess.run(["git", "-C", str(repo), *args], check=True,
                                       capture_output=True, text=True,
                                       env={**os.environ, **env})
    run("init", "-q")
    (repo / "LICENSE").write_text("Apache License, Version 2.0\n", encoding="utf-8")
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    run("add", ".")
    run("commit", "-q", "-m", "init")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return repo, head


def _catalogue(tmp_path, entries, root):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "root": str(root),
                                    "sources": entries}), encoding="utf-8")
    return path


def test_a_pinned_checkout_at_the_right_commit_is_found(git_repo, tmp_path):
    repo, head = git_repo
    catalogue = fetch_third_party.load_catalogue(
        _catalogue(tmp_path, {"demo": {"upstream": "x", "commit": head,
                                       "directory": "checkout"}}, tmp_path))
    record = fetch_third_party.inspect(catalogue, "demo")
    assert record["status"] == fetch_third_party.FOUND
    assert record["actual_commit"] == head
    assert record["license"] == "Apache-2.0"


def test_a_checkout_at_the_wrong_commit_is_a_mismatch(git_repo, tmp_path):
    repo, head = git_repo
    catalogue = fetch_third_party.load_catalogue(
        _catalogue(tmp_path, {"demo": {"upstream": "x", "commit": "0" * 40,
                                       "directory": "checkout"}}, tmp_path))
    record = fetch_third_party.inspect(catalogue, "demo")
    assert record["status"] == fetch_third_party.COMMIT_MISMATCH
    assert record["actual_commit"] == head


def test_a_missing_checkout_and_an_unpinned_entry_are_distinct(tmp_path):
    catalogue = fetch_third_party.load_catalogue(
        _catalogue(tmp_path, {"gone": {"upstream": "x", "commit": "a" * 40,
                                       "directory": "gone"},
                              "later": {"upstream": "y", "commit": None,
                                        "directory": "later"}}, tmp_path))
    assert fetch_third_party.inspect(catalogue, "gone")["status"] == \
        fetch_third_party.MISSING
    assert fetch_third_party.inspect(catalogue, "later")["status"] == \
        fetch_third_party.UNPINNED


def test_the_real_catalogue_pins_full_shas_and_lists_the_system_components():
    catalogue = fetch_third_party.load_catalogue()
    for name, entry in catalogue["sources"].items():
        commit = entry.get("commit")
        if commit:
            assert len(commit) == 40, f"{name} must pin the full sha"
    assert {"hccl", "hcomm", "cann-toolkit"} <= set(catalogue["system_provided"])
    assert "vendored-ascendep" in catalogue["vendored"]


def test_check_on_this_host_does_not_claim_a_missing_source_is_present():
    catalogue = fetch_third_party.load_catalogue()
    records = fetch_third_party.inspect_all(catalogue, only=["megatron-lm"])
    assert records[0]["status"] in (fetch_third_party.MISSING,
                                    fetch_third_party.COMMIT_MISMATCH)


# ---------------------------------------------------------------------------
# environments
# ---------------------------------------------------------------------------

def test_the_four_isolated_environments_are_declared():
    environments = env_manager.load_environments()
    isolated = {name for name, entry in environments.items() if entry.get("isolated")}
    assert {"deep_ep_lueelu", "deep_ep_v2", "moonep_npu", "moonep_dedup"} <= isolated
    # every isolated environment inherits the container's torch stack
    for name in isolated:
        assert environments[name].get("inherits") == "base"


def test_env_check_reports_the_base_environment():
    records = env_manager.check_all(only=["base"])
    assert records[0]["status"] == env_manager.PRESENT
    assert records[0]["isolated"] is False


def test_creating_an_environment_is_refused_without_a_usable_npu_stack(monkeypatch, tmp_path):
    monkeypatch.setattr(env_manager, "base_check",
                        lambda python=None: {"torch": None, "torch_npu": None,
                                             "npu_available": False, "usable": False})
    code = env_manager.main(["--create", "--root", str(tmp_path), "--only", "deep_ep_v2"])
    assert code == 2
    assert not (tmp_path / "deep_ep_v2").exists()


def test_base_check_reports_the_three_facts():
    info = env_manager.base_check()
    assert "python" in info
    assert "usable" in info
    assert isinstance(info["usable"], bool)


# ---------------------------------------------------------------------------
# native build
# ---------------------------------------------------------------------------

def test_in_repo_targets_and_third_party_targets_are_separated():
    targets = native_build.all_targets()
    assert "asc_comm" in targets and "l0_bridge" in targets
    assert "shmem" in targets and "deep-ep" in targets


def test_shmem_is_a_kernel_build_and_megatron_is_only_an_install():
    catalogue = native_build.load_catalogue()
    targets = native_build.third_party_targets(catalogue)
    assert targets["shmem"]["build"], "shmem needs build.sh"
    assert not targets["megatron-lm"]["build"], "megatron is an editable install"
    assert targets["megatron-lm"]["install"]


def test_a_dry_run_records_the_command_without_building():
    record = native_build.run_target("asc_comm", dry_run=True)
    assert record["status"] in ("DRY_RUN", native_build.BUILT_HOST_ONLY)
    assert any("build.sh" in part for part in record["command"])


def test_a_missing_checkout_is_blocked_not_failed():
    record = native_build.run_target("shmem")
    assert record["status"] == native_build.BLOCKED_DEPENDENCY
    assert "fetch_third_party" in record["reason"]


def test_artefact_state_reports_device_kernels_honestly():
    state = native_build.artefact_state("asc_comm") if (
        ROOT / native_build.IN_REPO["asc_comm"]["artifact"]).is_file() else {"present": False}
    if state.get("present"):
        # a host-only link must not be reported as device kernels
        assert "device_kernels_built" in state
    else:
        assert state["present"] is False
