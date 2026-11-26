"""CPU-only checks for the vendored ascendep adapters.

Nothing here needs an NPU, torch_npu or any native library: the adapters are only imported
from ``AscendEPBackend.initialize``, and these tests either stay on the resolution layer or
assert how a missing native dependency is reported.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.backend import BenchmarkUnavailable  # noqa: E402
from common.registry import registry  # noqa: E402
from operators import ascendep  # noqa: E402

VENDOR_ROOT = SCRIPTS / "operators" / "vendor" / "ascendep"
MANIFEST = VENDOR_ROOT / "VENDOR.json"
VENDORED_SCRIPT = SCRIPTS / "scripts" / "vendor_ascendep.py"
PINNED_COMMIT = "24ff84d3ab8c01e1e2071802465173e624e1f89d"
UPSTREAM_URL = "https://github.com/penn513/ascendep"
# Files in the vendored directory that are not upstream sources.
GENERATED = {"VENDOR.json", "README.md"}
# Every operator the registry routes through AscendEPBackend, MC2V2Backend, FusedGMMBackend
# or one of their subclasses, i.e. every operator that imports a vendored demo module.
ADAPTER_OPERATORS = (
    "allgather_fsdpturbo",
    "allgather_megatron",
    "allgather_overlap_mindspeed",
    "alltoall_fsdpturbo",
    "alltoall_megatron",
    "alltoallv_gmm_cann",
    "chunkmoe_fsdpturbo",
    "deepep_ascend",
    "deepep_cann",
    "deepep_direct_lueelu",
    "deepep_lueelu",
    "deepep_sglang",
    "domino_fsdpturbo",
    "ep_gmm_fused_lueelu",
    "fused_fsdpturbo",
    "mc2_dispatch_cann",
    "mc2_fsdpturbo",
    "megamoe_fsdpturbo",
    "megamoe_lueelu",
    "megamoe_td",
    "moonep",
)


@pytest.fixture(autouse=True)
def fresh_resolution():
    """Resolve the adapter source from scratch, so the environment decides every test."""
    ascendep.reset_source_cache()
    yield
    ascendep.reset_source_cache()


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_root_resolves_without_the_override(monkeypatch):
    monkeypatch.delenv("EP_ASCENDEP_ROOT", raising=False)
    root = ascendep.source_checkout()
    assert root == VENDOR_ROOT
    assert (root / "ep" / "runner.py").is_file()
    assert ascendep.source_checkout(None) == root


def test_manifest_digests_match_every_vendored_file():
    payload = manifest()
    listed = {entry["path"]: entry for entry in payload["files"]}
    on_disk = {path.relative_to(VENDOR_ROOT).as_posix()
               for path in VENDOR_ROOT.rglob("*")
               if path.is_file() and "__pycache__" not in path.parts
               and path.suffix != ".pyc"}
    assert on_disk - GENERATED == set(listed), "vendored files and VENDOR.json disagree"
    for relative, entry in listed.items():
        path = VENDOR_ROOT / relative
        assert entry["upstream_path"] == relative
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == digest(path), relative
    assert payload["upstream"]["commit"] == PINNED_COMMIT
    assert payload["upstream"]["url"] == UPSTREAM_URL
    assert payload["upstream"]["subdirectory"] == "ep"
    assert payload["license"]["status"] == "absent"
    assert (SCRIPTS / "operators" / "vendor" / "THIRD_PARTY_NOTICES.md").is_file()


def test_manifest_matches_the_vendoring_recipe():
    """VENDOR.json must be exactly what scripts/vendor_ascendep.py would copy."""
    spec = importlib.util.spec_from_file_location("vendor_ascendep_recipe", VENDORED_SCRIPT)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    assert set(recipe.selection()) == {entry["path"] for entry in manifest()["files"]}


def test_check_detects_tampering_missing_and_extra_files(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("vendor_ascendep_verifier", VENDORED_SCRIPT)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    copy = tmp_path / "ascendep"
    shutil.copytree(VENDOR_ROOT, copy)
    monkeypatch.setattr(recipe, "VENDOR_ROOT", copy)
    monkeypatch.setattr(recipe, "MANIFEST", copy / "VENDOR.json")
    assert recipe.check() == 0

    victim = copy / "ep" / "runner.py"
    victim.write_bytes(victim.read_bytes() + b"\n# drift\n")
    assert recipe.check() == 1, "a modified file must fail --check"
    shutil.copyfile(VENDOR_ROOT / "ep" / "runner.py", victim)
    assert recipe.check() == 0

    missing = copy / "ep" / "megatron.py"
    payload = missing.read_bytes()
    missing.unlink()
    assert recipe.check() == 1, "a missing file must fail --check"
    missing.write_bytes(payload)
    assert recipe.check() == 0

    extra = copy / "ep" / "moonep" / "extra.py"
    extra.write_text("", encoding="utf-8")
    assert recipe.check() == 1, "an unlisted file must fail --check"


def test_refresh_refuses_a_checkout_off_the_pinned_commit(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("vendor_ascendep_refresh", VENDORED_SCRIPT)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    (tmp_path / "ep").mkdir()
    (tmp_path / "ep" / "runner.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(recipe, "source_commit", lambda source: "f" * 40)
    before = MANIFEST.read_bytes()
    with pytest.raises(SystemExit) as refusal:
        recipe.refresh(tmp_path, allow_other_commit=False)
    assert refusal.value.code == 2
    assert MANIFEST.read_bytes() == before, "a refused refresh must not rewrite VENDOR.json"


def test_source_info_reports_the_vendored_origin(monkeypatch):
    monkeypatch.delenv("EP_ASCENDEP_ROOT", raising=False)
    info = ascendep.source_info()
    assert info["origin"] == "vendored"
    assert info["commit"] == ascendep.UPSTREAM_COMMIT == PINNED_COMMIT
    assert info["pinned"] == PINNED_COMMIT
    assert info["path"] == str(VENDOR_ROOT)
    assert info["manifest"] == str(MANIFEST)
    assert info["upstream"] == UPSTREAM_URL
    assert json.loads(json.dumps(info)) == info


def test_every_adapter_operator_has_a_vendored_module():
    specs = registry()
    for operator_id in ADAPTER_OPERATORS:
        module = ascendep.adapter_module(specs[operator_id])
        assert (VENDOR_ROOT / (module.replace(".", "/") + ".py")).is_file(), module
    # The registry maps deepep_direct_lueelu onto the deepep_ascend adapter.
    assert ascendep.adapter_module(specs["deepep_direct_lueelu"]) == "ep.deepep_ascend.demo"


def test_override_without_runner_is_a_source_problem(monkeypatch, tmp_path):
    monkeypatch.setenv("EP_ASCENDEP_ROOT", str(tmp_path))
    with pytest.raises(BenchmarkUnavailable) as error:
        ascendep.source_info()
    assert error.value.status == "source_unavailable"
    assert "EP_ASCENDEP_ROOT" in str(error.value)
    assert "ep/runner.py" in str(error.value)
    with pytest.raises(BenchmarkUnavailable) as checkout_error:
        ascendep.source_checkout()
    assert checkout_error.value.status == "source_unavailable"


def test_override_still_needs_the_pinned_checkout(monkeypatch, tmp_path):
    (tmp_path / "ep").mkdir()
    (tmp_path / "ep" / "runner.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("EP_ASCENDEP_ROOT", str(tmp_path))
    with pytest.raises(BenchmarkUnavailable) as error:
        ascendep.source_info()
    assert error.value.status == "source_unavailable"
    assert "EP_ASCENDEP_ROOT" in str(error.value)


def test_unvendored_adapter_module_is_reported_as_a_source_problem(monkeypatch):
    monkeypatch.delenv("EP_ASCENDEP_ROOT", raising=False)
    spec = SimpleNamespace(id="operator_without_an_adapter", upstream=None, capabilities=())
    with pytest.raises(BenchmarkUnavailable) as error:
        ascendep.AscendEPBackend(spec).initialize(object())
    assert error.value.status == "source_unavailable"
    assert "ep.operator_without_an_adapter.demo" in str(error.value)


@pytest.mark.skipif(importlib.util.find_spec("torch_npu") is not None,
                    reason="torch_npu is installed, so the adapter import cannot fail this way")
def test_missing_native_library_is_named_in_the_failure(monkeypatch):
    monkeypatch.delenv("EP_ASCENDEP_ROOT", raising=False)
    backend = ascendep.FusedGMMBackend(registry()["alltoallv_gmm_cann"])
    with pytest.raises(BenchmarkUnavailable) as error:
        backend.initialize(object())
    assert error.value.status == "dependency_missing"
    assert "torch_npu" in str(error.value)


def test_missing_vendored_tree_names_the_real_blocker(monkeypatch, tmp_path):
    monkeypatch.delenv("EP_ASCENDEP_ROOT", raising=False)
    monkeypatch.setattr(ascendep, "VENDORED_ROOT", tmp_path / "absent")
    with pytest.raises(BenchmarkUnavailable) as error:
        ascendep.source_info()
    assert error.value.status == "source_unavailable"
    assert "missing" in str(error.value)
    assert "vendor_ascendep.py --check" in str(error.value)


def test_vendored_manifest_commit_drift_is_reported(monkeypatch, tmp_path):
    monkeypatch.delenv("EP_ASCENDEP_ROOT", raising=False)
    copy = tmp_path / "ascendep"
    shutil.copytree(VENDOR_ROOT, copy)
    payload = json.loads((copy / "VENDOR.json").read_text(encoding="utf-8"))
    payload["upstream"]["commit"] = "0" * 40
    (copy / "VENDOR.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    monkeypatch.setattr(ascendep, "VENDORED_ROOT", copy)
    monkeypatch.setattr(ascendep, "VENDORED_MANIFEST", copy / "VENDOR.json")
    with pytest.raises(BenchmarkUnavailable) as error:
        ascendep.source_info()
    assert error.value.status == "source_unavailable"
    assert "vendor_ascendep.py --refresh" in str(error.value)


def test_importing_the_module_does_not_load_native_libraries(tmp_path):
    script = "\n".join([
        "import json, sys",
        f"sys.path.insert(0, {str(SCRIPTS)!r})",
        "import operators.ascendep as module",
        "print(json.dumps({'loaded': sorted(name for name in ('torch_npu', 'torch', 'deep_ep',",
        "                                       'fsdp_turbo', 'npu_patch', 'mindspeed', 'megatron')",
        "                             if name in sys.modules),",
        "                  'origin': module.source_info()['origin']}))",
    ])
    environment = {key: value for key, value in os.environ.items()
                   if key != "EP_ASCENDEP_ROOT"}
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert "torch_npu" not in payload["loaded"], payload
    assert payload["origin"] == "vendored"


def test_vendor_check_cli_passes():
    result = subprocess.run([sys.executable, str(VENDORED_SCRIPT), "--check"], cwd=SCRIPTS,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
