"""Tests for the environment doctor and the catalog it reads.

Why these tests are invariants instead of fixtures: the doctor reports the machine
it runs on, so asserting a particular status would fail on the next host.  Each
test below holds on *every* host:

* a dependency the probe could not find is never reported as ``pass``;
* ``available`` in the matrix means every dependency the catalog lists for that
  family passed, and ``blocked`` names at least one failing dependency;
* the status vocabulary is closed, and every check carries the evidence that
  justifies its verdict;
* the JSON the CLI prints round-trips and agrees with the in-process report.

The one host-specific expectation is the CPU-only one: on a machine without CANN
or an NPU, the CANN stack must be ``fail`` (the test skips itself on a machine
that has the stack).
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `pytest tests/dependencies` from another cwd
    sys.path.insert(0, str(ROOT))

from scripts import doctor  # noqa: E402  (path bootstrap must run first)

REQUIRED_KEYS = (
    "id", "category", "kind", "required_by", "version", "source_url", "commit", "license",
    "optional", "installable", "install_reason", "manual_step", "install", "check",
    "runtime_paths", "supported_soc", "supported_cann", "notes",
)
#: Dependencies that must be FAIL on a host with no CANN and no NPU.
CPU_ONLY_FAILURES = (
    "cann-toolkit", "torch-npu", "hccl", "shmem", "deep-ep", "mpi", "hccl-test", "msprof",
    "device-permissions",
)


@pytest.fixture(scope="module")
def manifest():
    return doctor.load_manifest(ROOT / "dependencies" / "manifest.yaml")


@pytest.fixture(scope="module")
def report():
    return doctor.run_doctor(profile="all", root=ROOT)


def groups_of(manifest):
    return [group["id"] for group in manifest["capability_groups"]]


def run_cli(arguments):
    return subprocess.run([sys.executable, *arguments], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=240)


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------
def test_manifest_is_valid(manifest):
    assert doctor.validate_manifest(manifest) == []


def test_every_entry_has_the_documented_keys(manifest):
    for entry in manifest["dependencies"]:
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        assert missing == [], f"{entry.get('id')}: missing {missing}"


def test_every_entry_has_a_valid_kind_and_category(manifest):
    for entry in manifest["dependencies"]:
        assert entry["kind"] in doctor.KINDS, entry["id"]
        assert entry["category"] in doctor.CATEGORIES, entry["id"]
        assert entry["id"] == entry["id"].strip()


def test_required_by_tokens_come_from_the_capability_groups(manifest):
    known = set(groups_of(manifest))
    for entry in manifest["dependencies"]:
        declared = doctor.tokens(entry["required_by"])
        assert declared, f"{entry['id']}: required_by must not be empty"
        unknown = [token for token in declared if token not in known]
        assert unknown == [], f"{entry['id']}: unknown required_by tokens {unknown}"


def test_operator_groups_match_the_operator_registry(manifest):
    registry = yaml.safe_load((ROOT / "operators" / "registry.yaml").read_text(encoding="utf-8"))
    registered = {f"operator:{name}" for name in registry["operators"]}
    groups = {group for group in groups_of(manifest) if group.startswith("operator:")}
    assert groups == registered


def test_every_group_is_required_by_something(manifest):
    covered = set()
    for entry in manifest["dependencies"]:
        covered.update(doctor.tokens(entry["required_by"]))
    unreachable = [group for group in groups_of(manifest) if group not in covered]
    assert unreachable == []


def test_status_vocabulary_is_closed(manifest):
    assert list(manifest["vocabulary"]["check_status"]) == list(doctor.STATUSES)
    assert list(manifest["vocabulary"]["matrix_status"]) == list(doctor.MATRIX_STATUSES)
    assert list(manifest["vocabulary"]["kinds"]) == list(doctor.KINDS)
    assert list(manifest["vocabulary"]["check_types"]) == list(doctor.CHECK_TYPES)
    assert dict(manifest["profiles"]) == {name: list(categories)
                                          for name, categories in doctor.PROFILES.items()}


# ---------------------------------------------------------------------------
# A probe never turns "not found" into "pass"
# ---------------------------------------------------------------------------
def _evidence_says_absent(evidence):
    if not isinstance(evidence, dict):
        return False
    if "find_spec" in evidence and evidence["find_spec"] is None:
        return True                                    # python module
    if "binary" in evidence and evidence.get("resolved") is None:
        return True                                    # one binary
    if "names" in evidence and evidence.get("resolved") == {}:
        return True                                    # every shared library missing
    if "binaries" in evidence and evidence.get("missing"):
        return True                                    # toolchain member
    if "patterns" in evidence and not evidence.get("matches"):
        return True                                    # device node
    return False


def test_absent_dependency_is_never_reported_as_pass(report):
    offenders = []
    for check in report.checks:
        if _evidence_says_absent(check.evidence) and check.status == "pass":
            offenders.append(check.id)
    assert offenders == [], f"absent dependencies reported as pass: {offenders}"


def test_every_check_carries_evidence_and_an_action(report):
    for check in report.checks:
        assert check.source == "probe"
        assert check.status in doctor.STATUSES
        assert check.evidence is not None, check.id
        assert check.detail, check.id
        if check.status in ("fail", "warn"):
            assert check.suggested_action, f"{check.id}: a failure needs an action"
        assert isinstance(check.blocking_for, tuple)


def test_counts_account_for_every_check(report):
    assert sum(report.counts.values()) == len(report.checks)
    observed = {status: 0 for status in doctor.STATUSES}
    for check in report.checks:
        observed[check.status] += 1
    assert report.counts == observed


# ---------------------------------------------------------------------------
# Matrix semantics
# ---------------------------------------------------------------------------
def test_matrix_statuses_are_closed(report):
    for row in report.matrix:
        assert row.status in doctor.MATRIX_STATUSES, row.id
        assert row.title, row.id


def test_matrix_available_means_every_dependency_passed(report):
    statuses = {check.id: check.status for check in report.checks}
    for row in report.matrix:
        if row.status == "available":
            assert not row.missing and not row.degraded, row.id
        elif row.status == "degraded":
            assert not row.missing, row.id
            assert row.degraded, row.id
            assert all(statuses[item] in ("warn", "skip") for item in row.degraded), row.id
        else:
            assert row.missing, row.id
            assert all(statuses[item] == "fail" for item in row.missing), row.id


def test_matrix_covers_every_group_when_nothing_is_filtered(report, manifest):
    in_scope = {row.id for row in report.matrix}
    out_of_scope = {item["id"] for item in report.out_of_scope}
    assert in_scope | out_of_scope == set(groups_of(manifest))
    assert in_scope & out_of_scope == set()


def test_partial_category_probe_has_no_matrix():
    partial = doctor.run_doctor(profile="npu", categories=("B",), root=ROOT)
    assert partial.matrix == ()
    assert all(check.category == "B" for check in partial.checks)
    assert partial.categories == ("B",)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        doctor.run_doctor(profile="gpu", root=ROOT)
    with pytest.raises(ValueError):
        doctor.run_doctor(profile="cpu", categories=("Z",), root=ROOT)


# ---------------------------------------------------------------------------
# The CPU-only expectation
# ---------------------------------------------------------------------------
def test_torch_is_probed_by_import(report):
    torch_check = next(check for check in report.checks if check.id == "torch")
    assert torch_check.status == "pass", torch_check.detail
    assert torch_check.evidence.get("imported") is True
    assert torch_check.evidence.get("version")


def test_npu_stack_fails_on_a_host_without_cann(report):
    if doctor.installed_version("torch-npu") or shutil.which("npu-smi"):
        pytest.skip("this host has an NPU stack; the CPU-only expectation does not apply")
    statuses = {check.id: check.status for check in report.checks}
    failing = {identifier for identifier in CPU_ONLY_FAILURES if statuses.get(identifier) == "fail"}
    assert failing == set(CPU_ONLY_FAILURES), sorted(set(CPU_ONLY_FAILURES) - failing)
    assert report.counts["fail"] >= 1
    assert report.exit_code == 1


def test_human_render_is_a_table_with_a_summary(report):
    text = doctor.render(report)
    assert "STATUS" in text and "summary:" in text
    assert f"profile={report.profile}" in text or report.profile in text
    assert "capability matrix" in text


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def test_json_round_trips_from_the_module_entry_point():
    completed = run_cli(["-m", "scripts.doctor", "--profile", "all", "--json"])
    assert completed.returncode in (0, 1), completed.stderr
    payload = json.loads(completed.stdout)
    assert set(payload) == {"schema_version", "profile", "categories", "platform", "counts",
                            "exit_code", "checks", "matrix", "out_of_scope", "wheelhouse"}
    assert sum(payload["counts"].values()) == len(payload["checks"])
    assert all(check["status"] in doctor.STATUSES for check in payload["checks"])
    assert all(row["status"] in doctor.MATRIX_STATUSES for row in payload["matrix"])
    expected = 1 if (payload["counts"]["fail"] or payload["wheelhouse"]["counts"]["fail"]) else 0
    assert payload["exit_code"] == expected == completed.returncode
    assert json.dumps(payload)  # re-serialisable


def test_json_round_trips_from_the_standalone_script():
    completed = run_cli(["scripts/doctor.py", "--json", "--profile", "cpu"])
    assert completed.returncode in (0, 1), completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["profile"] == "cpu"
    assert payload["categories"] == ["A"]
    local = doctor.run_doctor(profile="cpu", root=ROOT)
    assert payload["counts"] == local.counts
    assert payload["exit_code"] == local.exit_code == completed.returncode


def test_json_matches_the_in_process_report(report):
    completed = run_cli(["-m", "scripts.doctor", "--profile", "all", "--json"])
    payload = json.loads(completed.stdout)
    assert payload["counts"] == report.counts
    assert [check["id"] for check in payload["checks"]] == [check.id for check in report.checks]


# ---------------------------------------------------------------------------
# Offline wheelhouse
# ---------------------------------------------------------------------------
def _pins():
    pins = []
    for path in sorted((ROOT / "dependencies" / "python").glob("*.txt")):
        found, _ = doctor.parse_requirements(path)
        pins.extend((name, version) for name, version, _raw in found)
    return pins


def _wheelhouse(directory, drop=None):
    written = []
    for name, version in _pins():
        if name == drop:
            continue
        path = directory / f"{name}-{version}-py3-none-any.whl"
        path.write_bytes(b"")
        written.append(path.name)
    return written


def test_wheelhouse_can_satisfy_the_pins(tmp_path):
    written = _wheelhouse(tmp_path)
    assert written
    report = doctor.run_doctor(profile="cpu", root=ROOT, wheelhouse=tmp_path)
    assert report.wheelhouse["counts"]["fail"] == 0
    assert report.wheelhouse["counts"]["pass"] == len(_pins())
    assert report.exit_code == 0
    assert {record["status"] for record in report.wheelhouse["checks"]} == {"pass"}


def test_wheelhouse_reports_a_missing_pin(tmp_path):
    pins = _pins()
    dropped = pins[0][0]
    _wheelhouse(tmp_path, drop=dropped)
    report = doctor.run_doctor(profile="cpu", root=ROOT, wheelhouse=tmp_path)
    assert report.wheelhouse["counts"]["fail"] == 1
    assert report.exit_code == 1
    failing = [record for record in report.wheelhouse["checks"] if record["status"] == "fail"]
    assert failing and dropped.replace("_", "-") in failing[0]["id"].replace("_", "-")


def test_wheelhouse_requires_an_existing_directory(tmp_path):
    missing = tmp_path / "not-there"
    report = doctor.run_doctor(profile="cpu", root=ROOT, wheelhouse=missing)
    assert "error" in report.wheelhouse
    assert report.wheelhouse["checks"] == []


# ---------------------------------------------------------------------------
# The installer
# ---------------------------------------------------------------------------
def test_installer_passes_a_shell_syntax_check():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not on PATH on this host")
    completed = subprocess.run([bash, "-n", str(ROOT / "scripts" / "install_deps.sh")],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120)
    assert completed.returncode == 0, completed.stderr
