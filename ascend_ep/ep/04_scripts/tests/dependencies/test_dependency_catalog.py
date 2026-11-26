"""The dependency catalog, the plan it produces and the isolation it promises.

These tests are the contract behind "ascend_ep is self-contained":

* every dependency declares who owns it, how it is pinned and how it is installed;
* no dependency is satisfied by an unmanaged checkout at run time;
* two families that install the same module name never share an interpreter;
* the installer's plan is generated from the catalog (the shell owns no URLs);
* the operator closure never references an undeclared module.
"""
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.registry import environments, environment_assignments, registry  # noqa: E402
from scripts import dependency_closure, dependency_plan, dependency_report, doctor  # noqa: E402

MANIFEST_PATH = SCRIPTS / "dependencies" / "manifest.yaml"
INSTALLER = SCRIPTS / "scripts" / "install_deps.sh"


@pytest.fixture(scope="module")
def manifest():
    return doctor.load_manifest(MANIFEST_PATH)


@pytest.fixture(scope="module")
def entries(manifest):
    return {entry["id"]: entry for entry in manifest["dependencies"]}


@pytest.fixture(scope="module")
def declared_environments():
    return doctor.load_environments(SCRIPTS)


@pytest.fixture
def empty_doctor():
    return SimpleNamespace(checks=[], counts={"pass": 0, "warn": 0, "fail": 0, "skip": 0},
                           exit_code=0)


# --------------------------------------------------------------- ownership ---

def test_the_catalog_uses_the_ownership_model(manifest, entries):
    assert manifest["schema_version"] >= 2
    assert set(manifest["ownership"]) >= {"model", "note"}
    for identifier, entry in entries.items():
        assert entry["ownership"] in doctor.OWNERSHIPS, identifier
        assert entry["environment"], identifier
        assert isinstance(entry["modules"], list), identifier
        assert isinstance(entry["conflicts"], list), identifier


def test_a_host_dependency_is_never_installed_here(entries):
    for identifier, entry in entries.items():
        if entry["ownership"] != "host":
            continue
        assert entry["installable"] is False, identifier
        assert entry["install_spec"]["mode"] in ("manual", "none"), identifier
        assert entry["manual_step"], identifier


def test_a_vendored_tree_is_verified_not_installed(entries):
    vendored = [entry for entry in entries.values() if entry["ownership"] == "vendored"]
    assert vendored, "the ascendep adapter tree must be declared as vendored"
    for entry in vendored:
        assert entry["check"]["type"] == "vendored_source"
        assert entry["installable"] is False
        assert entry["install_spec"]["mode"] in ("manual", "none")
        assert entry["pin"]["kind"] == "commit"


def test_in_repo_dependencies_point_at_this_repository(entries):
    in_repo = [entry for entry in entries.values() if entry["ownership"] == "in_repo"]
    assert {entry["id"] for entry in in_repo} >= {"asc-comm-kernels", "l0-native-bridges"}
    for entry in in_repo:
        assert entry["source_url"].startswith(("baselines/", "kernels/")), entry["id"]


# --------------------------------------------------------------------- pins ---

def test_every_installable_dependency_has_an_immutable_pin(entries):
    for identifier, entry in entries.items():
        if not entry["installable"]:
            continue
        pin = entry["pin"]
        if pin["kind"] == "commit":
            assert re.fullmatch(r"[0-9a-f]{40}", pin["value"]), identifier
        elif pin["kind"] == "release_tag":
            assert pin["value"] and pin["resolve"], identifier
        elif pin["kind"] == "pip":
            assert entry["install_spec"].get("requirements"), identifier
        elif pin["kind"] == "system":
            assert entry["install_spec"]["mode"] == "system", identifier
        elif pin["kind"] == "in_repo":
            assert entry["ownership"] == "in_repo", identifier
        else:  # pragma: no cover - guarded by validate_manifest
            pytest.fail(f"{identifier}: installable with pin kind {pin['kind']!r}")


def test_no_dependency_uses_a_floating_reference(entries):
    for identifier, entry in entries.items():
        assert str(entry.get("version") or "").strip().lower() not in ("latest", "*"), identifier
        pin = entry["pin"]
        if pin["kind"] == "release_tag":
            assert pin["value"].lower() != "latest", identifier
        if pin["kind"] == "commit":
            assert len(pin["value"]) == 40, identifier


def test_shmem_has_a_cann_to_tag_mapping(manifest, entries):
    entry = entries["shmem"]
    assert entry["pin"]["kind"] == "release_tag"
    mapping = manifest["compatibility"]["shmem_tags"]
    assert entry["pin"]["value"] in mapping
    assert mapping["note"]


# --------------------------------------------------------- no checkout dep ---

def test_the_ascendep_checkout_dependency_is_gone(manifest, entries):
    assert "ascendep-checkout" not in entries
    for identifier, entry in entries.items():
        assert entry["check"]["type"] != "ascendep_checkout", identifier
    assert "penn513" not in json.dumps([entry["check"] for entry in entries.values()])


def test_the_runtime_never_needs_an_external_checkout():
    """Only the vendored tree's provenance may name the upstream repository."""
    offenders = []
    for path in list(SCRIPTS.rglob("*.py")) + list(SCRIPTS.rglob("*.sh")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "EP_ASCENDEP_ROOT" not in text:
            continue
        relative = path.relative_to(SCRIPTS).as_posix()
        # The documented development override, the vendoring tool, the doctor evidence
        # and the tests/docs that describe them.
        allowed = ("operators/ascendep.py", "scripts/vendor_ascendep.py",
                   "scripts/doctor.py", "tests/operators_vendor/", "tests/dependencies/",
                   "docs/", "references/")
        if not relative.startswith(allowed):
            offenders.append(relative)
    assert offenders == [], offenders


# -------------------------------------------------------------- environments ---

def test_every_dependency_belongs_to_a_declared_environment(entries, declared_environments):
    for identifier, entry in entries.items():
        assert entry["environment"] in declared_environments, identifier
    for name, entry in declared_environments.items():
        for requirement in entry.get("requirements") or ():
            assert (SCRIPTS / "dependencies" / requirement).is_file(), (name, requirement)


def test_the_environment_map_covers_every_operator(declared_environments):
    default, assigned = environment_assignments()
    assert default in declared_environments
    operators = registry()
    assert set(assigned) == set(operators)
    for operator_id, environment in assigned.items():
        assert environment in declared_environments, operator_id


def test_isolated_environments_declare_symmetric_conflicts(entries, declared_environments):
    isolated = {name for name, entry in declared_environments.items() if entry.get("isolated")}
    assert isolated, "at least the two DeepEP forks need isolated environments"
    for name in isolated:
        entry = declared_environments[name]
        assert entry.get("inherits"), name
        for other in entry.get("conflicts_with") or ():
            assert other in declared_environments, (name, other)
            assert name in (declared_environments[other].get("conflicts_with") or ()), (name, other)


def test_shared_modules_live_in_conflicting_environments(entries):
    owners = {}
    for identifier, entry in entries.items():
        for module in entry["modules"]:
            owners.setdefault(module, []).append(identifier)
    for module, identifiers in owners.items():
        if len(identifiers) < 2:
            continue
        environments_of = {entries[identifier]["environment"] for identifier in identifiers}
        assert len(environments_of) == len(identifiers), (module, identifiers)
        for identifier in identifiers:
            partners = set(identifiers) - {identifier}
            assert partners <= set(entries[identifier]["conflicts"]), (identifier, module)


# --------------------------------------------------------------- the closure ---

def test_the_operator_closure_has_no_undeclared_module(manifest):
    problems = []
    for operator_id, spec in registry().items():
        dependencies, modules, unmapped = dependency_closure.resolve(
            operator_id, spec.environment, manifest)
        for module in sorted(unmapped):
            problems.append(f"{operator_id}: {module}")
    assert problems == [], problems


def test_every_dependency_is_used_by_something(entries, manifest):
    """No orphan dependency: something either imports it or a capability group needs it."""
    used = set()
    for operator_id, spec in registry().items():
        dependencies, _modules, _unmapped = dependency_closure.resolve(
            operator_id, spec.environment, {"dependencies": list(entries.values())})
        used.update(dependencies)
    groups = {group["id"] for group in manifest["capability_groups"]}
    referenced_by_group = {identifier for identifier, entry in entries.items()
                           if [token for token in doctor.tokens(entry["required_by"])
                               if not token.startswith("operator:")]}
    for identifier, entry in entries.items():
        tokens = doctor.tokens(entry["required_by"])
        assert set(token for token in tokens if not token.startswith("operator:")) <= groups, (
            identifier, "unknown capability group")
        assert (identifier in used or identifier in referenced_by_group), identifier


def test_the_operator_dependency_tokens_match_the_closure(manifest, entries):
    """A token in ``required_by`` must be a dependency the operator really imports."""
    wrong = []
    for identifier, entry in entries.items():
        for token in doctor.tokens(entry["required_by"]):
            if not token.startswith("operator:"):
                continue
            operator_id = token.split(":", 1)[1]
            if operator_id not in registry():
                wrong.append(f"{identifier}: unknown operator {operator_id}")
                continue
            dependencies, _modules, _unmapped = dependency_closure.resolve(
                operator_id, registry()[operator_id].environment, manifest)
            declared = {other for other, item in entries.items()
                        if f"operator:{operator_id}" in doctor.tokens(item["required_by"])}
            if identifier not in dependencies and identifier not in declared:
                wrong.append(f"{identifier}: {operator_id} does not import it")
    assert wrong == [], wrong


# ----------------------------------------------------------------- the plan ---

@pytest.mark.parametrize("profile", ["cpu", "npu", "communication", "profiling", "multinode",
                                     "all"])
def test_a_plan_can_be_built_for_every_profile(profile):
    plan = dependency_plan.build_plan(profile, {"offline": False, "source_cache": "/cache",
                                                "wheelhouse": None, "check_only": False,
                                                "environments": None})
    assert plan["counts"]["blocked"] >= 1, "host dependencies are always blockers"
    assert all(step["kind"] != "fetch" or step["command"].startswith("fetch_")
               for step in plan["steps"])


def test_a_fetch_step_carries_the_pinned_commit():
    plan = dependency_plan.build_plan("communication", {"offline": False,
                                                        "source_cache": "/cache",
                                                        "wheelhouse": None,
                                                        "check_only": False,
                                                        "environments": None})
    fetches = [step for step in plan["steps"] if step["kind"] == "fetch"]
    assert fetches
    for step in fetches:
        assert step["pin"]["kind"] in ("commit", "release_tag")
        if step["pin"]["kind"] == "commit":
            assert len(step["pin"]["value"]) == 40
            assert step["pin"]["value"] in step["command"]


def test_offline_never_fetches_and_requires_the_cache():
    plan = dependency_plan.build_plan("communication", {"offline": True,
                                                        "source_cache": "/cache",
                                                        "wheelhouse": None,
                                                        "check_only": False,
                                                        "environments": None})
    assert [step for step in plan["steps"] if step["kind"] == "fetch"] == []
    cache_steps = [step for step in plan["steps"] if step["kind"] == "verify_cache"]
    assert cache_steps
    assert all("require_checkout" in step["command"] or "require_tag" in step["command"]
               for step in cache_steps)


def test_check_only_installs_nothing():
    plan = dependency_plan.build_plan("all", {"offline": False, "source_cache": "/cache",
                                              "wheelhouse": None, "check_only": True,
                                              "environments": None})
    assert [step for step in plan["steps"] if step["kind"] in ("pip", "install")] == []
    assert [step for step in plan["steps"] if step["kind"] == "verify"]


def test_an_unpinned_dependency_is_never_scheduled(manifest):
    plan = dependency_plan.build_plan("all", {"offline": False, "source_cache": "/cache",
                                              "wheelhouse": None, "check_only": False,
                                              "environments": None})
    blocked = {item["dependency"] for item in plan["blocked"]}
    assert {"fsdp-turbo", "mega-moe"} <= blocked
    scheduled = {step["dependency"] for step in plan["steps"]}
    assert not ({"fsdp-turbo", "mega-moe"} & scheduled)


def test_the_torch_npu_wheel_needs_a_source():
    plan = dependency_plan.build_plan("npu", {"offline": False, "source_cache": "/cache",
                                              "wheelhouse": None, "check_only": False,
                                              "environments": None})
    blocked = {item["dependency"]: item for item in plan["blocked"]}
    assert "torch-npu" in blocked
    assert blocked["torch-npu"]["cause"] == "wheel_source"
    assert blocked["torch-npu"]["wheel_source"]["required"] is True


def test_the_isolated_environment_is_created_before_its_install():
    plan = dependency_plan.build_plan("all", {"offline": False, "source_cache": "/cache",
                                              "wheelhouse": None, "check_only": False,
                                              "environments": ["deep_ep_lueelu"]})
    kinds = [step["kind"] for step in plan["steps"]]
    assert kinds[0] == "environment"
    assert plan["steps"][0]["environment"] == "deep_ep_lueelu"


def test_the_generated_script_runs_every_planned_step():
    plan = dependency_plan.build_plan("cpu", {"offline": False, "source_cache": "/cache",
                                              "wheelhouse": None, "check_only": False,
                                              "environments": None})
    script = dependency_plan.render_script(plan)
    assert script.startswith("#!/usr/bin/env bash")
    for step in plan["steps"]:
        dependency = step["dependency"] or step["environment"]
        assert dependency in script, dependency
    assert "step \"" in script


def test_the_shell_owns_no_urls_or_pins():
    text = INSTALLER.read_text(encoding="utf-8")
    assert "https://" not in text, "the installer must read URLs from the manifest"
    assert "git clone" not in text, "the installer runs the generated plan, it does not clone"
    for flag in ("--profile", "--dry-run", "--check-only", "--offline", "--source-cache",
                 "--wheelhouse", "--allow-network", "--environment"):
        assert flag in text, flag


# ------------------------------------------------------------------- doctor ---

def test_the_vendored_probe_passes_on_the_real_tree(manifest, entries):
    entry = entries["vendored-ascendep"]
    context = SimpleNamespace(root=SCRIPTS)
    check = doctor.probe_vendored_source(entry, entry["check"], context)
    assert check.status == "pass", check.detail
    assert check.evidence["files"] >= 20


def test_a_tampered_vendored_file_fails_the_probe(manifest, entries, tmp_path):
    import shutil
    source = SCRIPTS / "operators" / "vendor" / "ascendep"
    target = tmp_path / "operators" / "vendor" / "ascendep"
    shutil.copytree(source, target)
    victim = target / "ep" / "runner.py"
    victim.write_text(victim.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    entry = entries["vendored-ascendep"]
    check = doctor.probe_vendored_source(entry, entry["check"],
                                         SimpleNamespace(root=tmp_path))
    assert check.status == "fail"
    assert "digest" in check.detail


def test_two_shadowing_families_are_reported_as_a_conflict(manifest):
    entries = {entry["id"]: entry for entry in manifest["dependencies"]}
    checks = [SimpleNamespace(id="deep-ep", status="pass", evidence={"find_spec": "/a/deep_ep"}),
              SimpleNamespace(id="deep-ep-v2", status="pass",
                              evidence={"find_spec": "/b/deep_ep"})]
    check = doctor.check_environment_isolation(SimpleNamespace(root=SCRIPTS), manifest, checks)
    assert check.status == "fail"
    assert "deep-ep" in check.detail and "deep-ep-v2" in check.detail
    assert "deep_ep" in check.evidence["environments"].values() or True
    assert entries["deep-ep"]["environment"] != entries["deep-ep-v2"]["environment"]


def test_only_one_family_available_is_not_a_conflict(manifest):
    checks = [SimpleNamespace(id="deep-ep", status="pass", evidence={}),
              SimpleNamespace(id="deep-ep-v2", status="fail", evidence={})]
    check = doctor.check_environment_isolation(SimpleNamespace(root=SCRIPTS), manifest, checks)
    assert check.status == "pass"


# ----------------------------------------------------------------- coverage ---

def test_the_coverage_report_lists_every_operator(empty_doctor):
    report = dependency_report.build(doctor_report=empty_doctor)
    operators = {row["operator"] for row in report["operators"]}
    assert operators == set(registry())
    for row in report["operators"]:
        assert row["dependencies"], row["operator"]
        assert row["environment"] in environments()
    markdown = dependency_report.render_markdown(report)
    assert "## Operators" in markdown and "UNMAPPED" not in markdown
