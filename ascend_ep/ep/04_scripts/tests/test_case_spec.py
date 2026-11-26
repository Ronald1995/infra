"""CaseSpec contract: one definition per case, and an EP-size invariant matrix.

These tests protect the two properties the refactor exists for:

1. a case is defined in exactly one place, and every layer reads that definition;
2. EP8, EP64 and EP128 expand the *same* parameter matrix, so a difference in task
   count between EP sizes can only ever be a recorded status, never a missing test.
"""
from collections import defaultdict
from unittest.mock import patch

import pytest

from common import case_spec
from common.case_spec import (APPLICABILITY_RULES, CORRECTNESS_TOKENS, FIXED_OWNER,
                              L0_PARAMETERS, TRANSFORMS, catalog, canonical_cases,
                              case_counts, handler_for, owned_by, variants_for)
from common.plan import expand as expand_plan, load_plan
from common.registry import ROOT
from common.scheduler import check_parity, make_plan
from common.suites import select, suite

PLAN_PATH = ROOT / "configs/benchmark_plan.yaml"

OVERRIDE = {
    "schema_version": 1,
    "device": {"soc": "Ascend950", "cann_version": "test"},
    "channels": [
        {"case_id": "LINK-001", "channel": "AivUbMemChannel", "execution_resource": "AIV",
         "mechanisms": ["ub_memory"], "supported": True, "confidence": "probed",
         "reason": None, "evidence": ["bridge"]},
        {"case_id": "LINK-013", "channel": "shmem:UDMA", "execution_resource": "AIV",
         "mechanisms": ["shmem_udma"], "supported": True, "confidence": "probed",
         "reason": None, "evidence": ["shmem"]},
    ],
    "engines": [],
    "accelerators": [
        {"mode": "default", "execution_resource": None, "supported": True, "reason": None},
        {"mode": "aiv", "execution_resource": "AIV", "supported": True, "reason": None},
    ],
    "probes": [{"name": "override", "ok": True, "reason": None, "detail": {}}],
    "message_limits": {},
}


@pytest.fixture(scope="module")
def plan():
    return load_plan(PLAN_PATH)


def expand(plan_obj, **kwargs):
    from baselines.l0.topology import capabilities

    matrix = capabilities.discover({}, override=OVERRIDE)
    with patch("baselines.l0.topology.discovery.load", return_value=matrix):
        return expand_plan(plan_obj, **kwargs)


# ----------------------------------------------------------------- catalog ---

def test_the_catalog_is_the_only_case_table():
    specs = catalog()
    assert len(specs) == 83
    assert case_counts()["canonical"] == len(canonical_cases())
    for case_id, spec in specs.items():
        assert spec.id == case_id
        assert spec.handler, case_id
        assert set(spec.correctness) <= set(CORRECTNESS_TOKENS), case_id
        assert set(spec.applicability) <= set(APPLICABILITY_RULES), case_id
        assert spec.transform is None or spec.transform in TRANSFORMS
        assert variants_for(case_id) == spec.variants


def test_every_layer_reads_the_spec_instead_of_a_case_id():
    """No module may keep its own case-ID table."""
    import inspect

    import common.scheduler as scheduler
    import common.variants as variants

    for module in (scheduler, variants):
        source = inspect.getsource(module)
        for case_id in catalog():
            # A case ID inside a comment or a docstring is fine; a comparison is not.
            for pattern in (f'== "{case_id}"', f'!= "{case_id}"', f'"{case_id}" in',
                            f'in ("{case_id}"'):
                assert pattern not in source, f"{module.__name__} special-cases {case_id}"


def test_handler_coverage():
    from benchmarks.cases import known_handlers

    handlers = known_handlers()
    missing = sorted({spec.handler for spec in catalog().values()} - set(handlers))
    assert missing == []


def test_variant_owners_are_declared_once():
    assert variants_for("FUSE-005") == ("chunks_2", "chunks_4", "chunks_8")
    assert variants_for("FUSE-006") == ("tile_128_256_256", "tile_256_128_256",
                                        "tile_256_256_256")
    assert variants_for("FUSE-007") == ("copy_cores_1", "copy_cores_2", "copy_cores_4",
                                        "copy_cores_8")
    assert variants_for("LAYER-003") == ("fresh_plan", "cached_plan")
    assert variants_for("COMM-001") == ("default",)
    # The legacy collective case names the operation it actually runs.
    assert variants_for("HW-006") == ("allgather", "reducescatter")


def test_l0_variants_and_parameters_come_from_the_catalog():
    for case_id in ("LINK-001", "LINK-008", "LINK-012", "LINK-013", "EXEC-004"):
        assert case_spec.spec(case_id).requested() == L0_PARAMETERS[case_id]
    assert case_spec.spec("LINK-013").variants == ("device", "host")
    assert case_spec.spec("HW-008").variants == ("mte", "udma", "urma")


def test_fixed_owner_is_declared_on_the_case():
    assert owned_by("LAYER-003") == "alltoall_hccl"
    assert owned_by("LAYER-005") == "alltoall_hccl"
    assert owned_by("LAYER-006") == "alltoall_hccl"
    assert owned_by("COMM-001") is None
    assert FIXED_OWNER == "fixed_owner"


def test_applicability_rules_are_named_not_positional():
    spec = case_spec.spec("MOE-007")
    assert "concurrent_needs_two_ranks" in spec.applicability
    assert case_spec.spec("COMM-010").applicability[0] == "dedup_mode"
    assert "training_phase" in case_spec.spec("MOE-002").applicability


# ------------------------------------------------------- matrix invariance ---

def test_every_ep_size_expands_the_same_point_set(plan):
    run = expand(plan)
    counts = defaultdict(dict)
    for task in run.tasks:
        signature = (task["case_id"], task["level"], task["backend_id"],
                     task["parameters"].get("variant"))
        counts[task["ep_size"]][signature] = counts[task["ep_size"]].get(signature, 0) + 1
    reference = counts[run.plan.ep_sizes[0]]
    for size in run.plan.ep_sizes[1:]:
        assert counts[size] == reference, f"EP{size} expands a different point set"


def test_check_parity_rejects_a_collapsed_variant():
    tasks = [
        {"task_id": "a", "case_id": "FUSE-005", "level": "L2", "backend_id": "op",
         "ep_size": 8, "parameters": {"variant": "chunks_2"}},
        {"task_id": "b", "case_id": "FUSE-005", "level": "L2", "backend_id": "op",
         "ep_size": 8, "parameters": {"variant": "chunks_4"}},
        {"task_id": "c", "case_id": "FUSE-005", "level": "L2", "backend_id": "op",
         "ep_size": 64, "parameters": {"variant": "default"}},
        {"task_id": "d", "case_id": "FUSE-005", "level": "L2", "backend_id": "op",
         "ep_size": 64, "parameters": {"variant": "default"}, "duplicate": True},
    ]
    with pytest.raises(ValueError, match="not EP-size invariant"):
        check_parity(tasks)


def test_unsupported_points_keep_their_full_variant_expansion(plan):
    """A capability verdict may not collapse the parameter matrix."""
    run = expand(plan, ep_sizes=[8, 64], suites=["all"])
    variants = defaultdict(set)
    for task in run.tasks:
        if task["case_id"] != "FUSE-005":
            continue
        variants[task["ep_size"]].add(task["parameters"].get("variant"))
    assert variants[8] == variants[64] == {"chunks_2", "chunks_4", "chunks_8"}


def test_a_non_executable_point_is_still_a_task(plan):
    run = expand(plan, ep_sizes=[8], suites=["all"])
    statuses = {task["status"] for task in run.tasks}
    assert "unsupported" in statuses
    assert all(task["reason"] for task in run.tasks if task["status"] == "unsupported")
    assert all(task["status"] in {"not_run", "unsupported", "not_applicable",
                                 "not_implemented", "dependency_missing",
                                 "environment_unavailable"}
               for task in run.tasks)


def test_l0_is_not_multiplied_by_operators(plan):
    run = expand(plan, ep_sizes=[8], suites=["hardware", "memory", "transport",
                                             "collectives", "overlap"])
    l0 = [task for task in run.tasks if task["level"] == "L0"]
    assert l0
    assert {task["backend_id"] for task in l0} == {"hardware"}


def test_case_transform_is_declared_not_hardcoded():
    from common.scheduler import prepare_case

    config = {"workload": {"routing": "balanced", "tokens_per_rank": [64, 64]}}
    imbalanced = prepare_case(config, case_spec.spec("MOE-005"))
    assert imbalanced["workload"]["routing"] == "expert_hotspot"
    # The input must not be mutated: one configuration expands many cases.
    assert config["workload"]["routing"] == "balanced"
    dynamic = prepare_case(config, case_spec.spec("MOE-009"))
    assert dynamic["workload"]["token_steps"] == [[64, 64], [32, 32]]


def test_manifest_tasks_record_the_handler_and_correctness(plan):
    tasks = make_plan([expand(plan, ep_sizes=[8], suites=["decode"]).points[0].config],
                     operators=["alltoall_hccl"], cases=["COMM-001"])
    task = next(item for item in tasks if item["case_id"] == "COMM-001")
    assert task["handler"] == "l1.dispatch"
    assert task["correctness_requirements"] == ["dispatch_rows", "dispatch_counts"]
    assert task["case_phase"] == "inference"


def test_suites_still_only_select(plan):
    for suite_id in ("decode", "prefill", "overlap", "dedup", "quantized"):
        selected = suite(suite_id).case_ids_for()
        assert selected
        assert set(selected) <= set(catalog())
    assert set(select(["all"])) == set(canonical_cases())
