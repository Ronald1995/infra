"""Benchmark matrix tests: EP sizes, suites, expansion, capability and reports.

No NPU is used.  The L0 capability probe is replaced by an explicit override
fixture, so every assertion is about *planning* behaviour.  Report tests build
synthetic result rows only to exercise the aggregation logic; they are explicitly
not performance measurements.
"""
from copy import deepcopy
from collections import defaultdict
import json
from unittest.mock import patch

import pytest

from baselines.l0.topology import capabilities
from baselines.l0.transport import peers
from common.cases import (alias_map, canonical_cases, case_counts, catalog, l0_family)
from common.config import DEFAULT, LEVELS, load_configs, merge, validate
from common.controller import execute, results
from common.plan import expand, load_plan, matrix_metadata
from common.registry import ROOT, registry
from common.scheduler import check_completeness, make_plan
from common.storage import fingerprint
from common.suites import (ALL_SUITES, BUSINESS_SUITES, SUITES, select, suite)
from reporting import coverage
from schemas import SEVERITY, result
from run_benchmark import main

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
        {"case_id": "LINK-003", "channel": None, "execution_resource": "AIV",
         "mechanisms": ["ub_ctp"], "supported": False, "confidence": "probed",
         "reason": "channel_not_reported", "evidence": ["bridge"]},
    ],
    "engines": [],
    "accelerators": [
        {"mode": "default", "execution_resource": None, "supported": True, "reason": None},
        {"mode": "aiv", "execution_resource": "AIV", "supported": True, "reason": None},
    ],
    "probes": [{"name": "override", "ok": True, "reason": None, "detail": {}}],
    "message_limits": {},
}

MATRIX = capabilities.discover({}, override=OVERRIDE)


@pytest.fixture(scope="module")
def plan():
    return load_plan(PLAN_PATH)


def expand_plan(plan, **kwargs):
    with patch("baselines.l0.topology.discovery.load", return_value=MATRIX):
        return expand(plan, **kwargs)


# ---------------------------------------------------------------- catalog ---

def test_catalog_is_the_single_source_of_truth():
    cases = catalog()
    counts = case_counts()
    assert counts["total"] == len(cases) == 83
    assert counts["canonical"] == len(canonical_cases())
    assert sum(counts[level] for level in LEVELS) == counts["total"]
    for case in cases.values():
        assert case.category in ("memory", "transport", "execution", "collective",
                                 "compute", "overlap", "operator")
        assert case.level in LEVELS
    # The legacy L0 identifiers stay in the catalog as aliases.
    assert alias_map() == {"HW-007": "LINK-012", "HW-008": "LINK-013",
                           "HW-009": "MEM-001", "HW-012": "OVERLAP-001"}
    assert all(cases[index].alias_of == target for index, target in alias_map().items())
    assert l0_family(cases["OVERLAP-009"]) == "overlap"


# ------------------------------------------------------- EP size coverage ---

def test_every_ep_size_covers_the_same_complete_catalog(plan):
    run = expand_plan(plan)
    canonical = set(canonical_cases())
    assert len(run.coverage) == len(plan.ep_sizes) == 3
    for size in plan.ep_sizes:
        assert set(run.coverage[size]) == canonical
    assert len({frozenset(items) for items in run.coverage.values()}) == 1


def test_missing_ep_configuration_is_reported(plan):
    with pytest.raises(ValueError, match="EP16"):
        expand_plan(plan, ep_sizes=[16])


def test_manifest_metadata_records_the_matrix(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["decode"])
    metadata = run.metadata()
    assert metadata["ep_sizes"] == [8]
    assert metadata["levels"] == list(LEVELS)
    assert metadata["suites"] == ["decode"]
    assert metadata["case_counts"]["total"] == 83
    assert set(metadata["case_aliases"]) == set(alias_map())
    assert metadata["case_index"]["HW-007"]["status"] == "alias"
    point = metadata["config_points"][0]
    assert len(point["sources"]) >= 3
    assert point["fingerprint"]


def test_plan_integrity_rejects_a_missing_case(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["hardware"])
    with pytest.raises(ValueError, match="incomplete"):
        check_completeness(run.tasks, [8], expected=set(canonical_cases()))


def test_multi_yaml_layering_keeps_defaults_and_overrides_scalars(plan):
    run = expand_plan(plan, ep_sizes=[64], suites=["decode"])
    config = run.points[0].config
    # EP layer: environment comes from ep64, not from the decode suite file.
    assert config["distributed"]["ep_size"] == 64
    assert config["model"]["num_experts"] == 384
    # Suite layer: workload comes from decode.yaml.
    assert config["workload"]["tokens_per_rank"] == 16
    # Base layer: shared benchmark dimensions survive the composition.
    assert config["benchmark"]["message_bytes"] == [4096, 65536, 1048576, 16777216,
                                                    268435456, 1073741824]
    assert config["benchmark"]["levels"] == list(LEVELS)
    # Arrays are replaced, never concatenated, and every task keeps its source.
    task = next(task for task in run.tasks if task["case_id"] == "COMM-001")
    assert task["config_source"] and any("ep64_bf16.yaml" in item for item in task["config_source"])
    assert task["config_fingerprint"]


# ------------------------------------------------------------ L0 expansion ---

def test_l0_is_not_multiplied_by_the_operator_dimension(plan):
    one = expand_plan(plan, ep_sizes=[8], suites=["all"], operators=["deepep_ascend"])
    every = expand_plan(plan, ep_sizes=[8], suites=["all"], operators="all")
    low_one = {task["task_id"] for task in one.tasks if task["level"] == "L0"}
    low_every = {task["task_id"] for task in every.tasks if task["level"] == "L0"}
    assert low_one and low_one == low_every
    assert all(task["backend_id"] == "hardware" for task in one.tasks if task["level"] == "L0")


def test_l0_cases_exist_for_every_ep_size(plan):
    run = expand_plan(plan, ep_sizes=[8, 64, 128], suites=["all"])
    canonical_l0 = {case_id for case_id, case in canonical_cases().items() if case.level == "L0"}
    for size in (8, 64, 128):
        observed = {task["case_id"] for task in run.tasks
                    if task["level"] == "L0" and task["ep_size"] == size}
        assert observed == canonical_l0


def test_transport_fixed_and_all_peer_variants(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["transport"])
    link1 = [task for task in run.tasks if task["case_id"] == "LINK-001"]
    assert {task["parameters"]["peer_count"] for task in link1} == {1, 2, 4, 0}
    assert {task["parameters"]["peer_policy"] for task in link1} == {
        "ring", "same_server", "cross_server"}
    all_peers = [task for task in link1 if task["parameters"]["peer_count"] == 0]
    assert all(task["ep_size"] == 8 for task in all_peers)
    # SHMEM exchanges with the whole group, so it is not multiplied by peer policy.
    link12 = [task for task in run.tasks if task["case_id"] == "LINK-012"]
    assert {task["parameters"]["peer_policy"] for task in link12} == {"group"}


def test_execution_cases_use_the_configured_ep_group(plan):
    run = expand_plan(plan, ep_sizes=[64], suites=["collectives"])
    exec_tasks = [task for task in run.tasks if task["case_id"].startswith("EXEC-")]
    assert exec_tasks
    assert {task["operation"] for task in exec_tasks} == {
        "alltoall", "alltoallv", "allgather", "reducescatter", "allreduce", "p2p"}
    assert all(task["config"]["distributed"]["ep_size"] == 64 for task in exec_tasks)
    assert {task["parameters"]["accelerator"] for task in exec_tasks} == {
        "default", "aicpu_ts", "aiv", "aiv_only", "ccu_sched"}


def test_overlap_suite_contains_every_overlap_case(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["overlap"])
    cases = {task["case_id"] for task in run.tasks}
    assert {f"OVERLAP-{number:03}" for number in range(1, 10)} <= cases
    assert {f"FUSE-{number:03}" for number in range(1, 10)} <= cases
    overlap_tasks = [task for task in run.tasks if task["case_id"].startswith("OVERLAP-")]
    assert {task["parameters"]["message_bytes"] for task in overlap_tasks} == {
        4096, 1048576, 16777216}
    assert all(task["parameters"]["compute"] for task in overlap_tasks)


def test_legacy_aliases_never_duplicate_a_task(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["all"])
    assert not {task["case_id"] for task in run.tasks} & set(alias_map())
    explicit = expand_plan(plan, ep_sizes=[8], suites=["all"], cases=["HW-007"])
    assert explicit.tasks
    assert {task["case_id"] for task in explicit.tasks} == {"HW-007"}
    assert all(task["alias_of"] == "LINK-012" for task in explicit.tasks)
    # The legacy ID is checked against its canonical capability (LINK-013 is in
    # the override matrix), so it is not reported as unsupported by accident.
    legacy = expand_plan(plan, ep_sizes=[8], suites=["all"], cases=["HW-008"])
    assert legacy.tasks and all(task["status"] == "not_run" for task in legacy.tasks)


def test_l0_parameter_sets_are_consistent_across_ep_sizes(plan):
    run = expand_plan(plan, ep_sizes=[8, 64, 128], suites=["all"])
    sizes = {}
    for task in run.tasks:
        if task["level"] != "L0":
            continue
        sizes.setdefault((task["case_id"], task["parameters"].get("variant")), {}).setdefault(
            task["ep_size"], set()).add(task["parameters"].get("message_bytes"))
    assert sizes
    for key, per_ep in sizes.items():
        values = {frozenset(items) for items in per_ep.values()}
        assert len(values) == 1, key


# --------------------------------------------------------- L1-L4 expansion ---

def test_every_registered_operator_is_checked_for_an_operator_case(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["high_throughput"], cases=["COMM-001"])
    backends = {task["backend_id"] for task in run.tasks if task["case_id"] == "COMM-001"}
    assert backends == set(registry())


def test_unsupported_points_always_carry_a_reason(plan):
    run = expand_plan(plan, ep_sizes=[8, 64, 128], suites=["all"])
    for task in run.tasks:
        if task["status"] not in {"not_run", "passed"}:
            assert task["reason"], (task["case_id"], task["backend_id"], task["status"])
        assert task["status"] in set(SEVERITY) | {"not_implemented"}


@pytest.mark.parametrize("case_id", ["COMM-002", "COMM-004", "COMM-005", "COMM-006", "COMM-011",
                                     "MOE-002", "MOE-004", "MOE-008", "MOE-009", "LAYER-007",
                                     "LAYER-002", "LAYER-009"])
def test_required_cases_are_not_dropped(plan, case_id):
    run = expand_plan(plan, ep_sizes=[8], suites=["all"], cases=[case_id])
    assert case_id in {task["case_id"] for task in run.tasks}


def test_operator_capability_is_checked_per_case(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["quantized"])
    quantized = [task for task in run.tasks if task["case_id"] == "COMM-009"]
    unsupported = [task for task in quantized if task["status"] == "unsupported"]
    assert unsupported, "quantized dispatch must not be silently considered supported"
    compatible = {task["backend_id"] for task in quantized
                  if task["status"] in {"not_run", "passed"}}
    for backend in compatible:
        assert "quantized_dispatch" in registry()[backend].capabilities


# ----------------------------------------------------------------- suites ---

def test_all_suites_have_a_valid_mapping():
    known = set(catalog())
    for suite_id in ALL_SUITES:
        selection = SUITES[suite_id].case_ids_for()
        assert selection, suite_id
        assert set(selection) <= known, suite_id
        assert not set(selection) & set(alias_map()), suite_id


def test_l0_suites_filter_by_category_not_by_filename():
    assert {case_id for case_id in SUITES["hardware"].case_ids_for()}
    assert all(catalog()[case_id].category in {"collective", "compute"}
               for case_id in SUITES["hardware"].case_ids_for())
    assert all(catalog()[case_id].category == "memory"
               for case_id in SUITES["memory"].case_ids_for())
    assert all(catalog()[case_id].category == "transport"
               for case_id in SUITES["transport"].case_ids_for())
    assert all(catalog()[case_id].category == "execution"
               for case_id in SUITES["collectives"].case_ids_for())
    assert set(SUITES["hardware"].case_ids_for()) == {
        "HW-001", "HW-002", "HW-003", "HW-004", "HW-005", "HW-006", "HW-010", "HW-011"}
    assert {f"OVERLAP-{number:03}" for number in range(1, 10)} <= set(
        SUITES["overlap"].case_ids_for())
    assert len(SUITES["transport"].case_ids_for()) == 13


def test_all_suite_selects_the_whole_catalog_and_full_is_an_alias():
    assert set(select(["all"])) == set(canonical_cases())
    assert set(select(["full"])) == set(canonical_cases())
    assert suite("full").id == "all"


def test_business_suites_share_and_differ_as_intended():
    covered = set()
    for suite_id in BUSINESS_SUITES:
        covered |= set(SUITES[suite_id].case_ids_for())
    assert covered <= set(canonical_cases())
    # One case may belong to several suites.
    assert "COMM-001" in set(SUITES["decode"].case_ids_for())
    assert "COMM-001" in set(SUITES["prefill"].case_ids_for())
    # Decode and prefill share the case set and differ only by workload policy.
    assert set(SUITES["decode"].case_ids_for()) == set(SUITES["prefill"].case_ids_for())
    assert set(SUITES["strong_scaling"].case_ids_for()) == set(
        SUITES["weak_scaling"].case_ids_for())


def test_same_case_in_different_suites_uses_different_parameters(plan):
    decode = expand_plan(plan, ep_sizes=[8], suites=["decode"], cases=["COMM-001"])
    prefill = expand_plan(plan, ep_sizes=[8], suites=["prefill"], cases=["COMM-001"])
    decode_tokens = {task["config"]["workload"]["tokens_per_rank"] for task in decode.tasks}
    prefill_tokens = {task["config"]["workload"]["tokens_per_rank"] for task in prefill.tasks}
    assert decode_tokens == {16}
    assert prefill_tokens == {4096}
    assert {task["task_id"] for task in decode.tasks} != {task["task_id"] for task in prefill.tasks}


def test_identical_points_do_not_duplicate_tasks(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["all", "all"])
    ids = [task["task_id"] for task in run.tasks]
    assert len(ids) == len(set(ids))
    identities = {(task["ep_size"], task["case_id"], task["backend_id"],
                   fingerprint(task["parameters"]),
                   task.get("config_fingerprint")) for task in run.tasks}
    assert len(identities) == len(run.tasks)


def test_dedup_suite_has_a_none_control_group(plan):
    run = expand_plan(plan, ep_sizes=[64], suites=["dedup"])
    modes = {task["config"]["dedup"]["mode"] for task in run.tasks}
    assert modes == {"none", "rank", "server"}
    control = [task for task in run.tasks
               if task["config"]["dedup"]["mode"] == "none" and task["case_id"] == "COMM-001"]
    assert control, "the no-dedup control must be measured, not only declared"
    none_dedup = [task for task in run.tasks
                  if task["config"]["dedup"]["mode"] == "none"
                  and task["case_id"] in {"COMM-010", "LAYER-005", "LAYER-006"}]
    assert none_dedup and all(task["status"] == "not_applicable" for task in none_dedup)
    assert all(task["reason"] for task in none_dedup)


def test_load_balance_suite_covers_planner_cases():
    assert {"COMM-012", "MOE-005", "LAYER-003", "LAYER-004", "LAYER-008"} <= set(
        SUITES["load_balance"].case_ids_for())


# ---------------------------------------------------------------- scaling ---

def test_strong_scaling_keeps_the_global_workload_constant(plan):
    run = expand_plan(plan, ep_sizes=[8, 64, 128], suites=["strong_scaling"])
    tokens = {}
    for size in (8, 64, 128):
        config = next(point.config for point in run.points if point.ep_size == size)
        tokens[size] = config["workload"]["tokens_per_rank"] * size
        assert config["model"]["num_experts"] == 256
    assert len(set(tokens.values())) == 1
    assert tokens[8] == 32768
    assert len({frozenset(task["case_id"] for task in run.tasks if task["ep_size"] == size)
                for size in (8, 64, 128)}) == 1


def test_weak_scaling_keeps_the_per_rank_workload_constant(plan):
    run = expand_plan(plan, ep_sizes=[8, 64, 128], suites=["weak_scaling"])
    for size in (8, 64, 128):
        config = next(point.config for point in run.points if point.ep_size == size)
        assert config["workload"]["tokens_per_rank"] == 4096
        assert config["model"]["num_experts"] == 4 * size


def test_topology_and_rank_mapping_match_the_ep_size(plan):
    run = expand_plan(plan, ep_sizes=[8, 64, 128], suites=["all"])
    for point in run.points:
        servers = point.config["topology"]["server_ids"]
        supernodes = point.config["topology"]["supernode_ids"]
        assert len(servers) == len(supernodes) == point.ep_size
        assert servers == [rank // 8 for rank in range(point.ep_size)]
        assert supernodes == [rank // 16 for rank in range(point.ep_size)]


def test_peer_counts_never_exceed_the_available_peers(plan):
    run = expand_plan(plan, ep_sizes=[8, 64, 128], suites=["all"])
    transport = [task for task in run.tasks if task["level"] == "L0"
                 and task["case_family"] == "transport"]
    assert transport
    for task in transport:
        peers = task["parameters"].get("peer_count")
        if peers:
            assert peers <= task["ep_size"] - 1


def test_peer_count_beyond_the_group_is_explicit():
    config = load_configs(ROOT / "configs/smoke_cpu.yaml")[0]
    config = merge(config, {"benchmark": {"peer_counts": [4], "outstanding": [1],
                                          "message_bytes": [4096]}})
    tasks = make_plan([config], cases=["LINK-001"])
    # EP2 has only one peer, so a four-peer sweep must be reported, not dropped.
    assert tasks
    assert all(task["status"] == "unsupported" for task in tasks)
    assert all("peers" in task["reason"] for task in tasks)


# ---------------------------------------------------------------- reports ---

def _row(plan, task, status="passed", **overrides):
    record = result({"run_id": "synthetic", "environment": {}}, task, status)
    record["correctness"] = {"passed": True} if status == "passed" else None
    record["latency"] = ({"unit": "ms", "scope": "per_iteration_rank_max", "source": "measured",
                          "p50": 1.0, "p95": 1.2, "p99": 1.3} if status == "passed" else None)
    record["bandwidth"] = ({"unit": "bytes", "scope": "per_rank", "payload_bytes": 4096,
                            "payload_bandwidth_GBps": 1.0} if status == "passed" else None)
    record["throughput"] = ({"global_source_tokens_per_second": 100.0}
                            if status == "passed" else None)
    record["source"] = "measured"
    record.update(overrides)
    return record


def _ep_tasks(plan, ep_sizes, suite_id="decode", case_id="COMM-001", backend=None):
    run = expand_plan(plan, ep_sizes=ep_sizes, suites=[suite_id], cases=[case_id])
    tasks = [task for task in run.tasks if task["case_id"] == case_id]
    chosen = {}
    for task in tasks:
        if backend and task["backend_id"] != backend:
            continue
        chosen.setdefault(task["ep_size"], task)
    return chosen


def test_coverage_matrix_separates_passed_and_unsupported(plan):
    tasks = _ep_tasks(plan, [8, 64, 128])
    rows = [_row(plan, tasks[8]),
            _row(plan, tasks[64], "unsupported", error_reason="no adapter path"),
            _row(plan, tasks[128], "dependency_missing", error_reason="missing package")]
    manifest = {"run_id": "synthetic", "environment": {},
                "matrix": matrix_metadata(list(tasks.values()))}
    matrix = {record["case_id"]: record for record in coverage.case_matrix(rows, manifest)}
    record = matrix["COMM-001"]
    assert record["8"] == "passed"
    assert record["64"] == "unsupported"
    assert record["128"] == "dependency_missing"
    assert record["reasons"]["64"] == "no adapter path"


def test_failed_case_is_not_mistaken_for_a_missing_task(plan):
    tasks = _ep_tasks(plan, [8])
    row = _row(plan, tasks[8], "runtime_failed", error_reason="rank worker crashed")
    manifest = {"run_id": "synthetic", "environment": {},
                "matrix": matrix_metadata([tasks[8]])}
    matrix = {record["case_id"]: record for record in coverage.case_matrix([row], manifest)}
    assert matrix["COMM-001"]["8"] == "runtime_failed"
    # A case outside the plan scope is reported as such, and an expected case
    # without a record is missing rather than silently dropped.
    assert matrix["MOE-001"]["8"] == "out_of_scope"
    empty = {record["case_id"]: record for record in coverage.case_matrix([], manifest)}
    assert empty["COMM-001"]["8"] == "missing"


def test_operator_coverage_does_not_count_unsupported_as_passed(plan):
    tasks = _ep_tasks(plan, [8])
    rows = [_row(plan, tasks[8], "unsupported", error_reason="no adapter path"),
            _row(plan, tasks[8], "passed"),
            _row(plan, tasks[8], "dependency_missing", error_reason="missing package")]
    manifest = {"run_id": "synthetic", "environment": {},
                "matrix": matrix_metadata([tasks[8]])}
    records = coverage.operator_matrix(rows, manifest)
    assert len(records) == 1
    entry = records[0]
    assert entry["tasks"] == 3 and entry["passed"] == 1
    assert entry["unsupported"] == 1 and entry["dependency_missing"] == 1
    cells = coverage.case_operator_matrix(rows)
    assert len(cells) == 1
    assert cells[0]["status"] == "passed" and cells[0]["tasks"] == 3


def test_case_status_prefers_a_pass_but_never_hides_a_failure(plan):
    tasks = _ep_tasks(plan, [8])
    passed = _row(plan, tasks[8], "passed")
    unsupported = _row(plan, tasks[8], "unsupported", error_reason="no adapter path")
    failed = _row(plan, tasks[8], "runtime_failed", error_reason="rank worker crashed")
    manifest = {"run_id": "synthetic", "environment": {},
                "matrix": matrix_metadata([tasks[8]])}
    record, = [item for item in coverage.case_matrix([passed, unsupported], manifest)
               if item["case_id"] == "COMM-001"]
    assert record["8"] == "passed"
    record, = [item for item in coverage.case_matrix([passed, failed], manifest)
               if item["case_id"] == "COMM-001"]
    assert record["8"] == "runtime_failed"


def test_peer_selection_is_topology_aware_and_deterministic():
    domains = [0] * 8 + [1] * 8
    ring = peers.select(16, 3, "ring", 4, domains)
    assert len(ring) == 4 and 3 not in ring
    assert ring == peers.select(16, 3, "ring", 4, domains)
    same = peers.select(16, 3, "same_server", 4, domains)
    assert all(rank < 8 for rank in same)
    cross = peers.select(16, 3, "cross_server", 4, domains)
    assert all(rank >= 8 for rank in cross)
    assert len(peers.select(16, 3, "ring", 0, domains)) == 15


def test_performance_comparison_respects_shape_and_timing_boundary(plan):
    tasks = _ep_tasks(plan, [8, 64])
    first = _row(plan, tasks[8])
    second = _row(plan, tasks[64])
    comparison = coverage.ep_size_comparison([first, second], {})
    assert comparison["groups"] and comparison["groups"][0]["comparable"]
    # A different parameter variant must not be compared with the original point.
    changed = deepcopy(second)
    changed["case_parameters"] = {**changed["case_parameters"], "message_bytes": 8192}
    assert coverage.ep_size_comparison([first, changed], {})["groups"] == []
    # Different per-rank workload is not comparable either.
    changed = deepcopy(second)
    changed["config"]["workload"]["tokens_per_rank"] = 17
    assert coverage.ep_size_comparison([first, changed], {})["groups"] == []


def test_l0_unverified_path_is_not_comparable():
    task = {
        "task_id": "LINK-001-x", "case_id": "LINK-001", "level": "L0",
        "operation": "put", "backend_id": "hardware",
        "config": merge(load_configs(ROOT / "configs/smoke_cpu.yaml")[0], {
            "device": "npu", "benchmark": {"timing": "output_ready"},
            "distributed": {"ep_size": 8}}),
        "parameters": {"variant": "default", "message_bytes": 4096},
        "source_versions": {}, "baseline": True,
    }
    base = {
        "schema_version": 1, "run_id": "synthetic", "status": "passed",
        "benchmark_level": "L0", "operation": "put", "backend_id": "hardware",
        "case_id": "LINK-001", "task_id": "LINK-001-x", "case_parameters": {"message_bytes": 4096},
        "config": task["config"], "correctness": {"passed": True}, "source": "measured",
        "latency": {"unit": "ms", "scope": "per_iteration_rank_max", "p50": 1.0},
        "bandwidth": {"unit": "bytes", "scope": "per_rank", "payload_bytes": 4096,
                      "payload_bandwidth_GBps": 0.1},
        "throughput": None, "verification": {"status": "unverified"}, "observed": {},
    }
    other = deepcopy(base)
    other["config"] = merge(base["config"], {"distributed": {"ep_size": 64}})
    report = coverage.ep_size_comparison([base, other], {})
    assert report["groups"]
    assert all(not group["comparable"] for group in report["groups"])
    assert all(group["reason"] for group in report["groups"])


def test_coverage_report_files_are_written(tmp_path, plan):
    tasks = _ep_tasks(plan, [8, 64])
    rows = [_row(plan, tasks[8]), _row(plan, tasks[64])]
    manifest = {"run_id": "synthetic", "environment": {},
                "matrix": matrix_metadata(list(tasks.values()))}
    destination = coverage.write(tmp_path, manifest, rows)
    for path in destination.values():
        assert path.is_file()
    text = (tmp_path / "reports/coverage_matrix.md").read_text(encoding="utf-8")
    assert "EP8" in text and "EP64" in text
    comparison = (tmp_path / "reports/ep_size_comparison.md").read_text(encoding="utf-8")
    assert "Only measurements" in comparison or "same case" in comparison


def test_multi_ep_manifest_keeps_per_ep_indexes(tmp_path):
    from common.config import DEFAULT, validate
    from common.controller import create_manifest

    configs = []
    for ep_size in (2, 4):
        configs.append(validate(merge(DEFAULT, {
            "device": "cpu",
            "model": {"dtype": "float32", "num_experts": 8, "top_k": 2,
                      "hidden_size": 16, "intermediate_size": 24},
            "distributed": {"ep_size": ep_size},
            "workload": {"tokens_per_rank": 8},
        })))
    tasks = make_plan(configs, cases=["COMM-001"], operators=["alltoall_hccl"])
    root, manifest = create_manifest(tmp_path, tasks, "matrix-manifest")
    assert set(manifest["matrix"]["ep_sizes"]) == {2, 4}
    for ep_size in (2, 4):
        index = json.loads((root / f"manifests/ep{ep_size}.json").read_text(encoding="utf-8"))
        assert index["ep_size"] == ep_size and index["task_ids"]
    assert (root / "matrix.json").is_file()
    assert (root / "ep_summary.json").exists() is False


# -------------------------------------------------------------------- CLI ---

def test_plan_cli_composes_multiple_yaml_files(tmp_path, capsys):
    plan_file = tmp_path / "mini_plan.yaml"
    plan_file.write_text("\n".join([
        "schema_version: 1",
        "plan:",
        "  name: mini",
        "  ep_sizes: [8, 64]",
        "  operators: all",
        "  levels: [L1]",
        "  suites: [decode]",
        "  base_config: %s" % (ROOT / "configs/base_benchmark.yaml").as_posix(),
        "  ep_configs:",
        "    8: %s" % (ROOT / "configs/ep8_bf16.yaml").as_posix(),
        "    64: %s" % (ROOT / "configs/ep64_bf16.yaml").as_posix(),
        "  suite_configs:",
        "    decode: %s" % (ROOT / "configs/decode.yaml").as_posix(),
        "  topology: {server_size: 8, supernode_size: 16}",
    ]), encoding="utf-8")
    output = tmp_path / "runs"
    assert main(["plan", "--plan", str(plan_file), "--cases", "COMM-001",
                 "--operators", "alltoall_hccl", "--output", str(output),
                 "--run-id", "mini-plan"]) == 0
    manifest = json.loads((output / "mini-plan/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tasks"]) == 2
    assert {task["ep_size"] for task in manifest["tasks"]} == {8, 64}
    assert manifest["matrix"]["name"] == "mini"
    assert manifest["matrix"]["case_index"]["HW-007"]["status"] == "alias"
    resolved = (output / "mini-plan/resolved_config.yaml").read_text(encoding="utf-8")
    assert resolved.startswith("configs:")
    capsys.readouterr()
    assert main(["capabilities", "--plan", str(plan_file),
                 "--cases", "COMM-001", "--operators", "alltoall_hccl"]) == 0
    assert "EP sizes" in capsys.readouterr().out

# ------------------------------------------------------------ full matrix ---

@pytest.fixture(scope="module")
def full_matrix(plan):
    """One expansion of the default plan, shared by the composition tests."""
    return expand_plan(plan)


def _tasks(run, ep_size=None, suite=None, case_id=None):
    return [task for task in run.tasks
            if (ep_size is None or task["ep_size"] == ep_size)
            and (suite is None or suite in task["suite_ids"])
            and (case_id is None or task["case_id"] == case_id)]


def _tokens(run, suite_id, ep_size=8):
    return {task["config"]["workload"]["tokens_per_rank"]
            for task in _tasks(run, ep_size, suite_id)}


def test_full_matrix_composes_all_suite_overlays(plan, full_matrix):
    assert plan.mode == "full_matrix"
    expected_suites = {"all", "hardware", "memory", "transport", "collectives", "overlap",
                       *BUSINESS_SUITES}
    assert set(plan.suites) == expected_suites
    points = {(point.ep_size, point.suite) for point in full_matrix.points}
    for ep_size in plan.ep_sizes:
        assert {suite for size, suite in points if size == ep_size} == expected_suites
        observed = {suite for task in full_matrix.tasks if task["ep_size"] == ep_size
                    for suite in task["suite_ids"]}
        assert observed == expected_suites
    metadata = full_matrix.metadata()
    sources = {point["suite"]: point["sources"] for point in metadata["config_points"]}
    for suite_id, path in plan.suite_configs.items():
        assert any(path.name in source for source in sources[suite_id]), suite_id
    for point in metadata["config_points"]:
        assert point["task_count"] > 0, (point["ep_size"], point["suite"])


def test_full_matrix_preserves_suite_workload_variants(plan, full_matrix):
    assert _tokens(full_matrix, "decode") == {16}
    assert _tokens(full_matrix, "prefill") == {4096}
    assert _tokens(full_matrix, "training_tail") == {16, 64, 128, 512}
    assert _tokens(full_matrix, "high_throughput") == {4096, 8192, 16384}
    for ep_size in plan.ep_sizes:
        assert _tokens(full_matrix, "strong_scaling", ep_size) == {32768 // ep_size}
        assert _tokens(full_matrix, "weak_scaling", ep_size) == {4096}
        weak = next(point.config for point in full_matrix.points
                    if point.suite == "weak_scaling" and point.ep_size == ep_size)
        assert weak["model"]["num_experts"] == 4 * ep_size
    # Dedup really sweeps None / Rank / Server.
    dedup = _tasks(full_matrix, 64, "dedup", "COMM-010")
    modes = {task["config"]["dedup"]["mode"] for task in dedup}
    assert modes == {"none", "rank", "server"}
    # Load balance enables the planner for both training and inference.
    load_balance = _tasks(full_matrix, 64, "load_balance")
    assert any(task["config"]["load_balance"]["enabled"] for task in load_balance)
    assert {task["config"]["benchmark"].get("phase") for task in load_balance} == {"train", "forward"}
    # Topology expands the high-locality routing scenario.
    routings = {task["config"]["workload"]["routing"]
                for task in _tasks(full_matrix, 64, "topology")}
    assert {"balanced", "high_locality"} <= routings
    # Quantized keeps its own small-token scenario.
    quantized = _tasks(full_matrix, 8, "quantized")
    assert quantized and {task["config"]["workload"]["tokens_per_rank"]
                          for task in quantized} == {128}
    assert {task["case_id"] for task in quantized} == {"COMM-009", "FUSE-008", "MOE-006"}


def test_full_matrix_deduplicates_identical_tasks(full_matrix):
    task_ids = [task["task_id"] for task in full_matrix.tasks]
    assert len(task_ids) == len(set(task_ids))
    merged = [task for task in full_matrix.tasks if len(task["suite_ids"]) > 1]
    assert merged, "identical points from different suites must collapse into one task"
    for task in merged:
        suites_from_points = {label.split("/", 1)[1] for label in task["plan_points"]}
        assert suites_from_points == set(task["suite_ids"])
    identities = {(task["ep_size"], task["case_id"], task["backend_id"],
                   fingerprint(task["parameters"]), task["config_fingerprint"])
                  for task in full_matrix.tasks}
    assert len(identities) == len(full_matrix.tasks)


def test_full_matrix_keeps_distinct_workloads(plan, full_matrix):
    def configs(suite_id, ep_size, case_id):
        return {task["config_fingerprint"]
                for task in _tasks(full_matrix, ep_size, suite_id, case_id)}

    assert configs("decode", 8, "COMM-001") and configs("prefill", 8, "COMM-001")
    assert not configs("decode", 8, "COMM-001") & configs("prefill", 8, "COMM-001")
    for ep_size in plan.ep_sizes:
        assert not configs("strong_scaling", ep_size, "MOE-001") & configs(
            "weak_scaling", ep_size, "MOE-001")
    dedup = _tasks(full_matrix, 64, "dedup", "COMM-010")
    by_mode = defaultdict(set)
    for task in dedup:
        by_mode[task["config"]["dedup"]["mode"]].add(task["config_fingerprint"])
    assert set(by_mode) == {"none", "rank", "server"}
    for mode, fingerprints in by_mode.items():
        assert len(fingerprints) == 1, mode
    assert len({task["task_id"] for task in dedup}) == len(dedup)
    phases = {task["config"]["benchmark"]["phase"]
              for task in _tasks(full_matrix, 8, "load_balance")}
    assert phases == {"train", "forward"}


def test_full_matrix_expands_all_ep_sizes(plan, full_matrix):
    canonical = set(canonical_cases())
    assert len(canonical) == 79
    catalog_only = len(canonical)
    for ep_size in plan.ep_sizes:
        subset = [task for task in full_matrix.tasks if task["ep_size"] == ep_size]
        assert {task["case_id"] for task in subset} == canonical
        assert set(full_matrix.coverage[ep_size]) == canonical
        assert {task["level"] for task in subset} == set(LEVELS)
        assert len(subset) > catalog_only
    assert full_matrix.expected == canonical
    assert full_matrix.metadata()["case_counts"]["canonical"] == catalog_only


# --------------------------------------------------- overlay and topology ---

def _mini_plan(tmp_path, name, extra_plan="", suite_path=None):
    plan_file = tmp_path / f"{name}.yaml"
    suite_line = (f"    decode: {suite_path.as_posix()}"
                  if suite_path else "")
    plan_file.write_text("\n".join([
        "schema_version: 1",
        "plan:",
        f"  name: {name}",
        "  ep_sizes: [8]",
        "  operators: all",
        "  levels: [L0, L1, L2, L3, L4]",
        "  suites: [decode]",
        f"  base_config: {(ROOT / 'configs/base_benchmark.yaml').as_posix()}",
        "  ep_configs:",
        f"    8: {(ROOT / 'configs/ep8_bf16.yaml').as_posix()}",
        "  suite_configs:",
        suite_line,
        extra_plan,
    ]), encoding="utf-8")
    return plan_file


def test_suite_overlay_cannot_override_ep_environment(tmp_path):
    suite = tmp_path / "hostile.yaml"
    suite.write_text("\n".join([
        "extends: ep8_bf16.yaml",
        "suite_id: decode",
        "device: cpu",
        "distributed: {ep_size: 8, tp_size: 4}",
        "model: {hidden_size: 999, dtype: float16, num_experts: 24}",
        "topology: {server_ids: [9, 9, 9, 9, 9, 9, 9, 9]}",
        "workload: {tokens_per_rank: 16}",
        "matrix:",
        "  - distributed: {ep_size: 8}",
        "    device: cpu",
        "    model: {num_experts: 32, hidden_size: 1, dtype: float16}",
        "    topology: {server_ids: [9, 9, 9, 9, 9, 9, 9, 9]}",
        "    workload: {tokens_per_rank: 32}",
    ]), encoding="utf-8")
    run = expand_plan(load_plan(_mini_plan(tmp_path, "overlay", "  topology: {server_size: 8, supernode_size: 16}", suite)))
    config = run.points[0].config
    # EP environment wins over every suite/matrix declaration.
    assert config["distributed"]["ep_size"] == 8
    assert config["distributed"]["tp_size"] == 1
    assert config["device"] == "npu"
    assert config["model"]["hidden_size"] == 7168
    assert config["model"]["dtype"] == "bfloat16"
    assert config["topology"]["server_ids"] == [0] * 8
    # Scenario values still come from the suite, including the scaling expert count.
    assert config["workload"]["tokens_per_rank"] == 32
    assert config["model"]["num_experts"] == 32


def test_scaling_overlay_preserves_num_experts(plan, full_matrix):
    for ep_size in plan.ep_sizes:
        weak = next(point.config for point in full_matrix.points
                    if point.suite == "weak_scaling" and point.ep_size == ep_size)
        strong = next(point.config for point in full_matrix.points
                      if point.suite == "strong_scaling" and point.ep_size == ep_size)
        assert weak["model"]["num_experts"] == 4 * ep_size
        assert strong["model"]["num_experts"] == 256
        assert weak["distributed"]["ep_size"] == strong["distributed"]["ep_size"] == ep_size
        assert weak["model"]["hidden_size"] == strong["model"]["hidden_size"] == 7168


def test_topology_source_is_recorded(plan, full_matrix, tmp_path):
    for ep_size in plan.ep_sizes:
        record = full_matrix.metadata()["topology"][str(ep_size)]
        assert record["source"] == "plan"
        assert record["verified"] is False
        assert record["servers"] == max(1, ep_size // 8)
        assert record["supernodes"] == max(1, ep_size // 16)
    for point in full_matrix.points:
        assert point.config["topology"]["source"] == "plan"
        assert point.config["topology"]["verified"] is False
    # An explicit map is taken as-is and may declare itself verified.
    explicit = "\n".join([
        "  topology:",
        "    rank_to_server: [0, 0, 0, 0, 1, 1, 1, 1]",
        "    server_to_supernode: {0: 0, 1: 0}",
        "    verified: true",
    ])
    run = expand_plan(load_plan(_mini_plan(tmp_path, "explicit", explicit)))
    topology = run.metadata()["topology"]["8"]
    assert topology["source"] == "explicit" and topology["verified"] is True
    config = run.points[0].config["topology"]
    assert config["server_ids"] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert config["supernode_ids"] == [0, 0, 0, 0, 0, 0, 0, 0]


def test_cross_server_requires_valid_topology(plan, full_matrix, tmp_path):
    # Default derived map: EP8 is one server, EP64/EP128 have remote peers.
    for ep_size, expected in ((8, "not_applicable"), (64, "not_run"), (128, "not_run")):
        cross = [task for task in _tasks(full_matrix, ep_size, "transport", "LINK-001")
                 if task["parameters"].get("peer_policy") == "cross_server"]
        assert cross and {task["status"] for task in cross} == {expected}
        if expected == "not_applicable":
            assert all("cross-server" in task["reason"] for task in cross)
    # No topology at all: the topology-aware policies are refused, not degraded.
    config = validate(merge(load_configs(ROOT / "configs/transport.yaml")[0], {
        "benchmark": {"transport": {"enabled": ["ub_memory"],
                                    "peer_policies": ["ring", "cross_server"]}}}))
    with patch("baselines.l0.topology.discovery.load", return_value=MATRIX):
        tasks = make_plan([config], cases=["LINK-001"])
    cross = [task for task in tasks if task["parameters"]["peer_policy"] == "cross_server"]
    assert cross and all(task["status"] == "not_applicable" for task in cross)
    assert all("topology" in task["reason"] for task in cross)
    # An explicit two-server EP8 map makes the same case executable.
    explicit = validate(merge(load_configs(ROOT / "configs/transport.yaml")[0], {
        "topology": {"server_ids": [0, 0, 0, 0, 1, 1, 1, 1], "source": "explicit",
                     "verified": True},
        "benchmark": {"transport": {"enabled": ["ub_memory"],
                                    "peer_policies": ["cross_server"]}}}))
    with patch("baselines.l0.topology.discovery.load", return_value=MATRIX):
        tasks = make_plan([explicit], cases=["LINK-001"])
    assert tasks and all(task["status"] == "not_run" for task in tasks)


# ----------------------------------------------------- execution integrity ---

def _cpu_configs(ep_sizes):
    return [validate(merge(DEFAULT, {
        "device": "cpu",
        "model": {"dtype": "float32", "num_experts": 8, "top_k": 2,
                  "hidden_size": 16, "intermediate_size": 24},
        "distributed": {"ep_size": ep_size},
        "workload": {"tokens_per_rank": 8},
    })) for ep_size in ep_sizes]


def _manifest_with_statuses(tmp_path, tasks, statuses, run_id, require=True):
    from common.controller import create_manifest, persist_case, result_path

    matrix = matrix_metadata(tasks)
    matrix["execution"] = {**matrix.get("execution", {}),
                           "require_complete_execution": require}
    root, manifest = create_manifest(tmp_path, tasks, run_id, matrix=matrix)
    for task, status in zip(tasks, statuses):
        record = result(manifest, task, status, f"synthetic {status}")
        persist_case(result_path(root, task), record)
    return root, manifest


def _summary(root, manifest):
    from common.controller import _write_ep_summary
    return _write_ep_summary(root, manifest)


def test_not_run_marks_full_matrix_incomplete(tmp_path):
    from types import SimpleNamespace

    tasks = make_plan(_cpu_configs([2]), cases=["COMM-001"], operators=["alltoall_hccl"])
    for task in tasks:
        task["status"], task["reason"] = "not_run", "planner disabled"
    root, manifest = _manifest_with_statuses(tmp_path, tasks, ["not_run"] * len(tasks), "incomplete")
    summary = _summary(root, manifest)
    assert summary["completion_status"] == "incomplete"
    assert summary["planned_tasks"] == len(tasks)
    assert summary["completed_tasks"] == 0
    assert summary["not_run"] == len(tasks)
    assert summary["passed"] == 0
    assert summary["require_complete_execution"] is True
    # The controller refuses a silent success: an unresolved `not_run` fails the
    # run in completeness mode, and still reports the gap otherwise.
    assert execute(root, manifest, SimpleNamespace(nnodes=1, node_rank=0)) == 1
    assert len(summary["unresolved_tasks"]) == len(tasks)


def test_dependency_missing_is_not_passed(tmp_path):
    tasks = make_plan(_cpu_configs([2]), cases=["COMM-001"], operators=["alltoall_hccl"])
    root, manifest = _manifest_with_statuses(tmp_path, tasks, ["dependency_missing"] * len(tasks),
                                             "dependency")
    summary = _summary(root, manifest)
    assert summary["passed"] == 0
    assert summary["dependency_missing"] == len(tasks)
    assert summary["blocked"] == len(tasks)
    assert summary["executed"] == 0
    assert summary["completion_status"] == "complete"
    record = list(results(root, manifest))[0]
    assert record["status"] == "dependency_missing"


def test_unsupported_is_not_failed(tmp_path):
    tasks = make_plan(_cpu_configs([2]), cases=["COMM-001"], operators=["alltoall_hccl"])
    root, manifest = _manifest_with_statuses(tmp_path, tasks, ["unsupported"] * len(tasks),
                                             "unsupported")
    summary = _summary(root, manifest)
    assert summary["unsupported"] == len(tasks)
    assert summary["failed"] == 0
    assert summary["completed_tasks"] == len(tasks)
    assert summary["completion_status"] == "complete"


def test_not_applicable_is_terminal(tmp_path):
    tasks = make_plan(_cpu_configs([2]), cases=["COMM-001"], operators=["alltoall_hccl"])
    root, manifest = _manifest_with_statuses(tmp_path, tasks, ["not_applicable"] * len(tasks),
                                             "not-applicable")
    summary = _summary(root, manifest)
    assert summary["not_applicable"] == len(tasks)
    assert summary["not_run"] == 0
    assert summary["completion_status"] == "complete"
    assert summary["failed"] == 0


def test_complete_plan_requires_terminal_results(tmp_path):
    tasks = make_plan(_cpu_configs([2, 4]), cases=["COMM-001"], operators=["alltoall_hccl"])
    assert len(tasks) == 2
    root, manifest = _manifest_with_statuses(tmp_path, tasks,
                                             ["passed", "environment_unavailable"], "terminal")
    summary = _summary(root, manifest)
    assert summary["completion_status"] == "complete"
    assert summary["passed"] == 1
    assert summary["environment_unavailable"] == 1
    assert summary["blocked"] == 1
    # Re-open the plan with one unresolved task: it can no longer be complete.
    root, manifest = _manifest_with_statuses(tmp_path, tasks, ["passed", "not_run"], "unresolved")
    summary = _summary(root, manifest)
    assert summary["completion_status"] == "incomplete"
    assert summary["not_run"] == 1


def test_ep_summary_counts_are_consistent(tmp_path):
    tasks = make_plan(_cpu_configs([2, 4]), cases=["COMM-001"], operators=["alltoall_hccl"])
    assert len(tasks) == 2
    statuses = {2: "passed", 4: "runtime_failed"}
    root, manifest = _manifest_with_statuses(
        tmp_path, tasks, [statuses[task["ep_size"]] for task in tasks], "counts")
    summary = _summary(root, manifest)
    buckets = ("passed", "failed", "unsupported", "not_applicable", "not_implemented",
               "dependency_missing", "environment_unavailable", "source_unavailable", "not_run")
    assert sum(summary[bucket] for bucket in buckets) == summary["planned_tasks"]
    assert summary["consistency"] == {"planned_tasks": 2, "sum_of_buckets": 2,
                                      "consistent": True}
    per_ep = summary["ep_sizes"]
    assert set(per_ep) == {"2", "4"}
    for size, entry in per_ep.items():
        assert sum(entry[bucket] for bucket in buckets) == entry["planned_tasks"] == 1
        assert entry["consistency"]["consistent"] is True
        assert entry["completion_status"] == "complete"
    assert per_ep["2"]["passed"] == 1 and per_ep["2"]["failed"] == 0
    assert per_ep["4"]["failed"] == 1 and per_ep["4"]["passed"] == 0
    assert summary["failed"] == 1 and summary["passed"] == 1
    assert summary["executed"] == 2


# -------------------------------------------------- variant level coverage ---

def _manifest_for(run):
    return {"run_id": "synthetic", "environment": {}, "tasks": run.tasks,
            "matrix": run.metadata()}


def test_variant_coverage_accounts_for_every_manifest_task(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["decode"])
    rows = [_row(plan, task) for task in run.tasks]
    report = coverage.task_matrix(rows, _manifest_for(run))
    assert report["totals"]["planned"] == len(run.tasks)
    assert report["totals"]["missing"] == 0
    assert report["totals"]["matched"] == len(run.tasks)
    assert report["orphans"] == [] and report["cases_without_tasks"] == []
    for entry in report["summary"]:
        assert entry["accounted"] == entry["planned"] >= 1
    assert {cell["operator"] for cell in report["cells"]} == set(registry())


def test_variant_coverage_locates_missing_tasks(plan):
    run = expand_plan(plan, ep_sizes=[8], suites=["decode"], cases=["COMM-001"])
    rows = [_row(plan, task) for task in run.tasks]
    dropped = rows[:2]
    report = coverage.task_matrix(rows[2:], _manifest_for(run))
    assert report["totals"]["missing"] == 2
    missing = {cell["task_id"] for cell in report["cells"] if cell["status"] == "missing"}
    assert missing == {row["task_id"] for row in dropped}
    assert all(cell["reason"] for cell in report["cells"] if cell["status"] == "missing")
    # An unsupported variant is kept as its own terminal state, not as a pass.
    entries = [entry for entry in report["summary"] if entry["case_id"] == "COMM-001"]
    assert sum(entry["planned"] for entry in entries) == len(run.tasks)
    assert sum(entry["missing"] for entry in entries) == 2
    assert sum(entry["measured"] for entry in entries) == len(run.tasks) - 2
    # A result record that is not planned is reported, not counted as coverage.
    orphan = _row(plan, run.tasks[0])
    orphan["task_id"] = "COMM-001-orphan"
    report = coverage.task_matrix([*rows, orphan], _manifest_for(run))
    assert report["orphans"] == ["COMM-001-orphan"]
    assert report["totals"]["planned"] == len(run.tasks)

# ------------------------------------------------------------ L0 scale-out ---

def _l0_row(case_id, ep_size, parameters, status="passed", observed=None,
            verification="verified", task_suffix=None, reason=None):
    """Synthetic L0 row: report aggregation only, never a performance claim."""
    task = {
        "task_id": f"{case_id}-{ep_size}-{task_suffix or fingerprint(parameters)[:8]}",
        "case_id": case_id, "level": "L0", "operation": "put", "backend_id": "hardware",
        "config": merge(load_configs(ROOT / "configs/smoke_cpu.yaml")[0], {
            "device": "npu", "distributed": {"ep_size": ep_size}}),
        "parameters": dict(parameters), "source_versions": {}, "baseline": True,
    }
    row = result({"run_id": "synthetic", "environment": {}}, task, status, reason)
    row["case_parameters"] = dict(parameters)
    row["category"] = "transport"
    row["observed"] = dict(observed or {})
    row["verification"] = {"status": verification}
    if status == "passed":
        row["correctness"] = {"passed": True}
        row["latency"] = {"p50": 1.0}
        row["bandwidth"] = {"payload_bytes": parameters.get("message_bytes"),
                            "payload_bandwidth_GBps": 1.0}
    return row


def _groups(rows):
    return {(entry["case_id"], entry.get("variant"), entry.get("peer_policy"),
             entry.get("message_bytes"), entry["parameter_fingerprint"]): entry
            for entry in coverage.l0_scaleout(rows)}


def test_l0_scaleout_separates_variants():
    settings = {"variant": "device", "message_bytes": 4096, "peer_policy": "group",
                "peer_count": 0, "outstanding": 1, "mechanism": "shmem_udma"}
    host = {**settings, "variant": "host"}
    rows = [_l0_row("LINK-013", 8, settings), _l0_row("LINK-013", 64, settings),
            _l0_row("LINK-013", 8, host), _l0_row("LINK-013", 64, host)]
    entries = list(_groups(rows).values())
    assert len(entries) == 2
    assert {entry["variant"] for entry in entries} == {"device", "host"}
    for entry in entries:
        assert len(entry["points"]) == 2


def test_l0_scaleout_separates_peer_policies():
    settings = {"variant": "default", "message_bytes": 4096, "outstanding": 1,
                "peer_count": 0, "mechanism": "ub_memory"}
    rows = []
    for policy in ("ring", "same_server", "cross_server"):
        for ep_size in (8, 64):
            rows.append(_l0_row("LINK-001", ep_size, {**settings, "peer_policy": policy},
                                observed={"transport_protocol": "ub_memory",
                                          "execution_resource": "AIV"}))
    entries = list(_groups(rows).values())
    assert len(entries) == 3
    assert {entry["peer_policy"] for entry in entries} == {"ring", "same_server", "cross_server"}
    assert all(entry["mechanism"] == "ub_memory" for entry in entries)


def test_l0_scaleout_separates_compute_shapes():
    fixed = {"variant": "gmm1", "message_bytes": None, "shape_mode": "fixed",
             "shape": {"m": 8192, "k": 7168, "n": 3072, "groups": 8}}
    workload = {**fixed, "shape_mode": "workload", "shape": None}
    dgrad = {**fixed, "variant": "gmm1_dgrad"}
    rows = []
    for ep_size in (8, 64):
        for parameters in (fixed, workload, dgrad):
            rows.append(_l0_row("HW-010", ep_size, parameters, verification="verified"))
    entries = list(_groups(rows).values())
    assert len(entries) == 3
    assert {entry.get("shape_mode") for entry in entries} == {"fixed", "workload"}
    assert {entry["variant"] for entry in entries} == {"gmm1", "gmm1_dgrad"}


def test_l0_scaleout_does_not_overwrite_same_ep_points():
    parameters = {"variant": "default", "message_bytes": 4096, "outstanding": 1,
                  "peer_count": 0, "peer_policy": "ring", "mechanism": "ub_memory"}
    first = _l0_row("LINK-001", 8, parameters, task_suffix="a")
    second = _l0_row("LINK-001", 8, parameters, task_suffix="b")
    entry, = coverage.l0_scaleout([first, second])
    assert entry["points"] == []
    assert entry["ambiguous"] and entry["ambiguous"][0]["ep_size"] == 8
    assert entry["ambiguous"][0]["task_ids"] == [first["task_id"], second["task_id"]]
    assert entry["reason"] and entry["comparable"] is False
    # The same task measured twice is a re-run of one point, not an ambiguity.
    repeat = _l0_row("LINK-001", 8, parameters, task_suffix="a")
    entry, = coverage.l0_scaleout([first, repeat])
    assert [point["ep_size"] for point in entry["points"]] == [8]
    assert entry["ambiguous"] == []


def test_l0_scaleout_matches_only_identical_parameters():
    base = {"variant": "default", "message_bytes": 4096, "outstanding": 1,
            "peer_count": 0, "peer_policy": "ring", "mechanism": "ub_memory"}
    changed = {**base, "outstanding": 4}
    rows = [_l0_row("LINK-001", 8, base), _l0_row("LINK-001", 64, base),
            _l0_row("LINK-001", 64, changed)]
    entries = list(_groups(rows).values())
    assert len(entries) == 2
    shared = next(entry for entry in entries if entry["outstanding"] == 1)
    assert sorted(point["ep_size"] for point in shared["points"]) == [8, 64]
    separate = next(entry for entry in entries if entry["outstanding"] == 4)
    assert [point["ep_size"] for point in separate["points"]] == [64]
    # A different observed mechanism is never merged into the requested one.
    other = _l0_row("LINK-001", 128, base,
                    observed={"transport_protocol": "roce", "execution_resource": "AIV"})
    entries = coverage.l0_scaleout([*rows, other])
    assert len(entries) == 3
    assert any(entry["mechanism"] == "roce" for entry in entries)


def test_l0_scaleout_report_keeps_na_reason(tmp_path):
    base = {"variant": "default", "message_bytes": 4096, "outstanding": 1,
            "peer_count": 0, "peer_policy": "cross_server", "mechanism": "ub_memory"}
    rows = [
        _l0_row("LINK-001", 8, base, status="not_applicable",
                observed={"execution_resource": "AIV"}, verification="unverified",
                reason="no cross-server peer available at EP8"),
        _l0_row("LINK-001", 64, base,
                observed={"transport_protocol": "ub_memory", "execution_resource": "AIV"}),
    ]
    manifest = {"run_id": "synthetic", "environment": {}, "tasks": [],
                "matrix": {"ep_sizes": [8, 64], "expected_cases": ["LINK-001"],
                           "catalog": {"LINK-001": {"level": "L0", "category": "transport",
                                                    "alias_of": None}}}}
    written = coverage.write(tmp_path, manifest, rows)
    assert written["l0_scaleout"].is_file()
    text = written["l0_scaleout"].read_text(encoding="utf-8")
    assert "peer policy" in text and "cross_server" in text
    assert "not_applicable (no cross-server peer available at EP8)" in text
    assert written["variants"].is_file()
    assert (tmp_path / "reports/variant_coverage.csv").is_file()
