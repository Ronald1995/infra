"""Benchmark matrix tests: EP sizes, suites, expansion, capability and reports.

No NPU is used.  The L0 capability probe is replaced by an explicit override
fixture, so every assertion is about *planning* behaviour.  Report tests build
synthetic result rows only to exercise the aggregation logic; they are explicitly
not performance measurements.
"""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from baselines.l0.topology import capabilities
from common.cases import (alias_map, canonical_cases, case_counts, catalog, l0_family)
from common.config import LEVELS, load_configs, merge
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
    from types import SimpleNamespace

    from baselines.l0.transport.channel import _peers

    ctx = SimpleNamespace(size=16, rank=3,
                          config={"topology": {"server_ids": [0] * 8 + [1] * 8}})
    ring = _peers(ctx, {"peer_count": 4})
    assert len(ring) == 4 and ctx.rank not in ring
    assert ring == _peers(ctx, {"peer_count": 4})
    same = _peers(ctx, {"peer_count": 4, "peer_policy": "same_server"})
    assert all(rank < 8 for rank in same)
    cross = _peers(ctx, {"peer_count": 4, "peer_policy": "cross_server"})
    assert all(rank >= 8 for rank in cross)
    assert len(_peers(ctx, {"peer_count": 0})) == 15


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
