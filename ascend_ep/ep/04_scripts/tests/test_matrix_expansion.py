"""Matrix expansion: five EP sizes, every declared dimension, explicit N/A.

The properties these tests pin down are the ones the old matrix silently broke:

* ``cores`` / ``quiet_modes`` are expanded into every applicable combination
  instead of collapsing to the last configured value;
* a dimension the runtime cannot express is declared ``not_applicable`` with a
  reason (and a variant without the control keeps a fixed value), never faked;
* all five EP sizes produce the same task count and the same variant multiset;
* an impossible peer count stays in the manifest instead of shrinking the scope.
"""
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.case_spec import L0_PARAMETERS, spec  # noqa: E402
from common.config import load_configs  # noqa: E402
from common.plan import expand, load_plan  # noqa: E402
from common.scheduler import make_plan  # noqa: E402
from common.variants import (MECHANISM_DIMENSIONS, case_dimension_records,  # noqa: E402
                             case_dimensions, dimension_values, expand_variants)


@pytest.fixture(scope="module")
def settings():
    return load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]["benchmark"]


def test_shmem_udma_sweeps_every_core_and_quiet_combination(settings):
    sets = expand_variants(spec("LINK-013"), settings, 8)
    device = [item for item in sets if item["variant"] == "device"]
    host = [item for item in sets if item["variant"] == "host"]
    assert {item["cores"] for item in device} == {1, 2, 4}
    assert {item["quiet_mode"] for item in device} == {0, 1}
    # 6 message sizes x 3 cores x 2 quiet modes
    assert len(device) == 6 * 3 * 2
    # The host variant has no QP/quiet control: one fixed value, no fake sweep.
    assert {(item["cores"], item["quiet_mode"]) for item in host} == {(1, 0)}
    assert len(host) == 6


def test_every_device_combination_is_distinct(settings):
    sets = expand_variants(spec("LINK-013"), settings, 8)
    device = [(item["message_bytes"], item["cores"], item["quiet_mode"])
              for item in sets if item["variant"] == "device"]
    assert len(device) == len(set(device))


def test_a_mechanism_without_the_control_is_not_applicable(settings):
    dimensions = case_dimensions(spec("LINK-001"), settings, 8, "default")
    assert dimensions["cores"]["type"] == "not_applicable"
    assert "does not expose" in dimensions["cores"]["reason"]
    assert dimensions["quiet_mode"]["type"] == "not_applicable"
    # ... and no parameter set carries a value the runtime would ignore.
    for params in expand_variants(spec("LINK-001"), settings, 8):
        assert "cores" not in params and "quiet_mode" not in params


def test_dimension_support_is_declared_for_every_mechanism():
    from common.config import L0_MECHANISMS

    mechanisms = {params.get("mechanism") for params in L0_PARAMETERS.values()}
    declared = mechanisms & set(L0_MECHANISMS)
    assert declared == set(L0_MECHANISMS)
    assert declared <= set(MECHANISM_DIMENSIONS)


def test_fixed_and_sweep_dimensions_are_distinguishable(settings):
    dimensions = case_dimensions(spec("LINK-013"), settings, 8, "host")
    assert dimensions["cores"] == {"type": "fixed", "value": 1,
                                   "reason": dimensions["cores"]["reason"]}
    assert dimension_values(dimensions, "cores") == [1]
    device = case_dimensions(spec("LINK-013"), settings, 8, "device")
    assert dimension_values(device, "cores") == [1, 2, 4]


def test_dimension_records_name_the_consumer(settings):
    records = case_dimension_records([spec("LINK-013")], settings, 8)
    cores = records["LINK-013"]["device"]["cores"]
    assert cores["consumer"] == {"path": "baselines/l0/shmem/udma.py", "token": "cores"}
    assert (ROOT / cores["consumer"]["path"]).is_file()


def test_impossible_peer_count_stays_in_the_manifest(settings):
    config = load_configs(ROOT / "configs" / "smoke_cpu.yaml")[0]
    config["benchmark"].update({"peer_counts": [4], "outstanding": [1],
                                "message_bytes": [4096]})
    tasks = make_plan([config], cases=["LINK-001"])
    assert tasks
    task = tasks[0]
    assert task["parameters"]["peer_count"] == 4
    assert task["status"] in ("unsupported", "not_applicable")
    assert task["reason"]


@pytest.fixture(scope="module")
def plan():
    return load_plan(ROOT / "configs" / "benchmark_plan.yaml")


def test_five_ep_sizes_are_planned(plan):
    assert plan.ep_sizes == [8, 16, 32, 64, 128]
    for size in plan.ep_sizes:
        assert plan.ep_config(size).is_file()


def test_every_ep_size_produces_the_same_matrix(plan):
    run = expand(plan)
    counts = Counter(task["ep_size"] for task in run.tasks)
    assert sorted(counts) == [8, 16, 32, 64, 128]
    assert len(set(counts.values())) == 1, counts
    assert len(run.coverage) == 5


def test_three_experts_per_rank_is_the_unified_baseline(plan):
    for size in plan.ep_sizes:
        environment = load_configs(plan.ep_config(size))[0]
        assert environment["distributed"]["ep_size"] == size
        assert environment["model"]["num_experts"] == 3 * size


def test_the_ep64_historical_profile_is_a_separate_model(plan):
    historical = load_configs(ROOT / "configs" / "ep64_bf16_384.yaml")[0]
    assert historical["distributed"]["ep_size"] == 64
    assert historical["model"]["num_experts"] == 384
    # It is not part of the unified plan, so it can never be spliced into the
    # five-point scaling curve by accident.
    assert plan.ep_config(64).name == "ep64_bf16.yaml"


def test_plan_carries_dimensions_and_audit(plan):
    run = expand(plan, ep_sizes=[8])
    metadata = run.metadata()
    point = metadata["config_points"][0]
    assert point["dimensions"]["LINK-013"]["device"]["cores"]["type"] == "sweep"
    assert point["dimensions"]["LINK-001"]["default"]["cores"]["type"] == "not_applicable"
    assert point["config_provenance"]["distributed.ep_size"]["value"] == 8
    assert metadata["config_audit"]["ok"] is True


def test_transport_mechanisms_are_all_planned(plan):
    run = expand(plan, ep_sizes=[8], suites=["transport"])
    mechanisms = {task["parameters"].get("mechanism") for task in run.tasks
                  if task["case_category"] == "transport"}
    assert {"ub_memory", "urma", "ub_ctp", "ub_rtp", "hccs", "roce", "uboe",
            "shmem_mte", "shmem_udma"} <= mechanisms


def test_rank_imbalance_is_per_iteration():
    """The imbalance ratio must compare ranks inside one iteration."""
    from metrics import summary

    # Two iterations over four ranks.  Iteration 0 max/mean = 2.0/1.0, iteration 1
    # max/mean = 4.0/2.0: both ratios are 2.0.  The old formula compared
    # iteration 0's max (2.0) with the mean of the maxima (3.0) and reported 0.67.
    records = []
    for rank, per_iteration in enumerate(((2.0, 4.0), (1.0, 2.0), (1.0, 2.0), (0.0, 0.0))):
        records.append({"rank": rank, "samples": [
            {"latency_ns": value * 1e6, "host_issue_ns": 0, "host_wait_ns": 0,
             "output_ready_ns": value * 1e6} for value in per_iteration]})
    result = summary.latency(records)
    assert result["rank_max_mean_samples"] == [2.0, 2.0]
    assert result["rank_latency_max_mean_p50"] == 2.0
    assert result["rank_latency_max_mean_max"] == 2.0
    assert result["imbalance_ratio"] == 2.0
