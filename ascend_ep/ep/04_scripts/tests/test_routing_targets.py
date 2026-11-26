"""Target-distribution routing: generation, feasibility and target reporting.

The point of this module is that a load imbalance is a *specified* test condition,
not a hope: the generator receives a target (rank/expert max-mean, local ratio),
allocates integer quotas, and the achieved value is checked against it.  The two
honesty rules these tests enforce:

* a target the Top-K uniqueness rule cannot reach is reported ``not_applicable``
  with its feasible upper bound -- never silently clamped;
* a metric this generator does not control is rejected as a target instead of
  being "measured into existence".
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import routing  # noqa: E402
from common.backend import BenchmarkUnavailable  # noqa: E402
from common.config import load_configs  # noqa: E402
from common.plan import expand, load_plan  # noqa: E402

EP_FILES = {8: "configs/ep8_bf16.yaml", 16: "configs/ep16_bf16.yaml",
            32: "configs/ep32_bf16.yaml", 64: "configs/ep64_bf16.yaml",
            128: "configs/ep128_bf16.yaml"}


def target_config(ep=8, metric="rank_max_mean", value=4.0, hotspot=None,
                  tokens=None):
    config = load_configs(ROOT / EP_FILES[ep])[0]
    if tokens is not None:
        config["workload"]["tokens_per_rank"] = tokens
    config["workload"].update({
        "routing": "target_distribution",
        "routing_control": {
            "target_metric": metric, "target_value": value,
            "hotspot": hotspot or {"ranks": [0]},
            "tolerance": {"relative": 0.02},
        },
    })
    return config


def measure(config):
    """Generate every rank's routes and return the achieved global metrics."""
    ep, k = config["distributed"]["ep_size"], config["model"]["top_k"]
    experts = config["model"]["num_experts"]
    rank_total = np.zeros(ep, dtype=np.int64)
    expert_total = np.zeros(experts, dtype=np.int64)
    local = total = 0
    ids_by_rank = []
    for rank in range(ep):
        ids = routing.target_routes(config, rank)
        routing.validate_routes(ids, np.ones_like(ids, dtype=float) / k,
                                ids.shape[0], k, experts)
        owners = np.asarray(routing.geometry(config)[0])[ids]
        rank_total += np.bincount(owners.ravel(), minlength=ep)
        expert_total += np.bincount(ids.ravel(), minlength=experts)
        local += int((owners == rank).sum())
        total += ids.size
        ids_by_rank.append(ids)
    return {
        "rank_max_mean": routing.max_mean(rank_total.tolist()),
        "expert_max_mean": routing.max_mean(expert_total.tolist()),
        "local_route_ratio": local / total,
        "rank_total": rank_total.tolist(),
        "expert_total": expert_total.tolist(),
        "ids": ids_by_rank,
        "routes": total,
    }


@pytest.mark.parametrize("ep,value", [(8, 2.0), (8, 4.0), (16, 8.0), (16, 2.0)])
def test_rank_target_is_generated_exactly(ep, value):
    achieved = measure(target_config(ep=ep, value=value))
    assert achieved["rank_max_mean"] == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize("value", [1.0, 2.0, 4.0])
def test_expert_target_is_generated_exactly(value):
    achieved = measure(target_config(metric="expert_max_mean", value=value,
                                     hotspot={"expert": 0}))
    assert achieved["expert_max_mean"] == pytest.approx(value, rel=1e-9)


def test_a_uniform_expert_target_is_the_balanced_case():
    achieved = measure(target_config(metric="expert_max_mean", value=1.0,
                                     hotspot={"expert": 0}))
    assert achieved["expert_max_mean"] == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("value", [0.0, 0.25, 0.5])
def test_local_ratio_is_generated_exactly(value):
    achieved = measure(target_config(metric="local_route_ratio", value=value))
    assert achieved["local_route_ratio"] == pytest.approx(value, abs=1e-9)


def test_an_all_local_target_is_unreachable_with_three_experts_per_rank():
    """A token has K=6 slots but only 3 experts on one rank -> local ratio <= 0.5."""
    config = target_config(metric="local_route_ratio", value=1.0)
    plan = routing.target_plan(config)
    assert plan["status"] == "not_applicable"
    assert plan["bound"] == pytest.approx(0.5)
    assert "Top-K uniqueness" in plan["reason"]
    assert "local_route_ratio" in plan["reason"]


def test_the_trace_is_unique_and_conserved():
    config = target_config(value=2.0)
    achieved = measure(config)
    assert achieved["routes"] == sum(routing.geometry(config)[1]) * config["model"]["top_k"]
    assert sum(achieved["rank_total"]) == achieved["routes"]
    assert sum(achieved["expert_total"]) == achieved["routes"]


def test_feasible_bounds_match_the_topk_table():
    """The bound is computed from the placement, and matches E/K and P*min(K,E_r)/K."""
    expected = {8: 4.0, 16: 8.0, 32: 16.0, 64: 32.0, 128: 64.0}
    for ep, bound in expected.items():
        config = load_configs(ROOT / EP_FILES[ep])[0]
        bounds = routing.feasible_bounds(config)
        assert bounds["rank_max_mean"] == pytest.approx(bound)
        assert bounds["expert_max_mean"] == pytest.approx(
            config["model"]["num_experts"] / config["model"]["top_k"])


def test_unreachable_rank_target_is_not_applicable():
    config = target_config(ep=8, value=8.0)
    plan = routing.target_plan(config)
    assert plan["status"] == "not_applicable"
    assert plan["bound"] == pytest.approx(4.0)
    assert "Top-K uniqueness" in plan["reason"]
    with pytest.raises(BenchmarkUnavailable) as caught:
        routing.target_routes(config, 0)
    assert caught.value.status == "not_applicable"


def test_unreachable_expert_target_is_not_applicable():
    config = target_config(ep=8, metric="expert_max_mean", value=8.0,
                           hotspot={"expert": 0})
    plan = routing.target_plan(config)
    assert plan["status"] == "not_applicable"
    assert plan["bound"] == pytest.approx(4.0)


def test_a_statistical_metric_cannot_be_a_target():
    config = target_config(metric="server_max_mean", value=2.0)
    with pytest.raises(ValueError, match="statistical"):
        routing.target_plan(config)


def test_generation_is_deterministic():
    config = target_config(value=4.0)
    first = routing.target_routes(config, 0)
    second = routing.target_routes(config, 0)
    assert np.array_equal(first, second)


def test_unequal_source_tokens_are_supported():
    config = target_config(ep=8, value=2.0,
                           tokens=[4096, 2048, 1024, 512, 4096, 2048, 1024, 512])
    achieved = measure(config)
    assert achieved["rank_max_mean"] == pytest.approx(2.0, rel=1e-9)


def test_summarize_reports_the_target_control():
    config = target_config(value=4.0)
    achieved = measure(config)
    records = []
    for rank, ids in enumerate(achieved["ids"]):
        records.append(routing.summarize_local(
            routing.Routing(ids, np.ones_like(ids, dtype=float),
                            tuple(routing.geometry(config)[0]), tuple(
                                routing.geometry(config)[1]), 42),
            rank, None))
    tokens = routing.geometry(config)[1]
    route = routing.Routing(np.zeros((sum(tokens), config["model"]["top_k"]),
                                     dtype=np.int64),
                            np.ones((sum(tokens), config["model"]["top_k"])),
                            tuple(routing.geometry(config)[0]), tuple(tokens), 42)
    summary = routing.summarize(route, records, config)
    control = summary["target_control"]
    assert control["metric"] == "rank_max_mean"
    assert control["target"] == 4.0
    assert control["actual"] == pytest.approx(4.0)
    assert control["feasible_upper_bound"] == pytest.approx(4.0)
    assert control["status"] == "generated"
    assert control["met"] is True


def test_validation_rejects_a_bad_control_block():
    from common.config import validate

    config = target_config(value=2.0)
    config["workload"]["routing_control"]["target_value"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate(config)
    config = target_config(value=2.0)
    config["workload"]["routing_control"]["target_metric"] = "rank_tx_max_mean"
    with pytest.raises(ValueError, match="routing_control.target_metric"):
        validate(config)


def test_the_routing_imbalance_suite_expands_in_the_plan():
    plan = load_plan(ROOT / "configs" / "benchmark_plan.yaml")
    run = expand(plan, ep_sizes=[8], suites=["routing_imbalance"])
    assert run.tasks
    controls = {str(task["config"]["workload"].get("routing_control"))
                for task in run.tasks}
    assert len(controls) >= 4, "the matrix points must produce distinct targets"
    assert any(task["config"]["workload"]["routing"] == "balanced"
               for task in run.tasks)
