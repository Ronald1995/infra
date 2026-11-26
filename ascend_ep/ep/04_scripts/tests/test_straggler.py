"""Straggler controller: selection, injection placement, calibration and honesty.

The properties pinned here are the ones that make a slow-rank test a *measurement*:

* the selection is deterministic and identical on every rank (no collective needed);
* the injection happens inside the timed region and after the barrier, so it cannot
  be absorbed by rank alignment;
* a disabled block changes nothing (no sleep, no observation, no timing change);
* a method or stage without an implementation is reported, never approximated.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import straggler, timer  # noqa: E402
from common.backend import BenchmarkUnavailable  # noqa: E402
from common.config import load_configs, validate  # noqa: E402
from common.plan import expand, load_plan  # noqa: E402


def config_with(**overrides):
    config = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    block = {
        "enabled": True, "seed": 7,
        "selection": {"mode": "fixed", "ranks": [0]},
        "injection": {"stage": "before_operation", "method": "host_delay",
                      "delay_us": 20000},
        "trigger": {"probability": 1.0, "every_n_iterations": 1},
        "calibration": {"relative_tolerance": 0.5},
    }
    for key, value in overrides.items():
        if key in ("selection", "injection", "trigger", "calibration"):
            block[key] = {**block[key], **value}
        else:
            block[key] = value
    config["benchmark"]["straggler"] = block
    return config


# ----------------------------------------------------------------- selection ---

def test_a_disabled_block_changes_nothing():
    config = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    controller = straggler.StragglerController(config, 0, 8)
    assert controller.enabled is False
    assert controller.selected_ranks(0) == []
    assert controller.before_operation(0) is None
    summary = controller.summary()
    assert summary["injections"] == 0 and summary["observations"] == []


def test_fixed_selection_only_the_named_rank():
    controller = straggler.StragglerController(config_with(), 0, 8)
    assert controller.selected_ranks(0) == [0]
    assert controller.before_operation(0)["injected"] is True
    other = straggler.StragglerController(config_with(), 1, 8)
    assert other.selected_ranks(0) == [0]
    assert other.before_operation(0) is None
    assert other.summary()["injections"] == 0


def test_rotating_selection_moves_the_slow_rank():
    controller = straggler.StragglerController(
        config_with(selection={"mode": "rotating", "ranks": [0]}), 0, 8)
    assert [controller.selected_ranks(step)[0] for step in range(4)] == [0, 1, 2, 3]


def test_fraction_selection_is_rounded_and_deterministic():
    controller = straggler.StragglerController(
        config_with(selection={"mode": "fraction", "fraction": 0.125}), 0, 8)
    assert controller.selected_ranks(0) == [0]
    assert controller.selected_ranks(3) == [3]
    wide = straggler.StragglerController(
        config_with(selection={"mode": "fraction", "fraction": 0.25}), 0, 8)
    assert len(wide.selected_ranks(0)) == 2


def test_random_selection_is_reproducible_across_ranks():
    left = straggler.StragglerController(
        config_with(selection={"mode": "random", "fraction": 0.125}), 3, 8)
    right = straggler.StragglerController(
        config_with(selection={"mode": "random", "fraction": 0.125}), 5, 8)
    for step in range(5):
        assert left.selected_ranks(step) == right.selected_ranks(step)
    assert len({tuple(left.selected_ranks(step)) for step in range(6)}) > 1


def test_server_selection_needs_a_topology():
    config = config_with(selection={"mode": "server", "ranks": [0]})
    with pytest.raises(BenchmarkUnavailable) as caught:
        straggler.StragglerController(config, 0, 8)
    assert caught.value.status == "not_applicable"
    config["topology"] = {"server_ids": [0] * 8}
    controller = straggler.StragglerController(config, 0, 8)
    assert controller.selected_ranks(0) == list(range(8))


def test_hotspot_selection_follows_the_hot_rank():
    config = config_with(selection={"mode": "hotspot", "ranks": [0]})
    config["workload"]["hot_rank"] = 3
    controller = straggler.StragglerController(config, 3, 8)
    assert controller.selected_ranks(0) == [3]
    assert controller.before_operation(0)["injected"] is True
    config["workload"].pop("hot_rank")
    with pytest.raises(BenchmarkUnavailable):
        straggler.StragglerController(config, 3, 8)


# ------------------------------------------------------------------- trigger ---

def test_probability_zero_never_injects():
    controller = straggler.StragglerController(
        config_with(trigger={"probability": 0.0}), 0, 8)
    assert controller.should_inject(0) is False
    assert controller.before_operation(0) is None


def test_every_n_iterations_skips_the_others():
    controller = straggler.StragglerController(
        config_with(trigger={"every_n_iterations": 3}), 0, 8)
    assert controller.before_operation(0)["injected"] is True
    assert controller.before_operation(1) is None
    assert controller.before_operation(3)["injected"] is True


# ---------------------------------------------------------------- injection ---

def test_host_delay_is_measured_and_calibrated():
    controller = straggler.StragglerController(config_with(), 0, 8)
    observation = controller.before_operation(0)
    assert observation["injected"] is True
    assert observation["requested_delay_us"] == 20000
    assert observation["actual_delay_us"] >= 1000
    assert observation["calibration_passed"] is True
    assert controller.summary()["injections"] == 1


def test_stage_delay_only_fires_at_its_stage():
    controller = straggler.StragglerController(
        config_with(injection={"method": "stage_delay", "stage": "before_dispatch",
                               "delay_us": 1000}), 0, 8)
    assert controller.before_operation(0) is None
    assert controller.before_stage("before_combine", 0) is None
    assert controller.before_stage("before_dispatch", 0)["injected"] is True


def test_device_side_methods_are_declared_not_implemented():
    for method in ("device_delay", "compute_contention", "memory_contention",
                   "communication_contention"):
        config = config_with(injection={"method": method, "delay_us": 100})
        with pytest.raises(BenchmarkUnavailable) as caught:
            straggler.StragglerController(config, 0, 8)
        assert caught.value.status == "not_implemented"
        assert method in str(caught.value)


def test_a_relative_delay_needs_a_baseline():
    config = config_with(injection={"delay_us": None, "delay_ratio": 0.5})
    with pytest.raises(BenchmarkUnavailable) as caught:
        straggler.StragglerController(config, 0, 8)
    assert caught.value.status == "not_applicable"
    config = config_with(injection={"delay_us": None, "delay_ratio": 0.5,
                                    "baseline_us": 2000})
    controller = straggler.StragglerController(config, 0, 8)
    assert controller.before_operation(0)["requested_delay_us"] == 1000


def test_an_unsupported_stage_is_reported_as_such():
    controller = straggler.StragglerController(
        config_with(injection={"method": "stage_delay", "stage": "before_gmm"}), 0, 8)
    record = controller.unsupported_stage()
    assert record["status"] == "unsupported_stage_injection"
    assert record["stage"] == "before_gmm"


# ----------------------------------------------------------- timer placement ---

class FakeClock:
    def __init__(self):
        self.now = 0
        self.events = []

    def __call__(self):
        self.events.append("clock")
        self.now += 1_000_000
        return int(self.now)


def test_the_injection_is_inside_the_timed_region(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(timer, "perf_counter_ns", clock)

    class Context:
        device = type("D", (), {"type": "cpu"})()

        def barrier(self):
            clock.events.append("barrier")

        def synchronize(self):
            clock.events.append("synchronize")

    def operation(_arg):
        clock.events.append("operation")
        return None

    def straggler_hook(iteration):
        clock.events.append(f"inject:{iteration}")

    settings = {"timing": "output_ready", "warmup": 0, "iterations": 1}
    samples = timer.measure(Context(), settings, lambda index: index, operation,
                            straggler=type("S", (), {"before_operation": staticmethod(
                                straggler_hook)})())
    events = clock.events
    assert events.index("inject:0") > events.index("barrier")
    assert events.index("inject:0") > events.index("synchronize")
    assert events.index("inject:0") < events.index("operation")
    assert len(samples) == 1


# ------------------------------------------------------------------ config ---

def test_validation_rejects_a_bad_block():
    config = config_with(selection={"mode": "nonsense"})
    with pytest.raises(ValueError, match="selection.mode"):
        validate(config)
    config = config_with(injection={"method": "nonsense"})
    with pytest.raises(ValueError, match="injection.method"):
        validate(config)
    config = config_with(injection={"delay_us": -1})
    with pytest.raises(ValueError, match="delay_us"):
        validate(config)
    config = config_with(trigger={"probability": 2.0})
    with pytest.raises(ValueError, match="probability"):
        validate(config)


def test_the_straggler_suite_expands_with_a_baseline():
    plan = load_plan(ROOT / "configs" / "benchmark_plan.yaml")
    run = expand(plan, ep_sizes=[8], suites=["straggler"])
    assert run.tasks
    blocks = [task["config"]["benchmark"].get("straggler") or {} for task in run.tasks]
    assert any(block.get("enabled") is False for block in blocks)
    assert any(block.get("enabled") for block in blocks)
    modes = {((block.get("selection") or {}).get("mode")) for block in blocks
             if block.get("enabled")}
    assert {"fixed", "random", "rotating", "server", "hotspot"} <= modes
