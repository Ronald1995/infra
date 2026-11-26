"""Execution profiles and topology provenance.

A profile decides *how much* of the declared matrix one invocation executes; it must
never look like a different matrix.  These tests pin the difference between the four
profiles, the dimension policy they set, and the five-state topology vocabulary.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import profiles  # noqa: E402
from common.config import load_configs  # noqa: E402
from common.plan import (TOPOLOGY_SOURCES, expand, load_plan,  # noqa: E402
                         normalize_source)


@pytest.fixture(scope="module")
def plan():
    return load_plan(ROOT / "configs" / "benchmark_plan.yaml")


def test_every_profile_is_declared_and_resolvable():
    assert profiles.PROFILE_NAMES == ("smoke", "core", "full", "exhaustive")
    for name in profiles.PROFILE_NAMES:
        resolved = profiles.resolve(name, ["all", "hardware", "decode"],
                                    ["L0", "L1", "L2"])
        assert resolved["profile"] == name
        assert resolved["suites"]
        assert resolved["dimension_policy"] in ("all", "first")


def test_smoke_is_one_case_per_level_and_one_dimension_value(plan):
    run = expand(plan, ep_sizes=[8], profile="smoke")
    assert run.profile["profile"] == "smoke"
    levels = {}
    for task in run.tasks:
        levels.setdefault(task["level"], set()).add(task["case_id"])
    assert set(levels) <= {"L0", "L1"}
    for level, cases in levels.items():
        assert len(cases) == 1, (level, cases)
    for task in run.tasks:
        parameters = task["parameters"]
        if isinstance(parameters.get("message_bytes"), int):
            assert task["config"]["benchmark"]["dimension_policy"] == "first"


def test_smoke_sweeps_one_value_per_dimension():
    config = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    config["benchmark"]["dimension_policy"] = "first"
    from common.case_spec import spec
    from common.variants import case_dimensions, expand_variants

    dimensions = case_dimensions(spec("LINK-013"), config["benchmark"], 8, "device")
    assert dimensions["cores"]["values"] == [1]
    assert dimensions["quiet_mode"]["values"] == [0]
    assert dimensions["message_bytes"]["values"] == [4096]
    assert dimensions["cores"]["policy"] == "first"
    sets = expand_variants(spec("LINK-013"), config["benchmark"], 8)
    assert len({item["message_bytes"] for item in sets}) == 1


def test_core_covers_all_ep_sizes_and_standard_suites_but_not_specialised(plan):
    run = expand(plan, ep_sizes=[8, 16, 32, 64, 128], profile="core")
    suites = {task["config"]["suite_id"] for task in run.tasks}
    assert "decode" in suites and "transport" in suites
    assert "routing_imbalance" not in suites
    assert "straggler" not in suites and "long_tail" not in suites
    assert len({task["ep_size"] for task in run.tasks}) == 5
    assert len({task["case_id"] for task in run.tasks}) > 30


def test_full_includes_the_specialised_suites(plan):
    run = expand(plan, ep_sizes=[8], profile="full")
    suites = {task["config"]["suite_id"] for task in run.tasks}
    assert {"routing_imbalance", "straggler", "long_tail", "topology", "dedup",
            "quantized"} <= suites


def test_exhaustive_expands_more_than_full(plan):
    full = expand(plan, ep_sizes=[8], profile="full")
    exhaustive = expand(plan, ep_sizes=[8], profile="exhaustive")
    assert len(exhaustive.tasks) >= len(full.tasks)
    assert exhaustive.profile["aliases"] == "run"


def test_an_unknown_profile_is_rejected(plan):
    with pytest.raises(ValueError, match="execution profile"):
        expand(plan, ep_sizes=[8], profile="everything")


def test_metadata_records_the_profile(plan):
    run = expand(plan, ep_sizes=[8], profile="smoke")
    metadata = run.metadata()
    assert metadata["execution_profile"]["profile"] == "smoke"
    assert metadata["execution_profile"]["dimension_policy"] == "first"


def test_the_plan_declares_its_default_profile(plan):
    assert plan.profile == "full"
    assert plan.metadata()["profile"] == "full"


# ------------------------------------------------------------------ topology ---

def test_the_topology_vocabulary_has_five_states():
    assert TOPOLOGY_SOURCES == ("discovered", "user_provided", "inferred",
                                "synthetic", "unknown")


@pytest.mark.parametrize("value,expected", [
    ("discovered", "discovered"), ("user_provided", "user_provided"),
    ("inferred", "inferred"), ("synthetic", "synthetic"),
    ("environment", "user_provided"), ("explicit", "user_provided"),
    ("plan", "synthetic"), ("unavailable", "unknown"), (None, "unknown"),
    ("something else", "unknown")])
def test_legacy_and_unknown_sources_are_normalised(value, expected):
    assert normalize_source(value) == expected


def test_a_plan_pattern_is_synthetic_and_unverified(plan):
    run = expand(plan, ep_sizes=[8], profile="core")
    topology = run.points[0].config["topology"]
    assert topology["source"] == "synthetic"
    assert topology["verified"] is False
    record = run.metadata()["topology"]["8"]
    assert record["source"] == "synthetic"


def test_an_environment_map_is_user_provided(tmp_path):
    environment = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    environment["topology"] = {"server_ids": [0] * 8}
    # A directly composed config keeps the normalisation contract.
    from common.plan import _apply_topology

    class FakePlan:
        topology = {"server_size": 4}

    composed = _apply_topology(FakePlan(), environment, 8)
    assert composed["topology"]["source"] == "user_provided"
    assert composed["topology"]["server_ids"] == [0] * 8


def test_without_any_topology_the_source_is_unknown():
    from common.plan import _apply_topology

    config = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    config.pop("topology", None)

    class FakePlan:
        topology = {}

    composed = _apply_topology(FakePlan(), config, 8)
    assert composed["topology"]["source"] == "unknown"
    assert composed["topology"]["verified"] is False
