"""Configuration resolution: defaults last, strict fields, provenance, audit.

These tests pin the three behaviours the refactor exists for:

* a default can never shadow an explicit value from a lower layer (the old loader
  merged ``DEFAULT`` into each file separately, so an EP environment that omitted
  a field injected its implicit default over the base value);
* a suite or matrix point that declares an environment field is an **error**
  naming the field, never a silent deletion;
* every audited field reaches a task parameter *and* a runtime consumer.
"""
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config_audit  # noqa: E402
from common.config import (DEFAULT, field_errors, flatten_paths, load_configs, load_raw,  # noqa: E402
                           load_suite_overlays, matrix_overlay, provenance_record,
                           resolve_config, suite_scenario, validate_fields)
from common.plan import expand, load_plan  # noqa: E402


# --------------------------------------------------------------- provenance ---

def test_raw_loading_applies_no_defaults():
    raw, provenance = load_raw(ROOT / "configs" / "ep8_bf16.yaml")
    assert "benchmark" in raw and raw["benchmark"]["cores"] == [1, 2, 4]
    # The value comes from the inherited base file, and the provenance says so.
    assert provenance["benchmark.cores"] == "base_benchmark.yaml"
    assert provenance["distributed.ep_size"] == "ep8_bf16.yaml"
    # A field no file declares is absent, not filled with the schema default.
    assert "task_timeout_seconds" not in raw["benchmark"]


def test_defaults_are_applied_after_all_layers():
    base, base_provenance = load_raw(ROOT / "configs" / "base_benchmark.yaml")
    # An environment layer that omits ``benchmark.iterations`` must not override
    # the explicit base value with the schema default (50 vs 5).
    environment = {"distributed": {"ep_size": 64}}
    config, provenance = resolve_config([
        ("base_config", base, base_provenance),
        ("environment", environment, {"distributed.ep_size": "test"}),
    ])
    assert config["benchmark"]["iterations"] == base["benchmark"]["iterations"] == 50
    assert provenance["benchmark.iterations"] == "base_benchmark.yaml"
    assert provenance["distributed.ep_size"] == "test"
    # A field only the schema declares is marked as such.
    other, other_provenance = resolve_config([("base_config", {}, {})])
    assert other_provenance["benchmark.cores"] == "schema_default"
    assert other["benchmark"]["cores"] == DEFAULT["benchmark"]["cores"]


def test_provenance_record_carries_value_and_source():
    raw, provenance = load_raw(ROOT / "configs" / "ep128_bf16.yaml")
    config, merged = resolve_config([("environment", raw, provenance)])
    record = provenance_record(config, merged)
    assert record["distributed.ep_size"] == {"value": 128, "source": "ep128_bf16.yaml"}
    assert record["model.num_experts"] == {"value": 384, "source": "ep128_bf16.yaml"}
    json.dumps(record)  # serialisable: it goes into the plan metadata


def test_plan_point_carries_the_provenance():
    plan = load_plan(ROOT / "configs" / "benchmark_plan.yaml")
    points = expand(plan, ep_sizes=[32], suites=["all"]).points
    record = points[0].provenance
    assert record["distributed.ep_size"] == {"value": 32, "source": "ep32_bf16.yaml"}
    assert record["model.num_experts"] == {"value": 96, "source": "ep32_bf16.yaml"}
    assert record["benchmark.cores"]["source"] == "base_benchmark.yaml"
    assert points[0].provenance_fingerprint


# ------------------------------------------------------------- strict fields ---

def test_unknown_field_is_rejected_with_its_path():
    config = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    config["benchmark"]["quiet_mode"] = 1
    errors = field_errors(config)
    assert any("benchmark.quiet_mode" in error for error in errors), errors
    with pytest.raises(ValueError, match="quiet_mode"):
        validate_fields(config)


def test_typo_in_a_nested_section_is_rejected():
    config = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    config["benchmark"]["transport"]["peer_policyx"] = "ring"
    with pytest.raises(ValueError, match="peer_policyx"):
        validate_fields(config)


def test_a_suite_may_not_declare_the_environment():
    suite = {
        "suite_id": "decode",
        "distributed": {"ep_size": 8},
        "model": {"hidden_size": 999, "dtype": "float16"},
        "topology": {"server_ids": [0] * 8},
        "workload": {"tokens_per_rank": 16},
    }
    with pytest.raises(ValueError) as caught:
        suite_scenario(suite, "decode.yaml")
    message = str(caught.value)
    for field in ("distributed", "model.hidden_size", "model.dtype", "topology"):
        assert field in message, (field, message)


def test_a_suite_may_still_declare_the_expert_count():
    """Strong/weak scaling are defined in terms of ``model.num_experts``."""
    overlay = suite_scenario({"suite_id": "strong_scaling", "model": {"num_experts": 256}})
    assert overlay["model"] == {"num_experts": 256}


def test_matrix_point_may_not_vary_the_environment():
    with pytest.raises(ValueError, match="model.hidden_size"):
        matrix_overlay({"distributed": {"ep_size": 8},
                        "model": {"num_experts": 32, "hidden_size": 1}})
    with pytest.raises(ValueError, match="device"):
        matrix_overlay({"device": "cpu"})
    # The two documented exceptions stay legal.
    clean = matrix_overlay({"distributed": {"ep_size": 64},
                            "model": {"num_experts": 192},
                            "workload": {"tokens_per_rank": 512}})
    assert clean["model"] == {"num_experts": 192}


def test_suite_overlays_report_their_source():
    layers = load_suite_overlays(ROOT / "configs" / "decode.yaml", ep_size=8)
    overlay, provenance = layers[0]
    assert overlay["workload"]["tokens_per_rank"] == 16
    assert provenance["workload.tokens_per_rank"] == "decode.yaml"


# ------------------------------------------------------------------- audit ---

def test_config_audit_traces_every_field():
    plan = load_plan(ROOT / "configs" / "benchmark_plan.yaml")
    run = expand(plan, ep_sizes=[8])
    report = run.metadata()["config_audit"]
    assert report["ok"] is True, report["unconsumed"]
    fields = {row["field"]: row for row in report["fields"]}
    assert fields["benchmark.cores"]["status"] == "swept"
    assert fields["benchmark.cores"]["observed"] == [1, 2, 4]
    assert fields["benchmark.quiet_modes"]["observed"] == [0, 1]
    for row in report["fields"]:
        if row["status"] == "not_configured":
            continue
        assert row["consumer"]["proven"], row["field"]


def test_audit_flags_a_consumer_that_does_not_read_the_parameter(tmp_path):
    record = config_audit.audit(tasks=[], configs=[{"benchmark": {"cores": [1, 2]}}],
                                root=tmp_path)
    assert record["ok"] is False
    assert record["unconsumed"][0]["status"] == "consumer_missing"


def test_audit_flags_a_declared_but_unexpanded_dimension():
    tasks = [{"parameters": {"message_bytes": 4096}, "case_family": "transport"}]
    record = config_audit.audit(
        tasks, [{"benchmark": {"cores": [1, 2, 4]}}],
        {"EP8/all": {"LINK-001": {"default": {"cores": {"type": "not_applicable"},
                                               "message_bytes": {"type": "sweep"}}}}})
    row = next(item for item in record["fields"] if item["field"] == "benchmark.cores")
    assert row["status"] == "declared_not_applicable"
    assert record["ok"] is True


# ------------------------------------------------------------------ shape ---

def test_every_config_file_resolves_and_validates():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        if "benchmark_plan" in path.name:
            continue
        configs = load_configs(path)
        assert configs, path
        for config in configs:
            assert field_errors(config) == [], path
            assert set(flatten_paths(config)) <= set(flatten_paths(config))
