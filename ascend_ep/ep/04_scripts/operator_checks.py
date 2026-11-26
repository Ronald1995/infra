"""Shared contract assertions for the operator directories."""
from pathlib import Path

from common.config import load_configs
from common.registry import registry
from common.scheduler import make_plan


def check_capabilities(operator_id):
    from common.backend import EPBackend
    spec = registry()[operator_id]
    assert spec.api and spec.source.startswith("https://")
    assert spec.create().capabilities() == spec.capabilities
    assert isinstance(spec.create(), EPBackend)
    assert spec.capabilities or spec.reason


def check_correctness_entry(operator_id):
    path = Path(__file__).parent / "operators" / operator_id / "config.yaml"
    plan = make_plan(load_configs(path), levels=["L3"], cases=["MOE-001"])
    assert len(plan) == 1 and plan[0]["backend_id"] == operator_id
    assert plan[0]["reason"] is None or plan[0]["status"] != "passed"
