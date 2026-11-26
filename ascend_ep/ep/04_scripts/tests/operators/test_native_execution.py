"""Every adapter's representative case: real execution, real tensor checks.

`operator_checks.py` enforces registration, capability and initialisation honesty in
one process. This module proves the remaining two points of the operator contract on
two real ranks:

* a representative case **executes** through the adapter (not just plans), and
* the **tensor output** matches the reference, as recorded by the case validator.

It runs the CLI end to end over the adapters whose constraints allow the CPU device,
so the proof is a real Gloo execution rather than a mocked backend. The adapters that
require an Ascend device cannot be proven here; they are asserted to be *blocked* with
a recorded reason, never reported as passing.
"""
import json
import os
import subprocess
import sys

import pytest

from common.config import load_configs
from common.registry import ROOT, registry
from common.scheduler import make_plan

#: Representative cases: one dispatch, one dispatch+combine round trip, and a full
#: MoE forward+backward. Between them they cover the L1 and L3 timed blocks and the
#: dispatch/combine/GMM validators.
CASES = ("COMM-001", "COMM-004", "MOE-003")


def cpu_capable():
    return sorted(operator_id for operator_id, spec in registry().items()
                  if "cpu" in (spec.constraints.get("devices") or ()))


def _numeric_leaves(value, path=""):
    """Every numeric leaf in a correctness block, with its path.

    L1 records a flat ``{"dispatch_relative_l2": 0.0}``; L3 records per-step blocks with
    a nested ``relative_l2`` mapping. Both shapes carry the same information and both
    are checked, because a validator that recorded no number proves nothing.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _numeric_leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, f"{path}[{index}]")


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    try:
        import torch
    except (ImportError, OSError) as exc:  # pragma: no cover - environment guard
        pytest.skip(f"PyTorch runtime unavailable: {exc}")
    if not torch.distributed.is_gloo_available():  # pragma: no cover - environment guard
        pytest.skip("Gloo is unavailable in this PyTorch build")
    operators = cpu_capable()
    if not operators:  # pragma: no cover - registry guard
        pytest.skip("no adapter allows the CPU device")
    output = tmp_path_factory.mktemp("operators-gloo")
    run_id = "operator-contract"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run_benchmark.py"), "pipeline",
         "--config", str(ROOT / "configs/smoke_cpu.yaml"),
         "--cases", ",".join(CASES), "--operators", ",".join(operators),
         "--output", str(output), "--run-id", run_id],
        cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    report = output / run_id / "reports" / "benchmark_summary.json"
    payload = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
    assert completed.returncode == 0, (completed.stdout, completed.stderr,
                                      payload.get("status_counts"))
    return payload


def test_only_two_adapters_claim_the_cpu_device():
    """A guard on the harness: the CPU proof covers exactly the CPU-capable set."""
    assert cpu_capable() == ["allgather_hccl", "alltoall_hccl"]
    assert len(registry()) == 36


def test_every_representative_case_executes_and_passes(summary):
    assert summary["status_counts"] == {"passed": len(cpu_capable()) * len(CASES)}
    executed = {(row["backend_id"], row["case_id"]) for row in summary["results"]}
    for operator_id in cpu_capable():
        for case_id in CASES:
            assert (operator_id, case_id) in executed, (operator_id, case_id)
    for row in summary["results"]:
        assert row["status"] == "passed", (row["backend_id"], row["case_id"], row["error_reason"])
        assert row["correctness"]["passed"]


def test_the_recorded_tensor_check_is_numeric_and_tight(summary):
    """The adapter produced the reference tensor, not merely a successful call."""
    for row in summary["results"]:
        ranks = row["correctness"]["ranks"]
        assert ranks, row["case_id"]
        for rank in ranks:
            checks = [(path, value) for path, value in _numeric_leaves(rank)
                      if path != "rank" and "l2" in path]
            assert checks, (row["case_id"], rank)
            for path, value in checks:
                assert value == value, f"{row['case_id']} {path} is NaN"  # NaN is never a pass
                assert abs(value) < 0.05, (row["case_id"], path, value)


def test_the_execution_observation_names_the_real_api_and_transport(summary):
    for row in summary["results"]:
        observation = row["observation"]
        assert observation, row["case_id"]
        assert observation["actual_api"], "the adapter did not report its API"
        assert observation["actual_transport"] == "gloo"
        # HCCL does not expose its algorithm; "unknown" is the honest answer and a
        # guessed name would be a fabricated attribution.
        assert observation["actual_algorithm"] == "unknown"
        # A CPU process group is a functional stand-in, and says so.
        assert observation["fallback"]["used"] is True
        assert "functional stand-in" in observation["fallback"]["reason"]


def test_wire_bytes_are_not_fabricated(summary):
    for row in summary["results"]:
        byte_block = row["observation"]["bytes"]
        assert byte_block["logical_payload"] > 0
        assert byte_block["remote_payload"] is not None
        # No link counter and no profiler counter were collected, so the wire volume
        # stays unavailable.  An L3 row has no bandwidth record at all, which also
        # means nothing invented a physical bandwidth for it.
        assert byte_block["measured_wire"] is None
        assert byte_block["cross_card_source"] == "remote_payload_model"
        bandwidth = row.get("bandwidth")
        if bandwidth is not None:
            assert bandwidth["physical_bandwidth_GBps"] is None


def test_completion_proves_the_transfer(summary):
    for row in summary["results"]:
        completion = row["completion"]
        assert completion["level"] == "remote_visible"
        assert completion["proves_transfer"] is True
        assert row["handler"]


def test_npu_only_adapters_are_blocked_not_passed():
    """34 adapters need an Ascend device; on a CPU configuration they are unsupported."""
    config = load_configs(ROOT / "configs/smoke_cpu.yaml")
    npu_only = sorted(operator_id for operator_id, spec in registry().items()
                      if "cpu" not in (spec.constraints.get("devices") or ()))
    assert len(npu_only) == 34
    tasks = make_plan(config, operators=npu_only, levels=["L1"], cases=["COMM-001"])
    assert tasks
    for task in tasks:
        assert task["status"] == "unsupported", (task["backend_id"], task["status"])
        assert task["reason"] and "requires" in task["reason"]


def test_a_plan_never_reports_a_pass_before_execution():
    config = load_configs(ROOT / "configs/smoke_cpu.yaml")
    tasks = make_plan(config, operators=cpu_capable(), levels=["L1", "L2", "L3"],
                      cases=["COMM-001", "COMM-004", "MOE-003"])
    assert tasks
    assert all(task["status"] != "passed" for task in tasks)
