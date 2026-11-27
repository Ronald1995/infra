"""Execution-mode policy: formal tests are NPU-only and never fall back to CPU."""
import pytest

from common import execution_mode


def test_default_mode_is_npu_only():
    assert execution_mode.resolve(environ={}) == "npu_only"
    assert execution_mode.is_formal("npu_only")
    assert not execution_mode.is_formal("framework")


def test_precedence_is_explicit_then_environ_then_default():
    assert execution_mode.resolve("framework", {"EP_EXECUTION_MODE": "npu_only"}) == "framework"
    assert execution_mode.resolve(None, {"EP_EXECUTION_MODE": "framework"}) == "framework"
    assert execution_mode.resolve(None, {}) == "npu_only"


def test_unknown_mode_is_refused_instead_of_downgrading():
    with pytest.raises(execution_mode.FormalModeViolation):
        execution_mode.resolve("cpu", {})


def test_formal_mode_refuses_a_cpu_configuration():
    with pytest.raises(execution_mode.FormalModeViolation) as error:
        execution_mode.assert_formal("npu_only", {"device": "cpu"}, "COMM-001")
    assert "device" in str(error.value)


def test_framework_mode_allows_cpu_because_it_claims_no_operator_evidence():
    execution_mode.assert_formal("framework", {"device": "cpu"}, "self-check")


def test_cpu_pass_is_never_operator_evidence():
    passed_npu = {"device": "npu"}
    assert execution_mode.counts_as_operator_evidence("npu_only", passed_npu, "passed")
    # A CPU pass is a framework check, not an operator result.
    assert not execution_mode.counts_as_operator_evidence("npu_only", {"device": "cpu"}, "passed")
    # A framework-mode pass never counts either, whatever the device.
    assert not execution_mode.counts_as_operator_evidence("framework", passed_npu, "passed")
    # A blocked task is not evidence even when it "did not fail".
    assert not execution_mode.counts_as_operator_evidence("npu_only", passed_npu,
                                                          "dependency_missing")


def test_preflight_distinguishes_a_hardware_shortfall_from_a_broken_environment():
    short = execution_mode.preflight(device_count=8, requested_ranks=16)
    assert short["status"] == "environment_unavailable"
    assert short["blocked_kind"] == "hardware_shortfall"
    assert "never replaced by CPU" in short["reason"]

    unreachable = execution_mode.preflight(device_count=8, requested_ranks=8,
                                           device_reachable=False)
    assert unreachable["blocked_kind"] == "environment"

    ready = execution_mode.preflight(device_count=8, requested_ranks=8)
    assert ready["status"] is None and ready["blocked_kind"] is None


def test_describe_names_the_policy():
    assert "NPU_ONLY" in execution_mode.describe("npu_only")
    assert "never count" in execution_mode.describe("framework")
