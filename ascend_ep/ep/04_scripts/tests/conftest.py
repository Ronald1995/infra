"""Shared builders for the reporting tests.

The tests construct a *run directory* the same way the benchmark does -- manifest,
result records, then the real ``analyze`` + ``report`` entry points -- so a report bug
shows up here before it reaches a device run.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.generate_report import analyze_run
from common.cases import catalog
from common.config import DEFAULT, merge, validate
from common.plan import matrix_metadata
from common.storage import write_json
from reporting.report import report
from schemas import result

#: A deterministic environment block for one synthetic run.
NPU_ENVIRONMENT = {
    "host": "test-host", "platform": "Linux-5.10-aarch64", "python": "3.10.8",
    "packages": {"torch": "2.6.0", "torch-npu": "2.6.0", "PyYAML": "6.0.1"},
    "cann_version": "8.2.RC1", "cann_version_source": "env:ASCEND_TOOLKIT_VERSION",
    "cann_version_error": None, "driver_version": "24.1.0",
    "driver_version_source": "npu-smi info", "npu_models": ["Ascend910B4"],
    "git_commit": "deadbeefcafe", "git_changes": None, "device_visibility": "0,1,2,3",
}

CPU_ENVIRONMENT = {
    **NPU_ENVIRONMENT, "host": "cpu-host", "npu_models": None,
    "cann_version": None, "cann_version_source": "unavailable",
    "cann_version_error": "ASCEND_TOOLKIT_VERSION is unset", "driver_version": None,
    "driver_version_source": "unavailable", "packages": {"torch": "2.6.0+cpu"},
}


@pytest.fixture
def config_for():
    """``config_for(device, ep_size, **overrides)`` -> a validated configuration."""

    def build(device="cpu", ep_size=2, **overrides):
        base = {
            "device": device,
            "distributed": {"ep_size": ep_size, "tp_size": 1},
            "model": {"dtype": "bfloat16", "num_experts": 8, "top_k": 2,
                      "hidden_size": 128, "intermediate_size": 64},
            "workload": {"tokens_per_rank": 8, "routing": "balanced", "seed": 42},
            "benchmark": {"timing": "output_ready"},
        }
        for key, value in overrides.items():
            base[key] = merge(base.get(key) or {}, value) if isinstance(value, dict) else value
        return validate(merge(DEFAULT, base))

    return build


@pytest.fixture
def task_for(config_for):
    """``task_for(case_id, ...)`` -> one manifest task shaped like a real plan item."""

    def build(case_id, *, backend_id=None, ep_size=2, device="cpu", parameters=None,
              suite_ids=("test",), status="not_run", reason=None, index=0, config=None):
        case = catalog()[case_id]
        settings = deepcopy(config) if config is not None else config_for(
            device=device, ep_size=ep_size)
        backend = backend_id or ("hardware" if case.level == "L0" else "alltoall_hccl")
        params = dict(parameters) if parameters is not None else {"variant": "default"}
        return {
            "task_id": f"{case_id}-{backend}-{ep_size}-{index:03d}",
            "case_id": case_id, "level": case.level, "operation": case.operation,
            "backend_id": backend, "config": settings, "parameters": params,
            "status": status, "reason": reason, "source_versions": {"adapter_commit": "abc123"},
            "baseline": case.level == "L0", "ep_size": ep_size, "suite_ids": list(suite_ids),
            "case_category": case.category, "case_family": case.category,
            "case_tags": list(case.tags), "case_phase": case.phase, "handler": case.handler,
            "correctness_requirements": list(case.correctness), "alias_of": case.alias_of,
            "config_fingerprint": f"fingerprint-{case_id}-{backend}-{ep_size}",
        }

    return build


@pytest.fixture
def manifest_for():
    """``manifest_for(tasks, ...)`` -> a manifest with real matrix metadata."""

    def build(tasks, run_id="test-run", environment=None, matrix=None, report_options=None):
        metadata = matrix_metadata(list(tasks)) if matrix is None else dict(matrix)
        if report_options is not None:
            metadata["report"] = {**(metadata.get("report") or {}), **report_options}
        return {
            "schema_version": 1, "run_id": run_id,
            "environment": dict(environment if environment is not None
                                else (NPU_ENVIRONMENT if any(
                                    task["config"]["device"] == "npu" for task in tasks)
                                    else CPU_ENVIRONMENT)),
            "matrix": metadata, "tasks": list(tasks),
        }

    return build


@pytest.fixture
def record_for():
    """``record_for(manifest, task, ...)`` -> a result record with synthetic metrics."""

    def build(manifest, task, status="passed", *, p50=None, p95=None, p99=None,
              bandwidth=True, compute=False, throughput=None, correctness=True,
              latencies=None, extra=None):
        record = result(manifest, task, status)
        if status != "passed":
            record["correctness"] = None
            record.update(extra or {})
            return record
        p50 = 1.0 if p50 is None else p50
        record["latency"] = {
            "unit": "ms", "scope": "per_iteration_rank_max", "source": "measured",
            "p50": p50, "p95": p95 if p95 is not None else p50 * 1.1,
            "p99": p99 if p99 is not None else p50 * 1.15,
            "rank_min": latencies or [p50 * .95], "rank_mean": latencies or [p50],
            "rank_max": latencies or [p50], "rank_avg_p50_ms": p50 * .98,
            "cv": .02, "p95_p50_ratio": 1.1, "host_issue_p50_ms": p50 * .1,
            "host_wait_p50_ms": p50 * .05,
        }
        record["correctness"] = {"passed": True} if correctness else {"passed": False}
        record["source"] = "measured"
        if bandwidth:
            record["bandwidth"] = {
                "unit": "bytes", "bandwidth_unit": "GB/s", "scope": "per_rank",
                "source": "routing_trace_model", "payload_bytes": 4096,
                "payload_bandwidth_GBps": 4096 / (p50 * 1e6),
            }
        if compute:
            record["compute"] = {"source": "effective_expert_matrix_math",
                                 "effective_flops": 1e9, "effective_tflops": 1e9 / (p50 * 1e9)}
        if throughput is not None:
            record["throughput"] = {"global_source_tokens_per_second": throughput,
                                    "unit": "tokens/s", "scope": "EP group"}
        record["completion"] = {"level": "remote_visible", "detail": "test",
                                "proves_transfer": True}
        record.update(extra or {})
        return record

    return build


@pytest.fixture
def run_report(tmp_path):
    """Write a run directory, analyse it and generate every report."""

    def build(tasks, rows, run_id="test-run", environment=None, matrix=None,
              capabilities=None):
        root = Path(tmp_path) / run_id
        root.mkdir(parents=True, exist_ok=True)
        manifest = _manifest(tasks, run_id, environment, matrix)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for task, row in zip(tasks, rows):
            folder = "baseline_results" if task.get("baseline") else "operators"
            directory = root / folder / task["backend_id"] / task["task_id"]
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "metrics.json", row)
        analysis = analyze_run(root, list(rows))
        destination = report(root, manifest, list(rows), analysis, capabilities=capabilities)
        return SimpleNamespace(root=root, manifest=manifest, rows=list(rows),
                               destination=destination, analysis=analysis,
                               text=(destination / "benchmark_report.md").read_text(encoding="utf-8"))

    def _manifest(tasks, run_id, environment, matrix):
        metadata = matrix_metadata(list(tasks)) if matrix is None else dict(matrix)
        return {"schema_version": 1, "run_id": run_id,
                "environment": dict(environment or NPU_ENVIRONMENT),
                "matrix": metadata, "tasks": list(tasks)}

    return build


@pytest.fixture
def l0_matrix():
    """A capability matrix that supports one mechanism and rejects another."""
    return {
        "schema_version": 1, "source": "test",
        "channels": [
            {"channel": "UB Memory", "execution_resource": "AIV", "mechanisms": ["ub_memory"],
             "supported": True, "confidence": "declared", "reason": "test fixture"},
            {"channel": "URMA", "execution_resource": "AICPU_TS", "mechanisms": ["urma"],
             "supported": False, "confidence": "declared", "reason": "not present on this SoC"},
        ],
        "engines": [
            {"engine": "CCU", "execution_resource": "CCU", "mechanisms": ["ub_ctp"],
             "supported": True, "confidence": "declared", "reason": "test fixture"},
        ],
        "accelerators": [
            {"mode": "aiv", "execution_resource": "AIV", "supported": True, "reason": "probed"},
        ],
        "probes": [{"name": "transport_bridge", "ok": True, "detail": {}}],
    }
