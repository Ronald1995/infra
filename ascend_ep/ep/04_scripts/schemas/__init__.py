"""Versioned result vocabulary and shared status definitions."""
STATUSES = (
    "passed", "unsupported", "dependency_missing", "environment_unavailable", "source_unavailable",
    "not_implemented", "correctness_failed", "runtime_failed", "timeout", "oom", "not_run",
)
RETRYABLE = {"not_run", "runtime_failed", "timeout", "oom", "environment_unavailable",
             "dependency_missing", "source_unavailable"}


def result(manifest, task, status="not_run", reason=None):
    if status not in STATUSES:
        raise ValueError(f"invalid benchmark status {status!r}")
    return {
        "schema_version": 1, "run_id": manifest["run_id"], "task_id": task["task_id"],
        "case_id": task["case_id"], "suite_id": task["config"]["suite_id"],
        "backend_id": task["backend_id"], "benchmark_level": task["level"],
        "operation": task["operation"], "status": status, "source": "measured",
        "environment": manifest["environment"], "topology": task["config"]["topology"],
        "source_versions": task["source_versions"], "config": task["config"],
        "workload": None, "routing_trace_hash": None, "correctness": None,
        "raw_samples": [], "latency": None, "bandwidth": None, "throughput": None,
        "compute": None, "overlap": None, "memory": None, "load_balance": None,
        "profiling_artifacts": [], "logs": [], "error_reason": reason,
        "case_parameters": task["parameters"],
    }
