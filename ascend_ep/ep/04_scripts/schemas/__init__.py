"""Versioned result vocabulary and shared status definitions."""
STATUSES = (
    "passed", "unsupported", "dependency_missing", "environment_unavailable", "source_unavailable",
    "not_implemented", "correctness_failed", "runtime_failed", "timeout", "oom", "not_run",
    "not_applicable",
)
RETRYABLE = {"not_run", "runtime_failed", "timeout", "oom", "environment_unavailable",
             "dependency_missing", "source_unavailable"}

#: Statuses that mean "the case ran and passed", i.e. a real measurement exists.
MEASURED = {"passed"}
#: Statuses that mean "the plan could not run this point, for a recorded reason".
NOT_RUN = {"unsupported", "dependency_missing", "environment_unavailable", "source_unavailable",
           "not_implemented", "not_run", "not_applicable"}
#: Severity order from best to worst: the coverage matrix keeps the *highest*
#: index in a cell, so a failed attempt is never hidden by a passed one.
SEVERITY = ("passed", "not_run", "not_applicable", "unsupported", "source_unavailable",
            "dependency_missing", "environment_unavailable", "not_implemented",
            "oom", "timeout", "correctness_failed", "runtime_failed")


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
        # What actually ran and what it may claim: API, algorithm, transport,
        # fallback state, byte accounting and the completion guarantee.
        "observation": None, "completion": None, "handler": None,
    }
