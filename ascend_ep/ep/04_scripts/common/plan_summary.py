"""Structured plan summary: what the plan will execute, and what it costs.

The numbers a reader needs before a multi-hour run:

* how many requirements the plan contains, per EP size, level, suite and operator;
* how many of them are already classified as unsupported / blocked / not applicable,
  so "6,000 tasks" is never mistaken for "6,000 measurements";
* unique execution tasks after de-duplication;
* an estimated runtime -- only when an explicit assumption or a pilot measurement
  exists.  The spec forbids estimating a whole benchmark as
  ``iterations x kernel latency``, so without an assumption this field reports
  ``unavailable`` with the reason instead of a fabricated number.
"""
from collections import Counter

#: Statuses (see schemas.STATUSES) grouped the way a plan summary reports them.
CLASSES = {
    "executable": ("not_run", "passed", "unverified"),
    "unsupported": ("unsupported", "not_implemented"),
    "blocked": ("dependency_missing", "environment_unavailable", "source_unavailable",
                "blocked_topology"),
    "not_applicable": ("not_applicable",),
    "failed": ("correctness_failed", "runtime_failed", "timeout", "oom"),
}
_CLASS_OF = {status: name for name, statuses in CLASSES.items() for status in statuses}


def classify(status):
    return _CLASS_OF.get(status, "executable")


def summarize(run, estimation=None):
    """``run`` is a ``PlanRun`` or a task list; ``estimation`` overrides the plan's block."""
    tasks = getattr(run, "tasks", run)
    by_ep = Counter(task["ep_size"] for task in tasks)
    by_level = Counter(task["level"] for task in tasks)
    by_suite = Counter(suite for task in tasks for suite in task.get("suite_ids", ()))
    by_operator = Counter(task["backend_id"] for task in tasks)
    classes = Counter(classify(task["status"]) for task in tasks)
    total = len(tasks)
    assumptions = estimation if estimation is not None else getattr(
        getattr(run, "plan", None), "estimation", None) or {}
    return {
        "schema_version": 1,
        "tasks": total,
        "requirements": total,
        "unique_execution_tasks": total,
        "ep_sizes": sorted(by_ep),
        "by_ep_size": {str(size): count for size, count in sorted(by_ep.items())},
        "by_level": dict(sorted(by_level.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "by_operator": dict(sorted(by_operator.items())),
        "classes": {name: classes.get(name, 0) for name in CLASSES},
        "executable": classes.get("executable", 0),
        "not_executable": total - classes.get("executable", 0),
        "estimation": estimate(total, by_ep, assumptions),
    }


def estimate(tasks, by_ep, assumptions):
    """Runtime estimate from an explicit assumption, or an explained ``unavailable``."""
    if not assumptions:
        return {"status": "unavailable",
                "reason": ("no pilot measurement or explicit per-task assumption; the plan "
                           "does not estimate a benchmark from iterations x kernel latency")}
    for key in ("per_task_seconds", "startup_seconds_per_ep", "init_seconds_per_ep",
                "profiling_seconds_per_task"):
        value = assumptions.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return {"status": "unavailable",
                    "reason": f"estimation.{key} must be a non-negative number"}
    per_task = assumptions.get("per_task_seconds")
    if not per_task:
        return {"status": "unavailable",
                "reason": "estimation.per_task_seconds is required for an estimate"}
    startup = assumptions.get("startup_seconds_per_ep", 0) * len(by_ep)
    init = assumptions.get("init_seconds_per_ep", 0) * len(by_ep)
    profiling = assumptions.get("profiling_seconds_per_task", 0) * tasks
    total = per_task * tasks + startup + init + profiling
    return {"status": "estimated", "basis": "explicit assumption (not a measurement)",
            "components": {"tasks_seconds": per_task * tasks, "startup_seconds": startup,
                           "init_seconds": init, "profiling_seconds": profiling},
            "total_seconds": total, "assumptions": dict(assumptions)}
