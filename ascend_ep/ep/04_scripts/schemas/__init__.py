"""Versioned result vocabulary and shared status definitions.

Two vocabularies live here and nowhere else:

``STATUSES``
    the *raw* outcome of one task.  A report never renames or overwrites it.
``STATUS_GROUPS``
    the presentation buckets the summary tables count.  A bucket is a
    *classification of* a raw status, never a replacement: the raw status keeps
    being written to ``metrics.json`` and every detail table shows it.

The same module builds the initial result record.  It is the only place that
decides which metadata a record carries, so a task that failed before a single
measurement exists still arrives downstream with its case category, its
parameters, its level, its backend and its EP size.  That is what lets the L0
report show a failed transport case in the transport table instead of dropping it.
"""
STATUSES = (
    "passed", "unsupported", "dependency_missing", "environment_unavailable", "source_unavailable",
    "not_implemented", "correctness_failed", "runtime_failed", "timeout", "oom", "not_run",
    "not_applicable", "blocked_topology", "unverified",
)
RETRYABLE = {"not_run", "runtime_failed", "timeout", "oom", "environment_unavailable",
             "dependency_missing", "source_unavailable"}

#: Statuses that mean "the case ran and passed", i.e. a real measurement exists.
MEASURED = {"passed"}
#: Statuses that mean "the plan could not run this point, for a recorded reason".
NOT_RUN = {"unsupported", "dependency_missing", "environment_unavailable", "source_unavailable",
           "not_implemented", "not_run", "not_applicable", "blocked_topology"}
#: Statuses that mean "the case was attempted and failed".
FAILED = {"correctness_failed", "runtime_failed", "timeout", "oom"}
#: Severity order from best to worst: the coverage matrix keeps the *highest*
#: index in a cell, so a failed attempt is never hidden by a passed one.
SEVERITY = ("passed", "not_run", "not_applicable", "unsupported", "source_unavailable",
            "dependency_missing", "environment_unavailable", "not_implemented",
            "oom", "timeout", "correctness_failed", "runtime_failed")

#: Report-level status buckets.  Order is the order a summary table shows them.
#: ``Missing`` is not a task status: it is the state of a *planned* task that has
#: no result record at all, so it can only be produced by the report layer.
STATUS_GROUPS = (
    ("Passed", ("passed",)),
    ("Unverified", ("unverified",)),
    ("Failed", ("correctness_failed", "runtime_failed", "timeout", "oom")),
    ("Unsupported", ("unsupported", "not_implemented")),
    ("Dependency Missing", ("dependency_missing",)),
    ("Environment Unavailable", ("environment_unavailable",)),
    ("Source Unavailable", ("source_unavailable",)),
    ("Not Run", ("not_run",)),
    ("Not Applicable", ("not_applicable",)),
    ("Blocked Topology", ("blocked_topology",)),
    ("Missing", ("missing",)),
)
_STATUS_TO_GROUP = {status: label for label, statuses in STATUS_GROUPS for status in statuses}
_GROUP_ORDER = {label: index for index, (label, _statuses) in enumerate(STATUS_GROUPS)}


def status_group(status):
    """The report bucket for one raw status, or ``None`` when it is unknown.

    An unknown status is *not* silently mapped onto ``Not Run``: the caller has to
    deal with it, which is how a new status fails loudly instead of being hidden in
    a bucket that means something else.
    """
    return _STATUS_TO_GROUP.get(status)


def group_index(label):
    """Stable ordering key for a status bucket label."""
    return _GROUP_ORDER.get(label, len(_GROUP_ORDER))


def group_totals(statuses):
    """``{bucket: count}`` for an iterable of raw statuses, in display order."""
    counts = {label: 0 for label, _statuses in STATUS_GROUPS}
    for status in statuses:
        label = status_group(status)
        if label is None:
            counts.setdefault(status, 0)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _case_category(manifest, task):
    """The case category, from the manifest/task metadata or the case catalog.

    ``common.scheduler`` stamps ``case_category`` on every planned task, so the
    common path needs no catalog lookup.  A hand-built task or a manifest written
    by an older version falls back to the catalog, and only a case that is in
    neither is reported as ``None`` (never guessed from the ID text).
    """
    category = task.get("case_category")
    if category:
        return category
    case_id = task.get("case_id")
    if not case_id:
        return None
    try:
        from common.cases import catalog
        case = catalog().get(case_id)
    except Exception:  # a broken catalog must not stop a result from being written
        return None
    return getattr(case, "category", None)


def _ep_size(task):
    """The EP size of the resolved configuration, which is what actually ran.

    ``task["ep_size"]`` is the planned size and the two agree in a real run; the
    configuration wins because it is the value the distributed group was created
    from.
    """
    distributed = (task.get("config") or {}).get("distributed") or {}
    return distributed.get("ep_size") or task.get("ep_size")


def result(manifest, task, status="not_run", reason=None):
    """Build the initial result record for one planned task.

    Every field a report may group, filter or attribute by is filled here, before
    the case gets a chance to fail, so a failure keeps the same identity as the
    point that was planned.
    """
    if status not in STATUSES:
        raise ValueError(f"invalid benchmark status {status!r}")
    return {
        "schema_version": 1, "run_id": manifest["run_id"], "task_id": task["task_id"],
        "case_id": task["case_id"], "suite_id": task["config"]["suite_id"],
        "backend_id": task["backend_id"], "benchmark_level": task["level"],
        "operation": task["operation"], "category": _case_category(manifest, task),
        "ep_size": _ep_size(task),
        "status": status, "source": "measured",
        "environment": manifest["environment"], "topology": task["config"]["topology"],
        "source_versions": task["source_versions"], "config": task["config"],
        "workload": None, "routing_trace_hash": None, "correctness": None,
        "raw_samples": [], "latency": None, "bandwidth": None, "throughput": None,
        "compute": None, "overlap": None, "memory": None, "load_balance": None,
        "profiling_artifacts": [], "logs": [], "error_reason": reason,
        # The single standard parameter field.  Older records may still carry the
        # legacy `parameters` key; the reading layer accepts it, nothing new writes it.
        "case_parameters": dict(task.get("parameters") or {}),
        # What actually ran and what it may claim: API, algorithm, transport,
        # fallback state, byte accounting and the completion guarantee.
        "observation": None, "completion": None, "handler": None,
    }
