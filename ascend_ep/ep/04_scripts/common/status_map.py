"""Report classification: raw task status -> the delivery bucket.

``schemas.STATUSES`` is the *raw* vocabulary written into ``metrics.json`` and it
stays untouched.  This module is the single place that maps a raw status onto the
bucket a delivery report shows, so "the dependency was missing", "this host has
too few cards" and "the run crashed" never collapse into one undifferentiated
"failed" count.

Buckets (delivery vocabulary)
    ``PASSED``            a real measurement exists
    ``FAILED``            the task ran and did not pass
    ``BLOCKED_DEPENDENCY`` a Python package, native source or extension is missing
    ``BLOCKED_HARDWARE``  the host cannot supply the devices this point needs
    ``BLOCKED_PLATFORM``  the adapter or the built artifact does not support this
                          device generation; running it elsewhere will not help
    ``BLOCKED_ENVIRONMENT`` the platform can, but the environment could not be
                          prepared (device unreachable, runtime error, licence)
    ``BUILD_FAILED``      a kernel/extension build was attempted and failed
    ``SKIPPED_BY_USER``   excluded by the request, not by the environment
    ``UNKNOWN``           not classifiable; treated as "not passed" everywhere
"""
PASSED = "PASSED"
FAILED = "FAILED"
BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
BLOCKED_HARDWARE = "BLOCKED_HARDWARE"
BLOCKED_PLATFORM = "BLOCKED_PLATFORM"
BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
BUILD_FAILED = "BUILD_FAILED"
SKIPPED_BY_USER = "SKIPPED_BY_USER"
UNKNOWN = "UNKNOWN"

BUCKETS = (PASSED, FAILED, BLOCKED_DEPENDENCY, BLOCKED_HARDWARE, BLOCKED_PLATFORM,
           BLOCKED_ENVIRONMENT, BUILD_FAILED, SKIPPED_BY_USER, UNKNOWN)

#: Buckets that are a delivery result ("the task was attempted").
ATTEMPTED = (PASSED, FAILED)
#: Buckets that mean "the point could not be run here, for a recorded reason".
BLOCKED = (BLOCKED_DEPENDENCY, BLOCKED_HARDWARE, BLOCKED_PLATFORM,
           BLOCKED_ENVIRONMENT, BUILD_FAILED, UNKNOWN)

#: Raw statuses and the bucket they map to when no stronger evidence exists.
_STATUS_BUCKET = {
    "passed": PASSED,
    "correctness_failed": FAILED,
    "runtime_failed": FAILED,
    "timeout": FAILED,
    "oom": FAILED,
    "unverified": FAILED,
    "dependency_missing": BLOCKED_DEPENDENCY,
    "not_implemented": BLOCKED_DEPENDENCY,
    "build_failed": BUILD_FAILED,
    "environment_unavailable": BLOCKED_ENVIRONMENT,
    "blocked_topology": BLOCKED_ENVIRONMENT,
    "unsupported": BLOCKED_PLATFORM,
    "not_applicable": BLOCKED_PLATFORM,
    "skipped_by_user": SKIPPED_BY_USER,
    "missing": UNKNOWN,
    "not_run": UNKNOWN,
}

#: Raw statuses and the bucket they map to when no stronger evidence exists.
_STATUS_BUCKET = {
    "passed": PASSED,
    "correctness_failed": FAILED,
    "runtime_failed": FAILED,
    "timeout": FAILED,
    "oom": FAILED,
    "unverified": FAILED,
    "dependency_missing": BLOCKED_DEPENDENCY,
    # The pinned source checkout is a dependency: either it is absent (nothing
    # configured) or it is at the wrong revision.  Neither is fixed by waiting for
    # a different device generation, so it is not BLOCKED_PLATFORM.
    "source_unavailable": BLOCKED_DEPENDENCY,
    "not_implemented": BLOCKED_DEPENDENCY,
    "build_failed": BUILD_FAILED,
    "environment_unavailable": BLOCKED_ENVIRONMENT,
    "blocked_topology": BLOCKED_ENVIRONMENT,
    # The adapter declares the capability absent, or the built artifact targets
    # other silicon: running it here again cannot help.
    "unsupported": BLOCKED_PLATFORM,
    "not_applicable": BLOCKED_PLATFORM,
    "skipped_by_user": SKIPPED_BY_USER,
    "missing": UNKNOWN,
    "not_run": UNKNOWN,
}


def bucket(status, reason="", blocked_kind=None, build_result=False):
    """Classify one outcome.

    ``blocked_kind`` comes from the preflight (``"hardware_shortfall"``,
    ``"environment"``) and takes precedence over the raw status, because
    ``environment_unavailable`` alone cannot distinguish "too few cards" from
    "the runtime is broken".

    ``build_result`` marks a task whose blocker is a failed kernel/extension
    build, which is a build outcome rather than a dependency edge.

    ``reason`` is accepted for the report layer, which always has one, but the
    classification never depends on wording: a status whose meaning is ambiguous
    carries an explicit ``blocked_kind`` instead.
    """
    if build_result and status != "passed":
        return BUILD_FAILED
    if blocked_kind == "hardware_shortfall":
        return BLOCKED_HARDWARE
    return _STATUS_BUCKET.get(status, UNKNOWN)


def is_pass(value):
    return value == PASSED


def is_attempted(value):
    return value in ATTEMPTED


def summarise(pairs):
    """Count buckets from an iterable of ``(status, reason, blocked_kind)`` tuples."""
    counts = {name: 0 for name in BUCKETS}
    for item in pairs:
        status, reason, kind = (list(item) + [None, None, None])[:3]
        counts[bucket(status, reason or "", kind)] += 1
    return counts
