"""Report buckets: blocked is not failed, and hardware is not environment."""
from common import status_map


def test_pass_and_failure_map_directly():
    assert status_map.bucket("passed") == status_map.PASSED
    for status in ("correctness_failed", "runtime_failed", "timeout", "oom"):
        assert status_map.bucket(status) == status_map.FAILED


def test_dependency_blockers_are_not_failures():
    assert status_map.bucket("dependency_missing") == status_map.BLOCKED_DEPENDENCY
    assert status_map.bucket("not_implemented") == status_map.BLOCKED_DEPENDENCY


def test_hardware_shortfall_wins_over_the_generic_environment_status():
    # The raw status cannot express "too few cards", so the preflight kind does.
    assert status_map.bucket("environment_unavailable",
                             blocked_kind="hardware_shortfall") == status_map.BLOCKED_HARDWARE
    assert status_map.bucket("environment_unavailable",
                             blocked_kind="environment") == status_map.BLOCKED_ENVIRONMENT


def test_platform_restriction_is_distinguished_from_a_missing_package():
    assert status_map.bucket("unsupported", "the op is compiled for Ascend 910_93") == \
        status_map.BLOCKED_PLATFORM
    assert status_map.bucket("unsupported", "operation requires combine") == \
        status_map.BLOCKED_PLATFORM


def test_a_pinned_source_problem_is_a_dependency_blocker():
    # "wrong revision" and "not configured" are both dependency edges; neither
    # becomes true by waiting for different silicon.
    assert status_map.bucket("source_unavailable", "no checkout configured") == \
        status_map.BLOCKED_DEPENDENCY
    assert status_map.bucket("source_unavailable",
                             "expected an unchanged ascendep 24ff84d, found other") == \
        status_map.BLOCKED_DEPENDENCY


def test_a_failed_build_is_its_own_bucket():
    assert status_map.bucket("dependency_missing", "cmake failed", build_result=True) == \
        status_map.BUILD_FAILED
    assert status_map.bucket("passed", build_result=True) == status_map.PASSED


def test_unknown_status_never_counts_as_passed():
    assert status_map.bucket("something_new") == status_map.UNKNOWN
    assert not status_map.is_pass(status_map.UNKNOWN)
    assert not status_map.is_pass(status_map.BLOCKED_HARDWARE)


def test_summarise_counts_every_bucket():
    counts = status_map.summarise([("passed", "", None),
                                   ("dependency_missing", "", None),
                                   ("environment_unavailable", "", "hardware_shortfall")])
    assert counts[status_map.PASSED] == 1
    assert counts[status_map.BLOCKED_DEPENDENCY] == 1
    assert counts[status_map.BLOCKED_HARDWARE] == 1
    assert sum(counts.values()) == 3
