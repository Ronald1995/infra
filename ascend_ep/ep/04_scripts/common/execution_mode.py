"""Execution-mode policy: a formal operator test runs on a real Ascend NPU.

Two modes exist and they are not interchangeable backends:

``npu_only`` (default)
    Every task that measures an operator must carry ``device: npu``.  A CPU
    configuration is refused *before* the scheduler expands it, so a CPU result
    can never be counted as operator coverage.

``framework``
    Only for framework self-checks: configuration parsing, planner expansion,
    constraint filtering, reporting and helper utilities.  A CPU configuration is
    allowed here, but :func:`counts_as_operator_evidence` still refuses to treat
    the outcome as evidence about an operator.

The mode decides *policy*, never *device placement*: nothing in this module
substitutes CPU for NPU when a kernel or a rank is missing.  A missing kernel is
a recorded blocker, not a fallback.
"""
import os

#: Formal mode: the only mode allowed to produce operator coverage.
FORMAL_MODE = "npu_only"
EXECUTION_MODES = (FORMAL_MODE, "framework")
DEFAULT_MODE = FORMAL_MODE
MODE_ENVIRON = "EP_EXECUTION_MODE"

#: The device a formal task must run on.
FORMAL_DEVICE = "npu"


class FormalModeViolation(ValueError):
    """Raised when a configuration would route a formal test away from the NPU."""


def resolve(explicit=None, environ=None):
    """Return the effective execution mode.

    Precedence: explicit argument, then ``EP_EXECUTION_MODE``, then the default.
    An unknown value is an error: a typo must not silently downgrade the policy.
    """
    environ = os.environ if environ is None else environ
    value = (explicit or environ.get(MODE_ENVIRON) or DEFAULT_MODE).strip().lower()
    if value not in EXECUTION_MODES:
        raise FormalModeViolation(
            f"unknown execution mode {value!r}; choose from {EXECUTION_MODES}")
    return value


def is_formal(mode):
    return mode == FORMAL_MODE


def device_of(config):
    """The device a configuration declares, or ``None`` when it declares none."""
    if isinstance(config, dict):
        return config.get("device")
    return None


def assert_formal(mode, config, subject="task"):
    """Refuse a non-NPU configuration while the formal mode is active.

    ``framework`` mode is exempt because it never claims operator evidence; the
    caller is still responsible for keeping its results out of coverage, which
    :func:`counts_as_operator_evidence` enforces.
    """
    if not is_formal(mode):
        return
    device = device_of(config)
    if device != FORMAL_DEVICE:
        raise FormalModeViolation(
            f"{subject}: execution mode {FORMAL_MODE!r} requires device "
            f"{FORMAL_DEVICE!r}, configuration declares {device!r}. CPU execution is "
            f"limited to framework self-checks and is never operator evidence.")


def counts_as_operator_evidence(mode, config, status):
    """True only for a real NPU measurement taken in the formal mode.

    This is the single place that decides whether an outcome may enter operator
    coverage.  A CPU pass, a framework-mode pass, a skip and a blocked task all
    return ``False``.
    """
    if not is_formal(mode):
        return False
    if device_of(config) != FORMAL_DEVICE:
        return False
    return status == "passed"


def preflight(device_count, requested_ranks, device_reachable=True):
    """Classify whether a formal task can even be launched.

    Returns ``{status, blocked_kind, reason}`` using the benchmark's raw status
    vocabulary so downstream reporting keeps one naming scheme.  ``blocked_kind``
    distinguishes a hardware shortfall (too few devices) from an environment
    problem, which the report layer turns into ``BLOCKED_HARDWARE`` versus
    ``BLOCKED_ENVIRONMENT``.
    """
    if not device_reachable:
        return {"status": "environment_unavailable", "blocked_kind": "environment",
                "reason": "no Ascend NPU is reachable from this process"}
    if requested_ranks < 1:
        return {"status": "environment_unavailable", "blocked_kind": "environment",
                "reason": f"invalid rank count {requested_ranks}"}
    if device_count < requested_ranks:
        return {
            "status": "environment_unavailable",
            "blocked_kind": "hardware_shortfall",
            "reason": (f"requested {requested_ranks} NPU devices, found {device_count}; "
                       f"EP{requested_ranks} needs one device per rank and is never "
                       f"replaced by CPU execution"),
        }
    return {"status": None, "blocked_kind": None, "reason": None}


def describe(mode):
    """One-line human description used by reports and the CLI."""
    if is_formal(mode):
        return "NPU_ONLY: operator tests run on real Ascend NPU devices only"
    return ("framework: CPU self-checks allowed; results never count as operator "
            "coverage")
