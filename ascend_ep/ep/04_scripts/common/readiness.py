"""Per-operator dependency readiness, classified the way the delivery reports it.

``scripts/dependency_report.py`` already resolves each operator's dependency
closure and asks the doctor whether each dependency is present.  This module adds
the two questions the report needs on top of that:

* **Is the dependency even supported on this SoC?**  The manifest declares
  ``supported_soc`` per dependency.  ``fsdp-turbo`` for instance lists
  ``Ascend910B``/``Ascend910_93`` and not ``Ascend950``: on a 950DT host that is a
  *platform* blocker, and telling the user to "install the dependency" would send
  them down a path that cannot succeed.
* **Which single state describes the operator?**  The delivery vocabulary is
  ``READY``, ``MISSING``, ``VERSION_MISMATCH``, ``BUILD_FAILED``,
  ``SOURCE_UNAVAILABLE``, ``UNSUPPORTED_PLATFORM``.

Nothing here installs anything: it classifies, so the caller can decide.  The
install path stays ``scripts/install_deps.sh`` + ``scripts/vendor_ascendep.py``.
"""
from common import status_map

READY = "READY"
MISSING = "MISSING"
VERSION_MISMATCH = "VERSION_MISMATCH"
BUILD_FAILED = "BUILD_FAILED"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"

STATES = (READY, MISSING, VERSION_MISMATCH, BUILD_FAILED, SOURCE_UNAVAILABLE,
          UNSUPPORTED_PLATFORM)

#: State -> delivery bucket, so a report never invents a fourth vocabulary.
STATE_BUCKET = {
    READY: status_map.PASSED,
    MISSING: status_map.BLOCKED_DEPENDENCY,
    VERSION_MISMATCH: status_map.BLOCKED_DEPENDENCY,
    BUILD_FAILED: status_map.BUILD_FAILED,
    SOURCE_UNAVAILABLE: status_map.BLOCKED_DEPENDENCY,
    UNSUPPORTED_PLATFORM: status_map.BLOCKED_PLATFORM,
}

#: Order in which a blocker wins when an operator has several: "this host can
#: never run it" outranks "install something", which outranks "rebuild".
_PRECEDENCE = (UNSUPPORTED_PLATFORM, MISSING, VERSION_MISMATCH, SOURCE_UNAVAILABLE,
               BUILD_FAILED, READY)


def normalise_soc(model):
    """Map a device model string onto the manifest's SoC families.

    ``Ascend950DT_9582`` -> ``Ascend950``; ``Ascend910B4`` -> ``Ascend910B``;
    an A3 board reports ``Ascend910_93`` in its own naming.  An unrecognised model
    is returned unchanged, which makes every ``supported_soc`` check fail closed
    (reported as unsupported rather than silently assumed fine).
    """
    if not model:
        return None
    text = str(model)
    if "910_93" in text or text.startswith("Ascend910_9"):
        return "Ascend910_93"
    if "950" in text:
        return "Ascend950"
    if text.startswith("Ascend910"):
        return "Ascend910B"
    return text


def soc_supported(supported, soc):
    """True when ``soc`` is in a dependency's ``supported_soc`` list."""
    if not supported:
        return True                      # no declaration: do not invent a blocker
    if not soc:
        return True                      # unknown host: leave it as MISSING
    return "*" in supported or soc in supported


def _dep_state(entry, doctor_status, soc):
    """Classify one dependency from its manifest entry and the doctor verdict."""
    if entry is not None and not soc_supported(entry.get("supported_soc"), soc):
        return UNSUPPORTED_PLATFORM
    if doctor_status in ("pass", "warn"):
        return VERSION_MISMATCH if doctor_status == "warn" else READY
    if entry is None:
        return MISSING
    if entry.get("ownership") == "vendored":
        # A vendored tree is verified, never installed: absent means the tree is
        # missing from this checkout.
        return SOURCE_UNAVAILABLE
    if entry.get("install_spec") == "build" or entry.get("kind") == "native_build":
        # Distinguishing "never built" from "build failed" needs the build log; the
        # doctor reports a failed probe, so both are BUILD_FAILED only once a build
        # was attempted.  Without that evidence the honest state is MISSING.
        return MISSING
    return MISSING


def classify(rows, entries, soc):
    """Return ``{operator: {state, bucket, blockers, unsupported}}``.

    ``rows`` are the per-operator rows of ``scripts.dependency_report.build()``;
    ``entries`` maps dependency id -> manifest entry; ``soc`` is the normalised SoC
    family of the host being reported on.
    """
    result = {}
    for row in rows:
        states = {}
        for dep in row.get("dependencies") or []:
            states[dep] = _dep_state(entries.get(dep), (row.get("states") or {}).get(dep), soc)
        for module in row.get("unmapped_modules") or []:
            states.setdefault(f"module:{module}", MISSING)
        chosen = next((state for state in _PRECEDENCE if state in states.values()), READY)
        result[row["operator"]] = {
            "state": chosen,
            "bucket": STATE_BUCKET[chosen],
            "unsupported": sorted(dep for dep, state in states.items()
                                  if state == UNSUPPORTED_PLATFORM),
            "blockers": sorted(dep for dep, state in states.items() if state != READY),
            "dependencies": sorted(states),
        }
    return result


def refine_bucket(bucket, operator, readiness):
    """Let the SoC verdict correct a bucket that came from the run itself.

    A run reports ``dependency_missing`` for an operator whose dependency is simply
    absent; when the manifest also says that dependency cannot exist on this host,
    the delivery bucket must say so, because the next action is different.
    """
    if bucket != status_map.BLOCKED_DEPENDENCY:
        return bucket
    entry = (readiness or {}).get(operator)
    if entry and entry.get("state") == UNSUPPORTED_PLATFORM:
        return status_map.BLOCKED_PLATFORM
    return bucket
