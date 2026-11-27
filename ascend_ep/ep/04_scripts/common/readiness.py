"""Per-operator dependency readiness, classified the way the delivery reports it.

``scripts/dependency_report.py`` already resolves each operator's dependency
closure and asks the doctor whether each dependency is present.  This module adds
two questions on top of that:

* **Does the operator itself declare a hardware restriction?**  Only that may
  become ``BLOCKED_PLATFORM``: the adapter says which silicon it targets, e.g.
  ``dispatch_ffn_combine_vllm_ascend`` -> ``Atlas A3 (ascend910_93)``.
* **Is a missing dependency's release merely undocumented on this SoC?**  The
  manifest defines ``supported_soc`` as the SoCs the *release* is documented for,
  so ``fsdp-turbo`` listing only ``Ascend910B``/``Ascend910_93`` means "not
  documented on 950", not "cannot run on 950".  That state is
  ``UNDOCUMENTED_ON_SOC``: still a dependency blocker, never "do not retry here".

Nothing here installs anything: it classifies, so the caller can decide.  The
install path stays ``scripts/install_deps.sh`` + ``scripts/vendor_ascendep.py``.
"""
from common import platform_support, status_map

READY = "READY"
MISSING = "MISSING"
VERSION_MISMATCH = "VERSION_MISMATCH"
BUILD_FAILED = "BUILD_FAILED"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"
#: The dependency is absent *and* its release notes do not list this SoC.  That is
#: a documentation gap to resolve with a real run, not a statement that the
#: operator cannot work here -- the manifest itself says ``supported_soc`` is what
#: the release is "documented for".
UNDOCUMENTED_ON_SOC = "UNDOCUMENTED_ON_SOC"

STATES = (READY, MISSING, VERSION_MISMATCH, BUILD_FAILED, SOURCE_UNAVAILABLE,
          UNSUPPORTED_PLATFORM, UNDOCUMENTED_ON_SOC)

#: State -> delivery bucket, so a report never invents a fourth vocabulary.
STATE_BUCKET = {
    READY: status_map.PASSED,
    MISSING: status_map.BLOCKED_DEPENDENCY,
    VERSION_MISMATCH: status_map.BLOCKED_DEPENDENCY,
    BUILD_FAILED: status_map.BUILD_FAILED,
    SOURCE_UNAVAILABLE: status_map.BLOCKED_DEPENDENCY,
    UNSUPPORTED_PLATFORM: status_map.BLOCKED_PLATFORM,
    # Still a dependency the host lacks; the note is "confirm on this SoC",
    # never "do not retry here".
    UNDOCUMENTED_ON_SOC: status_map.BLOCKED_DEPENDENCY,
}

#: Order in which a blocker wins when an operator has several: a platform that
#: genuinely cannot run it outranks "install something", which outranks "rebuild",
#: which outranks "the release notes do not mention this SoC".
_PRECEDENCE = (UNSUPPORTED_PLATFORM, MISSING, SOURCE_UNAVAILABLE, BUILD_FAILED,
               UNDOCUMENTED_ON_SOC, VERSION_MISMATCH, READY)


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


def _dep_state(entry, doctor_status, soc, support=None):
    """Classify one dependency from the doctor verdict, its manifest entry and its
    platform-support evidence.

    Order matters.  An installed, importable package is READY even when its release
    notes never mentioned this SoC.  Only an *absent* dependency can be a SoC
    question, and ``UNSUPPORTED_PLATFORM`` requires a ``rejected`` verdict that
    carries evidence -- a release note that omits the SoC is ``undocumented``.
    """
    if doctor_status in ("pass", "warn"):
        return VERSION_MISMATCH if doctor_status == "warn" else READY
    if entry is None:
        return MISSING
    if platform_support.blocks(support):
        return UNSUPPORTED_PLATFORM
    if support == platform_support.UNDOCUMENTED or (
            support is None and not soc_supported(entry.get("supported_soc"), soc)):
        return UNDOCUMENTED_ON_SOC
    if entry.get("ownership") == "vendored":
        # A vendored tree is verified, never installed: absent means the tree is
        # missing from this checkout.
        return SOURCE_UNAVAILABLE
    return MISSING


def hardware_blocked(limits, soc):
    """True only when an operator's own ``constraints.hardware`` excludes this SoC.

    This is the one signal that may become ``BLOCKED_PLATFORM``: the adapter itself
    declares which silicon it targets (for example
    ``dispatch_ffn_combine_vllm_ascend`` -> ``Atlas A3 (ascend910_93)``).  A
    dependency's documentation is not such a signal.
    """
    if not limits or not soc:
        return False
    text = " ".join(str(item) for item in limits).lower()
    family = {
        "Ascend950": ("950", "a5"),
        "Ascend910_93": ("910_93", "a3"),
        "Ascend910B": ("910b", "a2"),
    }.get(soc, (soc.lower(),))
    return not any(token in text for token in family)


def classify(rows, entries, soc, operator_hardware=None, support_table=None):
    """Return ``{operator: {state, bucket, blockers, unsupported, undocumented}}``.

    ``rows`` are the per-operator rows of ``scripts.dependency_report.build()``;
    ``entries`` maps dependency id -> manifest entry; ``soc`` is the normalised SoC
    family of the host; ``operator_hardware`` maps operator -> its
    ``constraints.hardware`` list, which is the only source of a platform block.
    """
    operator_hardware = operator_hardware or {}
    support_table = support_table or {}
    result = {}
    for row in rows:
        states = {}
        for dep in row.get("dependencies") or []:
            entry = entries.get(dep)
            support = platform_support.status_for(support_table, dep, soc, entry)
            states[dep] = _dep_state(entry, (row.get("states") or {}).get(dep), soc, support)
        for module in row.get("unmapped_modules") or []:
            states.setdefault(f"module:{module}", MISSING)
        limits = operator_hardware.get(row["operator"]) or ()
        if hardware_blocked(limits, soc):
            states["operator:hardware"] = UNSUPPORTED_PLATFORM
        chosen = next((state for state in _PRECEDENCE if state in states.values()), READY)
        result[row["operator"]] = {
            "state": chosen,
            "bucket": STATE_BUCKET[chosen],
            "hardware_limits": list(limits),
            "unsupported": sorted(dep for dep, state in states.items()
                                  if state == UNSUPPORTED_PLATFORM),
            "undocumented": sorted(dep for dep, state in states.items()
                                   if state == UNDOCUMENTED_ON_SOC),
            "blockers": sorted(dep for dep, state in states.items() if state != READY),
            "dependencies": sorted(states),
        }
    return result


def refine_bucket(bucket, operator, readiness):
    """Let a *declared* platform limitation correct a bucket from the run itself.

    Only ``UNSUPPORTED_PLATFORM`` may do that, and that state now comes solely from
    an operator's own ``constraints.hardware``.  A dependency whose release notes
    simply omit this SoC stays ``BLOCKED_DEPENDENCY`` with a note, because claiming
    "do not retry here" on the strength of release notes is exactly the kind of
    invented blocker a report must not produce.
    """
    if bucket != status_map.BLOCKED_DEPENDENCY:
        return bucket
    entry = (readiness or {}).get(operator)
    if entry and entry.get("state") == UNSUPPORTED_PLATFORM:
        return status_map.BLOCKED_PLATFORM
    return bucket
