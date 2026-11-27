"""Platform-support evidence: documentation is a hint, evidence is a verdict.

A dependency's release notes saying nothing about Ascend950 must never become
``BLOCKED_PLATFORM``: eleven adapters were wrongly excluded that way once, and
several of them carry 950DT reference reports.  This module owns the four-state
vocabulary and the rule that only ``rejected`` -- which requires ``evidence`` --
may block a platform.

The data lives in ``dependencies/platform_support.yaml``.  Anything not listed
there falls back to the manifest's ``supported_soc``, which yields ``documented``
or ``undocumented`` and nothing stronger.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "dependencies" / "platform_support.yaml"

DOCUMENTED = "documented"
UNDOCUMENTED = "undocumented"
VALIDATED = "validated"
REJECTED = "rejected"

STATUSES = (DOCUMENTED, UNDOCUMENTED, VALIDATED, REJECTED)
#: Only this one may become BLOCKED_PLATFORM.
BLOCKING = (REJECTED,)
#: Statuses that still have to be proven by a real run on this SoC.
NEEDS_VALIDATION = (UNDOCUMENTED,)


class PlatformSupportError(ValueError):
    """The platform-support table cannot be trusted as written."""


def load(path=None):
    """Return ``{dep_id: {soc: {status, validation_required, note, evidence}}}``."""
    path = Path(path) if path else DEFAULT_PATH
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != 1:
        raise PlatformSupportError(f"{path.name}: unsupported schema_version")
    table = {}
    for dep_id, entry in (raw.get("dependencies") or {}).items():
        if not isinstance(entry, dict):
            raise PlatformSupportError(f"{path.name}:{dep_id}: entry must be a mapping")
        per_soc = {}
        for key, value in entry.items():
            if not isinstance(value, dict) or "status" not in value:
                continue
            status = value["status"]
            if status not in STATUSES:
                raise PlatformSupportError(
                    f"{path.name}:{dep_id}:{key}: unknown status {status!r}")
            if status == REJECTED and not value.get("evidence"):
                raise PlatformSupportError(
                    f"{path.name}:{dep_id}:{key}: a 'rejected' verdict needs evidence")
            per_soc[key] = {"status": status,
                            "validation_required": bool(value.get("validation_required",
                                                                  status == UNDOCUMENTED)),
                            "note": (value.get("note") or "").strip(),
                            "evidence": value.get("evidence")}
        table[dep_id] = per_soc
    return table


def _manifest_status(entry, soc):
    """What the manifest alone can say: documented, undocumented, or unknown."""
    if entry is None or not soc:
        return None
    socs = entry.get("supported_soc") or ["*"]
    if "*" in socs or soc in socs:
        return DOCUMENTED
    return UNDOCUMENTED


def status_for(table, dep_id, soc, entry=None):
    """Resolve the support status for one dependency on one SoC.

    Precedence: the explicit table, then the manifest's documented range, then
    ``None`` when nothing can be said (no entry and an unknown host).
    """
    explicit = (table.get(dep_id) or {}).get(soc)
    if explicit:
        return explicit["status"]
    return _manifest_status(entry, soc)


def detail_for(table, dep_id, soc, entry=None):
    """Status plus the note/evidence that justifies it."""
    explicit = (table.get(dep_id) or {}).get(soc)
    status = status_for(table, dep_id, soc, entry)
    if explicit:
        return {"status": status, "note": explicit["note"],
                "evidence": explicit["evidence"],
                "validation_required": explicit["validation_required"],
                "source": "platform_support.yaml"}
    return {"status": status, "note": "", "evidence": None,
            "validation_required": status in NEEDS_VALIDATION,
            "source": "manifest.supported_soc" if status else None}


def blocks(status):
    """True only for a verdict that was proven, never for a documentation gap."""
    return status in BLOCKING


def promote(table, dep_id, soc, status, evidence):
    """Move one (dependency, SoC) to a new status, refusing unevidenced rejections.

    Returns a new table; the caller decides where to persist it.  This is the only
    path from ``undocumented`` to ``rejected``, and it requires evidence such as a
    failing run id plus the error the run recorded.
    """
    if status not in STATUSES:
        raise PlatformSupportError(f"unknown status {status!r}")
    if status == REJECTED and not evidence:
        raise PlatformSupportError(
            f"{dep_id}@{soc}: refusing to reject a platform without evidence")
    updated = {dep: {key: dict(value) for key, value in per_soc.items()}
               for dep, per_soc in table.items()}
    entry = updated.setdefault(dep_id, {}).setdefault(soc, {})
    entry["status"] = status
    entry["evidence"] = evidence
    entry["validation_required"] = status in NEEDS_VALIDATION
    entry.setdefault("note", "")
    return updated


def validation_queue(table, soc):
    """Dependencies still awaiting a real run on ``soc``."""
    queue = []
    for dep_id, per_soc in sorted(table.items()):
        entry = per_soc.get(soc) or {}
        if entry.get("status") in NEEDS_VALIDATION:
            queue.append({"dependency": dep_id, "status": entry["status"],
                          "note": entry.get("note", ""),
                          "validation_required": True})
    return queue
