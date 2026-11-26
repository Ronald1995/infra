"""Capture runtime provenance without loading device runtimes during planning.

The CANN toolkit has no single authoritative query interface available from a plain
Python process, so the version is resolved through an ordered list of sources.  The
source is recorded next to the value: a report may say "CANN 8.2.RC1, from
``$ASCEND_HOME_PATH/version.cfg``" or "unavailable, no toolkit installation found",
but it must never present a guess as a version.
"""
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

#: Toolkit roots tried, in order, when the environment does not name one.
TOOLKIT_ROOTS = (
    "/usr/local/Ascend/ascend-toolkit/latest",
    "/usr/local/Ascend/latest",
    "/usr/local/Ascend/nnae/latest",
    "/usr/local/Ascend/asys/latest",
    "~/Ascend/ascend-toolkit/latest",
)

#: Files that may carry a toolkit version, in preference order.
VERSION_FILES = ("version.cfg", "ascend_toolkit_install.info", "version.info")

_VERSION_PATTERN = re.compile(r"^\s*(?:version|Version|CANN_VERSION|cann_version)\s*[=:]\s*(\S+)",
                              re.MULTILINE)


def versions():
    result = {}
    for package in ("torch", "torch-npu", "PyYAML"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def command(args, timeout=8):
    try:
        result = subprocess.run(args, capture_output=True, text=True, errors="replace",
                                timeout=timeout, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _read_version_file(path):
    """Parse one toolkit metadata file, or return ``None``."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _VERSION_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    # `version.cfg` may also be a bare `key=value` list without a `version=` key.
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() in {"version", "cann_version"}:
            return value.strip() or None
    return None


def cann_version(environ=None):
    """Resolve the CANN toolkit version and the source that produced it.

    Returns ``{"cann_version", "cann_version_source", "cann_version_error"}``.
    ``cann_version_source`` is one of ``env:ASCEND_TOOLKIT_VERSION``,
    ``file:<path>`` or ``unavailable``; a failure names every location that was
    tried so the report can explain the gap instead of printing an empty cell.
    """
    environ = os.environ if environ is None else environ
    value = (environ.get("ASCEND_TOOLKIT_VERSION") or "").strip()
    if value:
        return {"cann_version": value, "cann_version_source": "env:ASCEND_TOOLKIT_VERSION",
                "cann_version_error": None}

    candidates, tried = [], []
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        root = (environ.get(name) or "").strip()
        if not root:
            continue
        base = Path(root)
        # `ASCEND_OPP_PATH` normally points at <toolkit>/opp, so its parent is the
        # toolkit root that owns `version.cfg`.
        for directory in (base, base.parent):
            for filename in VERSION_FILES:
                candidate = directory / filename
                if candidate.is_file():
                    candidates.append(candidate)
    for root in TOOLKIT_ROOTS:
        for filename in VERSION_FILES:
            candidate = Path(root).expanduser() / filename
            if candidate.is_file():
                candidates.append(candidate)

    seen = set()
    for candidate in candidates:
        path = str(candidate)
        if path in seen:
            continue
        seen.add(path)
        value = _read_version_file(candidate)
        if value:
            return {"cann_version": value, "cann_version_source": f"file:{path}",
                    "cann_version_error": None}
        tried.append(path)

    reason = "ASCEND_TOOLKIT_VERSION is unset and no readable version file was found"
    if tried:
        reason += " (unreadable: " + ", ".join(tried) + ")"
    return {"cann_version": None, "cann_version_source": "unavailable",
            "cann_version_error": reason}


def driver_version(smi_info):
    """Best-effort driver version from ``npu-smi info`` output."""
    if not smi_info:
        return None
    match = re.search(r"Driver\s+Version\s*[:\s]\s*(\S+)", smi_info, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"Version\s*:\s*(\d+\.\d+[.\w]*)", smi_info)
    return match.group(1) if match else None


def capture(probe_npu=False):
    packages = versions()
    smi = shutil.which("npu-smi")
    info = {
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "packages": packages, "device_visibility": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "cann_root": os.getenv("ASCEND_HOME_PATH"),
        **cann_version(),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_changes": command(["git", "status", "--porcelain", "--untracked-files=no"]),
        "npu_smi": smi, "npu_smi_info": None, "driver_version": None,
        "driver_version_source": "unavailable",
    }
    if probe_npu and info["npu_smi"]:
        info["npu_smi_info"] = command([info["npu_smi"], "info"])
        if info["npu_smi_info"]:
            info["driver_version"] = driver_version(info["npu_smi_info"])
            info["driver_version_source"] = ("npu-smi info" if info["driver_version"]
                                             else "unavailable")
    return info
