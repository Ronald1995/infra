"""Runtime environment doctor for the EP benchmark.

Why this module exists: the runner already reports one status
(``dependency_missing``) for a missing wheel, a missing CANN component, a missing
host driver, a missing device node, an unbuilt kernel tree and a missing
compiler.  Those six situations need six different actions, and on a CPU-only
workstation five of them are expected while on a broken Ascend node only one is.
This module turns the single status into a per-dependency record plus a
per-family availability matrix, so a failure names the thing to install instead
of the thing that broke later.

Three rules the implementation keeps:

* a probe never guesses.  Observed values are quoted as evidence; an unreadable
  version stays ``None`` and is reported as unknown rather than filled in with a
  default;
* a missing dependency is never ``pass`` and never ``skip``.  ``skip`` is
  reserved for a probe that cannot decide (out-of-date doctor, an unknown check
  type, a version matrix with nothing installed), and it still degrades the
  families it would have blocked;
* the capability matrix is derived from ``dependencies/manifest.yaml``, so the
  catalog is the single place where "who needs what" is written down.

Nothing here installs anything, imports a device runtime, or touches the
network.  ``scripts/install_deps.sh`` is the installer; ``scripts/check_cann.py``,
``scripts/check_communication.py`` and ``scripts/check_environment.py`` are thin
views over the same records.
"""
from __future__ import annotations

import argparse
import ctypes.util
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
import glob
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys

import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
MANIFEST = SCRIPTS / "dependencies" / "manifest.yaml"
REQUIREMENTS = SCRIPTS / "dependencies" / "python"

#: Closed status vocabulary for one probe result.
STATUSES = ("pass", "warn", "fail", "skip")
#: Closed status vocabulary for one capability group.
MATRIX_STATUSES = ("available", "degraded", "blocked")
KINDS = ("pip", "native_build", "host_provided", "binary", "python_import")
CHECK_TYPES = (
    "python_version", "python_module", "shared_library", "binary", "env_var", "device_node",
    "driver", "cann_home", "toolchain", "ascendep_checkout", "kernel_artifact",
)
CATEGORIES = {
    "A": "python_base",
    "B": "cann",
    "C": "communication",
    "D": "multinode",
    "E": "profiling",
}
#: Profile -> dependency categories it proves.  Mirrored in the manifest and
#: cross-checked by tests/dependencies/test_manifest_consistency.py.
PROFILES = {
    "cpu": ("A",),
    "npu": ("A", "B"),
    "communication": ("A", "B", "C"),
    "profiling": ("A", "B", "E"),
    "multinode": ("A", "B", "D"),
    "all": ("A", "B", "C", "D", "E"),
}
#: Every key the catalog promises for one dependency entry.
REQUIRED_KEYS = (
    "id", "category", "kind", "required_by", "version", "source_url", "commit", "license",
    "optional", "installable", "install_reason", "manual_step", "install", "check",
    "runtime_paths", "supported_soc", "supported_cann", "notes",
)
SOC_PATTERN = re.compile(r"Ascend[\s_-]?(\d{3,4}\w*)", re.IGNORECASE)
PIN_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")
VERSION_PATTERN = re.compile(r"(\d+(?:\.\d+)+)")
ENV_NAMES = (
    "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_TOOLKIT_VERSION",
    "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_WORK_PATH", "HCCL_SOCKET_IFNAME",
    "HCCL_OP_EXPANSION_MODE", "EP_ASCENDEP_ROOT", "EP_HCCL_TEST_BINARY",
    "EP_L0_TRANSPORT_LIBRARY", "MASTER_ADDR", "MASTER_PORT", "VIRTUAL_ENV", "PIP_INDEX_URL",
)
GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    """One probed dependency (or platform fact) and how it was established."""

    id: str
    status: str
    detail: str
    suggested_action: str
    evidence: object
    source: str = "probe"
    blocking_for: tuple = ()
    category: str = "?"


@dataclass(frozen=True)
class Row:
    """One capability group's availability, derived from the checks above."""

    id: str
    title: str
    status: str
    missing: tuple = ()
    degraded: tuple = ()


@dataclass(frozen=True)
class Report:
    profile: str
    categories: tuple
    platform: dict
    checks: tuple
    matrix: tuple
    out_of_scope: tuple
    wheelhouse: dict
    counts: dict

    @property
    def exit_code(self):
        if self.counts.get("fail"):
            return 1
        if self.wheelhouse.get("counts", {}).get("fail"):
            return 1
        return 0

    def as_dict(self):
        return {
            "schema_version": 1,
            "profile": self.profile,
            "categories": list(self.categories),
            "platform": self.platform,
            "counts": dict(self.counts),
            "exit_code": self.exit_code,
            "checks": [
                {
                    "id": check.id, "status": check.status, "category": check.category,
                    "detail": check.detail, "suggested_action": check.suggested_action,
                    "evidence": check.evidence, "source": check.source,
                    "blocking_for": list(check.blocking_for),
                }
                for check in self.checks
            ],
            "matrix": [
                {"id": row.id, "title": row.title, "status": row.status,
                 "missing": list(row.missing), "degraded": list(row.degraded)}
                for row in self.matrix
            ],
            "out_of_scope": [dict(item) for item in self.out_of_scope],
            "wheelhouse": self.wheelhouse,
        }

    def as_json(self):
        return json.dumps(_jsonable(self.as_dict()), indent=2, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class Context:
    """Paths and cached host facts shared by every probe in one run."""

    root: Path
    manifest: Path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _command(args, timeout=15):
    """Run a diagnostic command.  Returns ``(returncode, stdout, stderr)``.

    ``returncode`` is ``None`` when the process could not start at all, which is
    a different fact from "it ran and failed" and is reported as such.
    """
    try:
        completed = subprocess.run(args, capture_output=True, text=True, errors="replace",
                                   timeout=timeout, check=False)
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def _jsonable(value):
    """Coerce evidence into JSON-safe primitives without losing the observation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def installed_version(distribution):
    """Installed distribution version, or ``None`` when the metadata is absent."""
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None
    except (OSError, ValueError):
        return None


def numeric_version(text):
    """Numeric prefix of a version string: ``2.6.0+cpu`` -> ``(2, 6, 0)``."""
    if text is None:
        return None
    match = VERSION_PATTERN.search(str(text))
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def satisfies(observed, constraint):
    """Return True/False, or None when the comparison is not meaningful.

    ``None`` means "not comparable" (a commit hash, an unparsable value, or an
    unknown observed version); callers report that as an unknown rather than as a
    pass, because a constraint that cannot be checked is not evidence.
    """
    if not constraint:
        return None
    observed_parts = numeric_version(observed)
    if observed_parts is None:
        return None
    decided = None
    for clause in str(constraint).split(","):
        clause = clause.strip()
        if not clause:
            continue
        match = re.match(r"(===|==|>=|<=|!=|>|<)\s*(\S+)", clause)
        if not match:
            return None
        operator, wanted = match.groups()
        wanted_parts = numeric_version(wanted)
        if wanted_parts is None:
            return None
        size = max(len(observed_parts), len(wanted_parts))
        left = observed_parts + (0,) * (size - len(observed_parts))
        right = wanted_parts + (0,) * (size - len(wanted_parts))
        if operator in ("==", "==="):
            decided = left == right
        elif operator == ">=":
            decided = left >= right
        elif operator == "<=":
            decided = left <= right
        elif operator == ">":
            decided = left > right
        elif operator == "<":
            decided = left < right
        else:
            decided = left != right
        if not decided:
            return False
    return decided


def tokens(value):
    """Flatten a ``required_by`` value (nested anchors) into unique tokens."""
    flattened = []
    for item in value or ():
        if isinstance(item, (list, tuple)):
            flattened.extend(tokens(item))
        elif isinstance(item, str):
            flattened.append(item)
    return list(dict.fromkeys(flattened))


def _expand(text):
    """Expand ``$VAR``/``${VAR}`` in a manifest path; ``None`` if a variable is unset."""
    if "$" not in str(text):
        return str(text)

    def substitute(match):
        name = match.group(1) or match.group(2)
        return os.getenv(name, "") if os.getenv(name) else "\0"

    expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", substitute, str(text))
    return None if "\0" in expanded else expanded


# ---------------------------------------------------------------------------
# Cached host facts
# ---------------------------------------------------------------------------
def reset_cache():
    """Drop the cached host facts (tests and long-lived hosts that re-source env)."""
    for probe in (_cann_installation, _npu_smi, _mpi, _infiniband):
        probe.cache_clear()


@lru_cache(maxsize=None)
def _cann_installation():
    """Return ``(root, version, version_file, evidence)`` for the CANN toolkit.

    The candidates are the documented environment variables, the conventional
    install root the spec names, and the Windows default, in that order.  The raw
    file content goes into the evidence so a report can be diffed between hosts.
    """
    declared = {name: os.getenv(name) for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")}
    candidates = [value for value in declared.values() if value]
    candidates += [
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/latest",
        os.path.join(os.environ.get("ProgramFiles", ""), "Huawei", "CANN"),
    ]
    scanned = []
    for root in candidates:
        if not root:
            continue
        scanned.append(root)
        for name in ("version.cfg", "ascend_toolkit_install.info"):
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return root, None, path, {"root": root, "version_file": path, "error": str(exc)}
            match = re.search(r"(?:version|Version)\s*[=:]\s*(\S+)", text)
            return (root, match.group(1) if match else None, path,
                    {"root": root, "version_file": path, "head": text.splitlines()[:6]})
    evidence = {"declared": declared, "candidates": scanned[:8],
                "error": "no version.cfg under any candidate root"}
    return None, None, None, evidence


@lru_cache(maxsize=None)
def _npu_smi():
    """Parse ``npu-smi info`` once: SoC candidates, chip rows, raw head."""
    binary = shutil.which("npu-smi")
    if not binary:
        return None
    returncode, stdout, stderr = _command([binary, "info"])
    if returncode != 0 or not stdout.strip():
        return {"command": [binary, "info"], "returncode": returncode,
                "error": stderr.strip()[:400], "soc": None, "chip_rows": 0, "raw_head": []}
    socs = sorted({match.group(0) for match in SOC_PATTERN.finditer(stdout)})
    chip_rows = len([line for line in stdout.splitlines() if re.match(r"^\|\s*\d+", line)])
    return {"command": [binary, "info"], "returncode": 0, "soc": socs[0] if socs else None,
            "soc_candidates": socs, "chip_rows": chip_rows,
            "raw_head": stdout.splitlines()[:40]}


def _mpi_implementation(text):
    lowered = text.lower()
    if "hydra" in lowered or "mpich" in lowered:
        return "MPICH/Hydra"
    if "open mpi" in lowered or "open-mpi" in lowered or "openmpi" in lowered:
        return "Open MPI"
    if "intel(r) mpi" in lowered or "intel mpi" in lowered:
        return "Intel MPI"
    if "microsoft mpi" in lowered:
        return "MS-MPI"
    return None


@lru_cache(maxsize=None)
def _mpi():
    """Identify the MPI implementation and version -- never just "command exists"."""
    binary = shutil.which("mpirun")
    if not binary:
        return None
    returncode, stdout, stderr = _command([binary, "--version"])
    text = (stdout + stderr).strip()
    version = None
    for line in text.splitlines()[:6]:
        match = VERSION_PATTERN.search(line)
        if match:
            version = match.group(1)
            break
    return {"binary": binary, "returncode": returncode,
            "implementation": _mpi_implementation(text), "version": version,
            "raw_head": text.splitlines()[:4]}


def _library_paths(root, extra=()):
    """Directories searched for a shared library, in loader-ish order."""
    candidates = []
    for variable in ("LD_LIBRARY_PATH", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        value = os.getenv(variable)
        if value:
            candidates.extend(part for part in value.split(os.pathsep) if part)
    home, _, _, _ = _cann_installation()
    if home:
        candidates.extend([os.path.join(home, "lib64"), os.path.join(home, "runtime", "lib64"), home])
    candidates.extend([
        "/usr/local/Ascend/driver/lib64",
        "/usr/local/Ascend/driver/lib64/common",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ])
    for item in extra:
        text = _expand(item) if isinstance(item, str) else str(item)
        if not text:
            continue
        candidates.append(text if os.path.isabs(text) else str(Path(root) / text))
    ordered, seen = [], set()
    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if os.path.isdir(resolved):
            ordered.append(resolved)
    return ordered


def _library_stem(name):
    """``libasc_comm_kernels.so`` -> ``asc_comm_kernels`` (the find_library name)."""
    stem = name
    for suffix in (".so", ".dylib", ".dll"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for prefix in ("lib",):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            stem = stem[len(prefix):]
    return stem


def _locate_library(name, directories):
    for directory in directories:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _find_spec(module):
    try:
        return find_spec(module), None
    except (ImportError, AttributeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _try_import(module):
    """Import a cheap module.  A probe must never abort the whole report, and the
    vendor extensions raise types this file cannot enumerate (ImportError, OSError,
    RuntimeError, and CANN-specific exceptions), so the catch is deliberately wide.
    """
    try:
        import_module(module)
        return True, None
    except Exception as exc:  # noqa: BLE001 - see docstring
        return False, f"{type(exc).__name__}: {exc}"[:300]


def _device_nodes(patterns):
    """Every path matching the device-node patterns, in stable order."""
    matches = []
    for pattern in patterns or ():
        matches.extend(sorted(glob.glob(pattern)))
    return matches


def _shm_info():
    info = {"path": "/dev/shm", "platform": sys.platform,
            "present": os.path.isdir("/dev/shm"), "total_bytes": None, "free_bytes": None}
    if not info["present"] or not hasattr(os, "statvfs"):
        return info
    try:
        stats = os.statvfs("/dev/shm")
    except OSError as exc:
        info["error"] = str(exc)
        return info
    info["total_bytes"] = stats.f_frsize * stats.f_blocks
    info["free_bytes"] = stats.f_frsize * stats.f_bavail
    return info


@lru_cache(maxsize=None)
def _infiniband():
    root = Path("/sys/class/infiniband")
    devices = sorted(entry.name for entry in root.iterdir()) if root.is_dir() else []
    tool = shutil.which("ibv_devices")
    output = None
    if tool:
        returncode, stdout, stderr = _command([tool])
        output = {"binary": tool, "returncode": returncode,
                  "stdout": stdout.splitlines()[:20], "stderr": stderr.strip()[:200]}
    return {"sysfs": str(root), "devices": devices, "ibv_devices": output}


def _ld_library_path():
    value = os.getenv("LD_LIBRARY_PATH")
    entries = [part for part in (value or "").split(os.pathsep) if part]
    home, _, _, _ = _cann_installation()
    required = []
    if home:
        required = [os.path.join(home, "lib64"), os.path.join(home, "runtime", "lib64")]
    covered = [path for path in required
               if any(os.path.abspath(entry) == os.path.abspath(path) for entry in entries)]
    return {"value": value, "entries": entries, "required": required, "covered": covered}


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------
def _check(identifier, category, status, detail, action, evidence, blocking_for=()):
    return Check(id=identifier, status=status, detail=detail, suggested_action=action,
                 evidence=evidence, blocking_for=tuple(blocking_for), category=category)


def _entry_check(entry, status, detail, action, evidence):
    return _check(entry["id"], entry.get("category", "?"), status, detail, action, evidence,
                  tokens(entry.get("required_by")))


def _catalog_hint(entry, check_type):
    if check_type == "python_module":
        distribution = entry["check"].get("distribution") or entry["check"].get("module")
        pinned = entry.get("version") or ""
        return (f"install {distribution} {pinned}".rstrip()
                + f" for {entry['id']}; see dependencies/python/ and dependencies/manifest.yaml")
    return (f"install {entry['id']} as the catalog describes "
            f"(kind={entry.get('kind')}); see dependencies/system/")


# ---------------------------------------------------------------------------
# Probes: one per ``check.type``
# ---------------------------------------------------------------------------
def probe_python_version(entry, check, ctx):
    observed = platform.python_version()
    constraint = entry.get("version")
    satisfied = satisfies(observed, constraint)
    evidence = {"version": observed, "constraint": constraint, "executable": sys.executable,
                "implementation": platform.python_implementation()}
    if satisfied is False:
        return _entry_check(entry, "fail",
                            f"python {observed} does not satisfy {constraint}",
                            "provision a supported interpreter (>=3.10,<3.13) and re-create the venv",
                            evidence)
    return _entry_check(entry, "pass", f"python {observed} ({platform.python_implementation()})",
                        "", evidence)


def probe_python_module(entry, check, ctx):
    module = check.get("module")
    distribution = check.get("distribution") or module
    version = installed_version(distribution)
    spec, spec_error = _find_spec(module)
    libraries = check.get("libraries") or []
    directory_hits = {}
    if libraries:
        directories = _library_paths(ctx.root)
        directory_hits = {name: _locate_library(name, directories) for name in libraries}
    requires_env = check.get("requires_env")
    env_value = os.getenv(requires_env) if requires_env else None
    evidence = {"module": module, "distribution": distribution, "version": version,
                "find_spec": getattr(spec, "origin", None) if spec else None,
                "spec_error": spec_error, "requires_env": requires_env,
                "env_value": env_value, "libraries": directory_hits}

    if spec is None:
        return _entry_check(entry, "fail", f"{module} is not importable",
                            _catalog_hint(entry, "python_module"), evidence)

    if check.get("import"):
        imported, error = _try_import(module)
        evidence["imported"] = imported
        evidence["import_error"] = error
        if not imported:
            return _entry_check(entry, "fail", f"{module} imports and fails: {error}",
                                "repair the installation; the metadata is present but the module is not usable",
                                evidence)

    satisfied = satisfies(version, entry.get("version"))
    evidence["version_satisfies"] = satisfied
    if libraries and not all(directory_hits.values()):
        missing = [name for name in libraries if not directory_hits.get(name)]
        return _entry_check(entry, "warn",
                            f"{module} is importable but {missing} was not found on the library path",
                            f"add the build output directory to LD_LIBRARY_PATH (missing: {missing})",
                            evidence)
    if satisfied is False:
        return _entry_check(entry, "warn",
                            f"{distribution} {version} does not satisfy {entry.get('version')}",
                            f"pin {distribution}{entry.get('version')} or update the manifest with the observed pairing",
                            evidence)
    if requires_env and not env_value:
        return _entry_check(entry, "warn",
                            f"{module} resolves, but {requires_env} is unset so the pinned tree cannot be verified",
                            f"export {requires_env} pointing at the pinned checkout", evidence)
    return _entry_check(entry, "pass", f"{module} {version or '(no metadata)'}", "", evidence)


def probe_shared_library(entry, check, ctx):
    names = check.get("names") or []
    require_all = check.get("require_all", True)
    extra = list(check.get("search") or [])
    directories = _library_paths(ctx.root, extra)
    find_results, resolved = {}, {}
    for name in names:
        find_results[name] = ctypes.util.find_library(_library_stem(name))
        located = _locate_library(name, directories)
        if located:
            resolved[name] = located
    evidence = {"names": names, "find_library": find_results, "resolved": resolved,
                "searched_directories": directories[:20]}
    missing = [name for name in names if name not in resolved]
    found = sorted(resolved.values())
    if (require_all and missing) or (not require_all and not found):
        detail = (f"{', '.join(missing)} not found on the library path"
                  if missing else f"none of {names} is present")
        return _entry_check(entry, "fail", detail,
                            "install the component that ships this library, or add its directory to LD_LIBRARY_PATH",
                            evidence)
    if missing:
        return _entry_check(entry, "warn", f"found {found}, missing {missing}",
                            "build the remaining bridges (see dependencies/manifest.yaml)", evidence)
    return _entry_check(entry, "pass", f"resolved {found}", "", evidence)


def probe_binary(entry, check, ctx):
    binary = check.get("binary")
    candidates = []
    for raw in check.get("extra_paths") or []:
        expanded = _expand(raw)
        if expanded:
            candidates.append(os.path.join(expanded, binary))
    resolved = shutil.which(binary)
    if not resolved:
        resolved = next((path for path in candidates if os.path.isfile(path)), None)
    evidence = {"binary": binary, "resolved": resolved, "extra_paths": candidates,
                "which": shutil.which(binary)}
    if not resolved:
        return _entry_check(entry, "fail", f"{binary} is not on PATH or in the documented directories",
                            _catalog_hint(entry, "binary"), evidence)

    version_args = check.get("version_args") or []
    if version_args:
        returncode, stdout, stderr = _command([resolved] + list(version_args))
        text = (stdout + stderr).strip()
        evidence.update({"version_arguments": version_args, "returncode": returncode,
                         "raw_head": text.splitlines()[:8]})
        if returncode not in (0, None) and not text:
            return _entry_check(entry, "warn",
                                f"{binary} exists but `{binary} {' '.join(version_args)}` produced no output (rc={returncode})",
                                f"run `{binary} {' '.join(version_args)}` by hand; the binary may be broken or incomplete",
                                evidence)
        if check.get("parse") == "mpi":
            implementation = _mpi_implementation(text)
            evidence["implementation"] = implementation
            evidence["version"] = _mpi()
            if implementation is None:
                return _entry_check(entry, "warn",
                                    f"{binary} exists but its implementation was not identified",
                                    "install Open MPI or MPICH/Hydra explicitly; the launch flags depend on it",
                                    evidence)
            return _entry_check(entry, "pass",
                                f"{binary} -> {implementation}", "", evidence)
        if check.get("parse") == "npu_smi":
            smi = _npu_smi() or {}
            evidence["npu_smi"] = smi
            if not smi.get("soc"):
                return _entry_check(entry, "warn",
                                    f"{binary} runs but no SoC name was parsed from `npu-smi info`",
                                    "check `npu-smi info` output by hand; the device may be unhealthy",
                                    evidence)
            return _entry_check(entry, "pass",
                                f"{binary} -> {smi['soc']} ({smi.get('chip_rows', 0)} chip rows)",
                                "", evidence)
        version = None
        for line in text.splitlines()[:6]:
            match = VERSION_PATTERN.search(line)
            if match:
                version = match.group(1)
                break
        evidence["parsed_version"] = version
        satisfied = satisfies(version, entry.get("version"))
        evidence["version_satisfies"] = satisfied
        if satisfied is False:
            return _entry_check(entry, "warn",
                                f"{binary} {version} does not satisfy {entry.get('version')}",
                                "use the version that belongs to this CANN release", evidence)
        return _entry_check(entry, "pass", f"{binary} {version or '(version not parsed)'}", "", evidence)
    return _entry_check(entry, "pass", f"{binary} resolves", "", evidence)


def probe_env_var(entry, check, ctx):
    names = check.get("names") or []
    mode = check.get("mode", "all")
    values = {name: os.getenv(name) for name in names}
    present = [name for name, value in values.items() if value]
    missing = [name for name in names if name not in present]
    evidence = {"values": values, "present": present, "missing": missing, "mode": mode}
    if mode == "any":
        if present:
            return _entry_check(entry, "pass", f"set: {present}", "", evidence)
        return _entry_check(entry, "fail", f"none of {names} is set",
                            "export the deployment's HCCL/launch variables before running", evidence)
    if not missing:
        return _entry_check(entry, "pass", f"all set: {present}", "", evidence)
    if present:
        return _entry_check(entry, "warn", f"partially set; missing {missing}",
                            f"export {missing} on every node in the job", evidence)
    return _entry_check(entry, "fail", f"{names} are not set",
                        "export the deployment variables (see dependencies/system/mpi.yaml)", evidence)


def probe_device_node(entry, check, ctx):
    patterns = check.get("patterns") or []
    access = check.get("access", "r")
    minimum = int(check.get("minimum", 1))
    matches = _device_nodes(patterns)
    permissions = {}
    for path in matches:
        permissions[path] = {
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
        }
    evidence = {"patterns": patterns, "matches": matches, "permissions": permissions,
                "platform": sys.platform, "access": access}
    if len(matches) < minimum:
        return _entry_check(entry, "fail",
                            f"no device node matched {patterns} (platform={sys.platform})",
                            "start the container with the Ascend device flags and install the host driver; "
                            "see dependencies/system/cann.yaml", evidence)
    unreadable = [path for path, info in permissions.items() if not info["readable"]]
    unwritable = [path for path, info in permissions.items() if access == "rw" and not info["writable"]]
    if unreadable:
        return _entry_check(entry, "fail", f"cannot read {unreadable}",
                            "add the run user to the HwHiAiUser group on the host and re-login", evidence)
    if unwritable:
        return _entry_check(entry, "warn", f"cannot write {unwritable}",
                            "grant write access via the container runtime; a run will fail while creating a context",
                            evidence)
    return _entry_check(entry, "pass", f"{len(matches)} device node(s) readable", "", evidence)


def probe_driver(entry, check, ctx):
    paths = [path for path in (check.get("paths") or []) if path]
    devices = _device_nodes(check.get("devices") or [])
    version_file, version_text = None, None
    for path in paths:
        if not os.path.isfile(path):
            continue
        version_file = path
        try:
            version_text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[:6]
        except OSError as exc:
            version_text = [f"unreadable: {exc}"]
        break
    driver_dir = any(os.path.isdir(path) for path in paths)
    evidence = {"paths": paths, "version_file": version_file, "version_head": version_text,
                "driver_dir": driver_dir, "devices": devices, "platform": sys.platform}
    if version_file:
        return _entry_check(entry, "pass", f"driver version file {version_file}", "", evidence)
    if driver_dir and devices:
        return _entry_check(entry, "warn", "driver directory present but version.info is missing",
                            "reinstall the host driver package; the installed tree looks incomplete",
                            evidence)
    if devices:
        return _entry_check(entry, "warn", "device nodes exist but /usr/local/Ascend/driver is absent",
                            "bind-mount /usr/local/Ascend/driver from the host; libascendcl cannot load the driver",
                            evidence)
    return _entry_check(entry, "fail", "no NPU driver (no version.info, no /dev/davinci_manager)",
                        entry.get("manual_step") or "install the host driver; see dependencies/system/cann.yaml",
                        evidence)


def probe_cann_home(entry, check, ctx):
    root, version, version_file, cached = _cann_installation()
    evidence = dict(cached)
    evidence["expected_version_file"] = check.get("version_file")
    if root is None or version_file is None:
        detail = "no CANN installation (no ASCEND_HOME_PATH and no /usr/local/Ascend/ascend-toolkit/latest)"
        return _entry_check(entry, "fail", detail,
                            entry.get("manual_step") or "install the CANN toolkit on the host", evidence)
    if version is None:
        return _entry_check(entry, "warn", f"CANN found at {root} but no version key in {version_file}",
                            "re-source set_env.sh; a toolkit without version.cfg is a partial install",
                            evidence)
    satisfied = satisfies(version, entry.get("version"))
    evidence["version_satisfies"] = satisfied
    if satisfied is False:
        return _entry_check(entry, "warn",
                            f"CANN {version} does not satisfy {entry.get('version')}",
                            "install the CANN release that matches the torch_npu pairing row", evidence)
    return _entry_check(entry, "pass", f"CANN {version} at {root}", "", evidence)


def probe_toolchain(entry, check, ctx):
    binaries = check.get("binaries") or []
    roots = [os.getenv(name) for name in (check.get("env") or []) if os.getenv(name)]
    directories = [os.path.join(root, sub) for root in roots for sub in (check.get("subdirs") or [])]
    resolved = {}
    for binary in binaries:
        found = shutil.which(binary)
        if not found:
            found = next((os.path.join(path, binary) for path in directories
                          if os.path.isfile(os.path.join(path, binary))), None)
        resolved[binary] = found
    missing = [binary for binary, path in resolved.items() if not path]
    evidence = {"binaries": resolved, "missing": missing, "searched": directories[:10]}
    if missing:
        return _entry_check(entry, "fail", f"missing toolchain members: {missing}",
                            entry.get("manual_step") or "install the missing toolchain members", evidence)
    return _entry_check(entry, "pass", f"resolved {sorted(resolved.values())}", "", evidence)


def probe_ascendep_checkout(entry, check, ctx):
    variable = check.get("env") or "EP_ASCENDEP_ROOT"
    value = os.getenv(variable)
    commit = check.get("commit")
    marker = check.get("marker") or "ep/runner.py"
    evidence = {"environment": variable, "value": value, "expected_commit": commit, "marker": marker}
    if not value:
        return _entry_check(entry, "fail", f"{variable} is not set",
                            f"git clone the ascendep checkout at {commit} and export {variable}", evidence)
    root = Path(value).expanduser()
    evidence["root"] = str(root)
    if not (root / marker).is_file():
        return _entry_check(entry, "fail", f"{root} has no {marker}",
                            "point the variable at the repository root, not at a subdirectory", evidence)
    evidence["marker_present"] = True
    if shutil.which("git"):
        # The doctor only ever queries the external checkout named by the
        # variable; it never runs git against this repository.
        returncode, stdout, stderr = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
        revision = stdout.strip()
        evidence.update({"revision": revision or None, "git_returncode": returncode,
                         "git_stderr": stderr.strip()[:200]})
        if returncode != 0:
            return _entry_check(entry, "warn", "checkout present but its revision could not be read",
                                "make sure the checkout is a git work tree, not a tarball", evidence)
        if commit and revision != commit:
            return _entry_check(entry, "fail",
                                f"checkout is at {revision[:12]}, expected {commit[:12]}",
                                f"git -C {root} checkout {commit}", evidence)
        return _entry_check(entry, "pass", f"pinned checkout {revision[:12]}", "", evidence)
    return _entry_check(entry, "warn", "checkout present but git is unavailable, so the commit is unverified",
                        "install git or verify the revision by hand", evidence)


def probe_kernel_artifact(entry, check, ctx):
    """Probe a kernel project through its own build manifest.

    The contract belongs to the project (``kernels/asc_comm/cmake/write_manifest.cmake``):
    ``build/manifest.json`` is written only after the link step and names the
    library per engine, so "the library exists" and "the build succeeded" cannot
    drift apart.  The doctor reads that file instead of guessing file names, and it
    does not load the library (a device kernel cannot be resolved on a CPU host;
    ``baselines/l0/transport/kernel_artifacts.py`` is the authority on ABI symbols).
    """
    source = ctx.root / (check.get("root") or "kernels/asc_comm")
    build = source / (check.get("build") or "build")
    manifest_name = check.get("manifest") or "manifest.json"
    build_manifest = build / manifest_name
    expected_libraries = list(check.get("libraries") or [])
    expected_engines = list(check.get("engines") or [])
    build_script = source / "build.sh"
    manifest, manifest_error, manifest_head = None, None, None
    if build_manifest.is_file():
        try:
            text = build_manifest.read_text(encoding="utf-8", errors="replace")
            manifest_head = text.splitlines()[:20]
            manifest = json.loads(text)
        except (OSError, ValueError) as exc:
            manifest_error = f"{type(exc).__name__}: {exc}"
    artifacts = sorted(path.name for path in build.iterdir() if path.is_file()) if build.is_dir() else []
    engines = {}
    if isinstance(manifest, dict):
        engines = {name: {"built": bool(info.get("built")), "library": info.get("library")}
                   for name, info in (manifest.get("engines") or {}).items()}
    built = sorted(name for name, info in engines.items() if info["built"])
    resolved = sorted(os.path.basename(info["library"]) for info in engines.values()
                      if info["library"] and (build / os.path.basename(info["library"])).is_file())
    evidence = {"source_root": str(source), "source_present": source.is_dir(),
                "build_script": str(build_script), "build_script_present": build_script.is_file(),
                "build_root": str(build), "build_present": build.is_dir(),
                "build_manifest": str(build_manifest), "build_manifest_present": build_manifest.is_file(),
                "manifest_error": manifest_error, "manifest_head": manifest_head,
                "engines": engines, "built_engines": built, "resolved_libraries": resolved,
                "artifacts": artifacts[:20]}
    if not source.is_dir():
        return _entry_check(entry, "fail", f"{source} is absent",
                            entry.get("manual_step") or "restore kernels/asc_comm/ and build it", evidence)
    if not build.is_dir() or not artifacts:
        detail = f"no build artifacts under {build}; the kernel build is BLOCKED"
        if not build_script.is_file():
            detail = f"no build artifacts and no {build_script.name} under {source}"
        return _entry_check(entry, "fail", detail,
                            entry.get("manual_step") or
                            "the CANN ASC-COMM toolchain is missing; see dependencies/system/asc_comm.yaml",
                            evidence)
    if manifest is None:
        return _entry_check(entry, "fail",
                            f"cannot read {build_manifest.name}: {manifest_error or 'missing'}",
                            "rebuild with the kernel project's build.sh; a build without a manifest is unverified",
                            evidence)
    missing_engines = [name for name in expected_engines if name not in engines]
    if missing_engines:
        return _entry_check(entry, "fail", f"the build manifest has no {missing_engines} entry",
                            "rebuild with the current kernels/asc_comm/cmake/write_manifest.cmake", evidence)
    if not built:
        return _entry_check(entry, "fail",
                            "the build manifest records no built engine (BLOCKED)",
                            entry.get("manual_step") or
                            "the device sources and the CANN ASC-COMM toolchain are missing", evidence)
    if expected_libraries and not any((build / name).is_file() for name in expected_libraries):
        return _entry_check(entry, "fail",
                            f"no {expected_libraries} under {build}",
                            "rebuild with the kernel project's build.sh", evidence)
    if len(built) < len(engines) or not resolved:
        return _entry_check(entry, "warn",
                            f"partial build: built={built}, resolved libraries={resolved}",
                            "the engines that were not built report UNSUPPORTED at run time", evidence)
    return _entry_check(entry, "pass", f"built engines {built} -> {resolved}", "", evidence)


PROBES = {
    "python_version": probe_python_version,
    "python_module": probe_python_module,
    "shared_library": probe_shared_library,
    "binary": probe_binary,
    "env_var": probe_env_var,
    "device_node": probe_device_node,
    "driver": probe_driver,
    "cann_home": probe_cann_home,
    "toolchain": probe_toolchain,
    "ascendep_checkout": probe_ascendep_checkout,
    "kernel_artifact": probe_kernel_artifact,
}


# ---------------------------------------------------------------------------
# Platform checks that are not a catalog entry but still block a family
# ---------------------------------------------------------------------------
def check_shm(ctx):
    info = _shm_info()
    if not info["present"]:
        return _check("mount.shm", "C", "fail",
                      f"no /dev/shm (platform={info['platform']})",
                      "run on Linux with a tmpfs /dev/shm of at least 1 GiB; SHMEM rendezvous and torch "
                      "shared-memory cannot work without it", info, ("l0_transport", "multinode"))
    if not info.get("total_bytes"):
        return _check("mount.shm", "C", "warn", "/dev/shm exists but its size is unreadable",
                      "check the mount options of /dev/shm", info, ("l0_transport", "multinode"))
    total = info["total_bytes"]
    if total < GIB:
        return _check("mount.shm", "C", "warn",
                      f"/dev/shm is {total / GIB:.2f} GiB (< 1 GiB)",
                      "remount /dev/shm with a larger size or point the SHMEM rendezvous at a bigger tmpfs",
                      info, ("l0_transport", "multinode"))
    return _check("mount.shm", "C", "pass", f"/dev/shm is {total / GIB:.1f} GiB", "", info,
                  ("l0_transport", "multinode"))


def check_rdma(ctx):
    info = _infiniband()
    devices = info["devices"]
    ibv = info["ibv_devices"] or {}
    parsed = [line for line in ibv.get("stdout", []) if re.match(r"^\s+\S+\s+\S+", line)]
    evidence = {"infiniband": info, "ibv_device_rows": parsed}
    if devices:
        return _check("network.rdma", "D", "pass", f"InfiniBand devices: {devices}", "", evidence,
                      ("multinode", "l0_transport"))
    if parsed:
        return _check("network.rdma", "D", "warn",
                      f"ibv_devices lists {len(parsed)} device(s) but /sys/class/infiniband is empty",
                      "verify the container sees /sys/class/infiniband; RDMA counters will be unavailable",
                      evidence, ("multinode", "l0_transport"))
    detail = ("no RDMA device (/sys/class/infiniband is empty, ibv_devices absent)"
              if ibv else "no RDMA device (no /sys/class/infiniband, no ibv_devices)")
    return _check("network.rdma", "D", "fail", detail,
                  "install the RDMA drivers / verbs stack and pass the devices into the container; "
                  "cross-server EP cannot use RoCE without them", evidence, ("multinode", "l0_transport"))


def check_ld_library_path(ctx):
    info = _ld_library_path()
    if info["value"] is None:
        return _check("env.ld_library_path", "B", "fail", "LD_LIBRARY_PATH is not set",
                      "source /usr/local/Ascend/ascend-toolkit/set_env.sh on every rank",
                      info, ("l0_collectives", "l0_hardware", "l0_transport", "l0_overlap"))
    if info["required"] and not info["covered"]:
        return _check("env.ld_library_path", "B", "warn",
                      f"LD_LIBRARY_PATH lacks {info['required']}",
                      "append $ASCEND_HOME_PATH/lib64 and $ASCEND_HOME_PATH/runtime/lib64; "
                      "ctypes.util.find_library does not consult LD_LIBRARY_PATH",
                      info, ("l0_collectives", "l0_hardware", "l0_transport", "l0_overlap"))
    if not info["required"]:
        return _check("env.ld_library_path", "B", "warn",
                      "LD_LIBRARY_PATH is set but no CANN installation was found to require",
                      "install the CANN toolkit first; the path entries can then be checked",
                      info, ("l0_collectives", "l0_hardware", "l0_transport", "l0_overlap"))
    return _check("env.ld_library_path", "B", "pass",
                  f"LD_LIBRARY_PATH covers {info['covered']}", "", info,
                  ("l0_collectives", "l0_hardware", "l0_transport", "l0_overlap"))


def compatibility(data):
    """Annotate the catalog's compatibility rows with the triple observed here.

    ``matches`` is ``True``/``False`` when every non-null field of the row could
    be compared, and ``None`` when a field is unknown -- an uncompared row is not
    a mismatch and is not a match either.
    """
    home, cann_version, version_file, _ = _cann_installation()
    observed = {"cann": cann_version, "torch": installed_version("torch"),
                "torch_npu": installed_version("torch-npu")}
    rows = []
    keys = ("cann", "torch", "torch_npu")
    for row in (data.get("compatibility") or {}).get("rows") or []:
        declared = [key for key in keys if row.get(key)]
        compared = [key for key in declared if observed.get(key)]
        if declared and len(compared) == len(declared):
            matches = all(_same_release(observed[key], row[key]) for key in declared)
        else:
            matches = None
        rows.append({"id": row.get("id"), "cann": row.get("cann"), "torch": row.get("torch"),
                     "torch_npu": row.get("torch_npu"), "soc": list(row.get("soc") or []),
                     "status": row.get("status"), "source_url": row.get("source_url"),
                     "evidence": row.get("evidence"), "matches": matches,
                     "compared_fields": compared})
    return {"observed": observed, "rows": rows, "cann_home": home, "version_file": version_file}


def check_cann_matrix(ctx, data):
    """Match the installed CANN/torch/torch_npu triple against the catalog table."""
    table = compatibility(data)
    observed = table["observed"]
    evidence = {"observed": observed, "cann_home": table["cann_home"],
                "version_file": table["version_file"],
                "rows": [row["id"] for row in table["rows"]]}
    if not observed["cann"] or not observed["torch_npu"]:
        return _check("cann.version_matrix", "B", "skip",
                      f"cannot evaluate the pairing (cann={observed['cann']}, torch_npu={observed['torch_npu']})",
                      "install CANN and torch_npu, then re-run; the pairing row is what makes the stack reproducible",
                      evidence, ())
    matched = [row for row in table["rows"] if row["matches"]]
    if matched:
        evidence["matched_row"] = matched[0]["id"]
        evidence["row"] = matched[0]
        return _check("cann.version_matrix", "B", "pass",
                      f"matches the known-good row {matched[0]['id']}", "", evidence, ())
    return _check("cann.version_matrix", "B", "fail",
                  f"installed triple {observed} matches no known-good row",
                  "install a documented CANN/torch/torch_npu triple from dependencies/manifest.yaml "
                  "compatibility.rows, or add the new triple to the catalog first",
                  evidence, ())


def _same_release(observed, expected):
    if observed is None or expected is None:
        return False
    if str(observed).strip() == str(expected).strip():
        return True
    left, right = numeric_version(observed), numeric_version(expected)
    return left is not None and left == right


PLATFORM_CHECKS = (
    ("mount.shm", check_shm),
    ("network.rdma", check_rdma),
    ("env.ld_library_path", check_ld_library_path),
)


# ---------------------------------------------------------------------------
# Manifest loading and validation
# ---------------------------------------------------------------------------
def load_manifest(path=None):
    """Load and validate the catalog; a broken catalog must not yield a matrix."""
    path = Path(path or MANIFEST)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = validate_manifest(data)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return data


def validate_manifest(data):
    """Return a list of catalog errors; empty means the catalog is coherent."""
    errors = []
    if not isinstance(data, dict):
        return ["manifest is not a mapping"]
    if data.get("schema_version") != 1:
        errors.append(f"unsupported schema_version {data.get('schema_version')!r}")
    vocabulary = data.get("vocabulary") or {}
    if list(vocabulary.get("check_status") or []) != list(STATUSES):
        errors.append("vocabulary.check_status does not match scripts.doctor.STATUSES")
    if list(vocabulary.get("matrix_status") or []) != list(MATRIX_STATUSES):
        errors.append("vocabulary.matrix_status does not match scripts.doctor.MATRIX_STATUSES")
    if list(vocabulary.get("kinds") or []) != list(KINDS):
        errors.append("vocabulary.kinds does not match scripts.doctor.KINDS")
    if list(vocabulary.get("check_types") or []) != list(CHECK_TYPES):
        errors.append("vocabulary.check_types does not match scripts.doctor.CHECK_TYPES")
    for name, categories in PROFILES.items():
        declared = (data.get("profiles") or {}).get(name)
        if declared is None:
            errors.append(f"profiles.{name} is missing")
        elif tuple(declared) != tuple(categories):
            errors.append(f"profiles.{name} is {declared}, scripts.doctor expects {list(categories)}")
    declared_categories = data.get("categories") or {}
    if sorted(declared_categories) != sorted(CATEGORIES):
        errors.append(f"categories must be exactly {sorted(CATEGORIES)}")

    groups = [group.get("id") for group in data.get("capability_groups") or []]
    if len(groups) != len(set(groups)):
        errors.append("duplicate capability group id")
    group_set = set(groups)
    entries = data.get("dependencies") or []
    if not entries:
        errors.append("no dependencies declared")
    seen = set()
    covered = set()
    for entry in entries:
        identifier = entry.get("id")
        if not identifier:
            errors.append("an entry has no id")
            continue
        if identifier in seen:
            errors.append(f"{identifier}: duplicate id")
        seen.add(identifier)
        missing_keys = [key for key in REQUIRED_KEYS if key not in entry]
        if missing_keys:
            errors.append(f"{identifier}: missing keys {missing_keys}")
        if entry.get("kind") not in KINDS:
            errors.append(f"{identifier}: invalid kind {entry.get('kind')!r}")
        if entry.get("category") not in CATEGORIES:
            errors.append(f"{identifier}: invalid category {entry.get('category')!r}")
        required_by = tokens(entry.get("required_by"))
        if not required_by:
            errors.append(f"{identifier}: required_by is empty")
        unknown = [token for token in required_by if token not in group_set]
        if unknown:
            errors.append(f"{identifier}: unknown required_by tokens {unknown}")
        covered.update(required_by)
        check = entry.get("check") or {}
        if check.get("type") not in CHECK_TYPES:
            errors.append(f"{identifier}: invalid check.type {check.get('type')!r}")
        if entry.get("installable") and not entry.get("install"):
            errors.append(f"{identifier}: installable but no install commands")
        if not entry.get("installable") and not entry.get("install_reason"):
            errors.append(f"{identifier}: not installable but no install_reason")
        if entry.get("kind") == "host_provided":
            if not entry.get("manual_step"):
                errors.append(f"{identifier}: host_provided without manual_step")
            if not entry.get("manual_step_owner"):
                errors.append(f"{identifier}: host_provided without manual_step_owner")
        if entry.get("kind") == "pip" and entry.get("installable") is not True:
            errors.append(f"{identifier}: kind pip must be installable")
        if entry.get("installable") is True and entry.get("kind") == "host_provided":
            errors.append(f"{identifier}: host_provided cannot be installable")
    for group in groups:
        if group and group not in covered:
            errors.append(f"capability group {group} has no dependency")
    return errors


# ---------------------------------------------------------------------------
# Availability matrix
# ---------------------------------------------------------------------------
def build_matrix(data, checks):
    """Derive per-group availability from the catalog plus the checks that ran.

    The dependency list of a group comes from the *catalog*, not from the checks:
    a ``--profile cpu`` run must still know that ``l0_transport`` needs
    libhccl.so, otherwise every group it did not probe would look available.
    Checks that are not catalog entries (``mount.shm`` and friends) contribute
    through their own ``blocking_for`` list.
    """
    titles = {group.get("id"): group.get("title", "") for group in data.get("capability_groups") or []}
    categories = {entry.get("id"): entry.get("category") for entry in data.get("dependencies") or []}
    required = {identifier: [] for identifier in titles}
    for entry in data.get("dependencies") or []:
        for group in tokens(entry.get("required_by")):
            if group in required:
                required[group].append(entry.get("id"))
    statuses = {}
    entry_ids = {entry.get("id") for entry in data.get("dependencies") or []}
    for check in checks:
        statuses[check.id] = check.status
        categories.setdefault(check.id, check.category)
        if check.id in entry_ids:
            continue          # already contributed by its catalog entry above
        for group in check.blocking_for:
            if group in required and check.id not in required[group]:
                required[group].append(check.id)
    rows, out_of_scope = [], []
    for identifier, title in titles.items():
        needs = required.get(identifier) or []
        if not needs:
            continue
        unchecked = sorted({categories.get(item) or "?" for item in needs if item not in statuses})
        if unchecked:
            out_of_scope.append({"id": identifier, "title": title,
                                 "needs_categories": unchecked, "needs": sorted(needs)})
            continue
        missing = [item for item in needs if statuses[item] == "fail"]
        degraded = [item for item in needs if statuses[item] in ("warn", "skip")]
        if missing:
            status = "blocked"
        elif degraded:
            status = "degraded"
        else:
            status = "available"
        rows.append(Row(id=identifier, title=title, status=status,
                        missing=tuple(missing), degraded=tuple(degraded)))
    return tuple(rows), tuple(out_of_scope)


# ---------------------------------------------------------------------------
# Wheelhouse
# ---------------------------------------------------------------------------
def parse_requirements(path):
    """Return ``(pins, unpinned)`` for one requirement file.

    ``pins`` is a list of ``(name, version, raw_line)``; anything that is not an
    ``==``/``===`` pin is returned in ``unpinned`` so a caller can object to it
    instead of silently ignoring it.
    """
    pins, unpinned = [], []
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = PIN_PATTERN.match(line)
        if match:
            pins.append((match.group(1), match.group(2), raw.strip()))
        else:
            unpinned.append(raw.strip())
    return pins, unpinned


def normalize_name(name):
    """Case- and separator-normalised name for comparing wheels against pins.

    Dots are deliberately preserved: the same helper is applied to a wheel file
    name, where the version (``2.6.0``) must keep its dots to line up with the
    pinned ``2.6.0``.
    """
    return re.sub(r"[-_]+", "-", str(name)).lower()


def _wheelhouse_record(identifier, status, detail, action, evidence):
    return {"id": identifier, "status": status, "detail": detail,
            "suggested_action": action, "evidence": evidence, "source": "probe",
            "blocking_for": []}


def check_wheelhouse(directory, root=None):
    """Prove that a local wheelhouse satisfies the pinned requirements offline."""
    root = Path(root or SCRIPTS)
    directory = Path(directory)
    if not directory.is_dir():
        return {"directory": str(directory), "checks": [], "counts": {status: 0 for status in STATUSES},
                "requirements": [], "error": f"{directory} is not a directory"}
    files = sorted(path.name for path in directory.iterdir() if path.is_file())
    normalized = [(normalize_name(name), name) for name in files]
    requirements = []
    records = []
    for path in sorted((root / "dependencies" / "python").glob("*.txt")):
        pins, unpinned = parse_requirements(path)
        for name, version, raw in pins:
            requirements.append({"file": path.name, "name": name, "version": version, "line": raw})
        for raw in unpinned:
            records.append(_wheelhouse_record(
                f"wheelhouse:{path.name}", "fail", f"unpinned requirement line {raw!r}",
                "pin every requirement so an offline wheelhouse is reproducible",
                {"file": path.name, "line": raw}))
    for requirement in requirements:
        name, version = requirement["name"], requirement["version"]
        prefix = f"{normalize_name(name)}-{version}"
        matches = []
        for normalized_name, original in normalized:
            if not normalized_name.startswith(prefix):
                continue
            remainder = normalized_name[len(prefix):]
            if remainder == "" or remainder[0] in "-+.":
                matches.append(original)
        record_id = f"wheelhouse:{name}"
        if matches:
            records.append(_wheelhouse_record(
                record_id, "pass", f"{name}=={version} satisfied by {matches[0]}", "",
                {"file": requirement["file"], "matches": matches[:5]}))
        else:
            records.append(_wheelhouse_record(
                record_id, "fail", f"{name}=={version} is missing from {directory.name}",
                f"pip download --no-deps --only-binary=:all: {name}=={version} -d {directory}",
                {"file": requirement["file"], "files_seen": len(files)}))
    counts = {status: 0 for status in STATUSES}
    for record in records:
        counts[record["status"]] += 1
    return {"directory": str(directory), "checks": records, "counts": counts,
            "requirements": requirements, "files": len(files)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def probe_platform(ctx):
    """Host facts shared by the report header and by scripts/check_environment.py."""
    home, version, version_file, cann_evidence = _cann_installation()
    smi = _npu_smi()
    mpi = _mpi()
    shm = _shm_info()
    return {
        "root": str(ctx.root),
        "host": socket.gethostname(),
        "os": platform.platform(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "cann_home": home,
        "cann_version": version,
        "cann_version_file": version_file,
        "cann_evidence": cann_evidence,
        "soc": (smi or {}).get("soc"),
        "soc_source": "npu-smi" if (smi or {}).get("soc") else "unavailable",
        "npu_smi": smi,
        "mpi": mpi,
        "devices": _device_nodes(("/dev/davinci[0-9]*", "/dev/davinci_manager", "/dev/devmm_svm")),
        "shm": shm,
        "infiniband": _infiniband(),
        "ld_library_path": _ld_library_path(),
        "env": {name: os.getenv(name) for name in ENV_NAMES},
    }


def probe_entry(entry, ctx):
    check = entry.get("check") or {}
    kind = check.get("type")
    prober = PROBES.get(kind)
    if prober is None:
        return _entry_check(entry, "skip", f"this doctor has no probe for check.type={kind!r}",
                            "update scripts/doctor.py (an unprobed dependency is not a satisfied one)",
                            {"check": check})
    result = prober(entry, check, ctx)
    if entry.get("optional") and result.status == "fail":
        return replace(result, status="warn", detail=f"{result.detail} (optional)")
    return result


def run_doctor(profile="all", categories=None, root=None, manifest=None, wheelhouse=None):
    """Probe the environment and return a :class:`Report`.

    ``categories`` restricts the probe to a subset of A-E and disables the matrix,
    because a matrix built from a partial probe would report the unprobed
    categories as blocked.  The focused CLIs use it; ``--profile`` does not.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    root = Path(root or SCRIPTS).resolve()
    data = load_manifest(manifest or (root / "dependencies" / "manifest.yaml"))
    if categories is None:
        probed = tuple(PROFILES[profile])
    else:
        probed = tuple(dict.fromkeys(categories))
        unknown = [category for category in probed if category not in CATEGORIES]
        if unknown:
            raise ValueError(f"unknown categories {unknown}; choose from {sorted(CATEGORIES)}")
    ctx = Context(root=root, manifest=Path(manifest or (root / "dependencies" / "manifest.yaml")))

    checks = [probe_entry(entry, ctx) for entry in data["dependencies"]
              if entry.get("category") in probed]
    for identifier, prober in PLATFORM_CHECKS:
        result = prober(ctx)
        if result.category in probed:
            checks.append(result)
    if "B" in probed:
        checks.append(check_cann_matrix(ctx, data))

    if categories is None:
        matrix, out_of_scope = build_matrix(data, checks)
    else:
        matrix, out_of_scope = (), ()

    counts = {status: 0 for status in STATUSES}
    for check in checks:
        counts[check.status] += 1
    if wheelhouse:
        wheelhouse_report = check_wheelhouse(wheelhouse, root)
    else:
        wheelhouse_report = {"directory": None, "checks": [], "counts": {status: 0 for status in STATUSES},
                             "requirements": []}
    return Report(profile=profile, categories=probed, platform=probe_platform(ctx),
                  checks=tuple(checks), matrix=matrix, out_of_scope=out_of_scope,
                  wheelhouse=wheelhouse_report, counts=counts)


def report_with(report, extra_checks, profile=None):
    """Copy ``report`` with extra checks appended and the counts recomputed.

    The focused CLIs (``check_cann`` / ``check_communication`` /
    ``check_environment``) add one or two platform checks that are not part of
    their category; the matrix is empty for a partial category probe, so nothing
    has to be recomputed for it.
    """
    checks = tuple(report.checks) + tuple(extra_checks)
    counts = {status: 0 for status in STATUSES}
    for check in checks:
        counts[check.status] += 1
    return replace(report, checks=checks, counts=counts, profile=profile or report.profile)


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------
DETAIL_WIDTH = 96


def _short(text, width=DETAIL_WIDTH):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _names(values, limit=6):
    values = list(values)
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f" (+{len(values) - limit} more)"


def render(report, show_matrix=True):
    """Render the human table; the JSON form carries the full evidence."""
    lines = []
    platform_info = report.platform
    lines.append("EP benchmark dependency doctor")
    lines.append(f"  profile    {report.profile}  (categories {','.join(report.categories)})")
    lines.append(f"  root       {platform_info['root']}")
    lines.append(f"  host/os    {platform_info['host']}  {platform_info['os']}")
    lines.append(f"  python     {platform_info['python']}  ({platform_info['executable']})")
    can_version = platform_info["cann_version"]
    lines.append(f"  CANN       {can_version or 'not installed'}"
                 f"  home={platform_info['cann_home'] or '-'}")
    lines.append(f"  SoC        {platform_info['soc'] or 'unknown'}"
                 f"  (source: {platform_info['soc_source']}, devices: {len(platform_info['devices'])})")
    mpi = platform_info.get("mpi") or {}
    lines.append(f"  MPI        {mpi.get('implementation') or 'not installed'}"
                 f" {mpi.get('version') or ''}".rstrip())
    shm = platform_info.get("shm") or {}
    shm_size = f"{shm['total_bytes'] / GIB:.1f} GiB" if shm.get("total_bytes") else "absent"
    lines.append(f"  /dev/shm   {shm_size}   RDMA devices: {len(platform_info['infiniband']['devices'])}")
    lines.append("")

    width_id = min(max((len(check.id) for check in report.checks), default=8), 34)
    lines.append(f"{'STATUS':<6}  {'ID':<{width_id}}  {'CAT':<3}  DETAIL")
    lines.append("-" * (6 + 2 + width_id + 2 + 3 + 2 + DETAIL_WIDTH))
    for check in report.checks:
        lines.append(f"{check.status:<6}  {check.id:<{width_id}}  {check.category:<3}  "
                     f"{_short(check.detail)}")

    attention = [check for check in report.checks if check.status in ("fail", "warn")]
    if attention:
        lines.append("")
        lines.append("needs attention")
        for check in attention:
            action = _short(check.suggested_action, 118) or "-"
            lines.append(f"  {check.status:<4} {check.id:<{width_id}}  {action}")

    if show_matrix and report.matrix:
        lines.append("")
        lines.append(f"capability matrix (profile={report.profile})")
        width_group = max((len(row.id) for row in report.matrix), default=8)
        lines.append(f"{'GROUP':<{width_group}}  {'STATUS':<9}  MISSING / DEGRADED")
        lines.append("-" * (width_group + 2 + 9 + 2 + DETAIL_WIDTH))
        for row in report.matrix:
            parts = []
            if row.missing:
                parts.append("missing: " + _names(row.missing))
            if row.degraded:
                parts.append("degraded: " + _names(row.degraded))
            lines.append(f"{row.id:<{width_group}}  {row.status:<9}  {_short('; '.join(parts) or '-')}")
    if report.out_of_scope:
        lines.append("")
        lines.append("out of scope for this profile (another profile proves these dependencies)")
        for item in report.out_of_scope[:12]:
            lines.append(f"  {item['id']:<34} needs categories {','.join(item['needs_categories'])}")
        if len(report.out_of_scope) > 12:
            lines.append(f"  ... and {len(report.out_of_scope) - 12} more groups "
                         f"(use --profile all for the full matrix)")
    if report.wheelhouse.get("directory"):
        lines.append("")
        lines.append(f"offline wheelhouse {report.wheelhouse['directory']}")
        for record in report.wheelhouse.get("checks", []):
            lines.append(f"  {record['status']:<6} {record['id']:<28} {_short(record['detail'], 90)}")
        counts = report.wheelhouse.get("counts", {})
        lines.append("  wheelhouse summary: " + " ".join(f"{key}={counts.get(key, 0)}" for key in STATUSES))

    lines.append("")
    summary = " ".join(f"{status}={report.counts.get(status, 0)}" for status in STATUSES)
    lines.append(f"summary: {summary}  exit={report.exit_code}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/doctor.py",
        description="Probe the third-party dependencies this benchmark needs and print a "
                    "per-capability availability matrix.")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="all",
                        help="dependency categories to prove (default: all)")
    parser.add_argument("--categories", default=None,
                        help="restrict the probe to a comma-separated list of A-E categories "
                             "(disables the matrix; used by the focused check_* CLIs)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--offline-wheelhouse", metavar="DIR", default=None,
                        help="prove that DIR can satisfy the pinned requirements without network")
    parser.add_argument("--manifest", default=None, help="catalog path (default: dependencies/manifest.yaml)")
    parser.add_argument("--root", default=None, help="SCRIPTS root (default: the parent of scripts/)")
    args = parser.parse_args(argv)

    categories = None
    if args.categories:
        categories = tuple(part.strip().upper() for part in args.categories.split(",") if part.strip())
    if args.offline_wheelhouse and not Path(args.offline_wheelhouse).is_dir():
        parser.error(f"--offline-wheelhouse {args.offline_wheelhouse} is not a directory")
    try:
        report = run_doctor(profile=args.profile, categories=categories, root=args.root,
                            manifest=args.manifest, wheelhouse=args.offline_wheelhouse)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"doctor: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write((report.as_json() if args.json else render(report)) + "\n")
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
