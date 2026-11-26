"""Probe the ASC-COMM kernel build outputs.

The AIV and CCU data planes are *device* programs.  They can only be used when the
kernel project under ``kernels/asc_comm/`` has been built for the installed CANN
toolchain, so the probe answers three separate questions and never conflates them:

``source_present``
    the kernel project exists in the tree;
``toolchain``
    ``ASCEND_HOME_PATH``/``/usr/local/Ascend`` plus a C++ compiler and CMake;
``artifacts``
    ``build/manifest.json`` describes a successful build, the shared library it
    names exists, and it exports the ABI symbols the host launcher needs.

A missing artifact is ``dependency_missing`` or ``not_implemented`` -- never
``unsupported`` and never a silent fallback to the host NBI path.
"""
from dataclasses import dataclass, field
import ctypes
from json import loads
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "kernels" / "asc_comm"
MANIFEST = PROJECT / "build" / "manifest.json"

#: ABI the host launcher resolves.  Kept next to the C header so a rename is caught
#: by the probe instead of by a segfault at run time.
REQUIRED_SYMBOLS = {
    "AIV": ("asc_comm_aiv_write_nbi", "asc_comm_aiv_drain"),
    "CCU": ("asc_comm_ccu_submit", "asc_comm_ccu_wait"),
}


@dataclass
class KernelArtifacts:
    engine: str
    source_present: bool
    toolchain: dict = field(default_factory=dict)
    library: str | None = None
    built: bool = False
    symbols: tuple = ()
    missing: list = field(default_factory=list)
    reason: str | None = None
    status: str = "dependency_missing"

    @property
    def available(self):
        return self.built and not self.missing and self.status == "ready"

    def as_dict(self):
        return {"engine": self.engine, "source_present": self.source_present,
                "toolchain": dict(self.toolchain), "library": self.library,
                "built": self.built, "symbols": list(self.symbols),
                "missing": list(self.missing), "status": self.status,
                "reason": self.reason}


def toolchain():
    """Locate the compiler and CANN installation the kernels need."""
    ascend_home = os.getenv("ASCEND_HOME_PATH")
    candidates = [ascend_home] if ascend_home else []
    candidates.extend(["/usr/local/Ascend/ascend-toolkit/latest",
                       "/usr/local/Ascend/latest"])
    present = next((path for path in candidates if path and Path(path).is_dir()), None)
    import shutil

    cmake = shutil.which("cmake")
    compiler = shutil.which("g++") or shutil.which("c++")
    return {"ascend_home": present, "cmake": cmake, "cxx": compiler,
            "complete": bool(present and cmake and compiler)}


def probe(engine):
    """Return the :class:`KernelArtifacts` for ``AIV`` or ``CCU``."""
    result = KernelArtifacts(engine=engine, source_present=PROJECT.is_dir(),
                             toolchain=toolchain())
    if not result.source_present:
        result.reason = f"the ASC-COMM kernel project is missing at {PROJECT}"
        result.status = "not_implemented"
        result.missing.append("kernel_source")
        return result
    if not MANIFEST.is_file():
        result.reason = (f"the ASC-COMM {engine} kernel has not been built; "
                         f"run kernels/asc_comm/build.sh (missing {MANIFEST})")
        for name in ("cmake", "cxx", "ascend_home"):
            if not result.toolchain.get(name):
                result.missing.append(name)
        if not result.missing:
            result.missing.append("build_artifacts")
        result.status = "not_implemented" if not result.toolchain["complete"] else "dependency_missing"
        return result
    try:
        manifest = loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result.reason = f"cannot read the kernel build manifest: {exc}"
        result.missing.append("build_manifest")
        return result
    entry = (manifest.get("engines") or {}).get(engine)
    if not entry:
        result.reason = f"the kernel build manifest has no {engine} entry"
        result.missing.append(f"{engine.lower()}_kernel")
        result.status = "not_implemented"
        return result
    result.library = entry.get("library")
    result.built = bool(entry.get("built")) and bool(result.library) and Path(result.library).is_file()
    if not result.built:
        result.reason = (f"the {engine} kernel library is not loadable: "
                         f"{entry.get('reason') or result.library}")
        result.missing.append(f"{engine.lower()}_library")
        return result
    result.symbols = _symbols(result.library)
    for name in REQUIRED_SYMBOLS[engine]:
        if name not in result.symbols:
            result.missing.append(name)
    if result.missing:
        result.reason = (f"the {engine} kernel library does not export "
                         f"{', '.join(result.missing)}")
        return result
    result.status = "ready"
    return result


def _symbols(library):
    try:
        module = ctypes.CDLL(str(library))
    except OSError:
        return ()
    found = []
    for engine, names in REQUIRED_SYMBOLS.items():
        del engine
        for name in names:
            if hasattr(module, name):
                found.append(name)
    return tuple(found)


def build_evidence(engine, artifacts=None):
    """Evidence facts a transport case may attach for a device data plane."""
    artifacts = artifacts or probe(engine)
    return {
        "engine": engine,
        "source_present": artifacts.source_present,
        "built": artifacts.built,
        "library": artifacts.library,
        "symbols": list(artifacts.symbols),
        "status": artifacts.status,
    }
