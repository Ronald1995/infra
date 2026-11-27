"""Distribution names versus import modules, and when a PyPI substitute is a lie.

Four dependencies here shadow an unrelated PyPI project name (see
``dependencies/distributions.yaml``).  ``pip install deep-gemm`` can succeed while
the code that runs is not DeepGEMM-Ascend at all, so the installer and the doctor
must answer two separate questions:

* which distribution provides this dependency (``package_name``),
* which module names it must expose (``import_modules``).

:func:`check_module_origin` closes the loop: after an install it verifies that the
imported module really lives inside the checkout the manifest pinned, instead of
accepting "the import worked".
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "dependencies" / "distributions.yaml"


class DistributionError(ValueError):
    """The distribution table is malformed."""


def load(path=None):
    path = Path(path) if path else DEFAULT_PATH
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != 1:
        raise DistributionError(f"{path.name}: unsupported schema_version")
    table = {}
    for dep_id, entry in (raw.get("distributions") or {}).items():
        if not isinstance(entry, dict):
            raise DistributionError(f"{path.name}:{dep_id}: entry must be a mapping")
        modules = entry.get("import_modules") or []
        if not isinstance(modules, list) or not all(isinstance(m, str) for m in modules):
            raise DistributionError(f"{path.name}:{dep_id}: import_modules must be a list")
        table[dep_id] = {
            "package_name": entry.get("package_name"),
            "import_modules": list(modules),
            "pypi_same_name": entry.get("pypi_same_name"),
            "substitutable": bool(entry.get("substitutable", False)),
            "reason": (entry.get("reason") or "").strip(),
        }
    return table


def package_for(table, dep_id):
    """The distribution to ask pip for, or ``None`` when only a checkout will do."""
    entry = table.get(dep_id) or {}
    return entry.get("package_name")


def modules_for(table, dep_id):
    entry = table.get(dep_id) or {}
    return tuple(entry.get("import_modules") or ())


def substitution_hazard(table, dep_id):
    """A one-line warning when PyPI has a lookalike that must not be used.

    Returns ``None`` when the name is safe (a real PyPI dependency, or no lookalike
    exists at all).
    """
    entry = table.get(dep_id) or {}
    lookalike = entry.get("pypi_same_name")
    if not lookalike or entry.get("substitutable"):
        return None
    return (f"{dep_id}: PyPI '{lookalike}' is a different project and must not be "
            f"substituted ({entry.get('reason') or 'pinned checkout required'})")


def module_origin(dep_id, module):
    """Where ``module`` was imported from, or ``None`` when it is not importable."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return spec.origin


def check_module_origin(table, dep_id, checkout=None):
    """Verify the imported modules resolve inside the expected checkout.

    Returns ``{"ok", "modules", "outside"}``.  ``outside`` lists the modules whose
    origin does not sit under ``checkout`` -- the signal that a similarly named PyPI
    package won the import race.
    """
    entry = table.get(dep_id) or {}
    modules = list(entry.get("import_modules") or [])
    result = {"ok": True, "modules": {}, "outside": []}
    root = Path(checkout).resolve() if checkout else None
    for module in modules:
        origin = module_origin(dep_id, module)
        result["modules"][module] = origin
        if origin is None:
            result["ok"] = False
            result["outside"].append(module)
            continue
        if root is not None:
            try:
                inside = Path(origin).resolve().is_relative_to(root)
            except AttributeError:  # Python < 3.9
                inside = str(Path(origin).resolve()).startswith(str(root))
            if not inside:
                result["ok"] = False
                result["outside"].append(module)
    return result


def resolve(table, dep_id):
    """Everything an installer needs for one dependency."""
    entry = table.get(dep_id) or {}
    return {
        "dependency": dep_id,
        "package_name": entry.get("package_name"),
        "import_modules": list(entry.get("import_modules") or ()),
        "substitutable": bool(entry.get("substitutable")),
        "hazard": substitution_hazard(table, dep_id),
        "reason": entry.get("reason"),
    }
