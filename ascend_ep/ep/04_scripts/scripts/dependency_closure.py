"""Which third-party modules does an operator actually import?

This is the bridge between the code and the dependency catalog. The manifest says what a
dependency provides (``modules``); this module answers the other direction by parsing the
imports of:

* ``operators/<id>/**``            the adapter itself,
* ``operators/vendor/ascendep/ep/<id>/**``  the vendored adapter it loads,
* every local module those files import from ``operators/`` (transitive),

and then maps each external top-level module to the dependency that provides it *in the
operator's environment*.  Two families that install the same module name therefore resolve
to the right one instead of "whichever came first".

``operators/ascendep.py`` is excluded on purpose: it is the loader, and its module-level
imports belong to the demo it loads lazily, not to every operator that imports the loader.
"""
import ast
from collections import defaultdict
from pathlib import Path
import sys

from common.registry import ROOT

STDLIB = set(sys.stdlib_module_names)
#: Packages implemented in this repository (never a dependency).
LOCAL_PACKAGES = {"common", "benchmarks", "baselines", "operators", "metrics", "analysis",
                  "reporting", "profiling", "schemas", "scripts", "operator_checks",
                  "run_benchmark"}
#: Adapter plumbing whose imports are attributed to the demo, not to every operator.
PLUMBING = {"operators/ascendep.py"}


def _read(path):
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return None


def scan_imports(path):
    """Every top-level module imported by one file (absolute imports only)."""
    tree = _read(path)
    if tree is None:
        return set()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def local_imports(relative, root=ROOT):
    """The local ``operators`` files one file imports."""
    root = Path(root)
    tree = _read(root / relative)
    if tree is None:
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            parts = node.module.split(".")
            if parts[0] != "operators" or len(parts) < 2:
                continue
            candidate = (root / "operators" / ("/".join(parts[1:]) + ".py"))
            if candidate.is_file():
                found.add(str(candidate.relative_to(root)).replace("\\", "/"))
            elif (root / "operators" / parts[1]).is_dir():
                for name in (root / "operators" / parts[1]).glob("*.py"):
                    if name.stem == (parts[2] if len(parts) > 2 else "__init__"):
                        found.add(str(name.relative_to(root)).replace("\\", "/"))
        elif isinstance(node, ast.ImportFrom) and node.level and node.module:
            base = (root / relative).parent
            candidate = base / (node.module.replace(".", "/") + ".py")
            if candidate.is_file():
                found.add(str(candidate.relative_to(root)).replace("\\", "/"))
    return found


def operator_files(operator_id, root=ROOT):
    """Every file an operator's runtime may import from."""
    root = Path(root)
    files = set()
    for path in (root / "operators" / operator_id).rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        files.add(str(path.relative_to(root)).replace("\\", "/"))
    demo = root / "operators" / "vendor" / "ascendep" / "ep" / operator_id
    if demo.is_dir():
        for path in demo.rglob("*.py"):
            files.add(str(path.relative_to(root)).replace("\\", "/"))
    return files


def closure(operator_id, root=ROOT):
    """The set of external top-level modules one operator's runtime imports."""
    root = Path(root)
    seen, queue = set(), list(operator_files(operator_id, root))
    while queue:
        relative = queue.pop()
        if relative in seen or relative in PLUMBING:
            continue
        seen.add(relative)
        queue.extend(item for item in local_imports(relative, root) if item not in seen)
    modules = set()
    for relative in seen:
        modules.update(scan_imports(root / relative))
    return {module for module in modules
            if module and module not in LOCAL_PACKAGES and module not in STDLIB}


def module_owners(manifest):
    """module -> dependency ids that provide it (more than one means a conflict)."""
    owners = defaultdict(list)
    for entry in manifest.get("dependencies") or ():
        for module in entry.get("modules") or ():
            owners[module].append(entry["id"])
    return {module: sorted(set(ids)) for module, ids in owners.items()}


def resolve(operator_id, environment, manifest, root=ROOT):
    """``(dependencies, modules, unmapped)`` for one operator in one environment."""
    modules = closure(operator_id, root)
    owners = module_owners(manifest)
    environments = {entry["id"]: entry["environment"]
                    for entry in manifest.get("dependencies") or ()}
    dependencies, unmapped = set(), set()
    for module in sorted(modules):
        candidates = owners.get(module) or []
        if not candidates:
            unmapped.add(module)
            continue
        same = [dep for dep in candidates if environments.get(dep) == environment]
        dependencies.add((same or candidates)[0])
    return dependencies, modules, unmapped
