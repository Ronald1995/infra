#!/usr/bin/env python3
"""Verify and refresh the vendored ascendep adapter sources.

``operators/vendor/ascendep/`` holds a byte-exact copy of the ascendep modules that
``operators/ascendep.py`` imports at runtime, so the benchmark no longer needs an
external ``penn513/ascendep`` checkout.  ``VENDOR.json`` records where the copy came
from and one sha256 digest per file; this script is the only supported way to touch
that tree.

    --check     recompute every digest and fail on drift, gaps or stale provenance
    --list      print the recorded manifest
    --refresh   re-copy the selected files from a pinned checkout and rewrite VENDOR.json

The selection is the import closure of ``operators/ascendep.py``: the ``demo.py`` of
every operator backed by ``AscendEPBackend`` (or a subclass), the support modules those
demos import at module level, and the adapter modules they pull in transitively.
Nothing else from upstream is copied: the CLI, report generators, suite runner, tests,
docs and configuration are excluded on purpose and listed in ``VENDOR.json``.
"""
import argparse
import ast
import fnmatch
import hashlib
import json
from datetime import date
from pathlib import Path
import re
import shutil
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
BOOTSTRAP = SCRIPTS / "operators" / "ascendep.py"
VENDOR_ROOT = SCRIPTS / "operators" / "vendor" / "ascendep"
MANIFEST = VENDOR_ROOT / "VENDOR.json"
MANIFEST_NAME = "VENDOR.json"
SCHEMA_VERSION = 1

# Files that live in the vendored directory but do not come from upstream.
GENERATED = ("VENDOR.json", "README.md")

UPSTREAM_URL = "https://github.com/penn513/ascendep"
UPSTREAM_SUBDIRECTORY = "ep"

# operator id -> backend class in operators/ that imports ep.<id>.demo
OPERATORS = {
    "allgather_fsdpturbo": "AscendEPBackend",
    "allgather_megatron": "AscendEPBackend",
    "allgather_overlap_mindspeed": "AscendEPBackend",
    "alltoallv_gmm_cann": "FusedGMMBackend",
    "alltoall_fsdpturbo": "AscendEPBackend",
    "alltoall_megatron": "AscendEPBackend",
    "chunkmoe_fsdpturbo": "AscendEPBackend",
    "deepep_ascend": "ExpandedDeepEPBackend",
    "deepep_cann": "AscendEPBackend",
    "deepep_lueelu": "ExpandedDeepEPBackend",
    "deepep_sglang": "AscendEPBackend",
    "domino_fsdpturbo": "AscendEPBackend",
    "ep_gmm_fused_lueelu": "FusedGMMBackend",
    "fused_fsdpturbo": "AscendEPBackend",
    "mc2_dispatch_cann": "MC2V2Backend",
    "mc2_fsdpturbo": "AscendEPBackend",
    "megamoe_fsdpturbo": "AscendEPBackend",
    "megamoe_lueelu": "AscendEPBackend",
    "megamoe_td": "AscendEPBackend",
    "moonep": "AscendEPBackend",
}

# Adapter modules reached from the selected demos, plus the support modules every
# demo imports at module level.
TRANSITIVE = {
    "ep/runner.py": "every vendored adapter does `from ep.runner import run` at module level",
    "ep/megatron.py": "`from ep.megatron import dispatch` in ep/allgather_megatron/demo.py "
                      "and ep/alltoall_megatron/demo.py",
    "ep/allgather_hccl/demo.py": "`from ep.allgather_hccl.demo import Gather` in "
                                 "ep/deepep_sglang/demo.py and ep/mc2_dispatch_cann/demo.py",
}

# Extra context for selections whose reason is more than "this operator needs it".
REASON_OVERRIDES = {
    "ep/deepep_ascend/demo.py": "operator deepep_ascend (ExpandedDeepEPBackend); also the registry "
                                "`upstream:` target of deepep_direct_lueelu and the source of "
                                "`expanded_forward` for deepep_lueelu",
    "ep/alltoallv_gmm_cann/demo.py": "operator alltoallv_gmm_cann (FusedGMMBackend); also supplies "
                                     "`weight_grad` to ep/ep_gmm_fused_lueelu/demo.py",
    "ep/allgather_hccl/demo.py": "not an ascendep-backed operator: imported by ep.deepep_sglang.demo "
                                 "and ep.mc2_dispatch_cann.demo for its `Gather` autograd function",
}

# Upstream files that are deliberately not vendored.  First match wins; a file that
# matches nothing aborts --refresh so new upstream files must be classified.
EXCLUSIONS = (
    ("ep/alltoall_hccl/demo.py",
     "reachable only through the lazy `from ep.alltoall_hccl.demo import forward as baseline` "
     "inside ep.runner.run(), i.e. upstream's own torchrun CLI. operators/ascendep.py imports the "
     "adapter modules directly and never calls run(); infra's alltoall_hccl operator is a "
     "CollectiveBackend that does not use ascendep at all"),
    ("ep/megamoe_cann/demo.py",
     "infra implements megamoe_cann directly on cann_ops_transformer (CANNMegaMoEBackend), so this "
     "adapter is never imported"),
    ("ep/hierarchical_alltoallv_mindspeed/demo.py",
     "infra implements hierarchical_alltoallv_mindspeed directly on MindSpeed/Megatron "
     "(MindSpeedHierarchicalBackend), so this adapter is never imported"),
    ("ep/suite.py",
     "upstream's torchrun suite orchestrator; infra plans and schedules cases through "
     "common/scheduler.py and run_benchmark.py"),
    ("ep/report_ep2.py", "upstream report generator, run only by hand from the CLI"),
    ("ep/report_profile.py", "upstream profiling report generator, run only by hand from the CLI"),
    ("ep/archive_profile.py", "imported only by ep/report_profile.py, which is itself a CLI report"),
    ("ep/backends.json", "backend inventory read only by ep/suite.py and the report scripts"),
    ("ep/megamoe_td/build_shmem_templates.sh",
     "native build helper for the SHMEM templates. Native libraries are not vendored; they are "
     "tracked by dependencies/manifest.yaml"),
    ("ascendep/*",
     "upstream's own `python -m ascendep` CLI/benchmark harness. No module under ep/ imports it and "
     "infra replaces it with common/ plus run_benchmark.py"),
    ("tests/*", "upstream's own pytest suite; the vendored copy is covered by tests/operators_vendor/"),
    ("configs/*", "upstream benchmark configuration data used only by its CLI"),
    ("scripts/*", "upstream cluster bootstrap and probe scripts; not importable modules"),
    ("examples/*", "upstream training example, not imported by any adapter"),
    ("docs/*", "upstream design and survey documents"),
    ("lb_planner/*", "upstream load-balancer planning notes, not imported by any adapter"),
    ("reports/*", "upstream measurement reports"),
    ("ep/README.md", "adapter catalogue documentation"),
    ("*/README.md", "per-module documentation"),
    ("README.md", "upstream project README"),
    ("AGENTS.md", "upstream authoring instructions for AI agents"),
    ("pyproject.toml", "builds the upstream `ascendep` distribution; infra has its own pyproject.toml"),
    (".gitignore", "upstream repository housekeeping"),
)


def fail(message):
    print(f"vendor_ascendep: {message}", file=sys.stderr)
    raise SystemExit(1)


def refuse(message):
    print(f"vendor_ascendep: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def pinned_commit():
    """The commit operators/ascendep.py declares, read without importing the module."""
    try:
        tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(f"cannot read {BOOTSTRAP}: {exc}")
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", None) == "UPSTREAM_COMMIT" for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if re.fullmatch(r"[0-9a-f]{40}", value.value):
                return value.value
            fail(f"{BOOTSTRAP} has a malformed UPSTREAM_COMMIT {value.value!r}")
    fail(f"{BOOTSTRAP} no longer assigns a 40-hex UPSTREAM_COMMIT")


def selection():
    """Relative path -> why the runtime needs it, in upstream order."""
    reasons = {}
    for operator, backend in OPERATORS.items():
        reasons[f"ep/{operator}/demo.py"] = f"operator {operator} ({backend})"
    reasons.update(TRANSITIVE)
    reasons.update(REASON_OVERRIDES)
    return dict(sorted(reasons.items()))


def exclusion_reason(path):
    for pattern, reason in EXCLUSIONS:
        if fnmatch.fnmatchcase(path, pattern):
            return reason
    fail(f"no vendoring decision recorded for upstream file {path!r}; "
         f"add it to selection() or EXCLUSIONS in {Path(__file__).name}")


def git(source, *arguments):
    try:
        return subprocess.check_output(["git", "-C", str(source), *arguments],
                                       encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        refuse(f"cannot read git metadata from {source}: {exc}")


def source_commit(source):
    revision = git(source, "rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        refuse(f"{source} is not a git checkout with a readable HEAD")
    return revision


def read_manifest():
    try:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(f"cannot read {MANIFEST}: {exc}")
    except ValueError as exc:
        fail(f"{MANIFEST} is not valid JSON: {exc}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        fail(f"{MANIFEST_NAME} schema_version {payload.get('schema_version')!r} is not "
             f"{SCHEMA_VERSION}")
    return payload


def vendored_files():
    """Relative posix paths of everything under the vendored root except generated files.

    ``__pycache__`` and ``*.pyc`` are produced the first time an adapter is imported and are
    not vendored sources, so they never make a tree stale.
    """
    found = set()
    for path in VENDOR_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(VENDOR_ROOT).as_posix()
        if relative in GENERATED or relative.endswith(".pyc") or "__pycache__" in path.parts:
            continue
        found.add(relative)
    return found


def check():
    manifest = read_manifest()
    problems = []
    commit = pinned_commit()
    recorded = manifest.get("upstream", {}).get("commit")
    if recorded != commit:
        problems.append(f"{MANIFEST_NAME} pins {recorded}, but operators/ascendep.py declares {commit}")
    listed = {}
    for entry in manifest.get("files", []):
        listed[entry["path"]] = entry
    if not listed:
        problems.append(f"{MANIFEST_NAME} lists no vendored files")
    for relative, entry in sorted(listed.items()):
        path = VENDOR_ROOT / relative
        if entry.get("upstream_path") != relative:
            problems.append(f"{relative}: upstream_path {entry.get('upstream_path')!r} does not "
                            f"describe the preserved layout")
        if not path.is_file():
            problems.append(f"{relative}: listed in {MANIFEST_NAME} but missing on disk")
            continue
        payload = path.read_bytes()
        digest = sha256(payload)
        if digest != entry.get("sha256"):
            problems.append(f"{relative}: sha256 {digest} does not match manifest "
                            f"{entry.get('sha256')}")
        if entry.get("bytes") is not None and len(payload) != entry["bytes"]:
            problems.append(f"{relative}: {len(payload)} bytes does not match manifest "
                            f"{entry['bytes']}")
    for relative in sorted(vendored_files() - set(listed)):
        problems.append(f"{relative}: present on disk but not listed in {MANIFEST_NAME}")
    if problems:
        print(f"vendor_ascendep: FAILED {len(problems)} check(s)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    total = sum(entry["bytes"] for entry in listed.values())
    print(f"vendor_ascendep: OK {len(listed)} files, {total} bytes, upstream "
          f"{manifest['upstream']['url']}@{manifest['upstream']['commit']}")
    return 0


def print_manifest(as_json=False):
    manifest = read_manifest()
    if as_json:
        print(json.dumps(manifest, indent=2))
        return 0
    upstream = manifest["upstream"]
    license_status = manifest.get("license", {})
    print(f"upstream      {upstream['url']}")
    print(f"commit        {upstream['commit']}")
    print(f"subdirectory  {upstream['subdirectory']}")
    print(f"vendored_at   {manifest.get('vendored_at')}")
    print(f"license       {license_status.get('status')} "
          f"({license_status.get('note', 'see operators/vendor/THIRD_PARTY_NOTICES.md')})")
    print()
    width = max((len(entry["path"]) for entry in manifest["files"]), default=4)
    for entry in manifest["files"]:
        print(f"{entry['path']:<{width}}  {entry['bytes']:>6}  {entry['sha256']}")
    total = sum(entry["bytes"] for entry in manifest["files"])
    print(f"{'total':<{width}}  {total:>6}  {len(manifest['files'])} files")
    excluded = manifest.get("excluded", [])
    print()
    print(f"excluded {len(excluded)} upstream file(s):")
    for entry in excluded:
        print(f"  {entry['path']}")
    return 0


def upstream_tracked(source, commit):
    listing = git(source, "ls-tree", "-r", "--name-only", commit)
    return [line.strip() for line in listing.splitlines() if line.strip()]


def refresh(source, allow_other_commit, as_json=False):
    source = Path(source).resolve()
    if not (source / UPSTREAM_SUBDIRECTORY / "runner.py").is_file():
        refuse(f"{source} does not look like an ascendep checkout "
               f"(no {UPSTREAM_SUBDIRECTORY}/runner.py)")
    commit = source_commit(source)
    pinned = pinned_commit()
    if commit != pinned and not allow_other_commit:
        refuse(f"{source} is at {commit}, but operators/ascendep.py pins {pinned}; "
               f"check out the pinned commit or pass --allow-other-commit")
    reasons = selection()
    tracked = set(upstream_tracked(source, commit))
    missing = sorted(set(reasons) - tracked)
    if missing:
        refuse(f"selected files are absent from {commit}: {', '.join(missing)}")
    excluded = [{"path": path, "reason": exclusion_reason(path)}
                for path in sorted(tracked - set(reasons))]
    VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    files = []
    for relative, reason in reasons.items():
        source_file = source / relative
        if not source_file.is_file():
            refuse(f"{source_file} is missing from the working tree of {source}")
        target = VENDOR_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source_file.read_bytes()
        shutil.copyfile(source_file, target)
        if target.read_bytes() != payload:
            fail(f"{target} differs from {source_file} after copying")
        files.append({"path": relative, "upstream_path": relative, "bytes": len(payload),
                      "sha256": sha256(payload), "reason": reason})
    for stale in sorted(vendored_files() - set(reasons)):
        path = (VENDOR_ROOT / stale).resolve()
        if not path.is_relative_to(VENDOR_ROOT.resolve()):
            refuse(f"refusing to delete {path}, which is outside {VENDOR_ROOT}")
        path.unlink()
        print(f"vendor_ascendep: pruned {stale}")
    for directory in sorted((path for path in VENDOR_ROOT.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    author = git(source, "log", "-1", "--format=%an <%ae>%n%aI%n%s", commit).splitlines()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": f"scripts/{Path(__file__).name}",
        "upstream": {
            "url": UPSTREAM_URL,
            "commit": commit,
            "subdirectory": UPSTREAM_SUBDIRECTORY,
            "commit_author": author[0] if author else None,
            "commit_date": author[1] if len(author) > 1 else None,
            "commit_subject": author[2] if len(author) > 2 else None,
        },
        "layout": ("Files keep their upstream paths, so adding this directory to sys.path makes "
                   "`import ep.<name>.demo` resolve exactly as it does in a full checkout. "
                   "operators/vendor/.gitattributes marks the tree -text so git stores these bytes "
                   "verbatim and the digests above stay valid on every platform."),
        "license": {
            "status": "absent",
            "file": None,
            "note": ("Upstream publishes no LICENSE, COPYING or NOTICE file at this commit and no "
                     "copyright or license header in any source file. Do not redistribute this "
                     "copy until the owner signs off; see operators/vendor/THIRD_PARTY_NOTICES.md."),
        },
        "vendored_at": date.today().isoformat(),
        "files": files,
        "excluded": excluded,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"vendor_ascendep: refreshed {len(files)} files "
          f"({sum(entry['bytes'] for entry in files)} bytes) from {source}@{commit}")
    print(f"vendor_ascendep: wrote {MANIFEST}")
    if as_json:
        print(json.dumps(manifest, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify or refresh the vendored ascendep adapter sources.",
        epilog="--check exits 0 only when every digest in VENDOR.json matches the tree on disk.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true",
                        help="recompute sha256 for every vendored file and fail on any drift")
    action.add_argument("--list", action="store_true", help="print the recorded manifest")
    action.add_argument("--refresh", action="store_true",
                        help="re-copy the selected files from --source and rewrite VENDOR.json")
    parser.add_argument("--source", type=Path,
                        help="pinned ascendep checkout to copy from (required by --refresh)")
    parser.add_argument("--allow-other-commit", action="store_true",
                        help="let --refresh copy from a checkout that is not at the pinned commit")
    parser.add_argument("--json", action="store_true", help="print machine readable output")
    args = parser.parse_args(argv)
    if args.refresh:
        if args.source is None:
            parser.error("--refresh requires --source <pinned ascendep checkout>")
        return refresh(args.source, args.allow_other_commit, args.json)
    if args.source is not None:
        parser.error("--source is only meaningful with --refresh")
    if args.list:
        return print_manifest(args.json)
    return check()


if __name__ == "__main__":
    sys.exit(main())
