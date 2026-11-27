"""Run and record the Native Build steps, without pretending a partial build is complete.

Three kinds of "native" work are easy to conflate, and the report must not:

* **in-repo kernels** (``kernels/asc_comm``, ``baselines/l0/transport/native``) --
  built here, and their own manifest says whether the *device* sources compiled;
* **third-party native extensions** (DeepEP, DeepGEMM-Ascend, npu_patch, SHMEM,
  cann-ops-transformer) -- each has its own build command from
  ``third_party/sources.yaml``;
* **pure-Python editable installs** (Megatron, MindSpeed, FSDPTurbo, Mega-MoE) --
  an install, not a compilation.  Reporting them as "built a kernel" would be a lie.

A step that fails is ``BUILD_FAILED`` with the tail of its log; a target whose
sources are absent is ``BLOCKED_DEPENDENCY``; a target that is present but has no
build steps is ``NOT_REQUIRED``.

Usage
    python scripts/native_build.py --list
    python scripts/native_build.py --run [--only asc_comm]
    python scripts/native_build.py --verify
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE = ROOT.parent.parent
CATALOGUE = PACKAGE / "third_party" / "sources.yaml"

BUILT = "BUILT"
BUILT_HOST_ONLY = "BUILT_HOST_ONLY"
BUILD_FAILED = "BUILD_FAILED"
BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
NOT_REQUIRED = "NOT_REQUIRED"
TIMEOUT = "TIMEOUT"

#: In-repo build targets.  `verify` reads the artifact the build claims to produce.
IN_REPO = {
    "asc_comm": {
        "build": ["bash", "kernels/asc_comm/build.sh", "--soc", "{soc}"],
        "artifact": "kernels/asc_comm/build/manifest.json",
        "kind": "npu_kernel",
    },
    "l0_bridge": {
        "build": ["bash", "baselines/l0/transport/native/build.sh"],
        "artifact": "baselines/l0/transport/native/build/libl0_transport_bridge.so",
        "kind": "host_extension",
    },
}


def load_catalogue(path=None):
    import yaml

    path = Path(path) if path else CATALOGUE
    if not path.is_file():
        return {"sources": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"sources": {}}


def artefact_state(target):
    """Read what a build actually produced, when the build writes a manifest."""
    path = ROOT / IN_REPO[target]["artifact"]
    if not path.is_file():
        return {"artifact": str(path), "present": False}
    state = {"artifact": str(path), "present": True}
    if path.suffix == ".json":
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {**state, "parse_error": True}
        engines = manifest.get("engines") or {}
        state["engines"] = {name: bool(value.get("built")) for name, value in engines.items()}
        state["reasons"] = {name: value.get("reason") for name, value in engines.items()
                            if not value.get("built")}
        state["device_kernels_built"] = any(state["engines"].values())
    return state


def third_party_targets(catalogue):
    """Sources whose catalogue entry declares a build step."""
    targets = {}
    for name, entry in (catalogue.get("sources") or {}).items():
        build = entry.get("build") or []
        install = entry.get("install") or []
        if not build and not install:
            continue
        targets[name] = {
            "directory": entry.get("directory"),
            "build": build,
            "install": install,
            "kind": "npu_kernel" if build else "python_install",
            "environment": entry.get("environment"),
        }
    return targets


def _run(command, cwd, timeout, dry_run):
    if dry_run:
        return {"status": "DRY_RUN", "command": command}
    try:
        result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": BUILD_FAILED, "command": command,
                "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - a hung toolchain must not crash the run
        return {"status": TIMEOUT, "command": command, "reason": str(exc)}
    if result.returncode != 0:
        tail = ((result.stderr or result.stdout or "").strip().splitlines()[-6:])
        return {"status": BUILD_FAILED, "command": command, "returncode": result.returncode,
                "tail": tail}
    return {"status": "OK", "command": command}


def run_target(name, timeout=1800, dry_run=False, soc="Ascend950", catalogue=None):
    """Build one target and record what it produced."""
    if name in IN_REPO:
        spec = IN_REPO[name]
        command = [part.format(soc=soc) for part in spec["build"]]
        outcome = _run(command, ROOT, timeout, dry_run)
        record = {"target": name, "kind": spec["kind"], "cwd": str(ROOT), **outcome}
        if not dry_run and outcome["status"] == "OK":
            state = artefact_state(name)
            record["artifact"] = state
            if spec["kind"] == "npu_kernel" and not state.get("device_kernels_built"):
                record["status"] = BUILT_HOST_ONLY
                record["reason"] = ("the host library linked, but no device kernel was "
                                    "built; the build manifest records which engine is "
                                    "missing")
            else:
                record["status"] = BUILT
        return record

    catalogue = catalogue or load_catalogue()
    targets = third_party_targets(catalogue)
    if name not in targets:
        raise SystemExit(f"unknown build target: {name}")
    entry = targets[name]
    directory = PACKAGE / "third_party" / (entry["directory"] or name)
    if not directory.is_dir():
        return {"target": name, "status": BLOCKED_DEPENDENCY,
                "reason": f"{directory} is not present; run "
                          f"`python scripts/fetch_third_party.py --check`"}
    steps = [[*step.split()] for step in entry["build"]] + \
            [[*step.split()] for step in entry["install"]]
    records = []
    for step in steps:
        outcome = _run(step, directory, timeout, dry_run)
        records.append(outcome)
        if outcome["status"] not in ("OK", "DRY_RUN"):
            return {"target": name, "kind": entry["kind"], "steps": records,
                    "status": outcome["status"], "reason": outcome.get("reason")}
    return {"target": name, "kind": entry["kind"], "steps": records,
            "status": NOT_REQUIRED if not steps else BUILT}


def all_targets(catalogue=None):
    catalogue = catalogue or load_catalogue()
    return list(IN_REPO) + list(third_party_targets(catalogue))


def verify(catalogue=None):
    """Report the artefact state of every build target without building anything."""
    catalogue = catalogue or load_catalogue()
    records = []
    for name in IN_REPO:
        state = artefact_state(name)
        records.append({"target": name, "artifact": state,
                        "status": BUILT if state.get("present") else BLOCKED_DEPENDENCY})
    for name, entry in third_party_targets(catalogue).items():
        directory = PACKAGE / "third_party" / (entry["directory"] or name)
        records.append({"target": name, "kind": entry["kind"],
                        "status": BLOCKED_DEPENDENCY if not directory.is_dir() else "PRESENT"})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--only", type=lambda v: [i.strip() for i in v.split(",") if i.strip()])
    parser.add_argument("--soc", default="Ascend950")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    catalogue = load_catalogue()
    if args.list:
        targets = all_targets(catalogue)
        print(json.dumps(targets, indent=2) if args.json else "\n".join(targets))
        return 0
    if args.verify:
        records = verify(catalogue)
        print(json.dumps({"targets": records}, indent=2) if args.json else
              "\n".join(f"{r['target']:28s} {r['status']}" for r in records))
        return 0 if all(r["status"] == BUILT for r in records) else 1

    names = args.only or all_targets(catalogue)
    records = [run_target(name, timeout=args.timeout, dry_run=args.dry_run,
                          soc=args.soc, catalogue=catalogue) for name in names]
    if args.json:
        print(json.dumps({"builds": records}, indent=2))
    else:
        for record in records:
            print(f"{record['target']:28s} {record['status']}"
                  + (f"  {record.get('reason') or ''}" if record.get("reason") else ""))
    return 0 if all(r["status"] in (BUILT, NOT_REQUIRED, "DRY_RUN") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
