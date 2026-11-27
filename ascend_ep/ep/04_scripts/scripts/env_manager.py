"""Manage the isolated dependency environments declared in dependencies/environments.

Four adapters cannot share one interpreter: ``deep-ep`` and ``deep-ep-v2`` both ship
``deep_ep``, and the two MoonEP branches both ship ``npu_patch``.  Each therefore
gets its own environment, and every one of them *inherits* the base interpreter's
torch / torch_npu stack rather than reinstalling it.

Two rules this tool enforces:

* It refuses to create an environment until the base interpreter proves it can run
  on the NPU (``--base-check``).  An isolated environment built on a broken NPU
  stack would fail much later and much less clearly.
* It never installs torch or torch_npu into a child environment.  ``inherits`` is
  the only supported route, which keeps the container's stack authoritative.

Usage
    python scripts/env_manager.py --base-check
    python scripts/env_manager.py --check
    python scripts/env_manager.py --create [--only deep_ep_lueelu]
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
DEFAULT_ENVS_ROOT = PACKAGE / "packages" / "envs"

PRESENT = "PRESENT"
ABSENT = "ABSENT"
BROKEN = "BROKEN"


def load_environments():
    from common.registry import environments

    return environments()


def venv_python(env_root, name):
    for candidate in ("bin/python", "Scripts/python.exe"):
        path = Path(env_root) / name / candidate
        if path.is_file():
            return path
    return None


def probe_interpreter(python):
    """Ask one interpreter what it can actually do, without loading a device."""
    script = (
        "import json,sys\n"
        "info={'python':sys.version.split()[0],'executable':sys.executable}\n"
        "try:\n"
        "    import torch\n"
        "    info['torch']=torch.__version__\n"
        "except Exception as exc:\n"
        "    info['torch']=None; info['torch_error']=f'{type(exc).__name__}: {exc}'\n"
        "try:\n"
        "    import torch_npu\n"
        "    info['torch_npu']=getattr(torch_npu,'__version__',None)\n"
        "    import torch as t\n"
        "    info['npu_available']=bool(t.npu.is_available())\n"
        "    info['npu_devices']=int(t.npu.device_count()) if info['npu_available'] else 0\n"
        "except Exception as exc:\n"
        "    info['torch_npu']=None; info['npu_available']=False; info['npu_devices']=0\n"
        "    info['torch_npu_error']=f'{type(exc).__name__}: {exc}'\n"
        "try:\n"
        "    import yaml\n"
        "    info['yaml']=True\n"
        "except Exception:\n"
        "    info['yaml']=False\n"
        "print(json.dumps(info))\n"
    )
    try:
        result = subprocess.run([str(python), "-c", script], capture_output=True,
                                text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {"error": (result.stderr or "").strip().splitlines()[-1:] or ["probe failed"]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "unparseable probe output"}


def base_check(python=None):
    """Is the base interpreter usable for NPU work at all?  (§6.1 of the spec)"""
    python = python or sys.executable
    info = probe_interpreter(python)
    info["usable"] = bool(info.get("torch") and info.get("torch_npu")
                          and info.get("npu_available"))
    return info


def check_env(name, entry, env_root, base_python=None):
    """Status of one declared environment (no device work)."""
    record = {"id": name, "isolated": bool(entry.get("isolated")),
              "inherits": entry.get("inherits"), "python_requirement": entry.get("python"),
              "members": list(entry.get("members") or []),
              "requirements": list(entry.get("requirements") or [])}
    python = venv_python(env_root, name)
    if not entry.get("isolated"):
        record.update(status=PRESENT, path=None,
                      note="not isolated: runs in the base interpreter")
        return record
    if python is None:
        record.update(status=ABSENT, path=str(Path(env_root) / name),
                      note="run --create to build it")
        return record
    info = probe_interpreter(python)
    record.update(path=str(python), interpreter=info)
    usable = bool(info.get("torch") and info.get("torch_npu"))
    record["status"] = PRESENT if usable else BROKEN
    if not usable:
        record["note"] = "inherits a base interpreter without torch/torch_npu"
    return record


def check_all(env_root=None, only=None):
    env_root = Path(env_root or DEFAULT_ENVS_ROOT)
    environments = load_environments()
    names = list(environments)
    if only:
        unknown = [item for item in only if item not in environments]
        if unknown:
            raise SystemExit(f"unknown environment(s): {', '.join(unknown)}")
        names = only
    return [check_env(name, environments[name], env_root) for name in names]


def create(name, entry, env_root, base_python=None, dry_run=False):
    """Create one isolated environment that inherits the base interpreter."""
    env_root = Path(env_root)
    target = env_root / name
    if not entry.get("isolated"):
        return {"id": name, "status": "SKIPPED", "reason": "not isolated"}
    if target.exists():
        return {"id": name, "status": "SKIPPED", "reason": "already exists",
                "path": str(target)}
    command = [str(base_python or sys.executable), "-m", "venv",
               "--system-site-packages", str(target)]
    if dry_run:
        return {"id": name, "status": "DRY_RUN", "command": command,
                "inherits": entry.get("inherits")}
    env_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {"id": name, "status": "FAILED",
                "reason": (result.stderr or "").strip()[-400:], "command": command}
    python = venv_python(env_root, name)
    info = probe_interpreter(python) if python else {"error": "no interpreter created"}
    usable = bool(info.get("torch") and info.get("torch_npu"))
    return {"id": name, "status": PRESENT if usable else BROKEN,
            "path": str(target), "interpreter": info,
            "command": command,
            "note": None if usable else "the inherited base stack is not usable"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--base-check", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--create", action="store_true")
    parser.add_argument("--only", type=lambda v: [i.strip() for i in v.split(",") if i.strip()])
    parser.add_argument("--root", type=Path, default=DEFAULT_ENVS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.base_check:
        info = base_check()
        print(json.dumps(info, indent=2) if args.json else
              f"python={info.get('python')} torch={info.get('torch')} "
              f"torch_npu={info.get('torch_npu')} npu_available={info.get('npu_available')} "
              f"devices={info.get('npu_devices')} usable={info.get('usable')}")
        return 0 if info.get("usable") else 1

    if args.check:
        records = check_all(args.root, args.only)
        if args.json:
            print(json.dumps({"environments": records}, indent=2))
        else:
            width = max(len(r["id"]) for r in records) + 2
            for record in records:
                detail = record.get("path") or record.get("note") or ""
                print(f"{record['id']:{width}s} {record['status']:10s} "
                      f"isolated={record['isolated']} inherits={record.get('inherits')} {detail}")
        return 0 if all(r["status"] in (PRESENT,) for r in records) else 1

    # --create refuses to build on a base interpreter that cannot use the NPU.
    base = base_check()
    if not base.get("usable") and not args.dry_run:
        print("refusing to create environments: the base interpreter cannot use the NPU "
              f"(torch={base.get('torch')} torch_npu={base.get('torch_npu')} "
              f"npu_available={base.get('npu_available')})", file=sys.stderr)
        return 2
    environments = load_environments()
    names = args.only or list(environments)
    records = [create(name, environments[name], args.root, dry_run=args.dry_run)
               for name in names]
    print(json.dumps({"environments": records}, indent=2) if args.json else
          "\n".join(f"{r['id']}: {r['status']} {r.get('reason') or r.get('path') or ''}"
                    for r in records))
    return 0 if all(r["status"] in (PRESENT, "SKIPPED", "DRY_RUN") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
