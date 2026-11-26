"""Host-environment diagnosis: devices, mounts, permissions, env vars, library path.

Why a separate command: "the benchmark hangs on rank 3" and "the container cannot
open /dev/davinci3" produce the same report today, and neither names the mount or
the group that is actually wrong.  This CLI prints the host facts first (devices,
/dev/shm, RDMA, the environment this process inherited) and then the checks that
turn those facts into a verdict: device readability, deployment variables, HCCL
configuration and the library-path coverage that decide whether a runtime library
is reachable at all.

Exit code is 0 only when every probed environment fact passed.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/check_environment.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.doctor import (  # noqa: E402  (path bootstrap must run first)
    GIB,
    SCRIPTS,
    STATUSES,
    check_ld_library_path,
    check_shm,
    render,
    report_with,
    run_doctor,
)


def _environment_lines(report):
    info = report.platform
    shm = info.get("shm") or {}
    shm_text = f"{shm['total_bytes'] / GIB:.1f} GiB" if shm.get("total_bytes") else "absent"
    lines = ["host environment", f"  root        {info['root']}",
             f"  host        {info['host']}  ({info['os']})",
             f"  python      {info['python']}  {info['executable']}",
             f"  SoC         {info['soc'] or 'unknown'}  (source: {info['soc_source']})",
             f"  devices     {len(info['devices'])} device node(s)"
             + (f": {', '.join(info['devices'])}" if info["devices"] else ""),
             f"  /dev/shm    {shm_text}",
             f"  RDMA        {len(info['infiniband']['devices'])} device(s)"
             + (f": {', '.join(info['infiniband']['devices'])}" if info["infiniband"]["devices"] else ""),
             "  environment"]
    for name, value in info["env"].items():
        lines.append(f"    {name:<28} {value if value is not None else '-'}")
    library_path = info["ld_library_path"]
    lines.append(f"    {'LD_LIBRARY_PATH':<28} {library_path['value'] if library_path['value'] is not None else '-'}")
    if library_path["required"]:
        lines.append(f"    {'required but not covered':<28} "
                     f"{library_path['required'] if not library_path['covered'] else '-'}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/check_environment.py",
        description="Check the host environment the benchmark depends on: device nodes and their "
                    "permissions, /dev/shm, RDMA devices, deployment variables, HCCL configuration "
                    "and LD_LIBRARY_PATH coverage.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--root", default=None, help="SCRIPTS root (default: the parent of scripts/)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else SCRIPTS
    try:
        report = run_doctor(profile="multinode", categories=("D",), root=root)
    except (OSError, ValueError) as exc:
        print(f"check_environment: {exc}", file=sys.stderr)
        return 2
    # Category D already brings `network.rdma`; add the two platform facts that are
    # classified elsewhere but decide whether a device is reachable at all.
    report = report_with(report, [check_shm(None), check_ld_library_path(None)], profile="environment")

    if args.json:
        payload = {
            "schema_version": 1,
            "platform": report.platform,
            "counts": {status: report.counts.get(status, 0) for status in STATUSES},
            "checks": report.as_dict()["checks"],
            "exit_code": report.exit_code,
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return report.exit_code

    print(_environment_lines(report))
    print()
    print(render(report, show_matrix=False))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
