"""Communication-runtime diagnosis: HCOMM, SHMEM, npu_patch, DeepEP, MoonEP, MPI, ASC-COMM.

Why a separate command: every operator family in this repository sits on a
different communication runtime, and the failure mode is always the same shape --
the adapter reports ``dependency_missing`` and the run stops somewhere far from the
cause.  This CLI probes exactly the C-category dependencies plus the two platform
facts those paths cannot work without: ``/dev/shm`` (SHMEM rendezvous) and a
library path that actually reaches the runtime (``LD_LIBRARY_PATH`` coverage).

Exit code is 0 only when every probed dependency passed; a warning keeps it 0 and
degrades the affected family in the matrix printed by ``scripts/doctor.py``.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/check_communication.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.doctor import (  # noqa: E402  (path bootstrap must run first)
    SCRIPTS,
    STATUSES,
    check_ld_library_path,
    render,
    report_with,
    run_doctor,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/check_communication.py",
        description="Check the communication runtimes (HCOMM, SHMEM, npu_patch, DeepEP, MoonEP, "
                    "MPI, ASC-COMM kernels) and the platform facts they need.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--root", default=None, help="SCRIPTS root (default: the parent of scripts/)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else SCRIPTS
    try:
        report = run_doctor(profile="communication", categories=("C",), root=root)
    except (OSError, ValueError) as exc:
        print(f"check_communication: {exc}", file=sys.stderr)
        return 2
    # `/dev/shm` (category C) is already part of the C probe; the library path is a
    # category B platform check, but without it no runtime library resolves.
    report = report_with(report, [check_ld_library_path(None)], profile="communication")

    if args.json:
        payload = {
            "schema_version": 1,
            "counts": {status: report.counts.get(status, 0) for status in STATUSES},
            "checks": report.as_dict()["checks"],
            "exit_code": report.exit_code,
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return report.exit_code

    print(render(report, show_matrix=False))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
