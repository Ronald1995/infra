"""CANN-side diagnosis: toolkit, driver, HCCL/HCOMM, npu-smi, torch_npu, pairing.

Why a separate command: the CANN stack is the only part of this benchmark where a
plausible-looking installation can still be unusable, because the pieces must come
from *one* release (driver, toolkit, HCCL, ccec, torch_npu) and the version the
site installed is often not the version the recipe assumed.  ``scripts/doctor.py
--profile npu`` answers the same questions but buries the pairing table in the
JSON; this CLI prints the triple first, then the B-category checks that prove it.

Exit code is 0 only when every probed CANN dependency passed and the observed
triple either matches a known-good row or cannot be compared at all.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python scripts/check_cann.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.doctor import (  # noqa: E402  (path bootstrap must run first)
    SCRIPTS,
    STATUSES,
    compatibility,
    load_manifest,
    render,
    run_doctor,
)


def _print_table(table):
    observed = table["observed"]
    lines = ["CANN / PyTorch / torch_npu pairing",
             f"  observed   cann={observed['cann'] or '-'}  torch={observed['torch'] or '-'}"
             f"  torch_npu={observed['torch_npu'] or '-'}",
             f"  CANN home  {table['cann_home'] or '-'}   version file  {table['version_file'] or '-'}",
             ""]
    lines.append(f"{'ROW':<26}  {'CANN':<10}  {'TORCH':<8}  {'TORCH_NPU':<14}  MATCH")
    lines.append("-" * 82)
    for row in table["rows"]:
        matches = row["matches"]
        verdict = "unknown" if matches is None else ("yes" if matches else "no")
        lines.append(f"{str(row['id']):<26}  {str(row['cann'] or '-'):<10}  "
                     f"{str(row['torch'] or '-'):<8}  {str(row['torch_npu'] or '-'):<14}  {verdict}")
    lines.append("")
    lines.append("  status=documented rows come from the vendor release matrix and are not "
                 "re-verified here;")
    lines.append("  status=probed rows were recorded on a machine.  An 'unknown' row means a "
                 "field could not be read.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/check_cann.py",
        description="Check CANN, the NPU driver, HCCL/HCOMM, npu-smi and torch_npu, and match the "
                    "installed triple against dependencies/manifest.yaml compatibility.rows.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--root", default=None, help="SCRIPTS root (default: the parent of scripts/)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else SCRIPTS
    try:
        data = load_manifest(root / "dependencies" / "manifest.yaml")
        report = run_doctor(profile="npu", categories=("B",), root=root)
    except (OSError, ValueError) as exc:
        print(f"check_cann: {exc}", file=sys.stderr)
        return 2
    table = compatibility(data)
    mismatched = [row["id"] for row in table["rows"] if row["matches"] is False]

    if args.json:
        payload = {
            "schema_version": 1,
            "observed": table["observed"],
            "cann_home": table["cann_home"],
            "version_file": table["version_file"],
            "compatibility": table["rows"],
            "matched": [row["id"] for row in table["rows"] if row["matches"]],
            "mismatched_rows": mismatched,
            "counts": {status: report.counts.get(status, 0) for status in STATUSES},
            "checks": report.as_dict()["checks"],
            "exit_code": 1 if (report.exit_code or mismatched) else 0,
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload["exit_code"]

    print(_print_table(table))
    print()
    print(render(report, show_matrix=False))
    if mismatched:
        print()
        print(f"note: the installed torch/torch_npu versions contradict row(s) {mismatched};")
        print("      install the triple from `python scripts/check_cann.py --json` -> compatibility[].matches")
    return 1 if (report.exit_code or mismatched) else 0


if __name__ == "__main__":
    sys.exit(main())
