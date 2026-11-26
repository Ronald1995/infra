"""Dependency readiness per operator: the coverage report the catalog exists for.

Answers, for every registered operator, three questions a reader actually has:

1. which dependency environment runs it,
2. which dependencies it needs (the transitive import closure, not a guess),
3. whether each of those is ready on this machine -- and if not, why.

```sh
python scripts/dependency_report.py                       # Markdown table
python scripts/dependency_report.py --json               # machine readable
python scripts/dependency_report.py --out reports/deps.md
```

The report is generated from the manifest, the operator registry and a doctor run, so it
cannot drift from the installation system: a dependency that is missing from the catalog
shows up as ``UNMAPPED`` instead of disappearing, and a blocked dependency names the
operator it blocks.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.registry import ROOT, environments, registry      # noqa: E402
from scripts import doctor, dependency_closure                 # noqa: E402

#: Statuses a dependency check can report, in the order a reader cares about.
ORDER = ("pass", "warn", "fail", "skip")


def build(root=None, doctor_report=None, profile="all"):
    """Assemble the per-operator dependency view."""
    root = Path(root or ROOT)
    data = doctor.load_manifest(root / "dependencies" / "manifest.yaml")
    entries = {entry["id"]: entry for entry in data["dependencies"]}
    report = doctor_report or doctor.run_doctor(profile=profile, root=root)
    status = {check.id: check.status for check in report.checks}
    detail = {check.id: check.detail for check in report.checks}
    operators = registry()
    environments_ = environments()
    rows = []
    declared = defaultdict(set)
    for entry in data["dependencies"]:
        for token in doctor.tokens(entry.get("required_by") or ()):
            if token.startswith("operator:"):
                declared[token.split(":", 1)[1]].add(entry["id"])
    for operator_id, spec in sorted(operators.items()):
        deps, modules, unmapped = dependency_closure.resolve(
            operator_id, spec.environment, data, root)
        deps = sorted(set(deps) | declared.get(operator_id, set()))
        states = {dep: status.get(dep, "skip") for dep in deps}
        blocking = [dep for dep, state in states.items() if state == "fail"]
        unprobed = [dep for dep, state in states.items() if state == "skip"]
        rows.append({
            "operator": operator_id,
            "status": spec.status,
            "environment": spec.environment,
            "isolated": bool(environments_.get(spec.environment, {}).get("isolated")),
            "dependencies": deps,
            "states": states,
            "ready": (None if unprobed else not blocking and not unmapped),
            "blocked_by": blocking,
            "not_probed": unprobed,
            "unmapped_modules": sorted(unmapped),
            "details": {dep: detail.get(dep) for dep in blocking},
        })
    environment_rows = []
    for name, entry in sorted(environments_.items()):
        members = sorted(dep for dep, item in entries.items() if item["environment"] == name)
        environment_rows.append({
            "id": name, "isolated": bool(entry.get("isolated")),
            "inherits": entry.get("inherits"),
            "requirements": entry.get("requirements") or [],
            "members": members,
            "ready": all(status.get(dep, "skip") in ("pass", "skip", "warn")
                         for dep in members),
        })
    return {
        "schema_version": 1,
        "profile": profile,
        "doctor_exit_code": report.exit_code,
        "counts": dict(report.counts),
        "operators": rows,
        "environments": environment_rows,
        "dependencies": [
            {"id": entry["id"], "ownership": entry["ownership"],
             "environment": entry["environment"], "pin": entry.get("pin", {}).get("kind"),
             "installable": entry["installable"], "status": status.get(entry["id"], "skip"),
             "detail": detail.get(entry["id"]),
             "blocks": sorted(token for token in doctor.tokens(entry.get("required_by") or ())
                              if token.startswith("operator:"))}
            for entry in sorted(data["dependencies"], key=lambda item: item["id"])
        ],
    }


def render_markdown(report):
    lines = [f"# Dependency coverage ({report['profile']})", "",
             f"doctor: {report['counts']} (exit {report['doctor_exit_code']})", "",
             "## Environments", "",
             "| Environment | Isolated | Inherits | Requirements | Members | Ready |",
             "| --- | --- | --- | --- | --- | --- |"]
    for entry in report["environments"]:
        lines.append(f"| {entry['id']} | {'yes' if entry['isolated'] else 'no'} | "
                     f"{entry['inherits'] or '-'} | {', '.join(entry['requirements']) or '-'} | "
                     f"{', '.join(entry['members'])} | "
                     f"{'yes' if entry['ready'] else 'no'} |")
    lines += ["", "## Operators", "",
              "| Operator | Adapter status | Environment | Dependencies | Ready | Blocked by |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in report["operators"]:
        blocked = ", ".join(f"{dep} ({row['details'].get(dep)})" for dep in row["blocked_by"])
        unmapped = ", ".join(f"UNMAPPED:{module}" for module in row["unmapped_modules"])
        reason = "; ".join(item for item in (blocked, unmapped) if item) or "-"
        ready = ("not probed" if row["ready"] is None
                 else "yes" if row["ready"] else "no")
        lines.append(f"| `{row['operator']}` | {row['status']} | {row['environment']} | "
                     f"{', '.join(row['dependencies']) or '-'} | {ready} | {reason} |")
    ready = sum(1 for row in report["operators"] if row["ready"] is True)
    unknown = sum(1 for row in report["operators"] if row["ready"] is None)
    summary = f"{ready}/{len(report['operators'])} operator(s) have every dependency ready"
    if unknown:
        summary += f"; {unknown} could not be judged under this profile"
    lines += ["", summary + ".", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="all", choices=sorted(doctor.PROFILES))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = build(profile=args.profile)
    text = json.dumps(report, indent=2) if args.json else render_markdown(report)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    ready = sum(1 for row in report["operators"] if row["ready"])
    return 0 if ready == len(report["operators"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
