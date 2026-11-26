"""Compare two benchmark manifests and explain every difference.

A refactor is only safe if the benchmark matrix can be proved not to have moved.
This tool answers that question in the terms the acceptance criteria use: EP size,
case, level, suite, operator, parameter variant, parameters, task count, status and
unsupported reason.

It classifies each difference instead of only listing it:

``identical``
    the same task signature and the same status/reason;
``status``
    the same point now resolves to a different status (a capability or rule change);
``parameter``
    the same (case, operator, variant) now carries different parameters;
``scope``
    a point exists in one manifest and not the other (a real coverage change);
``count``
    the same signature appears a different number of times (a duplicate/expansion
    change).

Exit code is 0 when the manifests agree on every *executable* point, 1 otherwise, so
it can gate a refactor without failing on an intentional, documented status change.

Usage::

    python scripts/manifest_diff.py OLD_MANIFEST NEW_MANIFEST [--json PATH]
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def task_signature(task):
    """The identity of one planned point, excluding its status."""
    parameters = task.get("parameters") or {}
    return {
        "ep_size": task.get("ep_size") or task["config"]["distributed"]["ep_size"],
        "case_id": task["case_id"],
        "level": task["level"],
        "operator": task["backend_id"],
        "variant": parameters.get("variant"),
        "message_bytes": parameters.get("message_bytes"),
    }


def _key(signature):
    return (signature["ep_size"], signature["case_id"], signature["level"],
            signature["operator"], signature["variant"], signature["message_bytes"])


def _parameters(task):
    return json.dumps(task.get("parameters") or {}, sort_keys=True, default=str)


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["tasks"] if isinstance(data, dict) else data


def compare(old_tasks, new_tasks, executable=("not_run", "passed")):
    """Return a structured comparison of two task lists."""
    old = {}
    new = {}
    for task in old_tasks:
        old.setdefault(_key(task_signature(task)), []).append(task)
    for task in new_tasks:
        new.setdefault(_key(task_signature(task)), []).append(task)

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    shared = sorted(set(old) & set(new))

    status_changes, parameter_changes, order_only = [], [], []
    for key in shared:
        before, after = old[key], new[key]
        pairs = list(zip(sorted(before, key=lambda t: t["task_id"]),
                         sorted(after, key=lambda t: t["task_id"])))
        for left, right in pairs:
            if (left["status"], left["reason"]) != (right["status"], right["reason"]):
                status_changes.append({
                    "point": _key(task_signature(left)),
                    "old": {"status": left["status"], "reason": left["reason"]},
                    "new": {"status": right["status"], "reason": right["reason"]},
                })
            elif _parameters(left) != _parameters(right):
                parameter_changes.append({"point": _key(task_signature(left)),
                                          "old": left.get("parameters"),
                                          "new": right.get("parameters")})
        if len(before) != len(after):
            order_only.append({"point": key, "old_count": len(before),
                               "new_count": len(after)})

    def by_ep(keys):
        return dict(sorted(Counter(key[0] for key in keys).items()))

    old_executable = {_key(task_signature(task)) for task in old_tasks
                      if task["status"] in executable}
    new_executable = {_key(task_signature(task)) for task in new_tasks
                      if task["status"] in executable}
    return {
        "schema_version": 1,
        "totals": {"old": len(old_tasks), "new": len(new_tasks),
                   "old_points": len(old), "new_points": len(new)},
        "per_ep": {
            "old": dict(sorted(Counter(
                (task.get("ep_size") or task["config"]["distributed"]["ep_size"])
                for task in old_tasks).items())),
            "new": dict(sorted(Counter(
                (task.get("ep_size") or task["config"]["distributed"]["ep_size"])
                for task in new_tasks).items())),
        },
        "status_changes": status_changes,
        "parameter_changes": parameter_changes,
        "count_changes": order_only,
        "scope": {
            "removed": [{"point": key} for key in only_old],
            "added": [{"point": key} for key in only_new],
            "removed_by_ep": by_ep(only_old),
            "added_by_ep": by_ep(only_new),
        },
        "executable": {
            "old": len(old_executable),
            "new": len(new_executable),
            "identical": old_executable == new_executable,
            "removed": sorted(old_executable - new_executable),
            "added": sorted(new_executable - old_executable),
        },
    }


def render(result):
    totals = result["totals"]
    lines = [f"tasks: {totals['old']} -> {totals['new']}",
             f"distinct points: {totals['old_points']} -> {totals['new_points']}",
             f"tasks per EP size: {result['per_ep']['old']} -> {result['per_ep']['new']}"]
    executable = result["executable"]
    lines.append(f"executable points: {executable['old']} -> {executable['new']} "
                 f"({'identical' if executable['identical'] else 'CHANGED'})")
    for name in ("removed", "added"):
        entries = executable[name]
        if entries:
            lines.append(f"  executable {name}: {len(entries)}")
            for entry in entries[:10]:
                lines.append(f"    {entry}")
    for label, key in (("status changes", "status_changes"),
                       ("parameter changes", "parameter_changes"),
                       ("count changes", "count_changes")):
        entries = result[key]
        lines.append(f"{label}: {len(entries)}")
        for entry in entries[:10]:
            lines.append(f"  {entry}")
    scope = result["scope"]
    lines.append(f"scope removed: {len(scope['removed'])} ({scope['removed_by_ep']})")
    lines.append(f"scope added:   {len(scope['added'])} ({scope['added_by_ep']})")
    for name in ("removed", "added"):
        for entry in scope[name][:10]:
            lines.append(f"  {name}: {entry['point']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--json", type=Path, help="write the structured diff here")
    parser.add_argument("--allow-executable-change", action="store_true",
                        help="exit 0 even when the executable point set changed")
    args = parser.parse_args(argv)
    result = compare(load(args.old), load(args.new))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(render(result))
    if result["executable"]["identical"] or args.allow_executable_change:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
