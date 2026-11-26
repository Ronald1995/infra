"""Turn the dependency manifest into an executable installation plan.

The manifest is the single source of truth. This module reads it and answers one
question: *for this profile, on this machine, what exactly must be fetched, built,
installed and verified?* It never downloads, builds or installs anything itself --
``scripts/install_deps.sh`` executes the plan -- which is what makes the plan reviewable
(dry run), testable (offline unit tests) and safe (the shell has no URLs or pins of its
own).

```sh
python scripts/dependency_plan.py --profile all                 # human summary
python scripts/dependency_plan.py --profile all --json          # the plan
python scripts/dependency_plan.py --profile all --script        # the bash to execute
python scripts/dependency_plan.py --profile all --offline \
    --source-cache /data/ep-sources --wheelhouse /data/ep-wheels
```

Plan rules:

* a dependency is selected when its category is in the profile;
* every step of an isolated environment runs with that environment's interpreter, so
  two packages that install the same module name never share a site-packages directory;
* ``offline`` forbids fetch steps: a source cache without the pinned checkout is a
  blocker, not a reason to clone;
* ``check_only`` keeps only the verify steps;
* a host-provided dependency is never a step: it becomes a blocker with its manual step;
* a source dependency is only installable when the manifest pins it. An unpinned entry is
  reported as a blocker, never resolved to a branch head.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.registry import ROOT                       # noqa: E402
from scripts import doctor                             # noqa: E402

MANIFEST = ROOT / "dependencies" / "manifest.yaml"
PYTHON = "python"

#: Blockers that mean the catalog itself is unusable for this machine (an entry the profile
#: needs that is marked not installable).  A missing wheel source or an unprepared offline
#: cache is *advisory*: it is printed with the exact manual step, the rest of the plan still
#: runs, and the doctor's verdict decides.  Only these make the planner exit non-zero.
ACTIONABLE_BLOCKERS = ("not-installable",)


class PlanError(ValueError):
    """A plan that cannot be built from the catalog."""


def load(root=None):
    root = Path(root or ROOT)
    data = doctor.load_manifest(Path(root) / "dependencies" / "manifest.yaml")
    environments = doctor.load_environments(root)
    return data, environments


def selected_entries(data, profile, environments, only_environments=None):
    """Entries of the profile, grouped by environment in installation order."""
    if profile not in doctor.PROFILES:
        raise PlanError(f"unknown profile {profile!r}; choose from {sorted(doctor.PROFILES)}")
    categories = set(doctor.PROFILES[profile])
    entries = [entry for entry in data["dependencies"] if entry["category"] in categories]
    if only_environments:
        unknown = set(only_environments) - set(environments)
        if unknown:
            raise PlanError(f"unknown environment(s) {sorted(unknown)}")
        entries = [entry for entry in entries if entry["environment"] in only_environments]
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["environment"]].append(entry)
    order = ["base"] + sorted(name for name in grouped if name != "base")
    return [(name, grouped[name]) for name in order if grouped[name]]


def _interpreter(environment, environments):
    """The interpreter a step runs with: base uses the current venv, isolated ones a copy."""
    if environment == "base" or not environments.get(environment, {}).get("isolated"):
        return PYTHON
    return f"$EP_ENV_ROOT/{environment}/bin/python"


def _pip_step(entry, environment, options, environments):
    spec = entry["install_spec"]
    python = _interpreter(environment, environments)
    packages = " ".join(spec.get("packages") or ())
    requirements = spec.get("requirements")
    wheel_source = spec.get("wheel_source") or {}
    flags = []
    if options["wheelhouse"]:
        flags += ["--no-index", f"--find-links {options['wheelhouse']}"]
    elif wheel_source.get("required") and wheel_source.get("env_var"):
        flags += ["--extra-index-url", f"\"${wheel_source['env_var']}\""]
    index = " ".join(flags)
    target = f"-r {requirements}" if requirements else packages
    command = f"{python} -m pip install {index} {target}".replace("  ", " ").strip()
    steps = [{"kind": "pip", "dependency": entry["id"], "environment": environment,
              "command": command, "requires_wheelhouse": bool(wheel_source.get("required")),
              "wheel_source": wheel_source,
              "key": (environment, requirements or packages,
                      tuple(spec.get("packages") or ()))}]
    return steps


def _source_steps(entry, environment, options, environments):
    spec = entry["install_spec"]
    source = spec["source"]
    pin = source.get("pin") or {}
    directory = source["directory"]
    target = (str(ROOT) if pin.get("kind") == "in_repo"
              else f"{options['source_cache']}/{directory}")
    python = _interpreter(environment, environments)
    steps = []
    if pin.get("kind") == "commit":
        fetch = (f"fetch_git {entry['id']} {source['url']} {target} {pin['value']}")
    elif pin.get("kind") == "release_tag":
        fetch = (f"fetch_tag {entry['id']} {source['url']} {target} {pin['value']}")
    elif pin.get("kind") == "in_repo":
        fetch = None
    else:
        raise PlanError(f"{entry['id']}: source dependency without a pin")
    if fetch:
        steps.append({"kind": "fetch", "dependency": entry["id"], "environment": environment,
                      "command": fetch, "offline": bool(options["offline"]),
                      "url": source["url"], "target": target, "pin": pin})
    for subdirectory in spec.get("subdirectories") or ():
        steps.append({"kind": "check_path", "dependency": entry["id"],
                      "environment": environment,
                      "command": f"require_path {entry['id']} {target}/{subdirectory}",
                      "path": f"{target}/{subdirectory}"})
    for command in spec.get("steps") or ():
        raw = command["run"]
        if raw.startswith(("pip ", "pip3 ")):
            raw = f"{python} -m {raw}"
        steps.append({"kind": "install", "dependency": entry["id"],
                      "environment": environment,
                      "command": f"run_in {target} {raw}",
                      "raw": command["run"], "target": target})
    return steps


def _verify_steps(entry, environment, options, environments):
    python = _interpreter(environment, environments)
    steps = []
    for command in entry["install_spec"].get("verify") or ():
        if command.strip().startswith("python "):
            command = command.replace("python ", f"{python} ", 1)
        steps.append({"kind": "verify", "dependency": entry["id"],
                      "environment": environment, "command": f"verify {entry['id']} {command}",
                      "raw": command})
    return steps


def build_plan(profile, options, root=None):
    """The full, ordered plan for one profile."""
    data, environments = load(root)
    plan = {"schema_version": 1, "profile": profile,
            "mode": "offline" if options["offline"] else "online",
            "check_only": bool(options["check_only"]),
            "source_cache": options["source_cache"], "wheelhouse": options["wheelhouse"],
            "environments": {name: {"isolated": bool(entry.get("isolated")),
                                    "inherits": entry.get("inherits"),
                                    "requirements": entry.get("requirements") or []}
                             for name, entry in environments.items()},
            "steps": [], "blocked": [], "skipped": [], "verify_only": []}
    seen_pip = set()
    for environment, entries in selected_entries(data, profile, environments,
                                                 options["environments"]):
        if environments.get(environment, {}).get("isolated") and not options["check_only"]:
            plan["steps"].append({
                "kind": "environment", "dependency": None, "environment": environment,
                "command": f"make_env {environment} "
                           f"{environments[environment].get('inherits') or 'base'}",
                "requirements": environments[environment].get("requirements") or [],
            })
        for entry in sorted(entries, key=lambda item: item["id"]):
            spec = entry["install_spec"]
            mode = spec["mode"]
            if mode in ("manual", "none"):
                if mode == "none" and (entry["ownership"] in ("vendored", "in_repo")
                                       or entry.get("provisioned_by")):
                    # Nothing to install: the source is here, it only has to be verified.
                    plan["verify_only"].append({
                        "dependency": entry["id"], "environment": environment,
                        "ownership": entry["ownership"],
                        "provisioned_by": entry.get("provisioned_by"),
                        "reason": (entry.get("install_reason")
                                   or ("provisioned by the %s" % entry["provisioned_by"]
                                       if entry.get("provisioned_by")
                                       else "already present in this repository")),
                    })
                    plan["steps"].extend(
                        _verify_steps(entry, environment, options, environments))
                else:
                    plan["blocked"].append(blocker(entry, "manual"))
                continue
            if not entry.get("installable"):
                plan["blocked"].append(blocker(entry, "not-installable"))
                continue
            if mode == "system":
                plan["blocked"].append(blocker(entry, "system"))
                continue
            if mode == "pip":
                wheel_source = spec.get("wheel_source") or {}
                offline_ok = bool(options["wheelhouse"]) or not wheel_source.get("required")
                if options["offline"] and not offline_ok:
                    plan["blocked"].append(blocker(entry, "wheelhouse"))
                    continue
                if wheel_source.get("required") and not (
                        options["wheelhouse"] or os.environ.get(
                            wheel_source.get("env_var") or "")):
                    plan["blocked"].append(blocker(entry, "wheel_source"))
                    continue
                steps = _pip_step(entry, environment, options, environments)
                if steps and steps[0]["key"] in seen_pip:
                    # One `pip install -r` per requirements file and environment; the
                    # per-dependency verify steps still run.
                    steps = []
                else:
                    seen_pip.update(step["key"] for step in steps)
            elif mode == "source":
                if options["offline"]:
                    plan["steps"].extend(_offline_source_steps(entry, environment, options,
                                                               environments))
                    continue
                steps = _source_steps(entry, environment, options, environments)
            else:  # pragma: no cover - validated by the catalog
                raise PlanError(f"{entry['id']}: unsupported install mode {mode!r}")
            if not options["check_only"]:
                plan["steps"].extend(steps)
            plan["steps"].extend(_verify_steps(entry, environment, options, environments))
    plan["counts"] = {
        "dependencies": len(plan["steps"]),
        "fetch": sum(1 for step in plan["steps"] if step["kind"] == "fetch"),
        "install": sum(1 for step in plan["steps"] if step["kind"] in ("pip", "install")),
        "verify": sum(1 for step in plan["steps"] if step["kind"] == "verify"),
        "blocked": len(plan["blocked"]),
        "verify_only": len(plan["verify_only"]),
    }
    return plan


def _offline_source_steps(entry, environment, options, environments):
    """Offline: the checkout must already be in the source cache at the pinned commit."""
    source = entry["install_spec"]["source"]
    pin = source.get("pin") or {}
    target = (str(ROOT) if pin.get("kind") == "in_repo"
              else f"{options['source_cache']}/{source['directory']}")
    steps = []
    if pin.get("kind") == "commit":
        steps.append({"kind": "verify_cache", "dependency": entry["id"],
                      "environment": environment,
                      "command": f"require_checkout {entry['id']} {target} {pin.get('value')}",
                      "target": target, "pin": pin})
    elif pin.get("kind") == "release_tag":
        steps.append({"kind": "verify_cache", "dependency": entry["id"],
                      "environment": environment,
                      "command": f"require_tag {entry['id']} {target} {pin.get('value')}",
                      "target": target, "pin": pin})
    # An in-repo build (l0-native-bridges) needs no cache: its source is in the tree.
    python = _interpreter(environment, environments)
    for command in entry["install_spec"].get("steps") or ():
        raw = command["run"]
        if raw.startswith(("pip ", "pip3 ")):
            raw = f"{python} -m {raw}"
        steps.append({"kind": "install", "dependency": entry["id"],
                      "environment": environment,
                      "command": f"run_in {target} {raw}",
                      "raw": command["run"], "target": target})
    steps.extend(_verify_steps(entry, environment, options, environments))
    return steps


def blocker(entry, reason):
    """One dependency the plan cannot install here, with the exact reason."""
    spec = entry["install_spec"]
    reasons = {
        "manual": "the dependency is provided by the deployment, not by this repository",
        "in-repo": "the dependency is already in this repository; verify it, do not install it",
        "not-installable": "the catalog marks this dependency as not installable here",
        "system": "installing it needs root (system package); run the command yourself",
        "wheelhouse": "offline install needs a wheelhouse for this wheel",
        "wheel_source": "the wheel is not on PyPI: set the index variable or pass --wheelhouse",
    }
    detail = {
        "dependency": entry["id"],
        "environment": entry["environment"],
        "kind": entry["kind"],
        "ownership": entry["ownership"],
        "reason": entry.get("install_reason") or reasons.get(reason, reason),
        "cause": reason,
        "manual_step": entry.get("manual_step") or spec.get("manual"),
    }
    if reason == "wheel_source" and not detail["manual_step"]:
        variable = (spec.get("wheel_source") or {}).get("env_var") or "EP_ASCEND_WHEEL_INDEX"
        detail["manual_step"] = (
            f"Pass --wheelhouse <directory with the Ascend wheels>, or export {variable} "
            f"pointing at the index that serves {entry['id']} for this CANN/torch triple.")
    if reason == "wheelhouse" and not detail["manual_step"]:
        detail["manual_step"] = (
            "An offline install can only use a local wheelhouse: pass --wheelhouse with the "
            "Ascend wheels, or run once with --allow-network and the index variable set.")
    if reason == "system" and spec.get("system"):
        detail["system"] = spec["system"]
    if reason == "wheel_source":
        detail["wheel_source"] = spec.get("wheel_source")
    return detail


# ------------------------------------------------------------------ rendering --
def render_summary(plan):
    lines = [f"profile {plan['profile']} ({plan['mode']}"
             + (", check-only" if plan["check_only"] else "") + ")",
             f"source cache: {plan['source_cache']}",
             f"wheelhouse: {plan['wheelhouse'] or '(none)'}"]
    for step in plan["steps"]:
        prefix = step["kind"]
        environment = f" [{step['environment']}]" if step.get("environment") else ""
        lines.append(f"  {prefix:<12} {step.get('dependency') or step['environment'] or ''}"
                     f"{environment}: {step['command']}")
    if plan.get("verify_only"):
        lines.append("verify only (already in this repository):")
        for item in plan["verify_only"]:
            lines.append(f"  {item['dependency']}: {item['reason']}")
    if plan["blocked"]:
        lines.append("blocked:")
        for item in plan["blocked"]:
            step = f"; manual: {item['manual_step']}" if item.get("manual_step") else ""
            lines.append(f"  {item['dependency']}: {item['reason']}{step}")
    counts = plan["counts"]
    lines.append(f"plan: {counts['fetch']} fetch, {counts['install']} install, "
                 f"{counts['verify']} verify, {counts['blocked']} blocked")
    return "\n".join(lines)


def render_script(plan):
    """The bash the installer executes. Every step keeps its own exit code and log."""
    lines = ["#!/usr/bin/env bash",
             "# Generated by scripts/dependency_plan.py -- do not edit, edit the manifest.",
             "set -uo pipefail",
             'EP_STEP_LOG="${EP_STEP_LOG:-$PWD/.install-steps.log}"',
             'printf "" >"$EP_STEP_LOG"',
             "",
             'run_in() { local dir="$1"; shift; ( cd "$dir" && "$@" ); }',
             "",
             'require_path() { local what="$1" path="$2";'
             ' if [ ! -e "$path" ]; then'
             ' echo "BLOCKED: $what: $path is missing (the pinned checkout does not contain it)" >&2;'
             ' return 1; fi; }',
             "",
             'require_checkout() { local what="$1" dir="$2" want="$3";'
             ' if [ ! -d "$dir/.git" ]; then'
             ' echo "BLOCKED: $what: no checkout at $dir; run once with the network" >&2; return 1; fi;'
             ' local have; have="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)";'
             ' if [ "$want" != "$have" ]; then'
             ' echo "BLOCKED: $what: checkout at $dir is $have, the manifest pins $want" >&2; return 1; fi;'
             ' echo "verified: $what at $have"; }',
             "",
             'require_tag() { local what="$1" dir="$2" tag="$3";'
             ' if [ ! -d "$dir/.git" ]; then'
             ' echo "BLOCKED: $what: no checkout at $dir; run once with the network" >&2; return 1; fi;'
             ' local commit; commit="$(git -C "$dir" rev-parse "refs/tags/$tag^{commit}" 2>/dev/null || true)";'
             ' if [ -z "$commit" ]; then'
             ' echo "BLOCKED: $what: tag $tag is not in $dir" >&2; return 1; fi;'
             ' echo "verified: $what tag $tag resolves to $commit"; }',
             "",
             'fetch_git() { local what="$1" url="$2" dir="$3" commit="$4";'
             ' if [ -d "$dir/.git" ]; then'
             ' have="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)";'
             ' if [ "$have" = "$commit" ]; then echo "cached: $what at $commit"; return 0; fi;'
             ' git -C "$dir" fetch --tags origin || return 1;'
             ' else mkdir -p "$(dirname "$dir")";'
             ' git clone --no-checkout "$url" "$dir" || return 1; fi;'
             ' git -C "$dir" checkout --quiet "$commit" || return 1;'
             ' have="$(git -C "$dir" rev-parse HEAD)";'
             ' if [ "$have" != "$commit" ]; then'
             ' echo "BLOCKED: $what: checked out $have, expected $commit" >&2; return 1; fi;'
             ' echo "fetched: $what at $have"; }',
             "",
             'fetch_tag() { local what="$1" url="$2" dir="$3" tag="$4";'
             ' if [ -d "$dir/.git" ]; then'
             ' have="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)";'
             ' want="$(git -C "$dir" rev-parse "refs/tags/$tag^{commit}" 2>/dev/null || true)";'
             ' if [ -n "$want" ] && [ "$have" = "$want" ]; then'
             ' echo "cached: $what tag $tag at $have"; return 0; fi;'
             ' git -C "$dir" fetch --tags origin || return 1;'
             ' else mkdir -p "$(dirname "$dir")";'
             ' git clone --no-checkout "$url" "$dir" || return 1; fi;'
             ' git -C "$dir" checkout --quiet "tags/$tag" || return 1;'
             ' have="$(git -C "$dir" rev-parse HEAD)";'
             ' echo "fetched: $what tag $tag resolved to $have (recorded as evidence)"; }',
             "",
             'make_env() { local name="$1" base="$2";'
             ' local dir="$EP_ENV_ROOT/$name";'
             ' if [ ! -x "$dir/bin/python" ]; then'
             ' echo "creating environment $name from $base";'
             ' "${EP_BASE_PYTHON:-python}" -m venv "$dir" || return 1; fi;'
             ' echo "environment: $name at $dir"; }',
             "",
             'record() { printf "%s\\t%s\\t%s\\n" "$1" "$2" "$3" >>"$EP_STEP_LOG"; }',
             "",
             'step() { local what="$1"; shift;'
             ' echo "+ $what";'
             ' "$@"; local status=$?;'
             ' record "$what" "$status" "$*";'
             ' if [ "$status" -ne 0 ]; then echo "BLOCKED: $what (exit $status)" >&2; return $status; fi; }',
             ""]
    for step in plan["steps"]:
        kind = step["kind"]
        dependency = step.get("dependency") or step.get("environment") or kind
        command = step["command"]
        if kind == "environment":
            lines.append(f'step "environment {step["environment"]}" make_env '
                         f'{step["environment"]} {step.get("requirements") and "base" or "base"}')
        elif kind in ("fetch", "verify_cache"):
            lines.append(f'step "{kind} {dependency}" {command}')
        elif kind == "check_path":
            lines.append(f'step "check {dependency}" {command}')
        elif kind in ("pip", "install"):
            lines.append(f'step "{kind} {dependency}" bash -c {shell_quote(command)}')
        elif kind == "verify":
            lines.append(f'step "verify {dependency}" bash -c {shell_quote(command)}')
        else:  # pragma: no cover
            raise PlanError(f"unsupported step kind {kind!r}")
    lines.append("")
    lines.append('echo "plan finished; run: bash scripts/install_deps.sh --check-only '
                 f'--profile {plan["profile"]}"')
    return "\n".join(lines)


def shell_quote(command):
    return "'" + str(command).replace("'", "'\\''") + "'"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="all", choices=sorted(doctor.PROFILES))
    parser.add_argument("--offline", action="store_true",
                        help="never fetch: require the pinned checkouts in --source-cache")
    parser.add_argument("--source-cache", default="$HOME/.cache/ep-benchmark/deps")
    parser.add_argument("--wheelhouse")
    parser.add_argument("--environment", action="append", dest="environments",
                        help="restrict the plan to one dependency environment")
    parser.add_argument("--check-only", action="store_true", help="only verify")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--script", action="store_true", help="print the bash to execute")
    parser.add_argument("--out", type=Path, help="write the plan JSON here")
    parser.add_argument("--no-fail", action="store_true",
                        help="exit 0 even when the plan has actionable blockers")
    args = parser.parse_args(argv)
    options = {"offline": args.offline, "source_cache": args.source_cache,
               "wheelhouse": args.wheelhouse, "check_only": args.check_only,
               "environments": args.environments}
    plan = build_plan(args.profile, options)
    if args.out:
        args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(plan, indent=2))
    elif args.script:
        print(render_script(plan))
    else:
        print(render_summary(plan))
    # A host-provided blocker is expected on a machine without CANN; an actionable one
    # (missing wheelhouse, unpinned installable package) means the plan cannot be run.
    actionable = [item for item in plan["blocked"]
                  if item["cause"] in ACTIONABLE_BLOCKERS]
    if actionable and not args.no_fail:
        print("BLOCKED: " + "; ".join(f"{item['dependency']} ({item['cause']})"
                                      for item in actionable), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
