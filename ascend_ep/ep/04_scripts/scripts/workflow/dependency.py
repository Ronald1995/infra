"""Phase 2: dependency classification and the permitted install.

The catalog (``dependencies/manifest.yaml``) is the only source of truth for what a
dependency is, whether it may be installed here, and what the manual step is when it may
not. This module adds the workflow policy on top:

* ``check_only`` requests never install. The dry-run output is still recorded, so the
  report can show what *would* have been installed.
* ``install_missing`` runs the installer only when the request also allows the network
  (or a wheelhouse is supplied); otherwise the installable items stay visibly blocked.
* ``sudo``, system packages, drivers, firmware, CANN and device nodes are never touched.
  An entry that needs them is recorded with its ``manual_step``.
* a dependency that only blocks some operators leaves the rest of the matrix runnable.

After a successful install the doctor is re-run; the phase passes on the *re-probed*
state, never on the installer's exit code alone.
"""
from pathlib import Path

import yaml

from common.registry import ROOT
from scripts.workflow.phases import PhaseOutcome

#: Kinds the installer is allowed to handle on this host.
INSTALLABLE_KINDS = ("pip", "python_import", "binary")
#: Anything whose install command needs these is a host action, not an automation step.
HOST_ACTION_MARKERS = ("sudo", "apt-get", "apt ", "dnf", "yum", "zypper", "rpm ",
                       "modprobe", "systemctl", "setcap", "chmod")


def load_manifest(root=None):
    path = Path(root or ROOT) / "dependencies" / "manifest.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = {}
    for entry in data.get("dependencies") or ():
        entries[entry["id"]] = entry
    return entries, data


def _requires_host_action(entry):
    commands = entry.get("install") or ()
    text = " ".join(commands) if isinstance(commands, (list, tuple)) else str(commands)
    return any(marker in text for marker in HOST_ACTION_MARKERS)


def classify(report, manifest=None):
    """Split the unsatisfied dependencies into actionable buckets.

    ``installable``  a wheel/module this process may install,
    ``buildable``    a source build whose checkout exists,
    ``host``         host-provided: driver, firmware, CANN, device node, system package,
    ``unsatisfied``  reported by the doctor but absent from the catalog.

    Every bucket keeps the doctor's own detail so a report never paraphrases a reason.
    """
    manifest = manifest if manifest is not None else load_manifest()[0]
    buckets = {"installable": [], "buildable": [], "host": [], "unsatisfied": [], "ok": []}
    for check in (report or {}).get("checks") or ():
        identifier = check.get("id")
        entry = manifest.get(identifier)
        status = check.get("status")
        if status == "pass":
            buckets["ok"].append(identifier)
            continue
        record = {"id": identifier, "status": status, "detail": check.get("detail"),
                  "category": check.get("category"),
                  "blocking_for": list(check.get("blocking_for") or ()),
                  "suggested_action": check.get("suggested_action")}
        if entry is None:
            buckets["unsatisfied"].append(record)
            continue
        record.update({
            "kind": entry.get("kind"), "version": entry.get("version"),
            "installable": bool(entry.get("installable")),
            "install_reason": entry.get("install_reason"),
            "manual_step": entry.get("manual_step"),
            "manual_step_owner": entry.get("manual_step_owner"),
            "optional": bool(entry.get("optional")),
        })
        if entry.get("kind") == "host_provided" or not entry.get("installable"):
            buckets["host"].append(record)
        elif entry.get("kind") == "native_build" or _requires_host_action(entry):
            checkout = (entry.get("check") or {}).get("path")
            present = bool(checkout) and Path(checkout).exists()
            record["source_present"] = present
            if present:
                buckets["buildable"].append(record)
            else:
                record.setdefault("install_reason",
                                  "the pinned source checkout is absent, so the build cannot run")
                buckets["host"].append(record)
        elif entry.get("kind") in INSTALLABLE_KINDS:
            if status == "warn" and record.get("optional"):
                buckets["ok"].append(identifier)
            else:
                buckets["installable"].append(record)
        else:
            buckets["host"].append(record)
    return buckets


def _blocking_groups(buckets):
    groups = set()
    for record in buckets["host"] + buckets["installable"] + buckets["buildable"]:
        groups.update(record.get("blocking_for") or ())
    return sorted(groups)


def handler_check(ctx):
    """``dependency_check``: classify what the doctor found."""
    report = ctx.data.get("doctor")
    if report is None:
        return PhaseOutcome.failed("the environment report is missing; "
                                   "`environment_discovery` did not run")
    manifest = load_manifest(ctx.root)[0]
    buckets = classify(report, manifest)
    path = ctx.write_json("dependency_report.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id,
        "policy": ctx.request.environment.dependency_policy,
        "allow_network": ctx.request.environment.allow_network,
        "allow_sudo": ctx.request.environment.allow_sudo,
        "wheelhouse": ctx.request.environment.wheelhouse,
        **{name: value for name, value in buckets.items()},
        "blocking_groups": _blocking_groups(buckets),
    })
    ctx.data["dependency"] = buckets
    hard_blocked = [record for record in buckets["host"]
                    if "python" == record["id"] or (
                        set(record.get("blocking_for") or ()) >= {"reporting"} and
                        record.get("status") == "fail")]
    installable = [record["id"] for record in buckets["installable"]]
    if hard_blocked:
        return PhaseOutcome.blocked(
            "a host dependency the workflow may not install is missing: "
            + "; ".join(f"{record['id']} ({record.get('detail')})" for record in hard_blocked),
            artifacts=[path], detail={"hard_blocked": hard_blocked})
    if installable or buckets["buildable"]:
        return PhaseOutcome.partial(
            f"{len(installable)} installable and {len(buckets['buildable'])} buildable "
            f"dependency/dependencies are unsatisfied: "
            + ", ".join(installable + [record["id"] for record in buckets["buildable"]]),
            artifacts=[path], detail={"installable": installable})
    if buckets["host"] or buckets["unsatisfied"]:
        names = [record["id"] for record in buckets["host"] + buckets["unsatisfied"]]
        return PhaseOutcome.partial(
            "some capabilities are blocked by host dependencies that cannot be installed "
            "here: " + ", ".join(names) + " (each keeps its manual step in the report)",
            artifacts=[path])
    return PhaseOutcome.passed([path])


def _installer_argv(ctx, dry_run=False):
    argv = ["bash", str(ctx.root / "scripts" / "install_deps.sh"),
            "--profile", _profile(ctx.request)]
    if ctx.request.environment.wheelhouse:
        argv += ["--wheelhouse", ctx.request.environment.wheelhouse]
    if dry_run:
        argv.append("--dry-run")
    return argv


def _profile(request):
    if request.environment.device == "cpu":
        return "cpu"
    return "all"


def handler_install(ctx):
    """``dependency_install``: run the installer when it is allowed to do anything."""
    buckets = ctx.data.get("dependency") or {}
    installable = buckets.get("installable") or []
    buildable = buckets.get("buildable") or []
    request = ctx.request
    if request.environment.dependency_policy != "install_missing":
        return PhaseOutcome.skipped(
            "the request's dependency policy is `check_only`: nothing was installed. "
            + (f"Installable items: {', '.join(record['id'] for record in installable)}."
               if installable else "No installable item is missing."))
    if not installable and not buildable:
        return PhaseOutcome.skipped("nothing installable is missing")
    offline = bool(request.environment.wheelhouse)
    if not offline and not request.environment.allow_network:
        return PhaseOutcome.blocked(
            "installing would need the network, which the request did not authorise "
            "(set environment.allow_network or supply a wheelhouse): "
            + ", ".join(record["id"] for record in installable))
    if any(_requires_host_action({"install": record.get("suggested_action")})
           for record in installable) and not request.environment.allow_sudo:
        return PhaseOutcome.blocked(
            "an installable dependency needs sudo, which the request did not authorise")
    dry = ctx.commands.run(_installer_argv(ctx, dry_run=True), log_name="install-dry-run.log")
    ctx.record("dependency_install", dry, mode="dry-run")
    install = ctx.commands.run(_installer_argv(ctx), log_name="install.log")
    ctx.record("dependency_install", install, mode="install")
    recheck = ctx.commands.benchmark("doctor", "--profile", _profile(request), "--json",
                                     log_name="doctor-after-install.json.log")
    ctx.record("dependency_install", recheck, mode="recheck")
    report = recheck.json()
    if not isinstance(report, dict):
        return PhaseOutcome.failed(
            "the doctor re-check produced no JSON report after installing; the install "
            f"cannot be proven (exit {recheck.exit_code})")
    buckets_after = classify(report, load_manifest(ctx.root)[0])
    path = ctx.write_json("dependency_after_install.json", {
        "schema_version": 1, "before": buckets, "after": buckets_after,
        "install_exit_code": install.exit_code,
        "install_log": str(install.log_path),
    })
    ctx.data["doctor"] = report
    ctx.data["dependency"] = buckets_after
    if install.exit_code != 0:
        return PhaseOutcome.failed(
            f"`install_deps.sh` exited {install.exit_code}; see {install.log_path}",
            artifacts=[path])
    if buckets_after["installable"]:
        return PhaseOutcome.partial(
            "some installable dependencies are still unsatisfied after the install: "
            + ", ".join(record["id"] for record in buckets_after["installable"]),
            artifacts=[path])
    return PhaseOutcome.passed([path], detail={"installed": [r["id"] for r in installable]})
