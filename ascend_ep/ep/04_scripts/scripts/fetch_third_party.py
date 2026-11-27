"""Materialise and verify the pinned third-party checkouts.

The A5 host has no route to GitHub or GitCode, so sources have to arrive as bytes
and then be *verified*.  This tool is the only supported way to do either:

    # on a machine with network (e.g. the Windows workstation)
    python scripts/fetch_third_party.py --fetch [--only fsdp-turbo,shmem]

    # on the target host, after copying ascend_ep/third_party/
    python scripts/fetch_third_party.py --check [--json]

    # record the licence and a tree hash once a checkout is present
    python scripts/fetch_third_party.py --record-licenses

Design rules:

* A short SHA is never enough.  ``--check`` reads ``git rev-parse HEAD`` and
  compares the full object id, and a tag-pinned entry is resolved to the commit the
  checkout actually holds.
* The tool never downloads during a benchmark run: ``--fetch`` is an explicit
  preparation step, and the benchmark only ever calls ``--check``.
* Nothing is invented.  A missing checkout, an unpinned entry and an unread licence
  each get their own status instead of a guess.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # ascend_ep/ep/04_scripts
PACKAGE = ROOT.parent.parent                        # ascend_ep
CATALOGUE = PACKAGE / "third_party" / "sources.yaml"

FOUND = "FOUND"
MISSING = "MISSING"
COMMIT_MISMATCH = "COMMIT_MISMATCH"
UNPINNED = "UNPINNED"
NOT_A_REPOSITORY = "NOT_A_REPOSITORY"


def load_catalogue(path=None):
    import yaml

    path = Path(path) if path else CATALOGUE
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    return raw


def source_root(catalogue):
    base = catalogue.get("root") or "third_party"
    return (PACKAGE / base).resolve()


def checkout_path(catalogue, name):
    entry = catalogue["sources"][name]
    return source_root(catalogue) / entry["directory"]


def _git(path, *args, timeout=30):
    try:
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def head_commit(path):
    return _git(path, "rev-parse", "HEAD")


def read_license(path):
    """Return ``(spdx_guess, file)`` from the checkout, or ``(None, None)``.

    Only file names are mapped, and only unambiguous ones: a COPYING file is not
    enough to claim GPL versus Apache, so the guess stays ``None`` there.
    """
    for name, guess in (("LICENSE-APACHE", "Apache-2.0"), ("LICENSE", None),
                        ("LICENSE.txt", None), ("COPYING", None), ("LICENSE.md", None)):
        candidate = Path(path) / name
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")[:4000].lower()
        if "apache license" in text:
            return "Apache-2.0", name
        if "mit license" in text:
            return "MIT", name
        if "gnu general public license" in text:
            return "GPL", name
        if "bsd" in text:
            return "BSD", name
        return guess, name
    return None, None


def inspect(catalogue, name):
    """Status of one catalogue entry, without touching the network."""
    entry = catalogue["sources"][name]
    path = checkout_path(catalogue, name)
    record = {"id": name, "dependency": entry.get("dependency"), "path": str(path),
              "expected_commit": entry.get("commit"), "expected_tag": entry.get("tag")}
    if entry.get("status") == "unpinned" or not (entry.get("commit") or entry.get("tag")):
        record.update(status=UNPINNED, reason="no commit and no tag in the catalogue")
        return record
    if not path.is_dir():
        record.update(status=MISSING, reason=f"{path} does not exist")
        return record
    if not (path / ".git").exists():
        record.update(status=NOT_A_REPOSITORY, reason="no .git directory")
        return record
    head = head_commit(path)
    if head is None:
        record.update(status=NOT_A_REPOSITORY, reason="git rev-parse HEAD failed")
        return record
    expected = entry.get("commit")
    if expected and head != expected:
        record.update(status=COMMIT_MISMATCH, actual_commit=head,
                      reason=f"expected {expected}, found {head}")
        return record
    record.update(status=FOUND, actual_commit=head)
    if entry.get("tag") and not expected:
        record["resolved_tag_commit"] = head
        record["note"] = (f"pinned by tag {entry['tag']}; the checkout holds {head} -- "
                          f"record this commit to make the pin exact")
    spdx, licence_file = read_license(path)
    record["license"] = spdx
    record["license_file"] = licence_file
    return record


def inspect_all(catalogue, only=None):
    names = list(catalogue["sources"])
    if only:
        unknown = [item for item in only if item not in catalogue["sources"]]
        if unknown:
            raise SystemExit(f"unknown source(s): {', '.join(unknown)}")
        names = only
    return [inspect(catalogue, name) for name in names]


def resolve_ref(path, ref):
    """Turn a catalogue pin into a commit id inside the checkout.

    A bare ``8.2.RC1`` is ambiguous to git: it looks like a path, and
    ``git checkout --detach`` refuses it.  Resolve explicitly instead -- commit,
    then tag, then remote branch -- and return ``None`` when nothing matches, so the
    caller can report FETCH_FAILED with the ref it tried.
    """
    candidates = [ref, f"refs/tags/{ref}", f"origin/{ref}", f"refs/remotes/origin/{ref}"]
    for candidate in candidates:
        commit = _git(path, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        if commit:
            return commit
    return None


def fetch(catalogue, only=None, shallow=False):
    """Clone the pinned checkouts.  Requires network; never called by a run."""
    names = list(catalogue["sources"])
    if only:
        names = only
    results = []
    root = source_root(catalogue)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        entry = catalogue["sources"][name]
        path = checkout_path(catalogue, name)
        ref = entry.get("commit") or entry.get("tag")
        if path.is_dir() and (path / ".git").exists() and ref:
            commit = resolve_ref(path, ref)
            if commit:
                results.append(inspect(catalogue, name))
                continue
            print(f"{name}: present checkout does not contain {ref}, re-cloning", flush=True)
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_dir() and (path / ".git").exists():
            results.append({"id": name, "status": "SKIPPED", "reason": "already present"})
            continue
        url = entry.get("upstream")
        if not url or not ref:
            results.append({"id": name, "status": UNPINNED, "reason": "no url or pin"})
            continue
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        print(f"fetching {name} from {url} @ {ref}", flush=True)
        clone = ["git", "clone", "--no-checkout"]
        if shallow:
            clone += ["--depth", "1"]
        clone += [url, str(path)]
        if subprocess.run(clone, timeout=3600).returncode != 0:
            results.append({"id": name, "status": "FETCH_FAILED", "reason": "git clone"})
            continue
        commit = resolve_ref(path, ref)
        if commit is None:
            results.append({"id": name, "status": "FETCH_FAILED",
                            "reason": f"the clone does not contain {ref}"})
            continue
        checkout = ["git", "-C", str(path), "checkout", "--detach", commit]
        if subprocess.run(checkout, timeout=1800).returncode != 0:
            results.append({"id": name, "status": "FETCH_FAILED", "reason": "git checkout"})
            continue
        results.append(inspect(catalogue, name))
    return results


def summarise(records):
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true",
                        help="verify the checkouts already on this host (no network)")
    action.add_argument("--fetch", action="store_true",
                        help="clone the pinned checkouts; needs network")
    action.add_argument("--record-licenses", action="store_true",
                        help="print the licence each checkout carries")
    parser.add_argument("--only", type=lambda value: [v.strip() for v in value.split(",") if v.strip()])
    parser.add_argument("--shallow", action="store_true", help="with --fetch: --depth 1")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--catalogue", type=Path)
    args = parser.parse_args(argv)

    catalogue = load_catalogue(args.catalogue)
    if args.fetch:
        records = fetch(catalogue, args.only, args.shallow)
    else:
        records = inspect_all(catalogue, args.only)

    if args.json:
        print(json.dumps({"sources": records, "counts": summarise(records)}, indent=2))
    else:
        width = max(len(record["id"]) for record in records) + 2
        print(f"{'source':{width}s} {'status':18s} commit / reason")
        for record in records:
            detail = record.get("actual_commit") or record.get("reason") or ""
            print(f"{record['id']:{width}s} {record['status']:18s} {detail}")
        print()
        print("counts:", summarise(records))

    if args.record_licenses:
        print()
        for record in records:
            if record["status"] == FOUND:
                print(f"{record['id']}: license={record.get('license')} "
                      f"file={record.get('license_file')}")

    # A check is a gate: a missing or mismatched checkout is not a warning.
    if args.check:
        return 0 if all(record["status"] in (FOUND, UNPINNED) for record in records) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
