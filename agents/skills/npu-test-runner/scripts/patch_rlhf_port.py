#!/usr/bin/env python3
"""Make the rlhf e2e conftest honour ``VLLM_RL_TEST_PORT``.

The container runs with ``--net=host`` and the rlhf conftest hardcodes
``port: int = 8770``, so two sessions running the same tests at the same time
fight over that port ("address already in use" while vLLM starts, or one
session's tests talking to the other session's server). ``npu-slot.sh`` exports
``VLLM_RL_TEST_PORT`` per session; this script teaches the conftest to read it,
falling back to a free ephemeral port when it is unset.

Properties:
  * idempotent - exits 0 without touching the file if it already reads the env var
  * newline preserving - CRLF stays CRLF, LF stays LF
  * decorator aware - the helper is inserted *above* ``@contextmanager``, so
    ``server`` keeps its decorator
  * fail closed - if the expected anchors are gone, nothing is written and the
    script exits 2 so a human can look at the file

Usage:
    patch_rlhf_port.py <repo-root> [--check]
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

CANDIDATES = (
    "tests/e2e/pull_request/one_card/rlhf/conftest.py",
    "tests/e2e/pull_request/rlhf/conftest.py",
)

HELPER_LINES = [
    "def _default_server_port() -> int:",
    '    """Return the HTTP port for the test server.',
    "",
    "    ``VLLM_RL_TEST_PORT`` (exported by ``npu-slot.sh``) pins the port so that",
    "    concurrent sessions sharing this host-networked container cannot collide.",
    "    Without it a free ephemeral port is used, so two sessions never share 8770.",
    '    """',
    '    env_port = os.environ.get("VLLM_RL_TEST_PORT")',
    "    if env_port:",
    "        return int(env_port)",
    "    from vllm.utils.network_utils import get_open_port",
    "",
    "    return get_open_port()",
]

RESOLVE_LINE = "    port = _default_server_port() if port is None else port"
OLD_DOC = "        port:            HTTP port to bind (caller is responsible for uniqueness)."
NEW_DOC = "        port:            HTTP port to bind; defaults to $VLLM_RL_TEST_PORT or a free port."
BASE_ANCHOR = "    base = _DUMMY_ARGS if dummy_weights else _BASE_ARGS"
SIG_PREFIX = "    port: int = "
SIG_EXPECTED = "    port: int = 8770,"
SIG_NEW = "    port: int | None = None,"


def patch_text(text: str) -> tuple[str, list[str]]:
    """Return (patched_text, notes). Raises ValueError when an anchor is missing."""
    notes: list[str] = []
    if "VLLM_RL_TEST_PORT" in text:
        return text, ["already reads VLLM_RL_TEST_PORT"]

    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    bare = [line.rstrip("\r\n") for line in lines]

    def with_eol(new_lines: list[str]) -> list[str]:
        return [line + eol for line in new_lines]

    # 1) server(): the hardcoded port default becomes None (resolved at call time)
    sig = [i for i, line in enumerate(bare) if line.startswith(SIG_PREFIX)]
    if not sig:
        raise ValueError(f"could not find a '{SIG_PREFIX}<n>,' parameter (expected '{SIG_EXPECTED}')")
    if bare[sig[0]] != SIG_EXPECTED:
        raise ValueError(
            f"found '{bare[sig[0]].strip()}' but expected '{SIG_EXPECTED.strip()}'; patch it by hand"
        )
    lines[sig[0]] = SIG_NEW + eol
    notes.append("server(): port default 8770 -> None")

    # 2) helper, inserted above any decorators belonging to server()
    server = next((i for i, line in enumerate(bare) if line == "def server("), None)
    if server is None:
        raise ValueError("could not find 'def server(' to insert the helper before")
    insert_at = server
    while insert_at > 0 and bare[insert_at - 1].startswith("@"):
        insert_at -= 1
    # two trailing blank lines: the original blank lines now precede the helper,
    # and PEP8/ruff want two blank lines between top-level definitions
    lines[insert_at:insert_at] = with_eol(HELPER_LINES) + [eol, eol]
    notes.append(f"inserted _default_server_port() before line {insert_at + 1}")

    # 3) resolve the port at the top of server()
    base = next((i for i, line in enumerate(lines) if line.rstrip("\r\n") == BASE_ANCHOR), None)
    if base is None:
        raise ValueError(
            "could not find the anchor to resolve the port:\n"
            f"    {BASE_ANCHOR.strip()}\nresolve `port` at the top of server() by hand"
        )
    lines.insert(base, RESOLVE_LINE + eol)
    notes.append("resolve port inside server()")

    # 4) docstring note
    for i, line in enumerate(lines):
        if line.rstrip("\r\n") == OLD_DOC:
            lines[i] = NEW_DOC + eol
            notes.append("updated the port docstring")
            break

    return "".join(lines), notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", help="repository root (e.g. /home/liurong/ws/alice)")
    ap.add_argument("--check", action="store_true", help="report status, never write")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser()
    target = next((repo / c for c in CANDIDATES if (repo / c).is_file()), None)
    if target is None:
        print(f"[port-patch] no rlhf conftest under {repo} - nothing to do")
        return 0

    # Read/write as UTF-8 with newline="" so neither the locale encoding (gbk on
    # Windows) nor newline translation can corrupt the file.
    with target.open("r", encoding="utf-8", newline="") as fh:
        original = fh.read()

    try:
        patched, notes = patch_text(original)
    except ValueError as exc:
        print(f"[port-patch] SKIPPED {target}: {exc}")
        return 2

    if patched == original:
        print(f"[port-patch] {target}: {'; '.join(notes)}")
        return 0

    if args.check:
        print(f"[port-patch] {target}: would patch ({'; '.join(notes)})")
        return 1

    with target.open("w", encoding="utf-8", newline="") as fh:
        fh.write(patched)
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=str(target),
        tofile=str(target),
        n=2,
    )
    sys.stderr.writelines(diff)
    print(f"[port-patch] patched {target}: {'; '.join(notes)}")
    print("[port-patch] review with `git diff` in your worktree before trusting the result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
