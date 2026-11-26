"""Conservative keyword scanning of runtime logs.

Nothing here turns an environment variable into an observation.  A log line that
mentions ``URMA`` proves that some component *said* URMA; it does not prove the
payload of this task travelled over URMA.  All claims produced by this module are
therefore ``heuristic`` and cannot promote a field to ``verified``.

The token tables below are the vocabulary the CANN runtime is known to print in
channel / resource names.  They are matched case-insensitively against whole
words so that ``ub_memory`` in a path or ``hccs`` in a hostname does not produce
a spurious claim.
"""
import os
import re
from pathlib import Path

from baselines.l0.verification.path_verifier import Evidence

#: Longest tokens first so ``ub_memory`` wins over ``ub``.
MECHANISM_TOKENS = (
    ("ub_memory", ("ub_memory", "ubmem", "ub mem")),
    ("ub_ctp", ("ub_ctp", "ctp")),
    ("ub_rtp", ("ub_rtp", "rtp")),
    ("urma", ("urma",)),
    ("roce", ("roce", "rdma")),
    ("hccs", ("hccs",)),
    ("uboe", ("uboe",)),
)
ACCELERATOR_TOKENS = (
    ("aiv_only", ("aiv_only",)),
    ("aicpu_ts", ("aicpu_ts", "aicpu", "aicpu ts")),
    ("aiv", ("aiv",)),
    ("ccu_ms", ("ccu_ms",)),
    ("ccu", ("ccu", "ccu_sched")),
)
CHANNEL_TOKENS = (
    "AivUbMemChannel", "AivUrmaChannel", "AicpuTsHccsChannel", "AicpuTsRoceChannel",
    "AicpuTsUrmaChannel", "CcuUrmaChannel", "HostCpuUrmaChannel",
)

MAX_LINES_PER_FILE = 20000


def candidate_paths(explicit=(), extra_roots=()):
    """Collect log locations without assuming a single CANN log layout."""
    paths = [Path(item) for item in explicit if item]
    for variable in ("HCCL_LOG_PATH", "HCOMM_LOG_PATH", "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH"):
        value = os.getenv(variable)
        if value:
            paths.append(Path(value))
    home = os.getenv("HOME") or os.getenv("USERPROFILE")
    if home:
        paths.append(Path(home) / "ascend" / "log")
    paths.extend(Path(root) for root in extra_roots if root)
    seen, unique = set(), []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _files(paths):
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*.log")):
                yield child


def _word(token):
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", re.IGNORECASE)


def _table_claims(line, table):
    hits = []
    for name, tokens in table:
        if any(_word(token).search(line) for token in tokens):
            hits.append(name)
    return hits


def scan(paths, limit=MAX_LINES_PER_FILE, max_files=32):
    """Return heuristic evidence items for every matching log line.

    Each item keeps the file path and the matching line so a reviewer can audit
    it; nothing is aggregated into a "detected transport" conclusion here.
    """
    evidence, inspected = [], []
    for number, path in enumerate(_files(paths)):
        if number >= max_files:
            break
        inspected.append(str(path))
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for index, line in enumerate(handle):
                    if index >= limit:
                        break
                    mechanisms = _table_claims(line, MECHANISM_TOKENS)
                    accelerators = _table_claims(line, ACCELERATOR_TOKENS)
                    channels = [name for name in CHANNEL_TOKENS
                                if _word(name).search(line)]
                    if not (mechanisms or accelerators or channels):
                        continue
                    claim = {}
                    for name in mechanisms:
                        claim["mechanism"] = name
                    for name in accelerators:
                        claim["accelerator"] = name
                    if channels:
                        claim["channel"] = channels[0]
                    evidence.append(Evidence(
                        source="hccl_log_keyword", claim=claim, path=str(path),
                        confidence="heuristic",
                        detail=f"line {index + 1}: {line.strip()[:400]}",
                    ))
        except OSError:
            continue
    return evidence, inspected
