"""Heuristic HCOMM log evidence, kept separate from HCCL runtime logs.

HCOMM owns the channel objects; an HCOMM log line naming a channel class is the
closest thing to a channel observation available in text form.  It is still a
keyword hit, so the claim stays heuristic and cannot verify a transport.
"""
from baselines.l0.verification import logs


def collect(explicit=(), extra_roots=()):
    paths = logs.candidate_paths(explicit, extra_roots)
    evidence, inspected = logs.scan(paths)
    for item in evidence:
        if "channel" in item.claim:
            item.source = "hcomm_log_keyword"
    return evidence, inspected


def configured_paths(config):
    section = ((config or {}).get("benchmark") or {}).get("verification") or {}
    return [item for item in section.get("hcomm_log_paths", ()) if item]
