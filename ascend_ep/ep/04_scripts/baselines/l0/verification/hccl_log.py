"""Heuristic HCCL runtime-log evidence.

HCCL logs are free text.  They can show that the runtime mentioned an
accelerator or a transport, and they are the only source available when the
native channel descriptor is not compiled in.  They never verify a path on their
own: see the module docstring of :mod:`baselines.l0.verification.logs`.
"""
from baselines.l0.verification import logs


def collect(explicit=(), extra_roots=()):
    """Return ``(evidence, inspected_files)`` for the available HCCL logs."""
    paths = logs.candidate_paths(explicit, extra_roots)
    return logs.scan(paths)


def configured_paths(config):
    """Log directories declared by a benchmark configuration."""
    section = ((config or {}).get("benchmark") or {}).get("verification") or {}
    return [item for item in section.get("hccl_log_paths", ()) if item]
