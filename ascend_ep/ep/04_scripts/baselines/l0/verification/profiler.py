"""Turn profiler kernel names into heuristic evidence about what executed.

The profiler observes kernels, not transports.  A kernel name can tell us that a
communication kernel ran and, in a few cases, which execution resource it used.
It cannot tell us which link carried the bytes, so such claims remain heuristic.
"""
from pathlib import Path

from baselines.l0.verification.path_verifier import Evidence
from profiling.kernel_classifier import classify

COMMUNICATION_HINTS = ("hccl", "shmem", "udma", "urma", "sdma", "mte", "ffts", "aiv")
ACCELERATOR_HINTS = (("AICPU", "aicpu"), ("AIV", "aiv"), ("CCU", "ccu"),
                     ("SDMA", "sdma"), ("MTE", "mte"))


def _claim_for(name):
    text = name.casefold()
    claim = {}
    for resource, token in ACCELERATOR_HINTS:
        if token in text:
            claim["accelerator"] = resource.casefold()
            break
    return claim


def collect(profile_root):
    """Return evidence for every communication kernel found under ``profile_root``.

    ``profile_root`` is a directory containing ``kernel_details.csv`` files, as
    produced by the existing profiling pipeline.
    """
    root = Path(profile_root)
    if not root.exists():
        return []
    evidence = []
    for path in sorted(root.rglob("kernel_details.csv")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            name = line.split(",", 1)[0].strip()
            if not name or classify(name) != "communication":
                continue
            if not any(hint in name.casefold() for hint in COMMUNICATION_HINTS):
                # Communication classification is transitive through fusion
                # kernels; only report kernels that name a mechanism.
                continue
            claim = _claim_for(name)
            evidence.append(Evidence(
                source="profiler_kernel_name", claim=claim, path=str(path),
                confidence="heuristic", detail=f"kernel {name}",
            ))
    return evidence
