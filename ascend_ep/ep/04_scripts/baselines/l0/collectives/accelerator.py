"""Discover the native HCCL test tool and the accelerator modes it accepts.

The mode names are read from the installed tool's ``--help`` output, never
hard-coded: different CANN releases list different subsets, and the released
``--help`` text is itself incomplete (it omits ``host_ts`` and ``aicpu`` even
though the option parser accepts them).  Only names the local ``--help`` prints
are offered, so a plan cannot request a mode the local tool rejects.

The native HCCL test tool ships with CANN oam-tools as ``tools/hccl_test`` and
must be built on the target host (see the module README).
"""
import os
import re
import shutil
import subprocess

from baselines.l0.topology.capabilities import Probe
from baselines.l0.verification.vocabulary import ACCELERATOR_RESOURCES

#: Test binary per operation.  ``p2p`` is absent from the tool: it ships
#: broadcast/reduce/scatter and collective all-* tests only.
BINARIES = {
    "alltoall": "alltoall_test",
    "alltoallv": "alltoallv_test",
    "allgather": "all_gather_test",
    "reducescatter": "reduce_scatter_test",
    "allreduce": "all_reduce_test",
    "broadcast": "broadcast_test",
    "reduce": "reduce_test",
    "scatter": "scatter_test",
}

#: Tokens the tool's option parser understands.  Used only to recognise values
#: that actually appear in ``--help``; never to invent one.
KNOWN_MODES = ("default", "host_ts", "aicpu_ts", "aiv", "aiv_only", "ccu_ms",
               "ccu_sched", "aicpu", "ccu_ms_only", "ccu_sched_only")

OPTION_PATTERN = re.compile(r"--accelerator|accelerator_config|-a\b")
VALUE_PATTERN = re.compile(r"([a-z_]+)")


def configured_binary(config):
    section = ((config or {}).get("benchmark") or {}).get("native_hccl") or {}
    return section.get("binary") or os.getenv("EP_HCCL_TEST_BINARY")


def locate(config=None):
    """Return the directory holding the test binaries, or the configured binary."""
    configured = configured_binary(config)
    if configured:
        return configured
    found = shutil.which("alltoall_test")
    if found:
        return os.path.dirname(found)
    candidates = [os.getenv("ASCEND_HOME_PATH")]
    for root in candidates:
        if not root:
            continue
        for relative in ("tools/hccl_test/bin", "tool/hccl_test/bin"):
            candidate = os.path.join(root, relative)
            if os.path.isdir(candidate):
                return candidate
    return None


def binary_path(config, operation):
    name = BINARIES.get(operation)
    if name is None:
        return None, f"the native HCCL test tool has no {operation!r} test binary"
    root = locate(config)
    if root is None:
        return None, ("native HCCL test tool not found; build CANN oam-tools hccl_test and set "
                      "benchmark.native_hccl.binary or EP_HCCL_TEST_BINARY")
    if os.path.isfile(root):
        return (root, None) if os.path.basename(root) == name else (
            None, f"configured binary {root} is not {name}")
    candidate = os.path.join(root, name)
    if not os.path.isfile(candidate):
        return None, f"{name} is not present under {root}"
    return candidate, None


def help_text(binary, timeout=15):
    try:
        completed = subprocess.run([binary, "--help"], capture_output=True, text=True,
                                   errors="replace", timeout=timeout, check=False)
        return (completed.stdout or "") + (completed.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"__error__ {exc}"


def parse_modes(text):
    """Return the accelerator modes the installed tool's help advertises."""
    if not text or text.startswith("__error__"):
        return [], "help output unavailable"
    for line in text.splitlines():
        if not OPTION_PATTERN.search(line):
            continue
        values = [token for token in VALUE_PATTERN.findall(line) if token in KNOWN_MODES]
        if values:
            return list(dict.fromkeys(values)), None
    if OPTION_PATTERN.search(text):
        return ["default"], "the help mentions the accelerator option but lists no mode names"
    return [], "the local --help output has no accelerator option"


def probe(config=None):
    """Probe result consumed by the capability matrix."""
    binary, reason = binary_path(config or {}, "alltoall")
    if binary is None:
        return Probe("native_hccl", False, reason, {"binaries": sorted(set(BINARIES.values()))})
    text = help_text(binary)
    modes, mode_reason = parse_modes(text)
    entries = []
    for mode in modes:
        resource = ACCELERATOR_RESOURCES.get(mode)
        entries.append({"mode": mode, "execution_resource": resource, "supported": True,
                        "reason": None, "evidence": [f"--help:{os.path.basename(binary)}"]})
    detail = {
        "binary_directory": os.path.dirname(binary),
        "binaries": sorted(set(BINARIES.values())),
        "operations": {operation: name for operation, name in BINARIES.items()},
        "modes": entries,
        "help_excerpt": [line for line in text.splitlines() if OPTION_PATTERN.search(line)][:6],
        "mode_parse_note": mode_reason,
        "implicit_default_note": (
            "When HCCL_OP_EXPANSION_MODE is unset, recent releases default the tool to "
            "ccu_sched on new-generation SoCs; the benchmark records the ambient variable "
            "and never assumes the default is neutral"),
        "ambient_expansion_mode": os.getenv("HCCL_OP_EXPANSION_MODE"),
    }
    if not modes:
        return Probe("native_hccl", False,
                     f"no accelerator mode could be read from --help ({mode_reason})", detail)
    return Probe("native_hccl", True, None, detail)
