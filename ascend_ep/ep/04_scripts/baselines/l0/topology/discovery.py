"""Single entry point for L0 capability discovery.

The scheduler asks this module which cases may run; the runner asks it which
channels the native bridge actually compiled in.  Both read the same cached
matrix so a plan never promises a capability the execution path cannot deliver.

An explicit matrix can be supplied (``EP_L0_CAPABILITIES=/path/matrix.json`` or
``--capability-matrix``).  Overrides are marked ``source: override`` in every
artifact and in the report, so a mocked run can never be mistaken for a probe.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from baselines.l0.topology import capabilities
from common.storage import fingerprint, load_json

_CACHE = {}


def _identity(config):
    benchmark = (config or {}).get("benchmark") or {}
    relevant = {
        "transport": benchmark.get("transport"),
        "native_hccl": benchmark.get("native_hccl"),
        "memory_library": benchmark.get("memory_library"),
        "shmem_udma_library": benchmark.get("shmem_udma_library"),
        "env": {name: os.getenv(name) for name in (
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_VERSION", "ASCEND_RT_VISIBLE_DEVICES",
            "EP_L0_TRANSPORT_LIBRARY", "EP_HCCL_TEST_BINARY", "EP_L0_CAPABILITIES",
        )},
    }
    return fingerprint(relevant)


def load(config=None, override_path=None, probe_runtime=False, refresh=False):
    """Return the capability matrix for ``config``."""
    path = override_path or os.getenv("EP_L0_CAPABILITIES")
    key = (path, probe_runtime, _identity(config))
    if not refresh and key in _CACHE:
        return deepcopy(_CACHE[key])
    if path:
        override = load_json(Path(path))
        matrix = capabilities.discover(config, override=override)
        matrix.setdefault("override_path", str(path))
    else:
        matrix = capabilities.discover(config, probe_runtime=probe_runtime)
    _CACHE[key] = matrix
    return deepcopy(matrix)


def clear_cache():
    _CACHE.clear()


def write(matrix, directory):
    """Persist the matrix next to a run so results stay self-describing."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "capabilities.json"
    from common.storage import write_json
    write_json(path, matrix)
    return path


def describe(matrix):
    """One-line-per-capability text used by ``run_benchmark.py`` and reports."""
    lines = [f"source: {matrix.get('source')}"]
    device = matrix.get("device") or {}
    for key in ("soc", "cann_version", "devices"):
        lines.append(f"{key}: {device.get(key)}")
    for entry in matrix.get("channels", ()):
        state = "supported" if entry["supported"] else "unsupported"
        lines.append(f"channel {entry.get('channel') or entry.get('case_id')} "
                     f"[{entry.get('execution_resource')}, "
                     f"{'/'.join(entry.get('mechanisms', ()))}]: {state}"
                     + (f" ({entry['reason']})" if entry.get("reason") else ""))
    for entry in matrix.get("engines", ()):
        state = "supported" if entry["supported"] else "unsupported"
        lines.append(f"engine {entry['engine']}: {state}"
                     + (f" ({entry['reason']})" if entry.get("reason") else ""))
    for entry in matrix.get("accelerators", ()):
        state = "supported" if entry["supported"] else "unsupported"
        lines.append(f"accelerator {entry['mode']}: {state}"
                     + (f" ({entry['reason']})" if entry.get("reason") else ""))
    return "\n".join(lines)


def for_case(matrix, case_id, mechanism=None, execution_resource=None):
    """Convenience wrapper returning ``(supported, reason)`` for a case."""
    return capabilities.support(matrix, case_id=case_id, mechanism=mechanism,
                                execution_resource=execution_resource)


def matrix_path(matrix):
    return matrix.get("override_path")


def dump(matrix):
    return json.dumps(matrix, indent=2, sort_keys=True)
