"""Collect device and software identity without inventing values.

Everything returned here is either read from a file, from an environment
variable, from a package metadata entry, or from a tool the user installed.
Unknown values stay ``None`` and carry a ``source`` note, because a fabricated
version string is worse than a missing one: it silently invalidates every
comparison the report makes.
"""
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import re
import shutil
import socket
import subprocess

PACKAGES = ("torch", "torch-npu", "PyYAML", "shmem", "ascend-comm", "hcomm")
SOC_PATTERN = re.compile(r"Ascend[\s_-]?(\d{3,4}\w*)", re.IGNORECASE)


def command(args, timeout=8):
    try:
        completed = subprocess.run(args, capture_output=True, text=True, errors="replace",
                                   timeout=timeout, check=True)
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def packages():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def ascend_home():
    for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        value = os.getenv(variable)
        if value:
            return value
    for candidate in (r"C:\Program Files\Huawei\CANN", "/usr/local/Ascend/ascend-toolkit/latest"):
        if os.path.isdir(candidate):
            return candidate
    return None


def cann_version():
    declared = os.getenv("ASCEND_TOOLKIT_VERSION")
    if declared:
        return declared, "environment:ASCEND_TOOLKIT_VERSION"
    home = ascend_home()
    if not home:
        return None, "unavailable:no CANN installation detected"
    for name in ("version.cfg", "ascend_toolkit_install.info"):
        path = os.path.join(home, name)
        if not os.path.isfile(path):
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        match = re.search(r"(?:version|Version)\s*=\s*(\S+)", text)
        if match:
            return match.group(1), f"file:{path}"
    return None, f"unavailable:no version file under {home}"


def library_present(names, roots=()):
    """Return the first existing library path for any of ``names``.

    ``roots`` default to the CANN installation and the usual loader paths.  The
    check is a fleet-wide probe used by capability discovery; it says nothing
    about whether the library can be loaded on this device.
    """
    search = [root for root in roots if root]
    home = ascend_home()
    if home:
        search.extend([home, os.path.join(home, "lib64"), os.path.join(home, "runtime", "lib64")])
    for root in search:
        for name in names:
            for candidate in (os.path.join(root, name),
                              os.path.join(root, "lib64", name),
                              os.path.join(root, "include", name)):
                if os.path.exists(candidate):
                    return candidate
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def npu_smi_summary():
    binary = shutil.which("npu-smi")
    if not binary:
        return None
    text = command([binary, "info"])
    if text is None:
        return None
    names = sorted({match.group(0) for match in SOC_PATTERN.finditer(text)})
    return {"command": [binary, "info"], "soc_candidates": names,
            "source": "npu-smi", "raw_head": text.splitlines()[:40]}


def torch_devices():
    """Return ``(device_names, error)`` using torch_npu, or ``(None, reason)``."""
    try:
        import torch
        import torch_npu  # noqa: F401
    except (ImportError, OSError) as exc:
        return None, f"torch_npu unavailable: {exc}"
    try:
        count = torch.npu.device_count()
        return [torch.npu.get_device_name(index) for index in range(count)], None
    except (RuntimeError, OSError) as exc:
        return None, f"device query failed: {exc}"


def summary(probe_npu=False):
    """Describe the device and software stack.

    ``probe_npu`` also queries torch_npu; the planning path keeps it off so that
    capability discovery never initialises a device.
    """
    version_string, version_source = cann_version()
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ascend_home": ascend_home(),
        "cann_version": version_string,
        "cann_version_source": version_source,
        "packages": packages(),
        "env": {name: os.getenv(name) for name in (
            "ASCEND_TOOLKIT_VERSION", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_WORK_PATH",
            "HCCL_OP_EXPANSION_MODE", "HCCL_ALGO", "HCCL_BUFFSIZE", "EP_L0_TRANSPORT_PROFILE",
        )},
        "libraries": {
            "libhccl": library_present(("libhccl.so", "libhccl.dll")),
            "libhcomm": library_present(("libhcomm.so", "libhcomm.dll")),
            "libascshmem": library_present(("libshmem.so", "libascshmem.so", "libshmem.dll")),
            "hccl_test": shutil.which("hccl_test"),
        },
        "soc": None,
        "soc_source": "unavailable",
        "devices": None,
        "device_source": "unavailable",
        "npu_smi": None,
    }
    npu_smi = npu_smi_summary()
    if npu_smi:
        info["npu_smi"] = npu_smi
        if npu_smi["soc_candidates"]:
            info["soc"] = npu_smi["soc_candidates"][0]
            info["soc_source"] = "npu-smi (heuristic match)"
    if probe_npu:
        names, error = torch_devices()
        if names is not None:
            info["devices"] = names
            info["device_source"] = "torch_npu"
            if names and info["soc"] is None:
                match = SOC_PATTERN.search(names[0])
                info["soc"] = match.group(0) if match else names[0]
                info["soc_source"] = "torch_npu device name (heuristic match)"
        else:
            info["device_error"] = error
    return info


def as_json(info):
    return json.dumps(info, indent=2, sort_keys=True)
