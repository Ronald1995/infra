"""Capture runtime provenance without loading device runtimes during planning."""
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import shutil
import subprocess
import sys


def versions():
    result = {}
    for package in ("torch", "torch-npu", "PyYAML"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def command(args, timeout=8):
    try:
        result = subprocess.run(args, capture_output=True, text=True, errors="replace",
                                timeout=timeout, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture(probe_npu=False):
    packages = versions()
    info = {
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "packages": packages, "device_visibility": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "cann_root": os.getenv("ASCEND_HOME_PATH"), "cann_version": os.getenv("ASCEND_TOOLKIT_VERSION"),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_changes": command(["git", "status", "--porcelain", "--untracked-files=no"]),
        "npu_smi": shutil.which("npu-smi"), "npu_smi_info": None,
    }
    if probe_npu and info["npu_smi"]:
        info["npu_smi_info"] = command([info["npu_smi"], "info"])
    return info
