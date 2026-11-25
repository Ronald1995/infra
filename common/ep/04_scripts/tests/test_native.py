"""Opt-in native smoke tests of normal benchmark entry points."""
import os
import subprocess
import sys

import pytest

from common.registry import ROOT

pytestmark = pytest.mark.npu


@pytest.mark.parametrize(("args", "required_devices"), [
    (["baseline", "--ep-size", "1", "--cases", "HW-010,HW-011"], 1),
    (["run", "--ep-size", "2", "--operators", "alltoall_hccl", "--cases", "COMM-001,COMM-003"], 2),
])
def test_native_smoke(tmp_path, args, required_devices):
    if os.getenv("EP_RUN_NPU_TESTS") != "1":
        pytest.skip("set EP_RUN_NPU_TESTS=1 on an Ascend host to run native smoke tests")
    import torch
    import torch_npu  # noqa: F401

    if torch.npu.device_count() < required_devices:
        pytest.skip(f"test requires {required_devices} visible Ascend devices")
    root = tmp_path / "runs"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run_benchmark.py"), *args,
         "--config", str(ROOT / "configs/smoke_npu.yaml"), "--output", str(root), "--run-id", "smoke-npu"],
        check=False, cwd=ROOT, text=True, capture_output=True, timeout=1200,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    from common.controller import results
    from common.storage import load_json

    run = root / "smoke-npu"
    rows = list(results(run, load_json(run / "manifest.json")))
    assert rows and all(r["status"] == "passed" and r["correctness"]["passed"] for r in rows)
