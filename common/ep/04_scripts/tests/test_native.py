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
    (["run", "--ep-size", "8", "--operators", "alltoall_hccl", "--cases", "MOE-003"], 8),
])
def test_native_smoke(tmp_path, args, required_devices):
    if os.getenv("EP_RUN_NPU_TESTS") != "1":
        pytest.skip("set EP_RUN_NPU_TESTS=1 on an Ascend host to run native smoke tests")
    torch = pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("torch_npu", exc_type=ImportError)

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
    if required_devices == 2:
        from profiling.parse_profile import parse_directory

        profiled = subprocess.run(
            [sys.executable, str(ROOT / "run_benchmark.py"), "profile",
             "--cases", "COMM-001", "--output", str(root), "--run-id", "smoke-npu"],
            check=False, cwd=ROOT, text=True, capture_output=True, timeout=1200,
        )
        assert profiled.returncode == 0, (profiled.stdout, profiled.stderr)
        dispatch = next(r for r in results(run, load_json(run / "manifest.json")) if r["case_id"] == "COMM-001")
        artifact, = dispatch["profiling_artifacts"]
        assert artifact["status"] == "passed"
        events = parse_directory((run / artifact["manifest"]).parent)
        assert {event.rank for event in events} == {0, 1}
        assert any(event.kernel_category == "communication" for event in events)


def test_native_cann_context_dispatch(tmp_path):
    if os.getenv("EP_RUN_NPU_TESTS") != "1":
        pytest.skip("set EP_RUN_NPU_TESTS=1 on an Ascend host to run native smoke tests")
    torch = pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("torch_npu", exc_type=ImportError)
    pytest.importorskip("cann_ops_transformer", exc_type=ImportError)
    if torch.npu.device_count() < 8:
        pytest.skip("context dispatch smoke requires an eight-device EP group")
    names = [torch.npu.get_device_name(rank).casefold() for rank in range(8)]
    if not (all("950" in name for name in names) or
            all("910_93" in name or "atlas a3" in name for name in names)):
        pytest.skip("context dispatch targets homogeneous 950DT or Atlas A3 devices")
    root = tmp_path / "runs"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run_benchmark.py"), "run", "--config",
         str(ROOT / "operators/mc2_dispatch_cann_v3/config.yaml"), "--cases", "COMM-001",
         "--output", str(root), "--run-id", "mc2-context"],
        check=False, cwd=ROOT, text=True, capture_output=True, timeout=1200,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    from common.controller import results
    from common.storage import load_json

    run = root / "mc2-context"
    row, = results(run, load_json(run / "manifest.json"))
    assert row["status"] == "passed" and row["correctness"]["passed"]
