"""Process-level CPU integration, using real Gloo collectives."""
import json
import os
import subprocess
import sys

import pytest

from common.registry import ROOT


def test_distributed_pipeline(tmp_path):
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyTorch runtime unavailable: {exc}")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable in this PyTorch build")
    command = [sys.executable, str(ROOT / "run_benchmark.py")]
    run_id = "local-gloo-test"
    arguments = ["--output", str(tmp_path), "--run-id", run_id]
    env = {**os.environ, "PYTHONUTF8": "1"}
    run = subprocess.run(
        [*command, "pipeline", "--config", str(ROOT / "configs/smoke_cpu.yaml"),
         "--cases", "COMM-004,COMM-005,MOE-003,LAYER-002", "--operators", "alltoall_hccl", *arguments],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    reports = tmp_path / run_id / "reports"
    summary_file = reports / "benchmark_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    assert run.returncode == 0, (run.stdout, run.stderr, [
        (row["case_id"], row["error_reason"]) for row in summary.get("results", [])
    ])
    assert summary["status_counts"] == {"passed": 4}
    for row in summary["results"]:
        assert row["correctness"]["passed"]
        assert [r["rank"] for r in row["raw_samples"]] == [0, 1]
        assert row["environment"]["device"] == "cpu"
    assert (reports / "benchmark_report.md").is_file()
    assert (reports / "benchmark_summary.csv").is_file()
