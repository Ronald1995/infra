"""Gates: what a phase accepts, and what it refuses to call success."""
import json


from common.registry import ROOT
from scripts.workflow import dependency, kernel
from scripts.workflow import phases as phase_runner
from scripts.workflow import request as request_module
from scripts.workflow import state as state_module


def build(prompt, **kwargs):
    return request_module.interpret(prompt, root=ROOT)


def run_workflow(workflow, prompt, **options):
    return workflow(prompt, **options)


# ------------------------------------------------------------ environment ---

def test_remote_ep_size_is_blocked_not_shrunk(workflow):
    final, ctx, fake = workflow("在 Ascend 950 上运行完整 EP Benchmark，EP8、EP64", devices=8)
    capacity = ctx.data["capacity"]
    assert capacity[8]["executable"] is True
    assert capacity[64]["executable"] is False
    assert "multi-node" in capacity[64]["reason"]
    assert final.phases["environment_discovery"].status == "partial"
    plan = ctx.data["plan"]
    assert plan["planned_ep_sizes"] == [8]
    assert plan["blocked_ep_sizes"][64]
    assert final.phases["plan_generation"].status == "partial"


def test_no_device_at_all_blocks_the_run(workflow):
    final, ctx, fake = workflow("在 Ascend 950 上运行完整 EP Benchmark，EP8", devices=0)
    assert final.phases["environment_discovery"].status == "blocked"
    assert final.status == "blocked"
    assert "no Ascend device was detected" in final.phases["environment_discovery"].reason


def test_cpu_capacity_is_a_process_policy_not_a_device_claim(workflow):
    final, ctx, fake = workflow("在当前 CPU 环境测试整个框架")
    capacity = ctx.data["capacity"]
    assert all(entry["executable"] for entry in capacity.values())
    assert final.phases["kernel_build"].status == "skipped"
    assert "CPU" in final.phases["kernel_build"].reason


# ------------------------------------------------------------- dependency ---

def test_host_dependency_blocks_instead_of_guessing(tmp_path, fake_benchmark):
    fake = fake_benchmark(checks=[
        {"id": "python", "status": "pass", "category": "A", "detail": "python 3.10.8",
         "suggested_action": "none", "blocking_for": ["reporting"], "evidence": {}},
        {"id": "npu-driver", "status": "fail", "category": "B",
         "detail": "the driver is absent", "suggested_action": "ask the host owner",
         "blocking_for": ["l0_transport"], "evidence": {}},
    ])
    report = {"checks": fake.checks}
    buckets = dependency.classify(report)
    assert [record["id"] for record in buckets["host"]] == ["npu-driver"]
    assert buckets["installable"] == []


def test_installable_dependency_without_authorization_is_skipped(workflow):
    final, ctx, fake = workflow(
        "完成完整 Benchmark，如果缺少依赖就尝试安装", checks=_missing_torch())
    install = final.phases["dependency_install"]
    assert install.status in ("blocked", "partial")
    assert "network" in (install.reason or "")


def test_install_runs_when_the_network_is_allowed(workflow):
    final, ctx, fake = workflow(
        "完成完整 Benchmark，如果缺少依赖就尝试安装，允许网络",
        checks=_missing_torch(), install_ok=True)
    install = final.phases["dependency_install"]
    assert install.status in ("passed", "partial")
    dry = [call for call in fake.calls if "--dry-run" in call]
    assert dry, "the dry run must be recorded before installing"
    assert any("install.log" in str(command.get("log", "")) or "install.log" in str(command)
               for command in install.commands)


def test_a_failed_install_is_recorded_with_its_log(workflow):
    final, ctx, fake = workflow(
        "完成完整 Benchmark，如果缺少依赖就尝试安装，允许网络",
        checks=_missing_torch(), install_ok=False)
    install = final.phases["dependency_install"]
    assert install.status == "failed"
    assert any(command["exit_code"] != 0 for command in install.commands)


def _missing_torch():
    return [
        {"id": "python", "status": "pass", "category": "A", "detail": "python 3.10.8",
         "suggested_action": "none", "blocking_for": ["reporting"], "evidence": {}},
        {"id": "torch", "status": "fail", "category": "A", "detail": "torch is absent",
         "suggested_action": "install torch==2.6.0",
         "blocking_for": ["l0_transport", "operator:alltoall_hccl"], "evidence": {}},
    ]


# ----------------------------------------------------------------- kernel ---

def test_missing_device_kernel_blocks_only_those_cases(monkeypatch, tmp_path, workflow):
    blocked = {"AIV": "AIV device source files are missing",
               "CCU": "CCU device source files are missing"}
    monkeypatch.setattr(kernel, "plan_build", lambda root: {
        "toolchain": {"complete": False}, "engines": {}, "ready": {},
        "buildable": [], "blocked": blocked,
        "native": {"transport_bridge": {"source_present": True}},
    })
    final, ctx, fake = workflow("完成所有可运行的 L0-L4 测试", devices=16)
    build = final.phases["kernel_build"]
    assert build.status == "partial"
    assert "AIV" in build.reason
    assert final.phases["plan_generation"].status in ("passed", "partial")
    assert final.status in ("partial", "passed")
    report = json.loads((ctx.output / "kernel_report.json").read_text(encoding="utf-8"))
    assert set(report["blocked"]) == {"AIV", "CCU"}


def test_kernel_build_is_never_told_to_clean(monkeypatch, tmp_path):
    calls = []

    class FakeCommands:
        log_dir = tmp_path / "logs"

        def run(self, argv, log_name=None, **kwargs):
            from scripts.workflow.phases import CommandResult
            calls.append([str(item) for item in argv])
            return CommandResult(argv=argv, exit_code=1, stdout="", stderr="blocked",
                                 log_path=tmp_path / "logs" / "build.log")

    monkeypatch.setattr(kernel, "plan_build", lambda root: {
        "toolchain": {"complete": True}, "engines": {}, "ready": {}, "buildable": ["AIV"],
        "blocked": {}, "native": {}})
    request = request_module.build({"intent": "partial_benchmark", "scope": {"ep_sizes": [8]}},
                                   root=ROOT)
    state = state_module.WorkflowState(workflow_id="w", run_id="w")
    context = phase_runner.WorkflowContext(
        state=state, request=request, store=state_module.WorkflowStore(tmp_path / "wf"),
        commands=FakeCommands(), root=ROOT, output=tmp_path)
    context.data["doctor"] = {"platform": {"soc": "Ascend950"}}
    outcome = kernel.handler(context)
    assert outcome.status == "partial"
    argv = calls[0]
    assert "--clean" not in argv
    assert "--strict" in argv
    assert argv[argv.index("--soc") + 1] == "Ascend950", "the SoC is read from the doctor"


# ------------------------------------------------------------------- plan ---

def test_a_plan_that_drops_an_ep_size_fails_the_gate(workflow):
    final, ctx, fake = workflow("只跑 EP8 的 L1", devices=16)
    assert final.phases["plan_generation"].status in ("passed", "partial")
    counts = ctx.data["plan"]["counts"]
    assert set(counts["by_ep_size"]) == {"8"}


def test_no_executable_task_blocks_the_workflow(workflow):
    final, ctx, fake = workflow("只跑 EP8 的 L1", devices=16, run_outcome="empty")
    assert final.phases["plan_generation"].status in ("passed", "partial")
    execution = final.phases["benchmark_execution"]
    assert execution.status == "partial"
    assert "no task passed" in execution.reason


# -------------------------------------------------------------- execution ---

def test_non_zero_exit_with_terminal_tasks_still_delivers(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=16,
                                run_outcome="partial")
    execution = final.phases["benchmark_execution"]
    assert execution.status == "passed"
    summary = json.loads((ctx.output / "execution_summary.json").read_text(encoding="utf-8"))
    assert summary["exit_code"] != 0
    assert final.phases["report_generation"].status == "passed"
    assert final.status == "partial"


def test_a_passed_task_without_correctness_fails_validation(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=16,
                                run_outcome="no_correctness")
    validation = final.phases["result_validation"]
    assert validation.status == "failed"
    assert "without a correctness record" in validation.reason
    assert final.status == "failed"


def test_correctness_failure_is_reported_not_retried(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=16,
                                run_outcome="correctness_failed")
    validation = final.phases["result_validation"]
    assert validation.status == "partial"
    assert "correctness" in validation.reason
    calls = [call for call in fake.calls if "resume" in call]
    assert not calls, "a correctness failure must not be retried blindly"


def test_incomplete_run_stops_the_delivery_at_partial(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=16,
                                run_outcome="incomplete")
    assert final.phases["benchmark_execution"].status == "partial"
    assert final.phases["result_validation"].status == "partial"
    assert final.status == "partial"


# -------------------------------------------------------------- profiling ---

def test_profiling_none_is_skipped_with_the_request_as_reason(workflow):
    final, ctx, fake = workflow("只生成测试计划")
    assert final.phases["profiling"].status == "skipped"
    assert "not part of" in final.phases["profiling"].reason


def test_profiling_failure_is_partial_not_fatal(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告，对主要算子执行 Profiling",
                                devices=16, profile_ok=False)
    profiling = final.phases["profiling"]
    assert profiling.status == "partial"
    assert "no parsable artifact" in profiling.reason
    assert final.status == "partial"


def test_profile_only_request_needs_an_existing_run(workflow):
    final, ctx, fake = workflow("对 EP8、EP64 的主要算子执行 Profiling", devices=16)
    profiling = final.phases["profiling"]
    assert profiling.status == "failed"
    assert "no manifest" in profiling.reason


# ----------------------------------------------------------------- report ---

def test_missing_report_file_fails_the_gate(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=16,
                                report_ok=False)
    report = final.phases["report_generation"]
    assert report.status == "failed"
    assert "benchmark_report.md" in report.reason


def test_final_validation_detects_an_accounting_mismatch(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=16)
    summary_path = ctx.run_output_dir / ctx.run_id / "ep_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["planned_tasks"] += 3
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    ctx.state.phases["final_validation"].status = "pending"
    ctx.store.save(ctx.state)
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id=ctx.state.workflow_id,
                                store_root=ctx.store.output_root, resume=True)
    validation = final.phases["final_validation"]
    assert validation.status == "failed"
    assert "sum to" in validation.reason
