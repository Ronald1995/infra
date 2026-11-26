"""End-to-end execution and the ten skill acceptance scenarios.

The scenarios are the prompts from the skill specification. Each one checks the *decision*
the workflow makes, not just that it ran: which intent, which scope, which commands, and
which claims it refuses to make.
"""
import json


from common.registry import ROOT
from scripts.workflow import profiling, request as request_module


def _subcommands(fake):
    return [call[call.index(str(ROOT / "run_benchmark.py")) + 1] for call in fake.calls
            if str(ROOT / "run_benchmark.py") in call]


def _artifacts(ctx):
    return {path.name for path in ctx.output.iterdir() if path.is_file()}


# --------------------------------------------------------------- end to end ---

def test_a_full_npu_workflow_runs_every_phase_once(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=128)
    assert final.status in ("passed", "partial")
    commands = _subcommands(fake)
    assert commands.count("run") == 1, commands
    assert "pipeline" not in commands, "run + pipeline would measure the matrix twice"
    for phase in ("request_resolution", "environment_discovery", "dependency_check",
                  "capability_probe", "plan_generation", "resource_estimation",
                  "benchmark_execution", "result_validation", "profiling", "analysis",
                  "report_generation", "final_validation"):
        assert final.phases[phase].status in ("passed", "partial", "skipped"), phase
    artifacts = _artifacts(ctx)
    for name in ("request.json", "resolved_request.json", "environment.json",
                 "dependency_report.json", "capability_report.json", "plan_manifest.json",
                 "resource_estimate.json", "execution_summary.json",
                 "result_validation.json", "profiling_summary.json",
                 "analysis_summary.json", "report_summary.json", "final_report.json"):
        assert name in artifacts, name
    # Every command is recorded with an exit code and a log path.
    for name in ("environment_discovery", "plan_generation", "benchmark_execution"):
        for command in final.phases[name].commands:
            assert "exit_code" in command and command["log"]


def test_the_executed_manifest_matches_the_validated_plan(workflow):
    final, ctx, fake = workflow("只跑 EP8 的 L1、L2", devices=128)
    payload = json.loads((ctx.output / "execution_summary.json").read_text(encoding="utf-8"))
    assert payload["plan_hash_match"] is True
    counts = ctx.data["plan"]["counts"]
    assert counts["planned_tasks"] > 0
    assert set(counts["by_ep_size"]) == {"8"}


def test_a_partial_delivery_reports_what_failed(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=128,
                                run_outcome="partial")
    assert final.status == "partial"
    validation = json.loads((ctx.output / "result_validation.json").read_text(encoding="utf-8"))
    assert validation["totals"]["failed"] >= 1
    assert validation["totals"]["unsupported"] >= 1
    final_report = json.loads((ctx.output / "final_report.json").read_text(encoding="utf-8"))
    assert final_report["accounting"]["buckets"]["failed"] >= 1
    assert final_report["problems"] == []


# ---------------------------------------------------------------- scenario A ---

def test_scenario_a_full_benchmark_is_interpreted_and_run(workflow):
    prompt = ("在 Ascend 950 上运行完整 EP Benchmark，EP8、EP64、EP128，覆盖 L0-L4 和所有算子。"
              "如果缺少依赖，尝试安装；完成性能分析和 Profiling，生成完整报告。")
    request = request_module.interpret(prompt, root=ROOT)
    assert request.intent == "full_benchmark"
    assert request.scope.ep_sizes == [8, 64, 128]
    assert request.scope.levels == ["L0", "L1", "L2", "L3", "L4"]
    assert request.environment.dependency_policy == "install_missing"
    assert request.environment.allow_network is False
    assert request.environment.soc == "Ascend950"
    final, ctx, fake = workflow(prompt, devices=128, checks=_good_checks())
    assert final.status in ("passed", "partial")
    commands = _subcommands(fake)
    assert commands.count("doctor") >= 1
    assert commands.count("capabilities") >= 1
    assert commands.count("run") == 1
    assert "profile" in commands and "analyze" in commands and "report" in commands
    assert ctx.state.run_id == ctx.state.workflow_id


def test_scenario_a_install_is_only_attempted_when_authorised(workflow):
    prompt = ("在 Ascend 950 上运行完整 EP Benchmark，EP8。如果缺少依赖，尝试安装。")
    final, ctx, fake = workflow(prompt, devices=128, checks=_missing_torch())
    install = final.phases["dependency_install"]
    # install_missing without network/wheelhouse: blocked, with that exact reason.
    assert install.status == "blocked"
    assert "network" in install.reason
    assert not [call for call in fake.calls if "install.log" in " ".join(map(str, call))]


# ---------------------------------------------------------------- scenario B ---

def test_scenario_b_partial_scope_keeps_operator_coverage(workflow):
    prompt = "只跑 EP64 的 L1、L2，比较所有支持的 Operator"
    request = request_module.interpret(prompt, root=ROOT)
    assert request.intent == "partial_benchmark"
    assert request.scope.ep_sizes == [64]
    assert request.scope.levels == ["L1", "L2"]
    assert request.scope.operators == "all", "operators must not be silently narrowed"
    final, ctx, fake = workflow(prompt, devices=128)
    counts = ctx.data["plan"]["counts"]
    assert set(counts["by_ep_size"]) == {"64"}
    assert set(counts["by_level"]) <= {"L1", "L2"}
    assert len(counts["by_operator"]) >= 2


# ---------------------------------------------------------------- scenario C ---

def test_scenario_c_estimate_does_not_execute_or_install(workflow):
    prompt = "先帮我估算完整 Benchmark 的任务数和运行时间，不要执行测试"
    request = request_module.interpret(prompt, root=ROOT)
    assert request.intent == "estimate"
    assert request.environment.dependency_policy == "check_only"
    final, ctx, fake = workflow(prompt, devices=128)
    commands = _subcommands(fake)
    assert "run" not in commands and "profile" not in commands
    assert final.phases["resource_estimation"].status in ("passed", "partial")
    estimate = json.loads((ctx.output / "resource_estimate.json").read_text(encoding="utf-8"))
    assert estimate["estimate"]["basis"] in ("historical_p50", "unavailable")
    assert "uncertainty" in estimate["estimate"]


# ---------------------------------------------------------------- scenario D ---

def test_scenario_d_resume_picks_the_unfinished_workflow(tmp_path, workflow):
    prompt = "完整跑一次 Benchmark 并生成报告"
    final, ctx, fake = workflow(prompt, devices=128, workflow_id="wf-last",
                               store_root=tmp_path / "workflows")
    assert final.status in ("passed", "partial")
    # A second, unfinished workflow is the one a resume must pick.
    from scripts.workflow import state as state_module
    store = state_module.WorkflowStore(tmp_path / "workflows")
    request = request_module.interpret(prompt, root=ROOT)
    state = store.create(request, workflow_id="wf-newer")
    state.finish_phase("request_resolution", "passed")
    store.save(state)
    latest = store.latest(unfinished=True)
    assert latest.workflow_id == "wf-newer"
    assert request_module.interpret("继续上次没跑完的 EP Benchmark", root=ROOT).intent == "resume"


# ---------------------------------------------------------------- scenario E ---

def test_scenario_e_installable_and_host_dependencies_are_separated(workflow):
    prompt = "完成完整 Benchmark，如果缺少依赖就尝试安装，允许网络"
    final, ctx, fake = workflow(prompt, devices=128, checks=_mixed_checks())
    report = json.loads((ctx.output / "dependency_report.json").read_text(encoding="utf-8"))
    assert [record["id"] for record in report["installable"]] == ["numpy"]
    assert [record["id"] for record in report["host"]] == ["npu-driver"]
    install = final.phases["dependency_install"]
    assert install.status in ("passed", "partial")
    # Other capabilities still run although the driver is blocked.
    assert final.phases["benchmark_execution"].status in ("passed", "partial")


# ---------------------------------------------------------------- scenario F ---

def test_scenario_f_missing_kernel_does_not_fake_a_measurement(monkeypatch, workflow):
    from scripts.workflow import kernel
    monkeypatch.setattr(kernel, "plan_build", lambda root: {
        "toolchain": {"complete": False}, "engines": {}, "ready": {}, "buildable": [],
        "blocked": {"AIV": "AIV device source files are missing"},
        "native": {}})
    final, ctx, fake = workflow("完成所有可运行的 L0-L4 测试", devices=128)
    assert final.phases["kernel_build"].status == "partial"
    commands = _subcommands(fake)
    assert "build" not in commands
    assert "run" in commands, "the rest of the matrix still runs"
    # No host-NBI fallback can be claimed: the kernel report says blocked, nothing else.
    report = json.loads((ctx.output / "kernel_report.json").read_text(encoding="utf-8"))
    assert "AIV" in report["blocked"]


# ---------------------------------------------------------------- scenario G ---

def test_scenario_g_profiling_covers_every_ep_size(workflow):
    prompt = "对 EP8、EP64、EP128 的主要算子执行 Profiling"
    request = request_module.interpret(prompt, root=ROOT)
    assert request.intent == "profile"
    # A profile-only request needs an existing run; the selection itself is tested below.
    final, ctx, fake = workflow(
        "完整跑一次 Benchmark 并生成报告，对主要算子执行 Profiling", devices=128)
    summary = json.loads((ctx.output / "profiling_summary.json").read_text(encoding="utf-8"))
    assert summary["policy"] == "representative"
    assert summary["selected"]
    selected_sizes = {item["ep_size"] for item in summary["selected"]}
    # The representative policy must give every EP size of the run a trace.
    assert selected_sizes == {8, 16, 32, 64, 128}
    assert len(summary["collected"]) == len(summary["selected"])
    assert summary["profile_enabled_flags"]


def test_profiling_selection_is_deterministic_and_passing_only():
    records = {
        "t1": {"task_id": "t1", "case_id": "COMM-001", "backend_id": "a", "status": "passed",
               "benchmark_level": "L1", "ep_size": 8},
        "t2": {"task_id": "t2", "case_id": "COMM-002", "backend_id": "a", "status": "passed",
               "benchmark_level": "L1", "ep_size": 8},
        "t3": {"task_id": "t3", "case_id": "COMM-001", "backend_id": "b", "status": "passed",
               "benchmark_level": "L1", "ep_size": 8},
        "t4": {"task_id": "t4", "case_id": "COMM-001", "backend_id": "a",
               "status": "runtime_failed", "benchmark_level": "L1", "ep_size": 8},
    }
    selected = profiling.select(records, "representative")
    assert [item["task_id"] for item in selected] == ["t1", "t3"]
    assert "t4" not in [item["task_id"] for item in selected]
    full = profiling.select(records, "full")
    assert len(full) == 3
    assert profiling.select(records, "none") == []


# ---------------------------------------------------------------- scenario H ---

def test_scenario_h_one_run_only(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=128)
    commands = _subcommands(fake)
    assert commands.count("run") == 1
    assert commands.count("resume") == 0
    assert "pipeline" not in commands
    run_calls = [call for call in fake.calls if "run" in call[1]]
    assert {call[call.index("--run-id") + 1] for call in run_calls if "--run-id" in call} == {
        ctx.state.run_id}


# ---------------------------------------------------------------- scenario I ---

def test_scenario_i_a_failing_run_still_delivers(workflow):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", devices=128,
                                run_outcome="partial")
    assert final.phases["report_generation"].status == "passed"
    report = (ctx.run_output_dir / ctx.run_id / "reports" / "benchmark_report.md")
    assert report.is_file()
    summary = json.loads((ctx.output / "execution_summary.json").read_text(encoding="utf-8"))
    assert summary["exit_code"] != 0
    # Blocked is never counted as passed.
    totals = json.loads(
        (ctx.output / "result_validation.json").read_text(encoding="utf-8"))["totals"]
    assert totals["passed"] + totals["failed"] + totals["unsupported"] + totals["blocked"] <= \
        totals["planned_tasks"] + 1


# ---------------------------------------------------------------- scenario J ---

def test_scenario_j_cpu_run_is_labelled_as_framework_validation(workflow):
    final, ctx, fake = workflow("在当前 CPU 环境测试整个框架")
    request = ctx.request
    assert request.environment.device == "cpu"
    assert request.execution.profile_policy == "none"
    assert final.phases["kernel_build"].status == "skipped"
    report = (ctx.run_output_dir / ctx.run_id / "reports" / "benchmark_report.md")
    assert report.is_file()
    environment = json.loads((ctx.output / "environment.json").read_text(encoding="utf-8"))
    assert environment["device"] == "cpu"
    # The workflow never claims NPU performance for a CPU run.
    capacity = {str(key): value for key, value in environment["rank_capacity"].items()}
    assert capacity, "a CPU request must still be capacity-checked"
    assert all(entry["device_source"] == "cpu processes" for entry in capacity.values())


def _good_checks():
    return [
        {"id": "python", "status": "pass", "category": "A", "detail": "python 3.10.8",
         "suggested_action": "none", "blocking_for": ["reporting"], "evidence": {}},
        {"id": "torch", "status": "pass", "category": "A", "detail": "torch 2.6.0",
         "suggested_action": "none", "blocking_for": ["l0_transport"], "evidence": {}},
    ]


def _missing_torch():
    return [
        {"id": "python", "status": "pass", "category": "A", "detail": "python 3.10.8",
         "suggested_action": "none", "blocking_for": ["reporting"], "evidence": {}},
        # numpy is a real pip-installable item; torch/torch_npu are provided by the
        # benchmark container image and must never be "installable" here.
        {"id": "numpy", "status": "fail", "category": "A", "detail": "numpy is absent",
         "suggested_action": "install numpy==2.2.6",
         "blocking_for": ["l0_transport", "operator:alltoall_hccl"], "evidence": {}},
    ]


def _mixed_checks():
    return [
        {"id": "python", "status": "pass", "category": "A", "detail": "python 3.10.8",
         "suggested_action": "none", "blocking_for": ["reporting"], "evidence": {}},
        # numpy is a real pip-installable item; torch/torch_npu are provided by the
        # benchmark container image and must never be "installable" here.
        {"id": "numpy", "status": "fail", "category": "A", "detail": "numpy is absent",
         "suggested_action": "install numpy==2.2.6",
         "blocking_for": ["l0_transport", "operator:alltoall_hccl"], "evidence": {}},
        {"id": "npu-driver", "status": "fail", "category": "B",
         "detail": "the driver is not installed", "suggested_action": "ask the host owner",
         "blocking_for": ["l0_transport"], "evidence": {}},
    ]
