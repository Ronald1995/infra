"""Resume: continue the unfinished phase, never the finished ones."""


from common.registry import ROOT
from scripts.workflow import state as state_module


def _commands_for(fake, phase_name):
    """The CLI subcommands recorded for one phase name in the fake call list."""
    return [call for call in fake.calls]


def test_a_workflow_resumes_at_the_first_unfinished_phase(workflow, tmp_path):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-resume",
                                store_root=tmp_path / "workflows")
    assert final.status in ("passed", "partial")
    # Pretend the last three phases never ran, as a crash would leave them.
    for name in ("analysis", "report_generation", "final_validation"):
        ctx.state.phases[name].status = "pending"
    ctx.store.save(ctx.state)
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-resume",
                                store_root=tmp_path / "workflows", resume=True)
    second = list(fake.calls)
    subcommands = [call[call.index(str(ROOT / "run_benchmark.py")) + 1] for call in second
                   if str(ROOT / "run_benchmark.py") in call]
    assert "doctor" not in subcommands, "a finished environment probe must not re-run"
    assert "plan" not in subcommands, "a validated plan must not be regenerated"
    assert "run" not in subcommands and "resume" not in subcommands
    assert "analyze" in subcommands and "report" in subcommands
    assert final.status in ("passed", "partial")


def test_an_interrupted_execution_continues_with_resume(workflow, tmp_path):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-int",
                                store_root=tmp_path / "workflows", run_outcome="incomplete")
    assert final.phases["benchmark_execution"].status == "partial"
    assert final.phases["result_validation"].status == "partial"
    # The run directory exists and is incomplete; the next attempt must resume it.
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-int",
                                store_root=tmp_path / "workflows", resume=True,
                                run_outcome="passed")
    second = list(fake.calls)
    subcommands = [call[call.index(str(ROOT / "run_benchmark.py")) + 1] for call in second
                   if str(ROOT / "run_benchmark.py") in call]
    assert "resume" in subcommands
    assert "run" not in subcommands, "a new run must never be started for an existing id"
    assert final.status in ("passed", "partial")
    assert final.phases["benchmark_execution"].status in ("passed", "skipped")


def test_a_completed_run_is_not_executed_again(workflow, tmp_path):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-again",
                                store_root=tmp_path / "workflows")
    run_calls = [call for call in fake.calls if "run" in call and call[-1] not in ("report",)]
    assert run_calls
    # Force only the execution phase to be re-run.
    ctx.state.phases["benchmark_execution"].status = "pending"
    ctx.store.save(ctx.state)
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-again",
                                store_root=tmp_path / "workflows", resume=True)
    second = list(fake.calls)
    subcommands = [call[call.index(str(ROOT / "run_benchmark.py")) + 1] for call in second
                   if str(ROOT / "run_benchmark.py") in call]
    assert "run" not in subcommands and "resume" not in subcommands
    assert final.phases["benchmark_execution"].status == "skipped"
    assert "already complete" in final.phases["benchmark_execution"].reason


def test_changed_request_invalidates_the_plan(tmp_path):
    from scripts.workflow import request as request_module
    store = state_module.WorkflowStore(tmp_path / "wf-store")
    request = request_module.build({"intent": "plan_only", "scope": {"ep_sizes": [8]}},
                                   root=ROOT)
    state = store.create(request, workflow_id="wf-change", repo_commit="abc")
    for name in ("request_resolution", "environment_discovery", "plan_generation",
                 "benchmark_execution", "analysis"):
        state.finish_phase(name, "passed")
    store.save(state)
    other = request_module.build({"intent": "plan_only", "scope": {"ep_sizes": [64]}},
                                 root=ROOT)
    assert state_module.needs_revalidation(state, other, "abc") == "plan_generation"
    state.invalidate_from("plan_generation", "the request changed")
    assert state.phases["request_resolution"].status == "passed"
    assert state.phases["plan_generation"].status == "pending"
    assert state.phases["analysis"].status == "pending"
    # A changed commit invalidates the same way.
    assert state_module.needs_revalidation(state, request, "def") == "plan_generation"


def test_revalidation_is_recorded_in_the_event_log(tmp_path):
    from scripts.workflow import request as request_module
    store = state_module.WorkflowStore(tmp_path / "wf-store")
    request = request_module.build({"intent": "plan_only", "scope": {"ep_sizes": [8]}},
                                   root=ROOT)
    state = store.create(request, workflow_id="wf-ev", repo_commit="abc")
    state.finish_phase("plan_generation", "passed")
    store.save(state)
    state.invalidate_from("plan_generation", "the commit changed")
    store.event(state, event="revalidated", from_phase="plan_generation")
    store.save(state)
    assert any(event["event"] == "revalidated" for event in store.events("wf-ev"))
    assert "invalidated" in store.load("wf-ev").phases["plan_generation"].reason


def test_phase_retry_keeps_the_attempt_history(workflow, tmp_path):
    final, ctx, fake = workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-attempts",
                                store_root=tmp_path / "workflows")
    attempts = ctx.state.phases["benchmark_execution"].attempts
    assert attempts >= 1
    ctx.state.phases["benchmark_execution"].status = "pending"
    ctx.store.save(ctx.state)
    workflow("完整跑一次 Benchmark 并生成报告", workflow_id="wf-attempts",
             store_root=tmp_path / "workflows", resume=True)
    assert ctx.store.load("wf-attempts").phases["benchmark_execution"].attempts > attempts
