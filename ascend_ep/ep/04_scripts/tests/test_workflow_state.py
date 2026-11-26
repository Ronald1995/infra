"""Workflow state: phases, persistence, events, invalidation."""
import json

import pytest

from common.registry import ROOT
from scripts.workflow import request as request_module
from scripts.workflow import state as state_module


@pytest.fixture
def base_request():
    return request_module.build({
        "intent": "plan_only",
        "plan": "configs/benchmark_plan.yaml",
        "scope": {"ep_sizes": [8, 64]},
    }, root=ROOT)


def test_phases_cover_the_documented_pipeline():
    assert state_module.PHASES[0] == "request_resolution"
    assert state_module.PHASES[-1] == "final_validation"
    for name in ("environment_discovery", "dependency_check", "dependency_install",
                 "capability_probe", "kernel_build", "plan_generation",
                 "resource_estimation", "benchmark_execution", "result_validation",
                 "profiling", "analysis", "report_generation"):
        assert name in state_module.PHASES


def test_a_new_state_starts_pending(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-1", repo_commit="abc")
    assert state.status == "pending"
    assert state.completed() == []
    assert state.resume_point() == "request_resolution"
    assert all(state.phases[name].status == "pending" for name in state_module.PHASES)
    assert (tmp_path / "wf-1" / "state.json").is_file()
    assert (tmp_path / "wf-1" / "request.json").is_file()


def test_state_round_trips_through_disk(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-2")
    state.start_phase("request_resolution")
    state.add_command("request_resolution", ["python", "run_benchmark.py", "plan"], 0, "log")
    state.finish_phase("request_resolution", "passed", artifacts=["a.json"])
    store.save(state)
    loaded = store.load("wf-2")
    assert loaded.phases["request_resolution"].status == "passed"
    assert loaded.phases["request_resolution"].commands[0]["exit_code"] == 0
    assert loaded.phases["request_resolution"].artifacts == ["a.json"]
    assert loaded.current_phase == "request_resolution"


def test_completed_phase_is_not_resumable(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-3")
    state.finish_phase("request_resolution", "passed")
    state.finish_phase("environment_discovery", "passed")
    assert state.completed() == ["request_resolution", "environment_discovery"]
    assert state.resume_point() == "dependency_check"


def test_skip_requires_a_reason(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-4")
    with pytest.raises(ValueError):
        state.skip_phase("kernel_build", None)
    state.skip_phase("kernel_build", "no device kernel is needed on CPU")
    assert state.phases["kernel_build"].status == "skipped"
    assert state.phases["kernel_build"].reason


def test_invalid_statuses_are_rejected(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-5")
    with pytest.raises(ValueError):
        state.finish_phase("request_resolution", "done")
    with pytest.raises(ValueError):
        state_module.PhaseRecord(name="x", status="ok")


def test_invalidation_marks_later_phases_pending(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-6")
    for name in ("request_resolution", "environment_discovery", "plan_generation",
                 "benchmark_execution"):
        state.finish_phase(name, "passed")
    state.invalidate_from("plan_generation", "the commit changed")
    assert state.phases["request_resolution"].status == "passed"
    assert state.phases["plan_generation"].status == "pending"
    assert state.phases["benchmark_execution"].status == "pending"
    assert "invalidated" in state.phases["benchmark_execution"].reason


def test_events_are_append_only(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-7")
    store.event(state, event="phase_started", phase="request_resolution")
    store.event(state, event="phase_finished", phase="request_resolution", status="passed")
    events = store.events("wf-7")
    assert [event["event"] for event in events] == ["created", "phase_started",
                                                    "phase_finished"]
    lines = (tmp_path / "wf-7" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line) for line in lines)


def test_store_finds_the_latest_unfinished_workflow(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    finished = store.create(base_request, workflow_id="wf-a")
    for name in state_module.PHASES:
        finished.finish_phase(name, "passed")
    store.save(finished)
    open_state = store.create(base_request, workflow_id="wf-b")
    open_state.finish_phase("request_resolution", "passed")
    store.save(open_state)
    assert store.latest().workflow_id in ("wf-a", "wf-b")
    assert store.latest(unfinished=True).workflow_id == "wf-b"


def test_needs_revalidation_on_a_changed_request_or_commit(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    state = store.create(base_request, workflow_id="wf-8", repo_commit="abc")
    assert state_module.needs_revalidation(state, base_request, "abc") is None
    assert state_module.needs_revalidation(state, base_request, "def") == "plan_generation"
    other = request_module.build({
        "intent": "plan_only", "scope": {"ep_sizes": [8]},
    }, root=ROOT)
    assert state_module.needs_revalidation(state, other, "abc") == "plan_generation"


def test_workflow_ids_are_unique_and_prefixed(tmp_path, base_request):
    store = state_module.WorkflowStore(tmp_path)
    first = store.new_id("full_benchmark")
    second = store.new_id("full_benchmark")
    assert first.startswith("ep-full-benchmark-")
    assert first != second
    with pytest.raises(FileExistsError):
        store.create(base_request, workflow_id="dup")
        store.create(base_request, workflow_id="dup")


def test_corrupt_state_is_reported(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "state.json").write_text("{not json", encoding="utf-8")
    store = state_module.WorkflowStore(tmp_path)
    with pytest.raises(ValueError, match="corrupt"):
        store.load("broken")


def test_the_request_matches_its_schema():
    import jsonschema

    schema = json.loads((ROOT / "scripts" / "workflow" / "request.schema.json")
                        .read_text(encoding="utf-8"))
    prompts = [
        "在 Ascend 950 上运行完整 EP Benchmark，EP8、EP64、EP128，覆盖 L0-L4 和所有算子。"
        "如果缺少依赖，尝试安装。",
        "只跑 EP64 的 L1、L2，比较所有支持的 Operator",
        "先帮我估算完整 Benchmark 的任务数和运行时间，不要执行测试",
        "继续上次没跑完的 EP Benchmark",
        "对 EP8、EP64、EP128 的主要算子执行 Profiling",
        "在当前 CPU 环境测试整个框架",
    ]
    for prompt in prompts:
        payload = request_module.interpret(prompt, root=ROOT).as_dict()
        payload.pop("sources")
        payload.pop("prompt")
        jsonschema.validate(payload, schema)


def test_the_request_schema_rejects_an_unknown_field():
    import jsonschema

    schema = json.loads((ROOT / "scripts" / "workflow" / "request.schema.json")
                        .read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"schema_version": 1, "intent": "full_benchmark",
                             "unexpected": True}, schema)
    with pytest.raises(request_module.RequestError):
        request_module.build({"intent": "full_benchmark", "unexpected": True}, root=ROOT)
