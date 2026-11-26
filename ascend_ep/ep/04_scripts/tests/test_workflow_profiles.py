"""The workflow carries the execution profile from the request to the plan.

A profile narrows one invocation without changing the declared matrix; the request
must therefore be able to ask for it, the plan command must receive it, and the
narrowed plan the workflow writes must record it.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.workflow import request as request_module  # noqa: E402


def test_explicit_profile_words_are_interpreted():
    assert request_module.interpret("run a smoke check", root=ROOT).scope.execution_profile == "smoke"
    assert request_module.interpret("做一次冒烟验证", root=ROOT).scope.execution_profile == "smoke"
    assert request_module.interpret("core baseline please", root=ROOT).scope.execution_profile == "core"
    assert request_module.interpret("exhaustive parameter study",
                                    root=ROOT).scope.execution_profile == "exhaustive"


def test_vague_wording_does_not_pick_a_profile():
    for prompt in ("run the full benchmark", "快速看一下", "执行 Benchmark"):
        assert request_module.interpret(prompt, root=ROOT).scope.execution_profile == ""


def test_the_scope_serialises_the_profile():
    request = request_module.interpret("smoke check", root=ROOT)
    assert request.scope.as_dict()["execution_profile"] == "smoke"
    data = request.as_dict()
    scope = data.get("scope") or {}
    assert scope.get("execution_profile") == "smoke"


def test_scope_args_carry_the_profile():
    from scripts.workflow import planning

    request = request_module.interpret("smoke check", root=ROOT)

    from types import SimpleNamespace

    argv = planning.scope_args(SimpleNamespace(request=request, root=ROOT))
    assert "--execution-profile" in argv
    assert argv[argv.index("--execution-profile") + 1] == "smoke"


def test_the_narrowed_plan_records_the_profile(tmp_path):
    from scripts.workflow import planning

    request = request_module.interpret("smoke check on the plan", root=ROOT)
    request.plan = "configs/benchmark_plan.yaml"
    request.scope.suites = ["all"]
    request.scope.levels = ["L0"]

    from types import SimpleNamespace

    path = planning.narrowed_plan(
        SimpleNamespace(root=ROOT, output=tmp_path, request=request), ep_sizes=[8])
    import yaml

    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert (spec.get("plan") or {}).get("profile") == "smoke"


def test_a_request_without_a_profile_keeps_the_plan_default(tmp_path):
    from scripts.workflow import planning

    request = request_module.interpret("run the full benchmark", root=ROOT)
    request.plan = "configs/benchmark_plan.yaml"
    request.scope.suites = ["all"]
    request.scope.levels = ["L0"]

    from types import SimpleNamespace

    path = planning.narrowed_plan(
        SimpleNamespace(root=ROOT, output=tmp_path, request=request), ep_sizes=[8])
    import yaml

    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert "profile" not in (spec.get("plan") or {})
