"""Source-only scope and quarantine checks; does not import the NPU stack."""

import ast
import copy
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "worker-rfc-consolidated"
BASELINE = "28065346d433f9d23157c77c2b5b79c959a61871"


def class_node(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)


def methods(cls):
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def reconstruct(cls, bases):
    cls = copy.deepcopy(cls)
    cls.bases = [ast.Name(id=name, ctx=ast.Load()) for name in bases]
    cls.body = list(methods(cls).values())
    for method in cls.body:
        method.decorator_list = [
            d for d in method.decorator_list if isinstance(d, ast.Name) and d.id == "property"
        ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    scope = dict(bases)
    exec(compile(ast.fix_missing_locations(module), "<source-declarations>", "exec"), scope)
    return scope[cls.name]


base_source = subprocess.check_output(
    ["git", "show", f"{BASELINE}:vllm_ascend/worker/worker.py"], cwd=ROOT
).decode("utf-8")
base = class_node(base_source, "NPUWorker")
current = class_node((ROOT / "vllm_ascend/worker/worker.py").read_text(encoding="utf-8"), "NPUWorker")
parent = class_node(Path(__file__).with_name("gpu_worker_parent.py").read_text(encoding="utf-8"), "Worker")
base_methods, current_methods, parent_methods = map(methods, (base, current, parent))
for name, method in base_methods.items():
    if name != "__init__":
        assert ast.dump(method) == ast.dump(current_methods[name]), name
print(f"UNCHANGED: {len(base_methods) - 1} existing non-constructor methods, including decorators")

interfaces = types.ModuleType("vllm_ascend.worker.worker_interfaces")
exec((ROOT / "vllm_ascend/worker/worker_interfaces.py").read_text(encoding="utf-8"), interfaces.__dict__)
blocked = set(interfaces.UNVERIFIED_GPU_WORKER_METHODS)
hooks = {"_make_elastic_ep_executor", "_validate_profiler_config"}
assert blocked == parent_methods.keys() - base_methods.keys() - hooks
assert current_methods.keys() - base_methods.keys() == blocked | hooks
for name, method in base_methods.items():
    if name == "__init__":
        continue
    references = {
        n.attr for n in ast.walk(method)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
    }
    assert not references & blocked, (name, references & blocked)
print(f"COVERED: all {len(blocked)} newly visible interfaces; existing methods do not reference the blocked helpers")

gpu = reconstruct(parent, {"WorkerBase": type("WorkerBase", (), {})})
npu = reconstruct(current, {"GPUWorker": gpu})
parent_module = types.ModuleType("vllm.v1.worker.gpu_worker")
parent_module.Worker = gpu
npu_module = types.ModuleType("vllm_ascend.worker.worker")
npu_module.NPUWorker = npu
spec = importlib.util.spec_from_file_location("inheritance_guard", ROOT / "tests/ut/worker/test_worker_inheritance.py")
guard = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    interfaces.__name__: interfaces,
    parent_module.__name__: parent_module,
    npu_module.__name__: npu_module,
}):
    spec.loader.exec_module(guard)
    guard.test_no_unverified_gpu_worker_interface_is_inherited()
    print("SOURCE GUARD: passes using actual source declarations (not runtime imports)")
    gpu.new_upstream_api = lambda self: None
    try:
        guard.test_no_unverified_gpu_worker_interface_is_inherited()
    except AssertionError:
        print("MUTATION: newly added upstream API correctly fails the guard")
    else:
        raise AssertionError("new upstream method escaped the guard")
    del gpu.new_upstream_api
    saved = npu.get_draft_model
    del npu.get_draft_model
    try:
        guard.test_no_unverified_gpu_worker_interface_is_inherited()
    except AssertionError:
        print("MUTATION: removed quarantine stub correctly fails the guard")
    else:
        raise AssertionError("missing stub escaped the guard")
    npu.get_draft_model = saved
    guard.test_no_unverified_gpu_worker_interface_is_inherited()

print("PASS: source checks and in-memory guard mutations")
