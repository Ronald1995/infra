"""Shared builders for the reporting tests.

The tests construct a *run directory* the same way the benchmark does -- manifest,
result records, then the real ``analyze`` + ``report`` entry points -- so a report bug
shows up here before it reaches a device run.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.generate_report import analyze_run
from common.cases import catalog
from common.config import DEFAULT, merge, validate
from common.plan import matrix_metadata
from common.storage import write_json
from reporting.report import report
from schemas import result

#: A deterministic environment block for one synthetic run.
NPU_ENVIRONMENT = {
    "host": "test-host", "platform": "Linux-5.10-aarch64", "python": "3.10.8",
    "packages": {"torch": "2.6.0", "torch-npu": "2.6.0", "PyYAML": "6.0.1"},
    "cann_version": "8.2.RC1", "cann_version_source": "env:ASCEND_TOOLKIT_VERSION",
    "cann_version_error": None, "driver_version": "24.1.0",
    "driver_version_source": "npu-smi info", "npu_models": ["Ascend910B4"],
    "git_commit": "deadbeefcafe", "git_changes": None, "device_visibility": "0,1,2,3",
}

CPU_ENVIRONMENT = {
    **NPU_ENVIRONMENT, "host": "cpu-host", "npu_models": None,
    "cann_version": None, "cann_version_source": "unavailable",
    "cann_version_error": "ASCEND_TOOLKIT_VERSION is unset", "driver_version": None,
    "driver_version_source": "unavailable", "packages": {"torch": "2.6.0+cpu"},
}


@pytest.fixture
def config_for():
    """``config_for(device, ep_size, **overrides)`` -> a validated configuration."""

    def build(device="cpu", ep_size=2, **overrides):
        base = {
            "device": device,
            "distributed": {"ep_size": ep_size, "tp_size": 1},
            "model": {"dtype": "bfloat16", "num_experts": 8, "top_k": 2,
                      "hidden_size": 128, "intermediate_size": 64},
            "workload": {"tokens_per_rank": 8, "routing": "balanced", "seed": 42},
            "benchmark": {"timing": "output_ready"},
        }
        for key, value in overrides.items():
            base[key] = merge(base.get(key) or {}, value) if isinstance(value, dict) else value
        return validate(merge(DEFAULT, base))

    return build


@pytest.fixture
def task_for(config_for):
    """``task_for(case_id, ...)`` -> one manifest task shaped like a real plan item."""

    def build(case_id, *, backend_id=None, ep_size=2, device="cpu", parameters=None,
              suite_ids=("test",), status="not_run", reason=None, index=0, config=None):
        case = catalog()[case_id]
        settings = deepcopy(config) if config is not None else config_for(
            device=device, ep_size=ep_size)
        backend = backend_id or ("hardware" if case.level == "L0" else "alltoall_hccl")
        params = dict(parameters) if parameters is not None else {"variant": "default"}
        return {
            "task_id": f"{case_id}-{backend}-{ep_size}-{index:03d}",
            "case_id": case_id, "level": case.level, "operation": case.operation,
            "backend_id": backend, "config": settings, "parameters": params,
            "status": status, "reason": reason, "source_versions": {"adapter_commit": "abc123"},
            "baseline": case.level == "L0", "ep_size": ep_size, "suite_ids": list(suite_ids),
            "case_category": case.category, "case_family": case.category,
            "case_tags": list(case.tags), "case_phase": case.phase, "handler": case.handler,
            "correctness_requirements": list(case.correctness), "alias_of": case.alias_of,
            "config_fingerprint": f"fingerprint-{case_id}-{backend}-{ep_size}",
        }

    return build


@pytest.fixture
def manifest_for():
    """``manifest_for(tasks, ...)`` -> a manifest with real matrix metadata."""

    def build(tasks, run_id="test-run", environment=None, matrix=None, report_options=None):
        metadata = matrix_metadata(list(tasks)) if matrix is None else dict(matrix)
        if report_options is not None:
            metadata["report"] = {**(metadata.get("report") or {}), **report_options}
        return {
            "schema_version": 1, "run_id": run_id,
            "environment": dict(environment if environment is not None
                                else (NPU_ENVIRONMENT if any(
                                    task["config"]["device"] == "npu" for task in tasks)
                                    else CPU_ENVIRONMENT)),
            "matrix": metadata, "tasks": list(tasks),
        }

    return build


@pytest.fixture
def record_for():
    """``record_for(manifest, task, ...)`` -> a result record with synthetic metrics."""

    def build(manifest, task, status="passed", *, p50=None, p95=None, p99=None,
              bandwidth=True, compute=False, throughput=None, correctness=True,
              latencies=None, extra=None):
        record = result(manifest, task, status)
        if status != "passed":
            record["correctness"] = None
            record.update(extra or {})
            return record
        p50 = 1.0 if p50 is None else p50
        record["latency"] = {
            "unit": "ms", "scope": "per_iteration_rank_max", "source": "measured",
            "p50": p50, "p95": p95 if p95 is not None else p50 * 1.1,
            "p99": p99 if p99 is not None else p50 * 1.15,
            "rank_min": latencies or [p50 * .95], "rank_mean": latencies or [p50],
            "rank_max": latencies or [p50], "rank_avg_p50_ms": p50 * .98,
            "cv": .02, "p95_p50_ratio": 1.1, "host_issue_p50_ms": p50 * .1,
            "host_wait_p50_ms": p50 * .05,
        }
        record["correctness"] = {"passed": True} if correctness else {"passed": False}
        record["source"] = "measured"
        if bandwidth:
            record["bandwidth"] = {
                "unit": "bytes", "bandwidth_unit": "GB/s", "scope": "per_rank",
                "source": "routing_trace_model", "payload_bytes": 4096,
                "payload_bandwidth_GBps": 4096 / (p50 * 1e6),
            }
        if compute:
            record["compute"] = {"source": "effective_expert_matrix_math",
                                 "effective_flops": 1e9, "effective_tflops": 1e9 / (p50 * 1e9)}
        if throughput is not None:
            record["throughput"] = {"global_source_tokens_per_second": throughput,
                                    "unit": "tokens/s", "scope": "EP group"}
        record["completion"] = {"level": "remote_visible", "detail": "test",
                                "proves_transfer": True}
        record.update(extra or {})
        return record

    return build


@pytest.fixture
def run_report(tmp_path):
    """Write a run directory, analyse it and generate every report."""

    def build(tasks, rows, run_id="test-run", environment=None, matrix=None,
              capabilities=None):
        root = Path(tmp_path) / run_id
        root.mkdir(parents=True, exist_ok=True)
        manifest = _manifest(tasks, run_id, environment, matrix)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for task, row in zip(tasks, rows):
            folder = "baseline_results" if task.get("baseline") else "operators"
            directory = root / folder / task["backend_id"] / task["task_id"]
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "metrics.json", row)
        analysis = analyze_run(root, list(rows))
        destination = report(root, manifest, list(rows), analysis, capabilities=capabilities)
        return SimpleNamespace(root=root, manifest=manifest, rows=list(rows),
                               destination=destination, analysis=analysis,
                               text=(destination / "benchmark_report.md").read_text(encoding="utf-8"))

    def _manifest(tasks, run_id, environment, matrix):
        metadata = matrix_metadata(list(tasks)) if matrix is None else dict(matrix)
        return {"schema_version": 1, "run_id": run_id,
                "environment": dict(environment or NPU_ENVIRONMENT),
                "matrix": metadata, "tasks": list(tasks)}

    return build


@pytest.fixture
def l0_matrix():
    """A capability matrix that supports one mechanism and rejects another."""
    return {
        "schema_version": 1, "source": "test",
        "channels": [
            {"channel": "UB Memory", "execution_resource": "AIV", "mechanisms": ["ub_memory"],
             "supported": True, "confidence": "declared", "reason": "test fixture"},
            {"channel": "URMA", "execution_resource": "AICPU_TS", "mechanisms": ["urma"],
             "supported": False, "confidence": "declared", "reason": "not present on this SoC"},
        ],
        "engines": [
            {"engine": "CCU", "execution_resource": "CCU", "mechanisms": ["ub_ctp"],
             "supported": True, "confidence": "declared", "reason": "test fixture"},
        ],
        "accelerators": [
            {"mode": "aiv", "execution_resource": "AIV", "supported": True, "reason": "probed"},
        ],
        "probes": [{"name": "transport_bridge", "ok": True, "detail": {}}],
    }


"""Offline harness for the prompt-driven workflow.

The workflow's job is to run *documented commands* and check their results. This harness
replaces the commands with a deterministic model of a benchmark host, so a complete
workflow -- doctor, dependencies, capability, plan, run, profile, analyze, report -- can be
exercised without a device, without installing anything, and without touching
``05_results``.

The fake still writes real files in the real shapes (``manifest.json``, ``ep_summary.json``,
``metrics.json``, ``analysis/results.json``, ``reports/*``), because the workflow only
learns about the world through those files.
"""
from common.storage import fingerprint
from scripts.workflow import phases as phase_runner
from scripts.workflow import request as request_module
from scripts.workflow import state as state_module

import yaml

#: Dependency checks the fake doctor reports.
GOOD_CHECKS = [
    {"id": "python", "status": "pass", "category": "A", "detail": "python 3.10.8",
     "suggested_action": "none", "blocking_for": ["reporting"], "evidence": {}},
    {"id": "torch", "status": "pass", "category": "A", "detail": "torch 2.6.0",
     "suggested_action": "none", "blocking_for": ["l0_transport"], "evidence": {}},
    {"id": "numpy", "status": "pass", "category": "A", "detail": "numpy 2.2.6",
     "suggested_action": "none", "blocking_for": ["reporting"], "evidence": {}},
]

MISSING_TORCH = [
    {**GOOD_CHECKS[0]},
    {"id": "torch", "status": "fail", "category": "A",
     "detail": "torch is not importable", "suggested_action": "install torch==2.6.0",
     "blocking_for": ["l0_transport", "operator:alltoall_hccl"], "evidence": {}},
    {**GOOD_CHECKS[2]},
]

HOST_BLOCKED = [
    {**GOOD_CHECKS[0]},
    {"id": "npu-driver", "status": "fail", "category": "B",
     "detail": "the NPU driver is not installed", "suggested_action": "ask the host owner",
     "blocking_for": ["l0_transport"], "evidence": {}},
]


class FakeBenchmark:
    """A deterministic stand-in for ``run_benchmark.py`` and the install script."""

    def __init__(self, root, workspace, *, device="npu", devices=16, soc="Ascend910B4",
                 checks=None, run_outcome="passed", profile_ok=True, install_ok=True,
                 report_ok=True, capabilities_ok=True):
        self.root = Path(root)
        self.workspace = Path(workspace)
        self.raw = self.workspace / "raw"
        self.device = device
        self.devices = devices
        self.soc = soc
        self.checks = deepcopy(checks if checks is not None else GOOD_CHECKS)
        self.run_outcome = run_outcome
        self.profile_ok = profile_ok
        self.install_ok = install_ok
        self.report_ok = report_ok
        self.capabilities_ok = capabilities_ok
        self.calls = []
        self.runtime_records = []
        self._counter = 0

    # -- entry points -----------------------------------------------------
    def run(self, argv, log_name=None, **kwargs):
        argv = [str(item) for item in argv]
        self.calls.append(argv)
        self._counter += 1
        log = self.workspace / "logs" / (log_name or f"{self._counter:03d}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        if "run_benchmark.py" in " ".join(argv):
            index = next(i for i, item in enumerate(argv) if item.endswith("run_benchmark.py"))
            command = argv[index + 1]
            result = getattr(self, f"_cmd_{command}", self._unknown)(argv[index + 2:])
        elif "install_deps.sh" in " ".join(argv):
            result = (0 if self.install_ok else 1), "installed", ""
        else:
            result = 0, "", ""
        exit_code, stdout, stderr = result
        log.write_text(f"$ {' '.join(argv)}\n# exit code: {exit_code}\n\n{stdout}\n{stderr}\n",
                       encoding="utf-8")
        return phase_runner.CommandResult(argv=argv, exit_code=exit_code, stdout=stdout,
                                          stderr=stderr, log_path=log)

    def benchmark(self, *args, **kwargs):
        return self.run([self.root / "run_benchmark.py", *[str(item) for item in args]],
                        **kwargs)

    def _unknown(self, argv):
        return 2, "", f"the fake does not implement {argv}"

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _option(argv, name, default=None):
        if name in argv:
            index = argv.index(name)
            if index + 1 < len(argv):
                return argv[index + 1]
        return default

    @staticmethod
    def _flags(argv, name):
        return [item for item in argv if item.startswith(name + "=")] or name in argv

    def _tasks(self, ep_sizes, levels):
        """Two adapters per (EP size, level): a small, complete matrix."""
        tasks = []
        for size in ep_sizes:
            for level in levels:
                for backend in ("alltoall_hccl", "allgather_hccl"):
                    case = {"L0": "LINK-001", "L1": "COMM-001", "L2": "FUSE-001",
                            "L3": "MOE-001", "L4": "LAYER-001"}[level]
                    category = "transport" if level == "L0" else "operator"
                    task_id = f"{case}-{backend}-{size}-{level}"
                    tasks.append({
                        "task_id": task_id, "case_id": case, "level": level,
                        "operation": "put" if level == "L0" else "dispatch",
                        "backend_id": backend,
                        "config": {
                            "suite_id": "test", "device": self.device,
                            "distributed": {"ep_size": size, "tp_size": 1},
                            "benchmark": {"timing": "output_ready", "iterations": 2,
                                          "warmup": 1},
                            "topology": {}, "model": {"num_experts": 8, "top_k": 2},
                            "workload": {"routing": "balanced", "seed": 42},
                            "dedup": {"mode": "none"}, "load_balance": {"enabled": False},
                            "profiling": {"enabled": True},
                        },
                        "parameters": {"variant": "default"},
                        "status": "not_run", "reason": None, "ep_size": size,
                        "suite_ids": ["test"], "case_category": category,
                        "source_versions": {}, "baseline": level == "L0",
                    })
        return tasks

    def _manifest(self, argv):
        plan_path = self._option(argv, "--plan")
        if plan_path:
            data = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
            spec = data.get("plan") or data.get("matrix") or {}
            ep_sizes = [int(size) for size in spec.get("ep_sizes") or []]
            levels = list(spec.get("levels") or ["L1"])
        else:
            config_path = self._option(argv, "--config")
            levels = ["L1", "L2"]
            if config_path:
                try:
                    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
                    levels = list((config.get("benchmark") or {}).get("levels") or levels)
                except (OSError, yaml.YAMLError):
                    pass
            ep_sizes = [int(self._option(argv, "--ep-size", 2))]
        return self._tasks(ep_sizes, levels)

    # -- commands ---------------------------------------------------------
    def _cmd_doctor(self, argv):
        platform = {"root": str(self.root), "host": "fake-host", "os": "Linux-fake",
                    "python": "3.10.8", "cann_home": "/usr/local/Ascend/latest",
                    "cann_version": "8.2.RC1",
                    "soc": self.soc if self.device == "npu" else None,
                    "npu_smi": {"chip_rows": self.devices} if self.device == "npu" else None,
                    "devices": [], "env": {}}
        report = {"schema_version": 1, "profile": "all", "platform": platform,
                  "counts": {}, "checks": self.checks, "matrix": [],
                  "out_of_scope": [], "wheelhouse": {"counts": {}}, "exit_code": 0}
        return 0, json.dumps(report), ""

    def _cmd_capabilities(self, argv):
        if not self.capabilities_ok:
            return 1, "capabilities failed", ""
        if "--plan" in argv:
            payload = {}
            for size in self._plan_ep_sizes(argv):
                payload[str(size)] = {
                    "tasks": 8, "statuses": {"not_run": 8},
                    "operators": {"alltoall_hccl": {"not_run": 4},
                                  "allgather_hccl": {"not_run": 4}},
                    "cases": {"COMM-001": {"alltoall_hccl": "not_run"}},
                }
            if "--runtime" in argv:
                for size in payload:
                    payload[size]["runtime"] = self._runtime(size)
                    payload[size]["matrix_source"] = "probe"
            return 0, json.dumps(payload), ""
        matrix = {"schema_version": 1, "source": "probe", "soC": self.soc,
                  "channels": [{"channel": "UB Memory", "mechanisms": ["ub_memory"]}],
                  "engines": [], "accelerators": [], "probes": []}
        if "--runtime" in argv:
            matrix["runtime"] = self._runtime("2")
        return 0, json.dumps(matrix), ""

    def _plan_ep_sizes(self, argv):
        plan = self._option(argv, "--plan")
        if not plan:
            return []
        data = yaml.safe_load(Path(plan).read_text(encoding="utf-8"))
        spec = data.get("plan") or data.get("matrix") or {}
        return [int(size) for size in spec.get("ep_sizes") or []]

    def _runtime(self, size):
        self.runtime_records.append(size)
        return {"schema_version": 1, "stages": ["soc", "compiled"],
                "records": [{"case_id": "LINK-001", "channel": "UB Memory",
                             "engine": "AIV", "mechanisms": ["ub_memory"],
                             "status": "supported", "supported": True, "released": True,
                             "stages": [{"stage": "soc", "state": "pass", "reason": None,
                                         "detail": {}}]}],
                "summary": {"supported": 1}}

    def _cmd_plan(self, argv):
        run_id = self._option(argv, "--run-id")
        output = Path(self._option(argv, "--output", self.workspace))
        tasks = self._manifest(argv)
        directory = output / run_id
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 1, "run_id": run_id, "tasks": tasks,
                    "environment": {"host": "fake-host"},
                    "matrix": {"ep_sizes": sorted({task["ep_size"] for task in tasks}),
                               "levels": sorted({task["level"] for task in tasks}),
                               "expected_cases": sorted({task["case_id"] for task in tasks}),
                               "topology": {}}}
        write_json(directory / "manifest.json", manifest)
        return 0, f"{directory} ({len(tasks)} tasks)", ""

    def _outcome_for(self, index, task):
        mode = self.run_outcome
        if mode == "passed":
            return "passed"
        if mode == "partial":
            return {0: "passed", 1: "passed", 2: "unsupported",
                    3: "runtime_failed"}.get(index % 8, "passed")
        if mode == "empty":
            return "unsupported"
        if mode == "incomplete":
            return "not_run" if index >= 2 else "passed"
        if mode == "no_correctness":
            return "passed"
        if mode == "correctness_failed":
            return "correctness_failed" if index == 0 else "passed"
        return "passed"

    def _cmd_run(self, argv):
        return self._execute(argv, resume=False)

    def _cmd_resume(self, argv):
        return self._execute(argv, resume=True)

    def _execute(self, argv, resume):
        run_id = self._option(argv, "--run-id")
        output = Path(self._option(argv, "--output", self.workspace))
        directory = output / run_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            if resume:
                return 2, "", "the run directory does not exist; nothing to resume"
            # `run` creates the manifest itself, exactly like create_manifest.
            tasks = self._manifest(argv)
            directory.mkdir(parents=True, exist_ok=True)
            write_json(manifest_path, {
                "schema_version": 1, "run_id": run_id, "tasks": tasks,
                "environment": {"host": "fake-host"},
                "matrix": {"ep_sizes": sorted({task["ep_size"] for task in tasks}),
                           "levels": sorted({task["level"] for task in tasks}),
                           "expected_cases": sorted({task["case_id"] for task in tasks}),
                           "topology": {}}})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = {"passed": 0, "failed": 0, "unsupported": 0, "not_run": 0,
                  "not_applicable": 0, "dependency_missing": 0, "environment_unavailable": 0,
                  "source_unavailable": 0, "not_implemented": 0}
        for index, task in enumerate(manifest["tasks"]):
            status = self._outcome_for(index, task)
            if resume and status == "not_run":
                status = "passed"
            record = self._record(task, status)
            folder = "baseline_results" if task.get("baseline") else "operators"
            write_json(directory / folder / task["backend_id"] / task["task_id"]
                       / "metrics.json", record)
            bucket = ("passed" if status == "passed" else
                      "unsupported" if status in ("unsupported", "not_implemented") else
                      "failed" if status in ("runtime_failed", "correctness_failed", "timeout",
                                             "oom") else status)
            counts[bucket] = counts.get(bucket, 0) + 1
        planned = len(manifest["tasks"])
        summary = {"schema_version": 1, "run_id": run_id, "planned_tasks": planned,
                   "completed_tasks": planned - counts["not_run"],
                   "completion_status": "complete" if not counts["not_run"] else "incomplete",
                   "passed": counts["passed"], "failed": counts["failed"],
                   "unsupported": counts["unsupported"],
                   "not_applicable": counts["not_applicable"],
                   "not_implemented": counts["not_implemented"],
                   "dependency_missing": counts["dependency_missing"],
                   "environment_unavailable": counts["environment_unavailable"],
                   "source_unavailable": counts["source_unavailable"],
                   "not_run": counts["not_run"],
                   "statuses": {}, "unresolved_tasks": [
                       task["task_id"] for index, task in enumerate(manifest["tasks"])
                       if self._outcome_for(index, task) == "not_run"],
                   "consistency": {"planned_tasks": planned,
                                   "sum_of_buckets": planned, "consistent": True}}
        write_json(directory / "ep_summary.json", summary)
        return (0 if not counts["failed"] and not counts["not_run"] else 1), run_id, ""

    def _record(self, task, status):
        record = {
            "schema_version": 1, "run_id": "fake", "task_id": task["task_id"],
            "case_id": task["case_id"], "suite_id": "test",
            "backend_id": task["backend_id"], "benchmark_level": task["level"],
            "operation": task["operation"], "category": task.get("case_category"),
            "ep_size": task["ep_size"], "status": status, "source": "measured",
            "environment": {"npu_models": [self.soc] if self.device == "npu" else None,
                            "cann_version": "8.2.RC1",
                            "packages": {"torch": "2.6.0"}},
            "topology": {}, "source_versions": {}, "config": task["config"],
            "workload": {"tokens_per_rank": [8] * task["ep_size"]},
            "routing_trace_hash": "trace", "correctness": None,
            "raw_samples": [], "latency": None, "bandwidth": None, "throughput": None,
            "compute": None, "overlap": None, "memory": None, "load_balance": None,
            "profiling_artifacts": [], "logs": [], "error_reason": None,
            "case_parameters": task["parameters"],
        }
        if status == "passed":
            record["correctness"] = ({"passed": True}
                                    if self.run_outcome != "no_correctness" else {})
            record["latency"] = {"unit": "ms", "scope": "per_iteration_rank_max",
                                 "p50": 1.0, "p95": 1.1, "p99": 1.2,
                                 "rank_max": [1.0, 1.0]}
            record["bandwidth"] = {"unit": "bytes", "bandwidth_unit": "GB/s",
                                   "scope": "per_rank",
                                   "source": "routing_trace_model",
                                   "payload_bytes": 4096,
                                   "payload_bandwidth_GBps": 0.004096}
            record["throughput"] = {"global_source_tokens_per_second": 1000.0}
            record["completion"] = {"level": "remote_visible"}
        elif status != "not_run":
            record["error_reason"] = f"{status} in the fake run"
        return record

    def _cmd_profile(self, argv):
        run_id = self._option(argv, "--run-id")
        directory = Path(self._option(argv, "--output", self.workspace)) / run_id
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        cases = (self._option(argv, "--cases") or "").split(",")
        operators = (self._option(argv, "--operators") or "").split(",")
        sizes = [item for item in (self._option(argv, "--ep-size") or "").split(",") if item]
        task_ids = [item for item in (self._option(argv, "--task-ids") or "").split(",") if item]
        selected = 0
        for task in manifest["tasks"]:
            if task_ids and task["task_id"] not in task_ids:
                continue
            if cases and cases != [""] and task["case_id"] not in cases:
                continue
            if operators and operators != [""] and task["backend_id"] not in operators:
                continue
            if sizes and str(task["ep_size"]) not in sizes:
                continue
            folder = "baseline_results" if task.get("baseline") else "operators"
            path = directory / folder / task["backend_id"] / task["task_id"] / "metrics.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "passed":
                continue
            selected += 1
            manifest_name = f"profiling/{task['task_id']}/profile_manifest.json"
            write_json(directory / manifest_name,
                       {"artifacts": [{"path": "trace.csv", "bytes": 10,
                                       "sha256": "0" * 64}],
                        "status": "passed" if self.profile_ok else "runtime_failed"})
            record.setdefault("profiling_artifacts", []).append(
                {"manifest": manifest_name,
                 "status": "passed" if self.profile_ok else "runtime_failed"})
            write_json(path, record)
        return 0, f"profiled {selected} task(s)", ""

    def _cmd_analyze(self, argv):
        run_id = self._option(argv, "--run-id")
        directory = Path(self._option(argv, "--output", self.workspace)) / run_id
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        groups = []
        by_pair = {}
        for task in manifest["tasks"]:
            folder = "baseline_results" if task.get("baseline") else "operators"
            path = directory / folder / task["backend_id"] / task["task_id"] / "metrics.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "passed":
                continue
            key = (task["case_id"], task["level"], task["ep_size"])
            by_pair.setdefault(key, []).append(record)
        for (case, level, size), records in by_pair.items():
            members = [{"backend": record["backend_id"], "task_id": record["task_id"],
                        "case": record["case_id"], "p50_ms": 1.0, "p95_ms": 1.1,
                        "iterations": 2, "tokens_per_second": 1000.0,
                        "payload_GBps": 0.004096, "effective_tflops": None,
                        "completion": "remote_visible", "ep_size": size,
                        "has_bandwidth": True, "has_compute": False,
                        "has_throughput": True} for record in records]
            groups.append({"key": fingerprint([case, level, size]), "kind":
                           "actual_comparison" if len(members) > 1 else "single_measurement",
                           "comparison": len(members) > 1, "case_id": case, "level": level,
                           "operation": "dispatch", "variant": "default", "parameters": {},
                           "ep_size": size, "device": self.device,
                           "timing": "output_ready", "members": members,
                           "backends": [item["backend"] for item in members],
                           "samples": {item["backend"]: 1 for item in members},
                           "spread_ratio": 1.0, "reason": None})
        results = {"comparison_details": {"groups": groups, "excluded": []},
                   "fusion": [], "scaling_groups": {}, "pareto": {},
                   "profiles": {}, "planner_comparisons": [],
                   "fixed_owner_comparisons": [], "dedup_comparisons": [],
                   "hypotheses": [], "profile_parse_errors": []}
        write_json(directory / "analysis" / "results.json", results)
        return 0, "analysis written", ""

    def _cmd_report(self, argv):
        if not self.report_ok:
            return 1, "", "the report generator failed"
        run_id = self._option(argv, "--run-id")
        directory = Path(self._option(argv, "--output", self.workspace)) / run_id
        reports = directory / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "ep_summary.json").read_text(encoding="utf-8"))
        files = ("benchmark_report.md", "benchmark_summary.json", "benchmark_summary.csv",
                 "coverage_matrix.md", "coverage_matrix.csv", "coverage_matrix.json",
                 "operator_coverage.md", "ep_size_comparison.md", "variant_coverage.md",
                 "failure_summary.md")
        for name in files:
            (reports / name).write_text(f"# {name}\n", encoding="utf-8")
        (reports / "benchmark_report.md").write_text(
            "# EP benchmark report\n\n## 0. Executive Summary\n\n"
            "## 9. Failures and Limitations\n\n"
            "`analysis/results.json` is [here](../analysis/results.json).\n",
            encoding="utf-8")
        (reports / "coverage_matrix.json").write_text(json.dumps({
            "ep_sizes": sorted({task["ep_size"] for task in manifest["tasks"]}),
            "cases": [], "operators": [], "case_operator": [],
            "case_operator_variant": {"totals": {"planned": summary["planned_tasks"],
                                                 "results": summary["planned_tasks"],
                                                 "matched": summary["planned_tasks"],
                                                 "missing": 0,
                                                 "measured": summary["passed"]}},
            "ep_size_comparison": {"groups": []}, "l0_scaleout": [],
        }), encoding="utf-8")
        (directory / "analysis").mkdir(exist_ok=True)
        (directory / "analysis" / "results.json").write_text(
            (directory / "analysis" / "results.json").read_text(encoding="utf-8")
            if (directory / "analysis" / "results.json").is_file() else "{}", encoding="utf-8")
        return 0, str(reports), ""


@pytest.fixture
def fake_benchmark(tmp_path):
    """``fake_benchmark(**options)`` -> a :class:`FakeBenchmark` rooted at the repo."""
    from common.registry import ROOT
    workspace = tmp_path / "workflow-home"
    workspace.mkdir()

    def build(**options):
        return FakeBenchmark(ROOT, workspace, **options)

    return build


@pytest.fixture
def workflow(tmp_path, fake_benchmark):
    """Run a complete workflow against the fake host and return its context.

    ``workflow(prompt="...", **fake_options)`` creates the workflow, executes every phase
    the intent selects, and returns ``(context, state)`` plus a ``read(name)`` helper for
    the artifacts it wrote.
    """
    from common.registry import ROOT

    def build(prompt, *, request_body=None, workflow_id="wf-test", store_root=None,
              resume=False, run_output=None, **options):
        fake = fake_benchmark(**options)
        if request_body is not None:
            request = request_module.build(request_body, root=ROOT)
        else:
            request = request_module.interpret(prompt, root=ROOT)
        run_root = Path(run_output) if run_output else (tmp_path / "runs")
        run_root.mkdir(exist_ok=True)
        output_root = Path(store_root) if store_root else (tmp_path / "workflows")
        store = state_module.WorkflowStore(output_root, root=ROOT)
        if resume:
            state = store.load(workflow_id)
            request = request_module.build(state.request, root=ROOT)
        else:
            state = store.create(request, workflow_id=workflow_id)
        ctx = phase_runner.WorkflowContext(
            state=state, request=request, store=store, commands=fake,
            root=ROOT, output=store.directory(state.workflow_id),
            run_output=run_root)
        final = phase_runner.execute_workflow(ctx, _handlers())
        return final, ctx, fake

    return build


def _handlers():
    from scripts.workflow import handlers
    return handlers()
