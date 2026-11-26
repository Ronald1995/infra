"""Persist task attempts and launch independent torchrun jobs."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from uuid import uuid4

import yaml

from common.environment import capture
from common.registry import ROOT
from common.storage import fingerprint, load_json, write_json
from schemas import RETRYABLE, result

DEFAULT_OUTPUT = ROOT.parent / "05_results" / "raw"


def checked_id(run_id):
    if not run_id or not all(c.isascii() and (c.isalnum() or c in "-_") for c in run_id):
        raise ValueError("run ID may contain letters, digits, hyphens and underscores")
    return run_id


def create_manifest(output, tasks, run_id=None):
    now = datetime.now(timezone.utc)
    run_id = checked_id(run_id or f"{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}")
    root = Path(output).resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    env = capture(probe_npu=any(t["config"]["device"] == "npu" for t in tasks))
    manifest = {"schema_version": 1, "run_id": run_id, "created_at": now.isoformat(),
                "environment": env, "tasks": tasks}
    write_json(root / "environment.json", env)
    write_json(root / "topology.json", {t["task_id"]: t["config"]["topology"] for t in tasks})
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump({t["task_id"]: t["config"] for t in tasks}, sort_keys=False), encoding="utf-8"
    )
    write_json(root / "manifest.json", manifest)
    return root, manifest


def result_path(root, task):
    folder = "baseline_results" if task["baseline"] else "operators"
    return root / folder / task["backend_id"] / task["task_id"] / "metrics.json"


def results(root, manifest):
    for task in manifest["tasks"]:
        location = result_path(root, task)
        yield load_json(location) if location.is_file() else result(
            manifest, task, task["status"], task["reason"]
        )


def wait_for(path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return load_json(path)
        time.sleep(.2)
    raise TimeoutError(f"timed out waiting for coordinator artifact {path}")


def environment_for(task):
    launch = task["config"].get("launch", {})
    override = launch.get("operators", {}).get(task["backend_id"], {})
    env = {**os.environ, **launch.get("env", {}), **override.get("env", {})}
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if os.name == "nt":
        env.setdefault("USE_LIBUV", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    return override.get("python", launch.get("python", sys.executable)), env


def launch_worker(task, manifest_path, attempt, profile, args):
    settings = task["config"]["distributed"]
    size = settings["ep_size"] * settings["tp_size"]
    local_size = args.nproc_per_node or size // args.nnodes
    if local_size * args.nnodes != size:
        raise ValueError(f"torchrun world size must equal EP x TP ({size})")
    python, env = environment_for(task)
    worker = [str(ROOT / "run_benchmark.py"), "_worker", str(manifest_path), task["task_id"], str(attempt)]
    if profile:
        worker.append("--profile-only")
    if size == 1 and task["config"]["device"] == "cpu":
        command = [python, *worker]
    elif os.name == "nt" and args.nnodes == 1 and task["config"]["device"] == "cpu":
        return launch_windows_gloo([python, *worker], env, attempt, size,
                                   task["config"]["benchmark"]["task_timeout_seconds"])
    else:
        port = args.master_port
        if port is None:
            with socket.socket() as sock:
                sock.bind((args.master_addr, 0))
                port = sock.getsockname()[1]
        command = [
            python, "-m", "torch.distributed.run", "--nnodes", str(args.nnodes),
            "--nproc-per-node", str(local_size), "--node-rank", str(args.node_rank),
            "--master-addr", args.master_addr, "--master-port", str(port), "--max-restarts", "0", *worker,
        ]
    attempt.mkdir(parents=True, exist_ok=True)
    log = attempt / f"node-{args.node_rank}.log"
    timeout = task["config"]["benchmark"]["task_timeout_seconds"]
    with log.open("w", encoding="utf-8") as output:
        try:
            process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                       cwd=ROOT, start_new_session=os.name != "nt")
        except OSError as exc:
            print(exc, file=output)
            return "dependency_missing"
        try:
            return None if process.wait(timeout=timeout) == 0 else "runtime_failed"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=output, stderr=subprocess.STDOUT, check=False)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            return "timeout"


def launch_windows_gloo(command, env, attempt, size, timeout):
    """Windows CPU PyTorch wheels may ship without the TCPStore libuv backend."""
    from contextlib import ExitStack

    attempt.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    workers = []
    with ExitStack() as files:
        try:
            for rank in range(size):
                log = files.enter_context((attempt / f"node-0-rank-{rank}.log").open("w", encoding="utf-8"))
                workers.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                env={**env, "RANK": str(rank), "LOCAL_RANK": str(rank),
                                                     "WORLD_SIZE": str(size),
                                                     "EP_GLOO_FILE_STORE": (attempt / "gloo.store").as_uri()}))
            while True:
                statuses = [worker.poll() for worker in workers]
                if all(value == 0 for value in statuses):
                    return None
                if any(value not in (None, 0) for value in statuses):
                    # Give peers time to persist bounded collective failures.
                    for worker in workers:
                        try:
                            worker.wait(timeout=min(1, max(0, deadline - time.monotonic())))
                        except subprocess.TimeoutExpired:
                            pass
                    return "runtime_failed"
                if time.monotonic() >= deadline:
                    return "timeout"
                time.sleep(.05)
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.wait()


def baseline_key(manifest, task, args):
    environment = manifest["environment"]
    cfg = task["config"]
    npu = cfg["device"] == "npu"
    if npu and (args.nnodes != 1 or not environment["npu_smi_info"]):
        return None
    python, _ = environment_for(task)
    if Path(python).resolve() != Path(sys.executable).resolve():
        return None
    return fingerprint([environment, task["task_id"], args.nnodes, args.nproc_per_node, python])


def persist_case(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path.with_name("correctness.json"), record["correctness"])
    write_json(path.with_name("status.json"), {
        "status": record["status"], "error_reason": record["error_reason"]
    })
    (path.with_name("config.yaml")).write_text(yaml.safe_dump(record["config"]), encoding="utf-8")
    with path.with_name("samples.jsonl").open("w", encoding="utf-8") as samples:
        import json
        for rank in record["raw_samples"]:
            for sample in rank["samples"]:
                samples.write(json.dumps({"rank": rank["rank"], **sample}) + "\n")
    write_json(path, record)


def execute(root, manifest, args, profile=False, resume=False, representative=False):
    from hashlib import sha256

    session = args.session if args.nnodes > 1 else uuid4().hex
    phase = "profile" if profile else "run"
    control = root / "logs" / f"control-{checked_id(session)}-{phase}"
    failed = False
    environments = {}
    profiled = set()
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        path = result_path(root, task)
        if args.node_rank != 0:
            action = wait_for(control / f"{task_id}.json", task["config"]["benchmark"]["task_timeout_seconds"])
            for attempt_id in action["attempts"]:
                attempt = root / ("profiling" if profile else "logs") / task_id / attempt_id
                ticket = wait_for(control / f"{attempt_id}.json", task["config"]["benchmark"]["task_timeout_seconds"])
                if not ticket["run"]:
                    break
                launch_worker(task, root / "manifest.json", attempt, profile, args)
            continue

        prior = load_json(path) if path.exists() else None
        executable = task["status"] == "not_run" and task["reason"] is None
        eligible = executable and ((prior and prior["status"] == "passed") if profile else
                                  (not prior or resume and prior["status"] in RETRYABLE))
        if profile:
            cases = getattr(args, "cases", None)
            eligible = eligible and (not cases or task["case_id"] in cases)
        if representative:
            config = task["config"]
            group = (task["backend_id"], task["level"], fingerprint(config))
            eligible = (eligible and config["device"] == "npu" and
                        config["profiling"].get("enabled", False) and group not in profiled)
            if eligible:
                profiled.add(group)
        preflight = None
        if eligible and task["config"]["device"] == "npu" and args.nnodes == 1:
            python, env = environment_for(task)
            identity = fingerprint([python, env])
            if identity not in environments:
                try:
                    probe = subprocess.run(
                        [python, "-c", "import torch_npu; import torch; print(torch.npu.device_count())"],
                        env=env, capture_output=True, text=True, errors="replace", check=True, timeout=30,
                    )
                    environments[identity] = (int(probe.stdout.strip().splitlines()[-1]), None)
                except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
                    detail = exc.stderr.strip().splitlines()[-1] if getattr(exc, "stderr", None) else str(exc)
                    environments[identity] = (0, detail)
            devices, error = environments[identity]
            requested = task["config"]["distributed"]["ep_size"] * task["config"]["distributed"]["tp_size"]
            if error:
                preflight = result(manifest, task, "dependency_missing", error)
            elif devices < requested:
                preflight = result(manifest, task, "environment_unavailable",
                                   f"requested {requested} NPU devices, found {devices}")
        if preflight and not profile:
            persist_case(path, preflight)
            failed = True
            continue
        if preflight and profile:
            eligible = False
        key = baseline_key(manifest, task, args) if task["baseline"] and not profile else None
        cache = root.parent / ".baseline_cache" / f"{key}.json" if key else None
        cache_hit = eligible and cache and cache.is_file()
        attempts = [uuid4().hex for _ in range(task["config"]["benchmark"]["retries"] + 1)] if (
            eligible and not cache_hit
        ) else []
        if args.nnodes > 1:
            write_json(control / f"{task_id}.json", {"attempts": attempts})
        if not eligible:
            if not prior and not profile:
                persist_case(path, result(manifest, task, task["status"], task["reason"]))
            continue
        if cache_hit:
            cached = load_json(cache)
            cached["reused_from"] = {"run_id": cached["run_id"], "task_id": cached["task_id"]}
            cached["run_id"] = manifest["run_id"]
            persist_case(path, cached)
            continue

        for attempt_index, attempt_id in enumerate(attempts):
            if args.nnodes > 1:
                write_json(control / f"{attempt_id}.json", {"run": True})
            attempt = root / ("profiling" if profile else "logs") / task_id / attempt_id
            launch_status = launch_worker(task, root / "manifest.json", attempt, profile, args)
            measured = attempt / "result.json"
            if launch_status is None and measured.is_file():
                record = load_json(measured)
            else:
                errors = [load_json(item) for item in sorted(attempt.glob("rank-*-error.json"))]
                if errors:
                    first = next((e for e in errors if e["status"] not in {"runtime_failed", "timeout"}), errors[0])
                    status = first["status"]
                    why = first["reason"]
                else:
                    status = launch_status or "runtime_failed"
                    why = "rank workers exited without an aggregate result"
                record = result(manifest, task, status, why)
            record["logs"] = [item.relative_to(root).as_posix() for item in sorted(attempt.glob("*.log"))]
            record["attempt"] = attempt.relative_to(root).as_posix()
            record["logs"].extend(item.relative_to(root).as_posix() for item in sorted(attempt.glob("rank-*-error.json")))
            if profile:
                artifacts = []
                for item in attempt.rglob("*"):
                    if item.is_file() and item.name not in {"result.json"}:
                        with item.open("rb") as source:
                            digest = sha256()
                            for block in iter(lambda: source.read(1 << 20), b""):
                                digest.update(block)
                        artifacts.append({"path": item.relative_to(root).as_posix(),
                                          "bytes": item.stat().st_size, "sha256": digest.hexdigest()})
                write_json(attempt / "profile_manifest.json", {"artifacts": artifacts, "status": record["status"]})
                prior["profiling_artifacts"].append({
                    "manifest": (attempt / "profile_manifest.json").relative_to(root).as_posix(),
                    "status": record["status"],
                })
                persist_case(path, prior)
            else:
                write_json(attempt / "status.json", record)
                persist_case(path, record)
                if cache and record["status"] == "passed":
                    write_json(cache, record)
            if record["status"] not in RETRYABLE:
                if args.nnodes > 1 and attempt_index + 1 < len(attempts):
                    write_json(control / f"{attempts[attempt_index + 1]}.json", {"run": False})
                break
        print(f"{task_id} {task['backend_id']} {record['status']}", flush=True)
        failed |= record["status"] not in {"passed", "unsupported"}
    return 1 if failed else 0
