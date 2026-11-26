# Execution flow of one task

This document follows a single manifest task from launch to report. The code paths are
named so each step can be checked in place.

## 1. Manifest

`common/controller.py::create_manifest` writes `manifest.json` (run id, environment,
matrix metadata, tasks), `matrix.json`, `resolved_config.yaml` (de-duplicated by config
fingerprint), `topology.json`, `manifests/ep{size}.json` and — when L0 cases are planned —
`capabilities.json` from `baselines/l0/topology/discovery.py`.

Each task already carries its resolved status (`not_run`, `unsupported`, `not_applicable`,
`not_implemented`) and the reason, so the controller never re-decides applicability.

## 2. Launch

`common/controller.py::execute` iterates `manifest["tasks"]` and computes `eligible`:

- `task["status"] == "not_run"` and `task["reason"] is None`;
- no prior `metrics.json`, or `--resume` with a status in `schemas.RETRYABLE`;
- for NPU single-node runs, a preflight check
  (`python -c "import torch_npu; import torch; print(torch.npu.device_count())"`) proves
  at least `ep_size * tp_size` devices. A failure is persisted as `dependency_missing` or
  `environment_unavailable` without launching anything.

`launch_worker` starts one process group per task: `torch.distributed.run --nnodes N
--nproc-per-node M --node-rank R --master-addr ... --master-port ...` for every multi-rank
or multi-node case, the direct interpreter when `world_size == 1` and the device is CPU, or
`launch_windows_gloo` on Windows CPU wheels that lack the TCPStore libuv backend (one
process per rank, `EP_GLOO_FILE_STORE` as rendezvous). Attempts live in
`logs/<task_id>/<attempt_id>/` (or `profiling/...` for profile runs) and are bounded by
`benchmark.retries + 1`.

## 3. Worker

`run_benchmark.py _worker` calls `common/runner.py::execute_worker`:

1. load the manifest, select the task by id, build an empty record with `schemas.result`;
2. `common/distributed.py::open_context` initialises `torch`, the process group, the EP
   group, the TP group and the metadata group;
3. `run_task` fills the record; rank 0 writes `<attempt>/result.json`;
4. on failure, every rank writes `rank-<n>-error.json` with `status`, `reason` and the
   traceback, and the process exits non-zero. The controller maps those files to a status
   through `common/runner.py::status_for_error`.

## 4. Workload and routing trace

Still before any backend exists, `run_task`:

- gathers `ctx.mapping()` into `record["topology"]` (logical rank, local rank, node, host,
  physical device, server, supernode);
- builds one workload per step (`common/workload.py::make_workload`) and one routing trace
  per step (`common/routing.py::summarize`), writing `rank-<n>-trace-<step>.json`;
- reads the number of steps from the operation: 1, the `token_steps` list for
  `dynamic_tokens`, the trace file for `trace_replay`, `benchmark.planner_steps` for
  `planner_training_steps`.

## 5. Case preparation

Inside the timed-out (but not timed) region the worker:

1. resolves the adapter (`common/registry.py` -> `operators/<id>/backend.py::create`) and
   calls `initialize`, then `prepare`, `prepare_communication` (L1 and the two serial fused
   cases) or `prepare_training` (planner training).
2. builds a `CaseRequest` (context, case, backend, works, traces, parameters, output) and
   calls `benchmarks/cases/__init__.py::prepare`, which resolves `CaseSpec.handler` in the
   lazy registry and returns a `common/prepared.py::PreparedCase`.
3. records `initialization_ns` = time from backend creation to the end of preparation.

The handler owns the timed block. It does not own the timer, the barrier or the
synchronisation — those stay in `common/timer.py` so that two adapters are timed the same
way.

## 6. Validation

`prepared.validate()` runs *before* the timed loop and executes the case at least once. A
failed check raises `common.validation.CorrectnessError`, which the controller records as
`correctness_failed` and a retryable status.

Examples: `validate_dispatch` compares expert counts, expert ids and inverse route slots
against the reference collective; `validate_moe` compares output and gradients against a
CPU FP32 oracle when the shapes are small enough (`< 2_000_000` elements) and against the
collective reference otherwise; `baselines/l0/collectives/torch_hccl.py` compares the
collective output element-wise.

## 7. The timed loop

`common/timer.py::measure` is the only timer. For every iteration
`i in range(warmup + iterations)`:

```
inputs = setup(i)                 # untimed: rebuild state, e.g. the backward graph
ctx.barrier(); ctx.synchronize()  # untimed
started  = perf_counter_ns()      # <-- TIMER START
results  = [run(input) ...]       # submit only; must not block on completion
issued   = perf_counter_ns()
for result in results: wait(result)   # real completion
waited   = perf_counter_ns()
ctx.synchronize()
completed = perf_counter_ns()     # <-- TIMER STOP
```

What is inside the boundary: `run` (submission) + `wait` (completion) + the trailing
synchronise. What is outside: setup, barrier, input construction, validation, profiler
collection.

Recorded per sample: `latency_ns` (`completed - started` for the default
`timing: output_ready`; the NPU event delta for `timing: device`; the issue time only for
`timing: host_api`), `host_issue_ns`, `host_wait_ns`, `output_ready_ns`, `operations`, and
optional `stage_ns` from `prepared.sample_metrics`.

Warmup samples are discarded. `timing: cold_start` is not measured here: the worker records
`initialization_ns` as the single sample. A backend with internal timing (`Prepared.direct`,
the native `hccl_test` path) is run once and its own samples are used.

## 8. From samples to metrics

`common/runner.py` writes `rank-<n>-samples.json`, gathers every rank's samples and calls
`metrics/summary.py`:

- `latency(ranks)` -> per-iteration rank max as the headline, plus `rank_min`,
  `rank_mean`, `p50`, `p95`, `p99`, `rank_avg_p50_ms`, `rank_min_p50_ms`, `imbalance_ratio`,
  `stddev`, `cv`, `p95_p50_ratio`, `host_issue_p50_ms`, `host_wait_p50_ms`,
  `output_ready_p50_ms`;
- `traffic(trace, ...)` + `bandwidth(volume, p50)` for L1 cases whose adapter has a
  `traffic_model`; L0 cases use the declared `Prepared.payload` instead;
- `compute(...)` for expert-matrix FLOPs and `throughput(...)` for tokens/s.

`observe(...)` builds the `ExecutionObservation` (see `docs/observability.md`) and attaches
`result["completion"] = {"level", "detail", "proves_transfer"}`.

## 9. Persistence

Rank artefacts (inside the attempt directory):

| File | Content |
|---|---|
| `rank-<n>-samples.json` | samples, correctness result, memory snapshot, `initialization_ns` |
| `rank-<n>-trace-<step>.json` | routing trace: top-k ids, weights, placement, seed, tokens |
| `routing_trace.json` | merged traces (rank 0) |
| `result.json` | the full record (rank 0) |
| `rank-<n>-error.json` | failure record, per rank |

`common/controller.py::persist_case` then writes, next to `metrics.json`:
`correctness.json`, `status.json`, `config.yaml`, `samples.jsonl`. Paths are
`baseline_results/<backend>/<task_id>/` for baselines and `operators/<backend>/<task_id>/`
otherwise (`result_path`).

## 10. Aggregation and report

`_write_ep_summary` writes `ep_summary.json`: mutually exclusive buckets
(`passed`, `failed`, `unsupported`, `not_applicable`, `not_implemented`,
`dependency_missing`, `environment_unavailable`, `source_unavailable`, `not_run`) whose
counts sum to `planned_tasks`. `not_run` after a run is an execution gap, not a pass;
`completion_status` is `complete` only when every task reached a terminal bucket, and
`execution.require_complete_execution` turns an incomplete run into a non-zero exit.

`analysis/generate_report.py::analyze_run` groups comparable points;
`reporting/report.py::report` writes `reports/benchmark_report.md`,
`benchmark_summary.{json,csv}`, the comparison tables, `reporting/coverage.py` coverage
matrices and `reporting/l0_report.py` for L0.

## 11. MOE-002 and MOE-003 (why they are not additive)

Both handlers call the same helper (`benchmarks/cases/common.py::moe_case`) with
`forward_only=False`; the difference is the `operation` the catalog gives the case:

| Case | `operation` | What `run` does | What `setup` does |
|---|---|---|---|
| MOE-001 | `forward` | one MoE forward, no gradients | nothing |
| MOE-002 | `backward` | `gradients(value, prepared)` only | `forward(value)` — builds the graph |
| MOE-003 | `forward_backward` | `forward(value)` **then** `gradients(...)` in the same call | nothing |

`moe_case` switches on `request.operation`, not on the case ID:

- `setup` returns `(index, forward(work))` only when `op == "backward"`; a backward graph can
  be differentiated once, so it is rebuilt every iteration *outside* the timer.
- `call` returns `gradients(value, prepared)` immediately for `op == "backward"` (MOE-002
  times the backward pass alone); for every other operation it runs `output =
  forward(value)` and then, for a training case, `gradients(value, output)` in the same call.

Consequences the report must respect: MOE-002's latency is **not** comparable to MOE-001's
forward (it contains no forward), and MOE-003 is a round trip measured once — **not** the sum
of MOE-001 and MOE-002. `metrics/summary.py` accordingly gives `forward_backward` the
`forward + backward` FLOP count, while `backward` gets `backward` only.

## 12. Failure handling

| Failure | Status recorded |
|---|---|
| `common.backend.BenchmarkUnavailable(status, reason)` | that status (`unsupported`, `dependency_missing`, `environment_unavailable`, `not_implemented`, `source_unavailable`) |
| `CorrectnessError` | `correctness_failed` |
| `ImportError` / `ModuleNotFoundError` | `dependency_missing` |
| out-of-memory text | `oom` |
| timeout text or `task_timeout_seconds` elapsed | `timeout` |
| anything else | `runtime_failed` |

All except `correctness_failed` are retryable (`schemas.RETRYABLE`), so a transient device
problem is retried within the same attempt budget while a wrong answer is not.
