---
name: ep-benchmark
description: Run, validate, profile and analyze distributed Ascend expert-parallel benchmarks.
---

# EP Benchmark: operating procedure

Ten steps, in order. Each step has a command and an acceptance gate; do not start the next
step until the gate passes, and never report a result that a gate did not produce.

Two rules apply to every step:

1. **Nothing is silently skipped.** A dependency that is missing leaves its task in the
   manifest with a recorded status (`dependency_missing`, `environment_unavailable`,
   `not_implemented`, `source_unavailable`). A task that disappears is a bug, not a saving.
2. **Nothing is "verified" without evidence.** CPU runs validate framework behaviour only.
   Every NPU-dependent claim needs a record from an Ascend run; until then the item belongs
   to `docs/blocked.md`.

Scope references: `operators/registry.yaml` (what each adapter claims),
`docs/benchmark_matrix.md` (matrix design), `docs/blocked.md` (what is not done).

## 1. Environment doctor

```sh
python run_benchmark.py doctor --profile all
python run_benchmark.py doctor --profile npu --json > doctor.json
python scripts/doctor.py --profile cpu          # focused view, same records
```

Gate: exit code 0, no `fail` row, and every `warn` carries an action you have decided to
accept. Read the per-family matrix, not just the counts: `l0_transport`, `l0_collectives`,
`profiling` and `multinode` must be `available` before the corresponding steps mean anything.

## 2. Dependency installation

```sh
bash scripts/install_deps.sh --profile npu --dry-run     # print, do not run
bash scripts/install_deps.sh --profile communication
bash scripts/install_deps.sh --profile all --wheelhouse /data/wheels
bash scripts/install_deps.sh --check-only --profile all  # verify, install nothing
```

Profiles: `cpu`, `npu`, `communication`, `profiling`, `multinode`, `all`
(`dependencies/manifest.yaml` categories A-E). The installer never installs the driver, the
firmware, the device nodes or CANN itself; it prints the exact manual step for each
(`manual_step`, `host_provided`).

Gate: the script ends by running `scripts/doctor.py`; exit 0 means the profile is proven.
A `BLOCKED:` line is a non-zero exit, never a warning to ignore.

## 3. Runtime capability probe

```sh
python run_benchmark.py capabilities --plan configs/benchmark_plan.yaml
python run_benchmark.py capabilities --plan configs/benchmark_plan.yaml --json
python run_benchmark.py capabilities --config configs/transport.yaml --matrix recorded.json
```

Gate: for each EP size the per-operator support counts match the expected matrix, and every
`unsupported` carries a reason string from the probe. Use `--matrix` only to replay a
recorded matrix; it is marked `source: override` in every report.

The staged probe in `baselines/l0/topology/runtime_probe.py` (`soc` -> `compiled` ->
`endpoint_compatible` -> `channel_acquired` -> `kernel_available` -> `resource_available`)
answers "is this channel usable *now*", which a compiled-in channel does not. Attach it to
the static matrix with:

```sh
python run_benchmark.py capabilities --config configs/transport.yaml --runtime --json
```

It is opt-in because it opens a bridge session and can be slow on a busy machine. Its
per-channel `status` distinguishes `supported`, `unsupported`, `environment_unavailable`
(a busy resource is *not* unsupported), `dependency_missing` and `not_implemented`. Without
the probe, `capabilities` prints the static matrix built by
`baselines/l0/topology/capabilities.py::discover`.

## 4. Build the required kernels

```sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "${ASCEND_HOME_PATH}/set_env.sh"
./kernels/asc_comm/build.sh --soc Ascend950 --strict
python run_benchmark.py capabilities --json      # re-check kernel_available
```

Gate: `build/manifest.json` exists and names each engine with the ABI symbols
(`asc_comm_aiv_write_nbi`, `asc_comm_aiv_drain`, `asc_comm_ccu_submit`,
`asc_comm_ccu_wait`), and `capabilities` no longer reports `dependency_missing` for those
channels. Until then `LINK-001`..`004` and `LINK-008` stay blocked — they must not be served
by the host NBI path (`docs/l0_data_plane.md`).

## 5. Validate the benchmark plan

```sh
python run_benchmark.py plan --plan configs/benchmark_plan.yaml
python run_benchmark.py plan --plan configs/benchmark_plan.yaml --ep-size 8 --suite overlap
python scripts/manifest_diff.py OLD/manifest.json NEW/manifest.json --json diff.json  # optional gate
```

Gate: `check_completeness` and `check_parity` pass (they raise otherwise); EP8, EP64 and
EP128 each expand 6753 tasks (`20259` total for the default plan); every task has either
`not_run` with no reason (runnable, not yet measured) or a non-executable status **with** a
reason; `alias_of` is recorded for the four legacy IDs. If `manifest_diff.py` is used, its
exit code must be 0 or the executable change must be justified in writing.

## 6. Run the benchmark matrix

```sh
python run_benchmark.py run --plan configs/benchmark_plan.yaml --run-id ep-matrix
python run_benchmark.py run --plan configs/benchmark_plan.yaml --ep-size 64
python run_benchmark.py run --config configs/smoke_npu.yaml --ep-size 8 --operators alltoall_hccl
python run_benchmark.py resume --run-id ep-matrix
```

Gate: `ep_summary.json` has `completion_status: complete` and
`consistency.consistent: true` (bucket sum == `planned_tasks`). With
`execution.require_complete_execution: true`, any remaining `not_run` is a failure of the
run, not a pass. `blocked` counts must equal the number of dependency/environment rows you
accepted in step 1-2 — an unexpected `dependency_missing` means step 2 did not do what you
think.

## 7. Correctness validation

Validation runs before every timed loop (`prepared.validate()` in
`common/runner.py::run_task`), so a run that reaches `passed` has already produced its
checks. Read them; do not trust the status alone.

```sh
python -c "import json;d=json.load(open('RUN/baseline_results/hardware/<task>/correctness.json'));print(d)"
python -c "import json;d=json.load(open('RUN/operators/alltoall_hccl/<task>/metrics.json'));print(d['correctness'])"
```

Gate: every `passed` task has a non-empty `correctness` block with the tokens its
`CaseSpec.correctness` promised (`dispatch_rows`, `combine_output`, `moe_forward_output`,
`input_gradient`, ...). `correctness_failed` is **not** retryable: fix the cause or record
the gap. A dispatch measured over a layout that was never validated is not a dispatch
measurement.

## 8. Profiling evidence

```sh
python run_benchmark.py profile --run-id ep-matrix --cases COMM-001,MOE-003
python -m pytest -q tests/  # offline checks for the analysis path
```

Gate: each profile attempt writes rank traces plus `profile_manifest.json` with a sha256 per
artefact, and the attempt is attached to the task record (`profiling_artifacts`). Kernel
names and overlap intervals may only enter a report when the trace actually contains them;
`msprof_kernel` evidence has no producer yet, so a kernel claim stays blocked
(`docs/blocked.md` section C).

## 9. Analyze results

```sh
python run_benchmark.py analyze --run-id ep-matrix
python run_benchmark.py analyze --run-id ep-matrix --compare-run ep-8 --compare-run ep-64
```

Gate: only comparable points are compared (same case, operator, parameters, model, per-rank
workload, timing boundary — `analysis/comparison.py::comparable`). For L0, only records with
`verification.status == "verified"` may enter a cross-mechanism comparison; that is a rule
you must apply yourself, because the analysis layer does not enforce it. On CPU nothing is
verified, so the analysis is a comparison of requested configurations and must be labelled as
such (`docs/observability.md`).

## 10. Generate reports

```sh
python run_benchmark.py report --run-id ep-matrix
python run_benchmark.py pipeline --plan configs/benchmark_plan.yaml   # run + analyze + report
```

Gate: `reports/coverage_matrix.*` accounts for every planned point
(`accounted == planned` per `(EP, case, operator, variant)`), `benchmark_report.md` lists the
outcome counts, `ep_summary.json` is consistent, and `l0_report.md` keeps
`requested`/`observed`/`verification` separate. The delivery must state, in this order:
what passed, what is blocked and why, which groups are comparable, and where the raw records
live. `not_run` is not a pass and not a failure — it is an unmeasured point.
