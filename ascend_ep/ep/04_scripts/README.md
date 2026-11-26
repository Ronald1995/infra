# Ascend EP benchmark

Distributed expert-parallel benchmarks for Ascend plus the CPU validation path. Scope:
operator survey (`../../ep.md`) and methodology (`../03_benchmark/ep_benchmark_methodology.md`);
per-operator status: [capability matrix](operators/README.md); procedure: [SKILL.md](SKILL.md).
One plan file expands `EP Size x Suite x Case x Operator x Parameter Variant`.

## Layout

```
common/          catalog, suites, plan, scheduler, config, runner, controller,
                 prepared case, observation, timer, validation
benchmarks/      case handlers: cases/{common,l1_communication,l2_fusion,l3_moe,
                 l4_router_planner}.py + l0_hardware.py (L0 routing)
baselines/       L0 hardware baselines and the reference collective/experts
operators/       one directory per adapter; operators/vendor/ascendep is the
                 vendored upstream adapter tree
metrics/         latency, traffic model, bandwidth, throughput, effective FLOPs
profiling/       trace collection and parsing
analysis/        comparability, comparison groups, L0 analysis, report entry point
reporting/       main report sections, formatting, specialty reports, coverage, L0 report
schemas/         versioned result/config/capability vocabularies
dependencies/    dependency catalog (manifest.yaml) and requirement files
scripts/         doctor.py, install_deps.sh, manifest_diff.py, vendor_ascendep.py,
                 benchmark_workflow.py + workflow/ (the autonomous phase machine)
references/      policies the skill reads: request interpretation, gates, dependencies,
                 execution, profiling, recovery, report acceptance
configs/         base + EP size + suite + plan configurations
kernels/asc_comm/ ASC-COMM AIV/CCU kernel project (device sources are blocked)
launchers/ tests/ multi-node helpers; offline tests (`npu` marked)
docs/            architecture, execution flow, data plane, evidence, observability,
                 manifest diff, blocked capabilities, matrix design, reporting, workflow
```

## Autonomous workflow

`SKILL.md` is the natural-language entry point; one instruction runs the whole pipeline:

```sh
python scripts/benchmark_workflow.py interpret "在 Ascend 950 上运行完整 EP Benchmark，EP8、EP64、EP128" \
  --out request.json
python scripts/benchmark_workflow.py execute --request request.json   # every phase
python scripts/benchmark_workflow.py plan    --request request.json   # plan + estimate only
python scripts/benchmark_workflow.py resume  --workflow-id <id>
python scripts/benchmark_workflow.py status  --workflow-id <id>
```

Phases: request -> environment -> dependencies (check, permitted install) -> capability
(static + runtime) -> kernel build -> plan -> estimate -> execution -> validation ->
profiling -> analysis -> report -> final validation. Every phase has a gate, writes a JSON
artifact and records each command with its exit code and log. State lives in
`05_results/workflows/<workflow_id>/`, so an interrupted run resumes at the unfinished phase
and keeps its run id. A blocked item blocks only its own scope: a missing operator dependency,
a missing AIV/CCU kernel or an EP size larger than the host never stops the rest of the
matrix, and nothing falls back to a cheaper measurement path.

## Quick start

```sh
python -m pip install -e '.[test,charts]'
python run_benchmark.py doctor            # per-dependency report + availability matrix
python run_benchmark.py list --capabilities
```

CPU smoke (framework behaviour only, not performance):

```sh
python run_benchmark.py pipeline --config configs/smoke_cpu.yaml \
  --cases COMM-001,COMM-004,COMM-005,MOE-003,LAYER-002 --run-id local-cpu
python run_benchmark.py plan --plan configs/benchmark_plan_smoke_cpu.yaml
```

NPU matrix (EP8, EP64, EP128 x L0-L4 x operators):

```sh
python run_benchmark.py doctor --profile all
bash scripts/install_deps.sh --profile all

python run_benchmark.py plan --plan configs/benchmark_plan.yaml          # 20259 tasks
python run_benchmark.py run  --plan configs/benchmark_plan.yaml          # sequential per EP size
python run_benchmark.py run  --plan configs/benchmark_plan.yaml --ep-size 64
python run_benchmark.py run  --plan configs/benchmark_plan.yaml --suite overlap

python run_benchmark.py profile --run-id RUN_ID --cases COMM-001,MOE-003
python run_benchmark.py analyze --run-id RUN_ID
python run_benchmark.py report  --run-id RUN_ID
```

The default plan is `mode: full_matrix`: `all` expands the complete catalog and each of the
other 15 suites expands with its own scenario configuration. Identical points collapse into
one task with the union of `suite_ids`; points with different parameters stay separate.
`plan.mode: catalog` is the old semantics. The full manifest is large (one full effective
configuration per task), so narrow a run with `--ep-size`, `--suite`, `--levels`, `--cases`
or `--operators`.

Single operator and public baselines:

```sh
python -m operators.deepep_ascend.benchmark --config operators/deepep_ascend/config.yaml --levels L3
python run_benchmark.py baseline --config configs/hardware.yaml --levels L0
python run_benchmark.py baseline --config configs/ep8_bf16.yaml --levels L1,L2
```

## Vendored ascendep adapters

The adapters that `operators/ascendep.py` imports are vendored at a pinned commit, so a run
needs **no external checkout**:

| | |
|---|---|
| Location | `operators/vendor/ascendep/ep/` (23 files) |
| Upstream | <https://github.com/penn513/ascendep> at `24ff84d3ab8c01e1e2071802465173e624e1f89d` |
| Provenance | `VENDOR.json` (per-file sha256), checked by `python scripts/vendor_ascendep.py --check` |
| Licence | **none declared upstream** — see `operators/vendor/THIRD_PARTY_NOTICES.md`; owner sign-off required before redistribution |

`EP_ASCENDEP_ROOT=/path/to/ascendep` remains a development override: validated the same way
and reported as `origin: override` (`operators/ascendep.py::source_info`). Known gap: there
is no `ep/mc2_combine_add_rmsnorm_cann/demo.py` upstream, so that operator reports
`source_unavailable` even though its host-side adapter exists.

## Dependencies and the doctor

`dependencies/manifest.yaml` is the single catalog: 33 entries, categories A-E
(`python_base`, `cann`, `communication`, `multinode`, `profiling`), kinds `pip`,
`native_build`, `host_provided`, `binary`, `python_import`, and an explicit `installable`
flag. The installer never touches the driver, firmware, device nodes or CANN itself.

```sh
python run_benchmark.py doctor --profile cpu|npu|communication|profiling|multinode|all
bash scripts/install_deps.sh --profile npu
bash scripts/install_deps.sh --check-only --profile all
bash scripts/install_deps.sh --profile all --wheelhouse /data/wheels
```

A missing dependency is never a silent skip: the task stays in the manifest with
`dependency_missing`, `environment_unavailable`, `not_implemented` or `source_unavailable`
and a reason.

## Matrix parity and `manifest_diff.py`

`common/scheduler.py::check_parity` enforces EP-size invariance of the
`(case, level, operator, variant)` multiset; `check_completeness` enforces case presence per
EP size. Current matrix: 6753 tasks at each of EP8/EP64/EP128 (20259 total), 719 executable points.

```sh
python scripts/manifest_diff.py OLD/manifest.json NEW/manifest.json --json diff.json
```

It classifies every difference (`identical`, `status`, `parameter`, `scope`, `count`),
reports the executable point set separately, and exits non-zero when that set changed. The
before/after numbers of this refactor are in [docs/manifest_diff.md](docs/manifest_diff.md).

## Results and statuses

Execution writes `manifest.json`, `matrix.json`, `resolved_config.yaml`, `topology.json`,
`capabilities.json`, per-attempt logs, per-task `metrics.json` + `correctness.json` +
`status.json` + `config.yaml` + `samples.jsonl`, and `ep_summary.json` with mutually exclusive
buckets. `not_run` is never a pass.

`report` writes `reports/benchmark_report.md` — the main analysis report, readable on its own
(executive summary, environment and capability matrix, coverage, L0 baseline, L1-L4 operator
performance, backend comparison, EP scaling, overlap, load balance, failures, evidence,
recommendations) — plus the specialty reports that back it up: `l0_report.md`,
`operator_comparison.md`, `bandwidth_analysis.md`, `overlap_analysis.md`,
`scalability_analysis.md`, `load_balance_analysis.md`, `planner_training.md`,
`dedup_comparison.md`, `failure_summary.md`, the coverage products
(`coverage_matrix.{md,csv,json}`, `case_operator_coverage.csv`, `variant_coverage.{md,csv}`,
`operator_coverage.md`, `ep_size_comparison.md`, `l0_scaleout.md`), `l0_metrics.json`,
`l0_curves.csv`, `benchmark_summary.{json,csv}` and the charts that had data.

Two rules make the report trustworthy: a comparison group needs two different backends under
identical conditions (one backend is a *single measurement*), and an analysis without data
still explains itself (`Status` / `Current Findings` / `Why Analysis Is Unavailable` /
`Required Conditions` / `Alternative Evidence` / `Conclusion`) instead of writing a bare
heading. Report layout, the data contract, the comparability rules and the formatting rules
are documented in [docs/reporting.md](docs/reporting.md).

Vocabulary: `passed`, `failed`, `unsupported`, `not_applicable`, `not_implemented`,
`dependency_missing`, `environment_unavailable`, `source_unavailable`, `not_run`,
`correctness_failed`, `runtime_failed`, `timeout`, `oom`. Only `passed` is a measurement. The
raw status is always kept; the report buckets (`schemas.STATUS_GROUPS`) only classify it.

## L0 hardware baselines

Six families, each with its own suite: memory (`MEM-*`), transport (`LINK-*`), execution
(`EXEC-*`), framework collectives (`HW-001`..`006`), compute (`HW-010`/`HW-011`) and overlap
(`OVERLAP-*`). Each result keeps `requested` and `observed` apart and reports `verified` /
`fallback` / `unverified` / `unsupported`; only `verified` records may be compared across
mechanisms. Detail: [baselines/l0/README.md](baselines/l0/README.md),
[docs/l0_data_plane.md](docs/l0_data_plane.md), [docs/evidence_capability.md](docs/evidence_capability.md).

```sh
python run_benchmark.py capabilities --config configs/transport.yaml
python run_benchmark.py run --config configs/transport.yaml --ep-size 8 --cases LINK-001,LINK-012
python run_benchmark.py run --config configs/collectives.yaml --ep-size 8 --cases EXEC-001
```

## Multi-node

One EP size per launch; every node runs the same configuration with a shared output root, run
id, session and rendezvous port (`launchers/run_ep64.sh --help`). We use TP1 for the expert
and router workloads. Physical domains must be declared in `topology.server_ids` /
`topology.supernode_ids`; a derived layout is recorded as `source: plan, verified: false`
and never reported as verified.

## Documentation

| Document | Content |
|---|---|
| [docs/architecture.md](docs/architecture.md) | module boundaries, call chain, before/after map, removed duplication |
| [docs/execution_flow.md](docs/execution_flow.md) | one task from manifest to report, the exact timing boundary, MOE-002/MOE-003 |
| [docs/l0_data_plane.md](docs/l0_data_plane.md) | control plane vs data plane, the three planes, the completion ladder, the delayed-write test |
| [docs/evidence_capability.md](docs/evidence_capability.md) | evidence kinds, verification ladder, runtime probe, capability matrix |
| [docs/observability.md](docs/observability.md) | observation fields, byte kinds, per-rank metrics, what HCCL can and cannot reveal |
| [docs/reporting.md](docs/reporting.md) | report artifacts, main-report layout, data contract, comparability rules, formatting, chart degradation |
| [docs/reporting_refactor.md](docs/reporting_refactor.md) | why the report system was rebuilt, what changed, test results, open gaps |
| [docs/workflow.md](docs/workflow.md) | the autonomous workflow: before/after, module map, state files, recorded runs |
| [references/](references/) | the policies the skill applies: request interpretation, gates, dependencies, execution, profiling, recovery, report acceptance |
| [docs/manifest_diff.md](docs/manifest_diff.md) | how to diff two manifests, and the numbers of this refactor |
| [docs/blocked.md](docs/blocked.md) | **what is not done**, with the exact missing interface |
| [docs/benchmark_matrix.md](docs/benchmark_matrix.md) | matrix design: layers, suites, variants, coverage reports |
| [SKILL.md](SKILL.md) | the ten-step operating procedure with acceptance gates |
| [operators/README.md](operators/README.md) | capability matrix per adapter |
| [dependencies/README.md](dependencies/README.md) | dependency catalog and profiles |
| [kernels/asc_comm/README.md](kernels/asc_comm/README.md) | the ASC-COMM kernel project and its blockers |

## What is blocked

No NPU and no CANN are present on this machine, so every NPU-dependent verification is
blocked: device kernels (AIV, CCU), any device execution, `HcclChannelAcquire` and
physical-link verification on hardware, msprof kernel evidence, EP64/EP128 on hardware, and
the multi-node/profiling installs. [docs/blocked.md](docs/blocked.md) is the complete list
with the exact missing environment and the condition that would close each item. CPU runs
validate framework behaviour; they say nothing about Ascend performance.

## Development

```sh
python -m pytest -q                                   # offline
python -m pytest -q tests/test_integration.py          # CPU two-rank Gloo, needs PyTorch
EP_RUN_NPU_TESTS=1 python -m pytest -m npu            # NPU tests, explicit runtime required
python scripts/vendor_ascendep.py --check              # vendored adapter integrity
```

On Windows CPU wheels without the libuv backend the launcher uses a Gloo file rendezvous; if
the temporary path is too long, pass a short `--basetemp` to pytest.
