# Architecture

This document describes the module boundaries of the benchmark framework and the call
chain from a plan file to a report. It is written against the code in this tree; every
claim names the module that implements it.

## 1. The one rule

A case is defined exactly once, in `common/case_spec.py`. Every other layer either
*selects* a case (`common/suites.py`), *parameterises* it (`common/variants.py`),
*schedules* it against an adapter (`common/scheduler.py`), or *executes* the handler the
spec names (`benchmarks/cases/`). No layer redefines what a case means.

`common/case_spec.py::CaseSpec` carries: identity (`id`, `level`, `operation`,
`category`), the adapter capability requirement (`requires`), parameter variants
(`variants`), the execution handler name (`handler`), sweep dimensions (`sweep`), the
message-size window (`size_window`), correctness tokens (`correctness`), selection tags,
phase, applicability rule names (`applicability`), the workload transform name
(`transform`), the fixed owner (`owner`), the legacy alias target (`alias_of`) and the
static request parameters (`parameters`).

## 2. The call chain

| Stage | Module / entry point | Responsibility | Must not do |
|---|---|---|---|
| Case catalog | `common/case_spec.py::specs` | The 83-row table (79 canonical + 4 legacy aliases); `_validate` rejects a bad row when the catalog is first built | Import torch, a device library or a baseline |
| Suite | `common/suites.py::select` | Select case IDs by level / category / explicit IDs; strong/weak scaling workload policy | Define a case, or narrow the level scope behind the plan's back |
| Config | `common/config.py::load_configs`, `validate`, `load_suite_overlays` | Load YAML, apply `extends`, expand `matrix`, validate every field; strip environment sections out of suite files (`suite_scenario`, `matrix_overlay`) | Let a suite file override `distributed` / `model` / `topology` / `device` |
| Plan | `common/plan.py::load_plan`, `plan_points`, `expand` | Compose base -> EP size -> suite overlay -> case transform -> topology, then call the scheduler and `check_completeness` | Contain case knowledge, or invent a task the scheduler did not produce |
| Scheduler | `common/scheduler.py::make_plan` | Expand `(EP size, case, operator, parameters)` into tasks; resolve status per case; keep every point in the manifest | Consult a capability verdict while expanding parameters |
| Manifest | `common/controller.py::create_manifest` | Persist `manifest.json`, `matrix.json`, `resolved_config.yaml`, `topology.json`, per-EP indexes | Execute anything |
| Controller | `common/controller.py::execute` | Iterate tasks, preflight NPU device count, launch workers, apply retries, persist results, write `ep_summary.json` | Time a case or interpret its numbers |
| Worker | `common/runner.py::execute_worker` -> `run_task` | One process per rank: workload, backend, prepared case, validation, timed loop, metrics, observation | Decide whether a task should run |
| Workload | `common/workload.py`, `common/routing.py` | Deterministic inputs, routing trace and its summary | Depend on the adapter under test |
| Backend | `common/registry.py::Operator.create` -> `operators/<id>/backend.py::create` | Adapter lifecycle (`initialize`, `prepare`, `prepare_communication`, `prepare_training`, `wait_until_ready`) | Change the timer boundary |
| PreparedCase | `common/prepared.py` | The single execution contract (`setup`, `run`, `wait`, `validate`, `cleanup`, `sample_metrics`, `request`, `evidence`, `payload`, `direct`, `observation`, `metadata`, `completion`) | Hide which completion level `wait` provides |
| Timer | `common/timer.py::measure` | Warmup + timed loop; `run` then `wait`; records `host_issue_ns`, `host_wait_ns`, `output_ready_ns` | Let setup, the barrier or validation fall inside the measured interval (the trailing `synchronize` is inside on purpose, so `output_ready` includes device work) |
| Validation | `common/validation.py` | Numerical/structural checks (`error`, `validate_dispatch`, `validate_moe`, `cpu_oracle`) | Run inside the timed loop |
| Metrics | `metrics/summary.py` | Latency aggregation, traffic model, bandwidth, throughput, effective FLOPs | Turn a local copy or a logical byte count into a cross-card bandwidth |
| Report | `reporting/report.py`, `reporting/coverage.py`, `reporting/l0_report.py` | Markdown/JSON/CSV, coverage matrices, EP comparison | Recompute a status |

### L0 chain

L0 is a shared hardware baseline, not an operator: `backend_id` is `hardware` and the
operator dimension is not expanded (`common/scheduler.py::make_plan`).

| Stage | Module | Responsibility | Must not do |
|---|---|---|---|
| Hardware baseline entry | `benchmarks/l0_hardware.py::prepare` | Route a case to its family (`memory`, `transport`, `execution`, `overlap`, `compute`, `collective`) and adapt the result to `PreparedCase` | Contain the measurement itself |
| Endpoint / channel | `baselines/l0/transport/channel.py::prepare` | Root-info exchange, `comm_init`, rank graph, buffer registration, `HcclChannelAcquire` | Claim a channel that `HcclChannelAcquire` did not return |
| Kernel submission | `baselines/l0/transport/dataplane.py` | Pick the data plane that must serve the declared execution resource | Serve `AIV`/`CCU` from the host NBI path |
| Completion | `baselines/l0/transport/completion.py` | Declare what `wait` proves (`submitted` < `fenced` < `local_complete` < `remote_visible`) | Let a `fenced` contract claim `proves_transfer` |
| Data validation | `baselines/l0/collectives/torch_hccl.py`, `baselines/l0/payload.py` | Element-wise comparison against an exactly computed reference | Compare a reduction against a tolerance nobody chose |
| Bandwidth | `metrics/summary.py::bandwidth` derived from `Prepared.payload` | Divide the declared payload by the measured rank-max latency | Report a physical bandwidth |

## 3. Before / after module map

| Module | Before | After | Why |
|---|---|---|---|
| `common/cases.py` | Case ID, level, operation and capability requirements (the catalog) | 35-line facade re-exporting `common.case_spec` (`Case = CaseSpec`) | Two definitions of a case drift apart; a facade cannot |
| `common/case_spec.py` | did not exist | The only case table, applicability rules, transforms, variants, L0 request parameters | One owner for identity + rules + parameters |
| `common/variants.py` | `CASE_VARIANTS` table and per-case size windows | Reads `spec.variants` / `spec.size_window`; only turns them into parameter dicts | A variant list must not live next to the expansion code |
| `common/scheduler.py` | `if case.id == ...` status chain and a fixed-owner exception list | Applies `APPLICABILITY_RULES` named on the spec, then the adapter contract (`status_for`) | Rules become data, ordered by the spec, and testable |
| `benchmarks/operator_cases.py` | `Operation` dispatcher with a per-case `if/elif` | 23-line compatibility shim over `benchmarks/cases` | One handler registry replaces the branch chain |
| `benchmarks/cases/{common,l1_communication,l2_fusion,l3_moe,l4_router_planner}.py` | did not exist (one file) | One module per level, 45 handlers in `HANDLERS`, resolved lazily by `benchmarks/cases/__init__.py::handler` | Adding a case is one row + one function + one registry line |
| `baselines/l0/contract.py::Prepared` | The L0 shape returned to the runner | Kept for L0 baselines; `common/prepared.py::PreparedCase.from_l0` adapts it | Existing baselines stay untouched; the runner sees one type |
| `common/prepared.py` | did not exist (two shapes: `Prepared` and `Operation`) | The union type plus `COMPLETION_LEVELS` | The runner has one execution path and one completion field |
| `common/observation.py` | Byte counts mixed into the flat result record | `ExecutionObservation` with four separate byte kinds and attribution | A local copy can no longer reach a cross-card bandwidth |
| `baselines/l0/transport/dataplane.py` | `engine` parameter bound to the HCOMM channel, transfer always submitted by the host | `DataPlaneBackend` with Host / AIV / CCU implementations | The data plane becomes a declared, gated object |
| `baselines/l0/transport/kernel_artifacts.py` | did not exist | Probes source / toolchain / built artifact / ABI symbols | A missing kernel is `dependency_missing` or `not_implemented`, never a host measurement |
| `baselines/l0/verification/evidence.py` | Descriptor + rank graph treated as a link proof | `EvidenceCollector`, seven evidence kinds, the verification ladder | Configuration is never evidence |
| `baselines/l0/topology/runtime_probe.py` | did not exist (`supported` meant "compiled in") | Staged probe with the `unsupported` / `environment_unavailable` / `dependency_missing` / `not_implemented` vocabulary | A compiled channel is not a usable channel |
| `dependencies/manifest.yaml`, `scripts/doctor.py`, `scripts/install_deps.sh` | One `dependency_missing` status for six different situations | Per-dependency catalog, per-family availability matrix, profile-driven installer | A CPU workstation and a broken Ascend node stop looking alike |
| `operators/vendor/ascendep/` | `EP_ASCENDEP_ROOT` checkout required | 23 files vendored at a pinned commit; the env var is an optional override | A run needs no external checkout, and provenance is recorded |

## 4. The duplication that was removed

| Fact | Old location(s) | New location |
|---|---|---|
| Case identity (ID, level, operation, category) | `common/cases.py` | `CaseSpec` in `common/case_spec.py` |
| Capability requirement per case | `common/cases.py` | `CaseSpec.requires` |
| Named parameter variants | `common/variants.py::CASE_VARIANTS` | `CaseSpec.variants` (one `_CASE_VARIANTS` table in the catalog) |
| Message-size window per case | inferred in `common/variants.py` | `CaseSpec.size_window` (and the family fallback in `variants.py::_window`) |
| Per-case status condition | `if case.id == ...` in `common/scheduler.py` | `APPLICABILITY_RULES` + `CaseSpec.applicability` |
| Fixed-owner adapter | exception list in `common/scheduler.py` | `CaseSpec.owner` (`_CASE_OWNER`), resolved by `scheduler.required_capabilities` |
| Workload transform | inline in the scheduler | `TRANSFORMS` + `CaseSpec.transform` |
| L0 request parameters | `common/cases.py::L0_PARAMETERS` | `common/case_spec.py::L0_PARAMETERS` (re-exported by `cases.py`) |
| Which operation a case times | `if/elif` in `benchmarks/operator_cases.py` | `CaseSpec.handler` + `benchmarks/cases/__init__.py::HANDLERS` |

One literal case-ID set remains in `common/scheduler.py::make_plan`: the `baselines`
command restricts L2 to `FUSE-001`/`FUSE-003` (the serial compositions that do not need a
fused adapter). It is a *selection* rule for one command, not a status rule. It is the only
case-ID literal left in the scheduling layer (`common/scheduler.py`, `common/variants.py`);
outside that layer, `benchmarks/l0_hardware.py` (family routing) and `common/runner.py`
(record fields, weights allocation) still name a few L0 IDs, which is routing and record
code, not case definition.

## 5. Invariants kept by the code

- **Parameter expansion is capability independent.** `common/variants.py::task_parameters`
  never reads a status; `scheduler.make_plan` attaches the verdict afterwards. This is what
  makes EP8, EP64 and EP128 produce the same point set.
- **Every point stays in the manifest.** An unsupported / not-applicable / not-implemented
  point is a task with a reason, never a missing task (`scheduler.make_plan`, checked by
  `tests/test_case_spec.py::test_a_non_executable_point_is_still_a_task`).
- **EP-size invariance is enforced, not assumed.**
  `common/scheduler.py::check_parity` compares the multiset of
  `(case, level, operator, variant)` per EP size and raises on any difference.
- **One execution path.** `common/runner.py::run_task` prepares every level through
  `benchmarks.cases.prepare` and times `run`/`wait` with `common/timer.py::measure`.
  L0-specific *record* fields (evidence, verification ladder, topology graph) are added
  afterwards; L0 does not have its own timer.
- **A completion guarantee is recorded with the number.** `PreparedCase.completion` flows to
  `result["completion"]`, and `result["observation"]` carries the byte accounting.
- **Configuration is never evidence.** `baselines/l0/verification/path_verifier.py` only
  promotes a field from an `Evidence` item whose source is in `STRONG_SOURCES`.

## 6. Adding a case

1. Add one row to `common/case_spec.py` (or one row to `_L1`..`_L4`), naming `handler`,
   `requires`, `variants`, `applicability` and `correctness`.
2. Add the handler function to the matching `benchmarks/cases/*.py` module and register it
   in that module's `HANDLERS`.
3. If the case needs a suite, add its ID to `common/suites.py` — a case that belongs to no
   suite is still reachable through `all`.

Nothing else needs an `if`/`elif`: the test
`tests/test_case_spec.py::test_every_layer_reads_the_spec_instead_of_a_case_id` fails when a
case ID comparison reappears in `common/scheduler.py` or `common/variants.py`.
