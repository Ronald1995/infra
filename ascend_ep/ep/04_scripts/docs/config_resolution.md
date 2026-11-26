# Configuration, expansion and coverage after the test-system refactor

This page documents the layers this refactor separated, why they had to be separated,
and how to verify that a configuration actually reached a task. The modules are
`common/config.py` (resolution), `common/variants.py` (dimension expansion),
`common/config_audit.py` (traceability), `common/plan.py` (composition) and
`common/scheduler.py` (integrity).

## 1. Layers and who may set what

| Layer | Sets | Must not set |
|---|---|---|
| `configs/base_benchmark.yaml` | measurement protocol: warmup, iterations, timing, message-size pool, outstanding/peer pools, verification, compute shape | EP size, topology, case lists |
| `configs/environments/ep{8,16,32,64,128}_bf16.yaml` (`configs/ep*_bf16.yaml`) | EP size, TP size, model shape, expert count, profiling switch | case lists, scenario matrices |
| `configs/*.yaml` suites | workload (tokens, routing, dedup, planner), benchmark phase, levels, one scenario matrix | `device`, `distributed` (EP size), `topology`, model shape |
| `configs/benchmark_plan.yaml` | EP range, operator scope, suite list, levels, execution and report policy, topology pattern | per-operator internals |
| `operators/registry.yaml` | adapter capabilities, constraints, dependencies | global scope |

`model.num_experts` is the single exception at suite level: strong/weak scaling are
*defined* in terms of the expert count, so the scaling block and its matrix points may
set it. Every other environment field in a suite or matrix point is a hard error that
names the field and the file (`common/config.py::suite_scenario`,
`::matrix_overlay`). Before this change those fields were silently deleted, so a file
said one thing and the run did another.

## 2. Resolution order and provenance

```
schema_default -> base_config -> EP environment -> suite -> matrix point -> CLI
```

Every layer is read **raw** (`load_raw`, which follows `extends` but applies no schema
default), composed by `resolve_config`, and validated exactly once at the end. The old
loader merged `DEFAULT` into each file separately, which is how an EP environment that
merely omitted a field could inject an implicit default over an explicit base value.

`resolve_config` returns `(effective, provenance)`. Each plan point and each task
manifest records `config_provenance` as `{field: {value, source}}`:

```json
"config_provenance": {
  "distributed.ep_size": {"value": 32, "source": "ep32_bf16.yaml"},
  "benchmark.cores": {"value": [1, 2, 4], "source": "base_benchmark.yaml"},
  "topology.server_ids": {"value": [0, ...], "source": "plan pattern (synthetic, unverified)"}
}
```

A field that no file declares reports `schema_default`, which is how a reader tells an
intentional value from a default.

## 3. Dimension expansion

`common/variants.py` now decides per case, per mechanism and per variant which
dimensions are `sweep`, `fixed` or `not_applicable`, and expands **every** applicable
combination:

| Dimension | Where it sweeps | Where it is not applicable |
|---|---|---|
| `message_bytes` | every L0 family (per case size window) | compute cases |
| `outstanding` | transport cases that declare it | single-transfer cases (fixed 1) |
| `peer_count` | transport cases with an explicit peer sweep | cases that use the whole group (fixed 0) |
| `cores` | LINK-013 `shmem_udma` **device** variant (`udma.py` passes it as the QP count) | every other mechanism: the native bridge ABI has no core-count control |
| `quiet_mode` | LINK-013 `shmem_udma` **device** variant (`udma.py` passes it into `ep_l0_udma_submit`) | the `host` variant (no QP/quiet control) and every other mechanism |

The old code took `cores[-1]` / `quiet_modes[-1]`, so `[1, 2, 4] x [0, 1]` ran exactly
one point (`4`, `1`) and the host variant even recorded a `quiet_mode` it never used.
Now the device variant produces 6 message sizes × 3 cores × 2 quiet modes = 36 points,
the host variant 6 points with the fixed `(1, 0)`, and a mechanism without the control
carries no `cores`/`quiet_mode` key at all.

`case_dimensions(...)` is written into the plan metadata per EP point, so
"why is there no `cores=4` task for LINK-001?" is answered by the manifest itself:
`{"type": "not_applicable", "reason": "ub_memory does not expose any core-count control; ..."}`.

## 4. Config audit: config field -> task -> runtime consumer

`common/config_audit.py` traces the fields whose declared-vs-executed gap has bitten
this matrix:

| Field | Task parameter | Runtime consumer |
|---|---|---|
| `benchmark.message_bytes` | `message_bytes` | `baselines/l0/overlap/runner.py` |
| `benchmark.outstanding` | `outstanding` | `baselines/l0/transport/channel.py` |
| `benchmark.peer_counts` | `peer_count` | `baselines/l0/transport/peers.py` |
| `benchmark.cores` | `cores` | `baselines/l0/shmem/udma.py` |
| `benchmark.quiet_modes` | `quiet_mode` | `baselines/l0/shmem/udma.py` |
| `benchmark.execution.modes` | `accelerator` | `baselines/l0/collectives/accelerator.py` |
| `benchmark.transport.enabled` | `mechanism` | `baselines/l0/contract.py` |

Statuses: `swept` (all configured values reached a task), `partial`, `unconsumed`,
`consumer_missing`, `declared_not_applicable`, `not_selected`, `not_configured`. The plan
writes the report to `config_audit` in its metadata; `unconsumed` must be empty.

The audit does not prove physical effect at runtime — that is the result record's job
(`requested` vs `observed`, verification status). It proves that the configuration is
reachable and that the code that claims to consume it does mention it.

## 5. Five EP sizes and the unified model baseline

`configs/benchmark_plan.yaml` now plans EP8, EP16, EP32, EP64 and EP128, all with the
same case catalog, the same suites, the same parameter matrix and the same
`require_complete_execution`. The unified model baseline keeps **three experts per rank**
at every size, so an EP-scaling curve changes the communication scale and not the local
expert work:

| EP | Global experts | Experts/rank | Tokens/rank | Hidden | Intermediate | Top-K |
|---|---|---|---|---|---|---|
| 8 | 24 | 3 | 4096 | 7168 | 3072 | 6 |
| 16 | 48 | 3 | 4096 | 7168 | 3072 | 6 |
| 32 | 96 | 3 | 4096 | 7168 | 3072 | 6 |
| 64 | 192 | 3 | 4096 | 7168 | 3072 | 6 |
| 128 | 384 | 3 | 4096 | 7168 | 3072 | 6 |

`configs/ep64_bf16_384.yaml` keeps the historical six-experts-per-rank profile. It is a
*different model profile*: run it as its own plan and report it separately, never spliced
into the five-point scaling curve.

Current expansion (CPU, no hardware consulted):

```
$ python -c "from common.plan import load_plan, expand; run = expand(load_plan('configs/benchmark_plan.yaml')); ..."
tasks 63145   per ep {8: 12629, 16: 12629, 32: 12629, 64: 12629, 128: 12629}
config_audit ok True   (cores 480 tasks, quiet_modes 480 tasks, both swept)
```

The exact task count is generated, never quoted from documentation. Equal per-EP counts
are a *necessary* condition, not a sufficient one: `check_completeness` and
`check_parity` also require the same canonical case set and the same variant multiset at
every size, and an unsupported point stays in the manifest with its reason instead of
shrinking the scope.

## 5.5 Routing as a specified test condition

`common/routing.py` gained the `target_distribution` strategy. It does not "generate
randomly and then check": it allocates integer quotas for the requested metric, then
generates Top-K ids that satisfy them exactly.

| Target metric | Generated? | Bound |
|---|---|---|
| `rank_max_mean` | yes | `P * min(K, experts_per_rank) / K` |
| `expert_max_mean` | yes (<= 1 is the uniform case) | `E / K` |
| `local_route_ratio` | yes | `min(K, experts_per_rank) / K` |
| `server_max_mean`, `rank_tx/rx_max_mean`, `cross_server_ratio` | **no** -- statistical only | -- |

* The plan is a pure function of the seed, the token counts, the placement and the
  target, so every rank derives the whole quota table itself and keeps its own row:
  all operators therefore measure the same trace without a collective.
* A target above the bound is reported `not_applicable` with the bound and the reason
  (`topk_unique_expert_constraint`), never clamped. For example EP8 with three experts
  per rank cannot reach an 8x rank target, because a token has only three experts to
  send to on one rank; the bound is 4x.
* The achieved value is checked against the target with a relative tolerance in
  `summarize()` (`target_control.met`); a miss is a failed test point, not a silent pass.
* `routing_control` is validated: an unimplemented metric is rejected at configuration
  time instead of being "measured into existence".

`configs/routing_imbalance.yaml` is the suite that exercises it (balanced control,
rank 2x/4x, expert 2x, local 0/50%, and a decode-sized trace). It is registered in
`configs/benchmark_plan.yaml`, so the five EP sizes plan the same routing matrix.

## 5.6 Slow-rank injection

`common/straggler.py` separates "a rank has more work" (routing imbalance) from "a rank
is slower" (straggler). A request declares `benchmark.straggler`:

```yaml
straggler:
  enabled: true
  seed: 42
  selection: {mode: fixed, ranks: [0]}      # fixed|random|rotating|fraction|server|hotspot|non_hotspot
  injection: {stage: before_operation, method: host_delay, delay_us: 500}
  trigger: {probability: 1.0, every_n_iterations: 1}
  calibration: {relative_tolerance: 0.10}
```

* The selection is a pure function of the seed and the iteration, so every rank agrees
  on who is slow without a collective; `server` needs `topology.server_ids` and
  `hotspot`/`non_hotspot` need `workload.hot_rank`, and without them the test point is
  `not_applicable` instead of running as if it had been injected.
* `host_delay` and `stage_delay` are implemented. `device_delay`,
  `compute_contention`, `memory_contention` and `communication_contention` raise
  `BenchmarkUnavailable("not_implemented", ...)`: a device-side straggler is not
  approximated with a host sleep.
* The delay is injected **inside the timed region and after the barrier**
  (`common/timer.py`): before the barrier it would be absorbed by rank alignment,
  after the timer it would not be measured. The stage variant is called by an adapter
  at a boundary it really exposes; a fused path is never split just to inject.
* Every injection records requested vs measured delay and whether calibration passed;
  the per-rank summary lands in the task result (`record["straggler"]`) and the
  baseline point of `configs/straggler.yaml` disables the injection so the slowdown
  ratio has a denominator measured under the same configuration.

## 5.7 Long tail and coverage metrics

`analysis/long_tail.py` pairs every stressed point with a baseline measured under the
same condition and reports `slowdown_ratio`, `throughput_retention`, `amplification`
(extra group latency / injected delay) and the calibration error of the injection:

* a straggler point pairs with the same routing and the straggler disabled;
* a routing-target point pairs with the same case/operator/EP/tokens/model under
  `routing: balanced`;
* no matching baseline -> `not_applicable` with the reason. A ratio is never computed
  against a different configuration.

`configs/long_tail.yaml` holds the nine combined scenarios from the spec (C1 ideal
baseline, C2 load-only, C3 hardware-only, C4 hotspot routing + hotspot slow, C5 tail
propagation, C6 dynamic hotspot, C7 rotating slow, C8 12.5% slow, C9 planner on).

`common/coverage.py` keeps four numbers apart instead of one "pass rate":

| Metric | Denominator | What it proves |
|---|---|---|
| planning | planned requirements | the matrix expansion is complete |
| execution | applicable requirements (`not_applicable` excluded) | the run actually reached them |
| correctness | executed requirements | the numbers are right |
| verified performance | applicable requirements | the mechanism was proven, not a fallback |

`unverified`, `blocked_topology`, `not_run` and the failure statuses all lower the
right metric: a mechanism with no implementation can reach 100% planning coverage and
still cannot reach 100% verified coverage, and the report now says so explicitly
(`analysis/results.json` carries `coverage_metrics` and `long_tail`; the report has a
dedicated long-tail section).

## 5.8 Execution profiles and topology provenance

`common/profiles.py` separates "what the plan declares" from "how much of it this
invocation executes":

| Profile | Suites | Levels | Dimensions | Aliases |
|---|---|---|---|---|
| `smoke` | `all` only | L0, L1 | first value | dedupe |
| `core` | standard suites | plan | all | dedupe |
| `full` (default) | plan's list | plan | all | dedupe |
| `exhaustive` | plan's list | plan | all | run |

`--execution-profile` (or `plan.profile`) selects it; the resolved profile is written to
the manifest (`execution_profile`) and every plan point records
`benchmark.dimension_policy`, so a narrowed run is visible instead of looking like a
smaller matrix.

Topology provenance uses exactly five states, normalised on read:

| State | Meaning |
|---|---|
| `discovered` | read from the running deployment (runtime probe) |
| `user_provided` | an explicit rank map in the environment or the plan |
| `inferred` | derived from a declared grouping rule |
| `synthetic` | the plan's `server_size`/`supernode_size` pattern (never verified) |
| `unknown` | no topology was obtainable |

Legacy strings (`environment`, `explicit`, `plan`, `unavailable`) are mapped onto the
five states, and an unrecognised value becomes `unknown` -- a claim of "verified" is
never invented.

## 5.9 One transport registry, and the long-tail chart

`common/transport.py` is the only place a transport mechanism is declared. Each entry
describes four independent axes (programming model, protocol, execution engine,
physical link), its LINK/OVERLAP cases, its claim, and which dimensions it sweeps.
`config.L0_MECHANISMS`, `variants.MECHANISM_DIMENSIONS` and the tests all read it, so
adding a mechanism cannot leave one of the three tables behind:

```python
"shmem_udma": {
    "programming_model": "shmem", "protocol": None, "physical_link": "ub",
    "execution_engines": ("aiv",),
    "dimensions": {"cores": "sweep", "quiet_mode": "sweep",
                   "variants": {"device": "sweep", "host": "fixed"}},
    "link_cases": ("LINK-013",), "overlap_cases": ("OVERLAP-007",),
    "claim": "SHMEM UDMA submission with a measurable QP count and async-submit flag",
}
```

The report's long-tail section now also renders a chart
(`charts/long-tail-slowdown.png`) of the measured slowdown per stressed point, with a
1.0 reference line; when no pair has a baseline, no chart is produced and the section
says so instead of showing an empty figure.

## 5.10 Requested vs observed parameter evidence

A configured dimension is a claim about the runtime until the handler records the value
it actually used. `common/parameter_evidence.py` writes, per task,
`dimension_evidence = {parameters: {name: {requested, observed, verification, reason}},
verified, unverified}`:

* `verified` -- the handler reported the value at the runtime boundary (the SHMEM UDMA
  path records the QP count, quiet flag, outstanding count and slot size it handed to
  `ep_l0_udma_submit` / `shmem.core.put`);
* `unverified` -- no runtime record exists, so the configured value is an intention;
  the record says which dimensions and why.

The verdict flows into coverage: a `passed` row with an unverified dimension no longer
counts towards *verified performance* coverage, even though its correctness passed.
Copying the configuration into `observed` is never accepted as evidence, and a task
that sweeps no audited dimension claims nothing (and is trivially consistent).

## 6. What is still open (tracked, not hidden)

| Item | State |
|---|---|
| Routing target generation | **done for rank/expert/locality** (`target_distribution`, see 5.5); server-level and cross-server generation stay statistical until a server quota allocator exists |
| Straggler controller | **done for host/stage delay** (section 5.6); device delay and resource contention are declared `not_implemented` |
| Profiling / report sections for routing, straggler, combined long tail | pending the two items above |
| `blocked_topology` / `unverified` result statuses | the status vocabulary has `unsupported`, `not_implemented`, `not_applicable`; topology verification is recorded on the topology record, not yet as a task status |
| Transport registry as a single declaration | mechanisms are declared in `config.py::L0_MECHANISMS` and `variants.py::MECHANISM_DIMENSIONS`; they agree, but the merge into one registry is pending |
| Execution profiles (`smoke` / `core` / `full` / `exhaustive`) | the plan has `mode` and per-suite selection; the four named profiles are not implemented yet |

Each of these is a behaviour that would otherwise be a fake dimension or a fabricated
result, so it is listed here instead of being approximated.
