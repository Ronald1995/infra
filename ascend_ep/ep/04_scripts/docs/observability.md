# Observability

Every result record answers four questions: how long did it take, how many bytes crossed a
card boundary, what actually ran, and where did each number come from. This document maps
the fields to the code that fills them.

## 1. `ExecutionObservation`

`common/observation.py::ExecutionObservation` is the single shape every backend may fill.
It is serialised into `result["observation"]` by `common/runner.py::observe`.

| Field | Meaning |
|---|---|
| `actual_api` | the API actually called (`EPBackend.actual_api()`, else the registry's `api`) |
| `dispatch_bytes_sent` / `dispatch_bytes_received` | per-direction dispatch bytes, when a backend can separate them |
| `combine_bytes_sent` / `combine_bytes_received` | per-direction combine bytes |
| `logical_payload_bytes` | what the model asked to move, counted once per logical transfer |
| `remote_payload_bytes` | bytes leaves the card: local rows excluded, replication to several peers included |
| `measured_wire_bytes` | a counter or profiler volume; `None` means "not measured", never zero |
| `local_copy_bytes` | bytes moved inside one card (packing, reordering, quantisation) |
| `rank_max_latency_ms` / `rank_avg_latency_ms` | the cross-rank latency the runner measured |
| `actual_algorithm` | the algorithm actually used, or `None` -> serialised as `"unknown"` |
| `actual_transport` | the process-group backend actually in use, or `None` -> `"unknown"` |
| `fallback_used` / `fallback_reason` | the adapter ran on a different path than its name implies |
| `evidence` | the evidence items attached to this observation |
| `source` | `measured` / `backend_reported` / `routing_trace_model` / `declared_payload_model` / `unavailable` |
| `confidence` | `verified` / `declared` / `unknown` |
| `unavailable` | byte fields this observation cannot provide, with the reason |

`__post_init__` refuses an unknown `source`/`confidence` and refuses `fallback_used=True`
without a `fallback_reason`. `require(*fields)` records a field as unavailable instead of
leaving it to be read as zero.

## 2. Four kinds of byte count

| Kind | What it counts | Cross-card? |
|---|---|---|
| `logical_payload_bytes` | tokens x top_k x hidden x dtype, once per logical transfer | no — a model of the work, not of the wire |
| `remote_payload_bytes` | sum over the EP group of bytes that leave the sending card | yes, but it is still a model |
| `measured_wire_bytes` | what a counter or profiler reported | yes, measured |
| `local_copy_bytes` | bytes moved inside one card | **never** |

`cross_card_bytes` returns `measured_wire_bytes` when present, otherwise
`remote_payload_bytes`, otherwise `None`; `cross_card_source` names which of the three it
used. `cross_card_bandwidth(ms)` returns `None` when `cross_card_bytes` is `None` or the
latency is zero.

**Rule: a local copy never enters a cross-card bandwidth.** Dividing
`local_copy_bytes` — or `logical_payload_bytes` — by a link latency is the error this module
exists to prevent. `local_copy_bytes` is reported for completeness only, and
`metrics/summary.py::bandwidth` never reads it.

## 3. Latency aggregation

`metrics/summary.py::latency` requires an equal, non-zero sample count on every rank and
computes per iteration:

| Field | Definition |
|---|---|
| `rank_min`, `rank_mean`, `rank_max` | per-iteration min / mean / max across ranks (ms) |
| `p50`, `p95`, `p99` | percentiles of the per-iteration **rank max** — the headline numbers |
| `rank_avg_p50_ms`, `rank_min_p50_ms` | p50 of the per-iteration rank mean and rank min, reported next to the headline so an imbalance is visible |
| `imbalance_ratio` | first-iteration rank max / mean of rank maxima |
| `stddev`, `cv` | spread of the rank maxima |
| `p95_p50_ratio` | tail ratio of the headline series |
| `host_issue_p50_ms`, `host_wait_p50_ms`, `output_ready_p50_ms` | the three timing components from the timer |

The headline is rank max on purpose: a collective is only as fast as its slowest rank, so an
average would hide the rank that gates the step.

## 4. Where each number lands

| Number | Result field | Producer |
|---|---|---|
| samples (latency, issue, wait, ready, stage) | `raw_samples[].samples[]`, `samples.jsonl` | `common/timer.py::measure` |
| latency aggregate | `latency` | `metrics/summary.py::latency` |
| payload / traffic volume | `bandwidth` (L0: `declared_payload_model`; L1: `routing_trace_model`) | `Prepared.payload` or `metrics/summary.py::traffic` |
| bandwidth | `bandwidth.{logical,payload}_bandwidth_GBps` | `metrics/summary.py::bandwidth` |
| token rate | `throughput` | `metrics/summary.py::throughput` |
| effective FLOPs | `compute` | `metrics/summary.py::compute` |
| correctness checks | `correctness.ranks[].*` | the case handler's `validate()` |
| what ran | `observation` | `common/runner.py::observe` |
| completion guarantee | `completion.{level,detail,proves_transfer}` | `PreparedCase.completion` |
| handler metadata (operation, timing_includes, ...) | `handler_metadata` | the case handler |
| environment, versions, host | `environment` | `common/environment.py::capture` |
| rank mapping and topology | `topology`, `topology_graph` | `common/distributed.py::mapping` |
| routing trace summary | `load_balance`, `routing_trace_hash` | `common/routing.py::summarize` |
| memory snapshot | `memory` | `common/memory.py` |
| logs and attempt path | `logs`, `attempt` | `common/controller.py::execute` |

Two fields are deliberately always `None` in `bandwidth`: `physical_bandwidth_GBps` and
`hardware_bandwidth_utilization`, with
`hardware_unavailable_reason = "hardware link counters unavailable"`. A physical bandwidth
is only derivable from wire counters, and this framework does not collect them.

## 5. HCCL adapters: what can and cannot be observed

For an operator that goes through `torch.distributed` (for example `alltoall_hccl`,
`allgather_hccl`, implemented by `baselines/reference/collectives.py::CollectiveBackend`):

Observable:

- the process-group backend, via `dist.get_backend(ep_group)` -> `observation.actual_transport`
  (recorded by `common/runner.py::_transport`);
- the API name, from the adapter's `ACTUAL_API` or the registry `api` entry;
- the completion level the adapter promises: `CollectiveBackend.COMPLETION` is
  `("remote_visible", "the collective returns after the operation completed on this rank")`,
  so the local buffer holds the peer's data; the base `EPBackend` default is
  `("local_complete", ...)` for adapters that only wait a handle;
- a detected fallback: `_fallback_reason` reports `the process group backend is <x>, not
  HCCL` on NPU, and `CPU process group (<x>) is a functional stand-in, not an Ascend
  transport` for a non-L0 case on CPU;
- the routing-trace traffic model (`routing_trace_model`, `declared` confidence), including
  `dedup_opportunity_ratio` and the cross-server byte models.

Not observable, and therefore recorded as such:

- **the collective algorithm** (ring, hierarchical, ...). `CollectiveBackend` exposes
  `actual_algorithm()` and returns `ACTUAL_ALGORITHM = None` by design instead of guessing
  from the backend name, so the field is serialised as `"unknown"` unless a log or profiler
  source names it; `baselines/l0/collectives/torch_hccl.py` sets `transport_protocol: "auto"`
  for the same reason;
- **the wire volume** — `measured_wire_bytes` is `None` and
  `unavailable["measured_wire_bytes"] = "no link counter or profiler counter was collected"`
  for every `routing_trace_model` observation;
- **the kernel** — no `kernel` or `msprof_kernel` evidence is produced by any L1-L4 path;
  only `L0` has an evidence collector, and even there no production call site adds a
  profiler record yet (see `docs/blocked.md`).

An L2-L4 handler builds its `PreparedCase` without a `completion` argument, so
`benchmarks/cases/__init__.py::prepare` fills it from the adapter's own contract
(`EPBackend.completion_semantics()`: base `EPBackend` `local_complete`, `CollectiveBackend`
`remote_visible`). `result["completion"]["level"]` is therefore never `"none"` for a case
that ran; if a handler ever states nothing and there is no adapter, `"none"` means exactly
that — no completion guarantee was recorded — and must not be read as a transfer guarantee.

L2-L4 rows also carry byte volume now: `common/runner.py` derives
`logical_payload_bytes` and `remote_payload_bytes` from the routing trace even when the level
does not model wire traffic (this is why an L2/L3 row is not blank), and still leaves
`measured_wire_bytes` unavailable. `result["bandwidth"]` remains L1-only, so nothing invents a
physical bandwidth for a level that does not model one.

## 6. L0 specifics

The L0 path reports payload bytes through `Prepared.payload`, not through a routing trace:

- `result["bandwidth"]` is built by `common/runner.py` from
  `prepared.payload = {"per_rank_bytes", "global_bytes", "direction", "definition"}` and is
  labelled `source: "declared_payload_model"`, with `scope: "per_rank"` and
  `global_payload_bytes` kept beside it;
- `result["observation"]` for L0 is the explicit `unavailable(...)` observation
  ("this level does not model cross-card traffic; see compute and throughput"), because
  `observe()` has no traffic volume for `is_hw` tasks. Its byte fields are null by design,
  and its `evidence` list still receives the completion item
  (`"<level>: <detail>"`) that `run_task` appends;
- the declared payload is *not* a wire measurement. A `LINK-*` bandwidth is therefore a
  payload model over a measured rank-max latency, and it cannot be compared with a
  counter-derived bandwidth.

## 7. What a report may compare

`analysis/comparison.py::comparable` accepts a row only when it is `passed`, its `source` is
`measured`, its `correctness.passed` is true and its `latency.p50` is positive. Groups are
then keyed by device, NPU model, CANN version, package set, topology, EP/TP size, workload,
routing-trace hash, model, timing mode, dedup/load-balance state, level, operation and case
parameters, so two adapters land in one group only when all of those agree.

For L0, `baselines/l0/verification/path_verifier.py::status_allows_comparison` states that
only `verified` records may be compared across mechanisms, and `comparable()` enforces it: an
L0 row whose `verification.status` is not `verified` never enters a group. On this host
nothing reaches `verified` (no bridge, no device), so no cross-mechanism L0 ranking is
produced at all; a report must say that instead of presenting requested configurations as a
measured ranking.
