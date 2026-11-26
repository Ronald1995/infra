# Evidence and capability

Two questions are answered here, and they are separate:

- *What may a finished case claim?* — `baselines/l0/verification/`.
- *What may be attempted on this machine at all?* — `baselines/l0/topology/`.

## 1. Evidence kinds

`baselines/l0/verification/evidence.py::KINDS` is a closed vocabulary. Each kind maps to the
`Evidence.source` string that `path_verifier.py` accepts:

| Kind | Source string | Produced by | Strength |
|---|---|---|---|
| `descriptor` | `native_descriptor` | the native bridge's self-description | strong |
| `rank_graph` | `rank_graph` | the runtime rank-graph query | strong |
| `endpoint` | `endpoint` | an HCOMM endpoint was created | strong |
| `channel_kind` | `channel_kind` | `HcclChannelAcquire` returned a channel class | strong |
| `kernel` | `kernel` | a kernel launch was recorded | strong |
| `msprof_kernel` | `msprof_kernel` | a profiler record names the kernel | strong |
| `runtime_execution` | `runtime_execution` | the peer's payload was validated | strong |

Heuristic sources exist for log scraping (`hccl_log_keyword`, `hcomm_log_keyword`,
`profiler_kernel_name`, `soc_name_heuristic`). A heuristic claim is attached to the record
but **never** promotes a field to `verified` (`path_verifier.verify`, branch
`item.confidence == "strong"`).

## 2. What a record may claim

`baselines/l0/verification/path_verifier.py` applies five rules:

1. a field is `verified` only when a strong source names it;
2. a heuristic source may report a field but leaves it `unverified` (the claim is kept for
   audit);
3. if strong evidence contradicts the request, the record status is `fallback`;
4. a required field with no evidence at all makes the record `unverified`
   (`DEFAULT_REQUIRED = ("execution_resource", "transport_protocol")`);
5. configuration values are never evidence.

The aggregate status is the *worst* required field (`vocabulary.worst`, severity order
`verified < unverified < fallback < unsupported/environment_unavailable < failed`).
`status_allows_comparison(status)` states the rule that only a `verified` record may enter a
cross-mechanism comparison. Note that it is currently a predicate without a call site: the
analysis layer (`analysis/comparison.py::comparable`) filters on `passed` status, a passed
correctness block and a positive p50, and does **not** read the verification status. The gate
is therefore a documented rule plus a recorded field, not an enforced filter — a reader must
check `verification.status` before comparing two mechanisms.

Every record therefore carries both blocks: `requested` (intent) and `observed` (what a
runtime answer established). They are produced by different code paths and are never
copied into each other.

## 3. The verification ladder

`evidence.py::LADDER`, weakest first:

| Rung | Reached when | Rung is not reached by |
|---|---|---|
| `configured` | the request names an execution resource or protocol | — |
| `endpoint_verified` | an HCOMM endpoint was created | a configured endpoint |
| `channel_acquired` | `HcclChannelAcquire` returned a channel class | a compiled-in channel |
| `kernel_verified` | a kernel name or profiler record exists | a kernel that is merely available |
| `execution_verified` | the peer payload was validated | a returned call |
| `physical_link_verified` | the link is named by runtime evidence (see below) | a descriptor, a protocol constant, a configuration value |

`EvidenceCollector.resolve_link()` decides the link from evidence only:

| Evidence present | Result | Confidence |
|---|---|---|
| channel kind + rank-graph protocol that agree | that family | `verified` |
| channel kind + profiler record (+ agreeing protocol) | that family | `verified` |
| channel kind alone | family implied by the class name | `inferred` |
| rank-graph protocol alone | family implied by the protocol | `inferred` |
| nothing | `unverified` | `unknown` |

Two properties of the implementation must be read carefully:

- the rungs are **independent booleans**. `levels()["physical_link_verified"]` is derived
  from the link resolution, not from the rungs below it. A case with a channel kind and an
  agreeing rank-graph protocol resolves `physical_link` as `verified` while
  `kernel_verified` and `execution_verified` are still `False`. A reader must inspect the
  individual rungs, not only `strongest_level()`, which returns the deepest *reached* rung.
- `verification_level*` (the ladder) and `verification` (the field-level block) come from
  two places: the ladder from `EvidenceCollector.levels()`, stored in
  `Prepared.levels` and copied to `result["verification_levels"]` / `["verification_level"]`
  by `common/runner.py`; the field-level block from `path_verifier.annotate(record,
  request, prepared.evidence)`. A case can be `physical_link_verified` on the ladder and
  `unverified` in `verification.status` when the collector recorded no evidence item for
  the requested fields.

### Why a descriptor cannot verify a physical link

A descriptor says which channel class the runtime *accepted* and which protocol the bridge
was *built* with. A configuration says what was *requested*. Neither observes a byte
crossing a link: a URMA-family channel class does not prove that URMA carried the payload,
and a rank-graph protocol string does not prove that the write used it. Only an independent
runtime answer that agrees with the channel kind promotes the link to `verified`; and even
then, `execution_verified` still requires the peer's payload check.

## 4. Runtime capability probe

`baselines/l0/topology/runtime_probe.py::probe` walks a fixed stage sequence and records the
first stage that fails:

| Stage | Fails when | Status |
|---|---|---|
| `soc` | the SoC model cannot be read | `unsupported` when the platform cannot offer it |
| `compiled` | the bridge was not built with a channel for this resource/mechanism | `unsupported` |
| `endpoint_compatible` | the SoC/endpoint protocol rejects it | `unsupported` |
| `channel_acquired` | `HcclChannelAcquire` did not run or failed | `environment_unavailable` |
| `kernel_available` | the ASC-COMM kernel does not export the ABI | `dependency_missing` |
| `resource_available` | the device resource is busy | `environment_unavailable` |

The vocabulary is deliberate and not interchangeable:

- `unsupported` — the platform cannot offer it (SoC, protocol, compiled channel);
- `environment_unavailable` — the platform can, but a resource is busy right now;
- `dependency_missing` — a runtime library or binary is absent;
- `not_implemented` — the probe or the code path does not exist yet;
- `supported` — every stage above passed.

**A busy resource is never `unsupported`.** `STAGE_STATUS[("resource_available", False)]`
and `STAGE_STATUS[("channel_acquired", False)]` both map to `environment_unavailable`;
`tests/test_evidence.py::test_a_busy_resource_is_not_reported_as_unsupported` pins this.
Resources acquired by a probe are released before the measurement starts.

Implementation status: the staged probe is a library. It is exercised by
`tests/test_evidence.py` and has no CLI entry point and no production call site. What the
`capabilities` command prints is the *capability matrix* (§5), built by
`baselines/l0/topology/capabilities.py::discover` from the bridge, `hccl_test`, SHMEM and
memory-library probes.

## 5. Capability matrix

Schema: `schemas/l0-capabilities-v1.json` (required keys `schema_version`, `source`,
`device`, `channels`, `engines`, `accelerators`, `probes`; each channel entry requires
`supported` and `confidence`). `source` is one of `probe`, `override`, `unavailable`, and
`capabilities.discover` records `declared` versus `probed` confidence per channel: only a
probed entry may be `supported`.

The scheduler consumes the matrix per L0 parameter set
(`common/scheduler.py::_l0_status` -> `capability_table.support`): a channel the matrix
marks unsupported produces `unsupported` with the probe's own reason, and a legacy alias is
looked up as its canonical case.

Example: the injectable inputs from `tests/test_evidence.py`, which describe a machine
without pretending one was probed:

```python
DESCRIPTOR = {"channels": [{"channel": "AivUbMemChannel", "execution_resource": "AIV",
                            "mechanisms": ["UB_MEM"]}]}
DEVICE = {"soc": "Ascend950"}

runtime_probe.probe({}, channels=[
    {"case_id": "LINK-001", "channel": "AivUbMemChannel",
     "execution_resource": "AIV", "mechanisms": ["ub_memory"]}],
    descriptor=DESCRIPTOR, device=DEVICE)
```

| Input variation | `compiled` | `channel_acquired` | Recorded status |
|---|---|---|---|
| descriptor present (as above) | `pass` | `unknown` (no bridge session opened) | `environment_unavailable` |
| `descriptor={}` (bridge not built) | `unknown` | `unknown` | `dependency_missing` |
| channel not in the descriptor (`AicpuTsRoceChannel`) | `fail` | not reached | `unsupported` |

The `unknown` states are the honest part: without a bridge session the probe cannot know
whether `HcclChannelAcquire` would succeed, and it says so instead of guessing.

## 6. What cannot be verified here

This machine has no NPU, no CANN and no device nodes (`doctor`: `SoC unknown`, devices 0).
Therefore no endpoint can be created, no channel acquired, no kernel recorded in a profiler,
no payload delivered to a peer, and no physical link resolved from runtime evidence. An L0
transport case does not even reach the ladder here: `channel.prepare` raises `unsupported`
("native transport baselines require Ascend devices") before a collector exists. If it did
run, the ladder would stop at `configured`, because every rung above it needs a runtime
answer. `channel_acquired` and above are BLOCKED; see `docs/blocked.md`.
