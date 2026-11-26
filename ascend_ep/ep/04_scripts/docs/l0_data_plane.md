# L0 data plane

The L0 transport baseline (`LINK-*`) separates three things that used to be one code path:
the control plane, the data plane, and the completion guarantee. This document describes
each, and states which parts are implemented and which are blocked.

## 1. Control plane vs data plane

| Plane | What it does | Where |
|---|---|---|
| Control plane | Open a native bridge session, exchange root information, initialise the HCCL communicator, query the rank graph, register symmetric buffers, acquire one channel per peer, read the peer's memory descriptors, allocate an engine thread | `baselines/l0/transport/channel.py::prepare`, `baselines/l0/transport/bridge.py` |
| Data plane | Move the bytes: issue the remote write, drain the engine, report completion | `baselines/l0/transport/dataplane.py` |
| Completion | Say what "done" means for the primitive that was used | `baselines/l0/transport/completion.py` |

The control plane is real for every `LINK-*` case, including the blocked ones: the channel
would be acquired by the host. What used to be wrong is that the *data plane* was also the
host: after binding the channel to `AIV` or `CCU`, the old driver called the host NBI
transfer API. The engine parameter changed; the bytes were still moved by the host.

## 2. The three data-plane implementations

`baselines/l0/transport/dataplane.py` declares one interface
(`initialize`, `prepare`, `submit`, `wait`, `validate`, `finalize`, `completion`,
`blocker`) and three implementations:

| Class | `kind` | Engine | Completion contract | State today |
|---|---|---|---|---|
| `HostDataPlane` | `host` | `HOST_CPU`, `AICPU_TS` | `HOST_NBI_DRAIN` (`fenced`) | implemented; the only plane that can run |
| `AIVDataPlane` | `aiv` | `AIV` | `AIV_DRAIN` (`remote_visible`) | host side complete, device kernel missing -> blocked |
| `CCUDataPlane` | `ccu` | `CCU` | `CCU_COMPLETION` (`remote_visible`) | host side complete, device kernel missing -> blocked |

The mapping from a case's declared execution resource to a plane is explicit:

```python
KIND_FOR_RESOURCE = {"HOST_CPU": "host", "AICPU_TS": "host", "AIV": "aiv", "CCU": "ccu"}
```

`dataplane.kind_for(resource)` raises `BenchmarkUnavailable("unsupported", ...)` for a
resource with no plane, so an unknown engine cannot silently fall back to the host.

## 3. Why the host NBI path is not an AIV or CCU data plane

An `AIV` case asks: "what does a device-side AIV kernel achieve?" The host NBI path
answers a different question ("what does host submission achieve?") and its completion is a
fence, not a transfer. Three mechanisms in the code enforce the separation:

1. `baselines/l0/transport/channel.py::_data_plane` asks the data plane for a blocker
   *before* touching the bridge. A non-host plane with a missing kernel raises
   `dependency_missing` (toolchain or artifact absent) or `not_implemented` (artifact
   present, buffer registration never executed on hardware). It never returns a host plane
   for an `AIV`/`CCU` case.
2. `baselines/l0/transport/kernel_artifacts.py::probe` answers three separate questions —
   is the kernel project present, is a *complete* toolchain present (`ASCEND_HOME_PATH`,
   CMake, a C++ compiler), and does a built library export the ABI symbols
   (`asc_comm_aiv_write_nbi`, `asc_comm_aiv_drain`, `asc_comm_ccu_submit`,
   `asc_comm_ccu_wait`). A partial build cannot pass as a working plane.
3. `kernels/asc_comm/common/abi.h` is the contract; `kernels/asc_comm/build.sh` exits
   non-zero with `BLOCKED` and a status when the toolchain or the device source is missing,
   and writes `build/manifest.json` only after a successful link.

Affected cases: `LINK-001`..`LINK-004` (AIV) and `LINK-008` (CCU) declare those execution
resources in `common/case_spec.py::L0_PARAMETERS`. `LINK-005`..`LINK-007`, `LINK-009`..
`LINK-011` declare `AICPU_TS`, and `LINK-012` `HOST_CPU`; those are served by the host
plane by definition. `OVERLAP-*` cases declare execution resources in their parameters but
run through `baselines/l0/overlap/runner.py`, which does not use `DataPlaneBackend` at all.

## 4. The completion ladder

Ordered weakest to strongest (`baselines/l0/transport/completion.py::LEVELS`):

| Level | Meaning | May a bandwidth use it? |
|---|---|---|
| `submitted` | the driver accepted the request | no |
| `fenced` | a local fence or drain returned; ordering only | no |
| `local_complete` | the local engine reports the operation finished | yes |
| `remote_visible` | the peer can read the data | yes |

`CompletionContract.__post_init__` rejects a contract that claims `proves_transfer` at
`submitted` or `fenced`, so the wrong claim cannot be constructed:

```python
CompletionContract("bad", "fenced", True, "claims too much")   # ValueError
```

Declared contracts in this tree:

| Contract | Level | `proves_transfer` | Used by |
|---|---|---|---|
| `host_nbi_fence` | `fenced` | no | degraded AIV/CCU request (test-only path) |
| `host_nbi_drain` | `fenced` | no | `HostDataPlane` |
| `cpu_engine_queued` | `submitted` | no | a CPU engine that returns on queueing |
| `notify_round_trip` | `remote_visible` | yes | a peer acknowledgement |
| `aiv_drain` | `remote_visible` | yes | `AIVDataPlane` |
| `ccu_completion` | `remote_visible` | yes | `CCUDataPlane` |
| `torch_collective` | `remote_visible` | yes | `torch.distributed` collectives |
| `sync_exchange` | `remote_visible` | yes | a synchronous request/response exchange |

`completion.for_engine(engine, device_plane=...)` selects the contract. Note that
`for_engine("AIV")` without a device plane returns `host_nbi_fence`: if an AIV transfer is
ever issued by the host, the honest contract is a fence.

## 5. The `.wait()` contract

`common/prepared.py` fixes the timing boundary:

```
Timer Start -> run()   # submit only, must not block on completion
            -> wait()  # real completion: the data is visible on the receiving side
            -> Timer Stop
```

`run` may not block, and `wait` may not return before the guarantee its contract names. The
runner stores `result["completion"] = {"level", "detail", "proves_transfer"}` next to the
latency, and `PreparedCase.transfers_distinct_from_submission()` is true only for
`local_complete` and `remote_visible`.

For the host plane the guarantee is `fenced`. `HostDataPlane.wait` calls
`session.quiet()` and raises if it fails; it does not claim the remote write landed. A
`LINK-001`-style result produced by that plane must therefore not be read as a completed
device transfer, and today it cannot be produced at all for AIV/CCU cases (see §3).

## 6. Anti-early-return test (delayed write)

The test exists to falsify the claim "`wait()` returning means the peer can read the
data". Design (also described in `kernels/asc_comm/aiv/README.md`):

1. rank 0 submits one remote write of a known pattern to rank 1's registered buffer;
2. rank 1 spins on the destination buffer **before** any barrier, flag or collective —
   it must not synchronise with rank 0 first;
3. rank 0 must not report completion until rank 1's read matches the pattern;
4. the write is deliberately delayed (the peer is busy, the engine queue is deep, or the
   kernel is asked to sleep before retiring the write), so an early `wait()` return is
   observable as a mismatch rather than as a lucky timing.

What it proves when it passes: the completion primitive is not a fence. What it proves when
it fails: the contract must be downgraded to `fenced` in
`baselines/l0/transport/completion.py`, and the case may no longer sit inside a bandwidth
denominator.

**Status: BLOCKED.** It cannot run on this machine. It requires:

- two Ascend devices with UB links between them (this host has no NPU and no CANN);
- the ASC-COMM AIV kernel built on the target (`kernels/asc_comm/build.sh --strict`
  producing `build/manifest.json`), because a host-side write has no kernel to delay;
- a peer-side spin read in the same process group, i.e. a real two-rank launch, not the
  CPU Gloo fallback;
- the resulting record attached to a run: kernel name from msprof, `completion.level ==
  remote_visible`, and the delayed-write mismatch count (must be zero).

Until those exist, no run in this tree may claim that its `wait()` was proven by this test.

## 7. What a result may claim today

| Case group | Served by | Claim allowed |
|---|---|---|
| `LINK-001`..`004`, `LINK-008` | none (blocked) | `dependency_missing` / `not_implemented`, never a bandwidth |
| `LINK-005`..`007`, `LINK-009`..`011` | `HostDataPlane` (`AICPU_TS`) | latency of a host-issued transfer, completion `fenced`; no transfer proof |
| `LINK-012`, `LINK-013` | `baselines/l0/shmem` | its own prepared object; not a `DataPlaneBackend` |
| `EXEC-001`..`006` | the native `hccl_test` tool via `baselines/l0/collectives/native_hccl.py` | the tool's own average latency and `check_result`; the framework does not time this path, so `completion` stays the `Prepared` default (`submitted`) and no `wait` contract is exercised |

`tests/test_dataplane.py` pins the invariants: a fence cannot prove a transfer, the host
drain is a fence and says so, AIV/CCU planes are not host planes, a device plane without a
kernel reports its blocker and never falls back, and the Python `AscCommTransfer` field
order still matches `abi.h`.
