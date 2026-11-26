# ASC-COMM: the AIV and CCU data planes for the L0 transport baseline

## Why this project exists

The L0 transport baseline used to take an `engine` parameter, bind the HCOMM channel
to that engine, and then submit the transfer itself through the **host** NBI path.
A case labelled `AIV` or `CCU` therefore measured host submission and reported it as
a device data plane. That is the single largest attribution error in the benchmark:
the number was real, the label was not.

`ASC-COMM` is the separate kernel project that provides the real device data plane.
The rule it enforces is simple:

> Control plane: the host creates the endpoint, acquires the channel, registers the
> buffers and hands over addresses.
> Data plane: a compiled device program performs the transfer.
> A host-submitted copy is **never** a substitute for either device plane.

## Status: BLOCKED on hardware and toolchain

| part | state | evidence |
| --- | --- | --- |
| `host/launcher.cpp`, `host/registration.cpp` | written, no CANN API used | compiles with any C++17 compiler |
| `common/abi.h` | written and mirrored in Python | `tests/test_abi.py` checks field order and symbols |
| `cmake/write_manifest.cmake`, `CMakeLists.txt`, `build.sh` | written and probe-driven | `build.sh` exits non-zero with the exact blocker |
| `aiv/write_nbi/kernel.cpp` | **not written** | needs the AIV/HCOMM device headers of the installed toolkit |
| `ccu/dsl/kernel.cpp` | **not written** | needs the CCU DSL frontend of the installed toolkit |
| any device execution | **never executed** | needs an Ascend 950/A3 device with UB links |

Consequences, by design:

* `LINK-001`, `LINK-002`, `LINK-003`, `LINK-004` (AIV) and `LINK-008` (CCU) report
  `not_implemented` while the device sources are missing, and
  `dependency_missing` when even the toolchain is absent.
* They never fall back to the host path, so a `LINK-0xx` result can no longer be a
  host measurement wearing an AIV label.
* `tests/test_kernels.py` asserts exactly this behaviour, including that the probe
  refuses a library that does not export the ABI symbols.

**Nothing in this directory may be described as working until it has been built and
executed on hardware and the resulting `build/manifest.json` plus the msprof kernel
record are attached to a run.**

## Layout

```
kernels/asc_comm/
├── README.md                  this file
├── CMakeLists.txt             real build; refuses to run without ASCEND_HOME_PATH
├── build.sh                   probe + build + manifest, or a precise blocker
├── cmake/write_manifest.cmake the file the capability probe reads
├── common/abi.h               the ABI contract (host <-> kernel)
├── aiv/
│   ├── README.md              what the AIV kernel must do, and how to verify it
│   └── write_nbi/             device source goes here (see README)
├── ccu/
│   ├── README.md              what the CCU kernel must do, and how to verify it
│   └── dsl/                   CCU DSL source goes here (see README)
├── host/
│   ├── launcher.cpp           validates descriptors, forwards to the engine, never copies
│   └── registration.cpp       registered-buffer bounds checks
└── tests/                     ABI/Python cross-checks
```

## ABI

`common/abi.h` declares one transfer record and six entry points:

```c
typedef struct { uint64_t src, dst, bytes; uint32_t peer, flags; } asc_comm_transfer;

int asc_comm_aiv_write_nbi(uint32_t channel, const asc_comm_transfer*, uint32_t count);
int asc_comm_aiv_drain(uint32_t channel, uint64_t timeout_ns);
int asc_comm_ccu_submit(uint32_t channel, const asc_comm_transfer*, uint32_t count);
int asc_comm_ccu_wait(uint32_t channel, uint64_t timeout_ns);
int asc_comm_build_info_get(asc_comm_build_info* out);
```

The Python mirror is `baselines/l0/transport/dataplane.py::AscCommTransfer`. The
loader resolves exactly these symbols and refuses a library that is missing any of
them (`baselines/l0/transport/kernel_artifacts.py::REQUIRED_SYMBOLS`), so a partial
build cannot masquerade as a working data plane.

## Build

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "${ASCEND_HOME_PATH}/set_env.sh"
./kernels/asc_comm/build.sh --soc Ascend950
```

* Without `ASCEND_HOME_PATH` the build stops with `dependency_missing` and says why.
* With the toolchain present but the device sources absent it warns and produces a
  library whose AIV/CCU entry points return `ASC_COMM_ERR_UNSUPPORTED`. Use
  `--strict` to turn that into a hard failure.
* `build/manifest.json` is written only after the library linked.

## Completion semantics (this is the point of the project)

| data plane | completion symbol | contract | proves the transfer? |
| --- | --- | --- | --- |
| host NBI | `HcclChannelQuiet` (drain) | `fenced` | **no** |
| AIV | `asc_comm_aiv_drain` | `remote_visible` | yes, after peer validation |
| CCU | `asc_comm_ccu_wait` | `remote_visible` | yes, after peer validation |

The host fence is a fence: it orders the local channel and says nothing about the
peer. The runner records the contract next to the latency (`result["completion"]`),
so a fenced sample is never presented as a completed transfer. See
`docs/l0_data_plane.md`.

## Verification plan (what must be attached before this is called done)

1. `./build.sh --soc <soc> --strict` succeeds on the target machine, and
   `build/manifest.json` names both engines.
2. `python run_benchmark.py capabilities --json` shows `kernel_available` for the
   AIV/CCU channels instead of `dependency_missing`.
3. `LINK-002` (AIV) and `LINK-008` (CCU) run with `completion.level ==
   remote_visible`, `observation.source == measured`, and a correct element-wise
   payload check.
4. The msprof record contains the kernel name from `aiv/write_nbi/README.md`, and
   the evidence bundle contains `msprof_kernel` + `kernel` evidence.
5. A delayed-write test proves `wait()` does not return before the peer observes the
   data: see `docs/l0_data_plane.md` §"Anti-early-return test".

Until all five exist, every AIV/CCU case is reported as blocked, and the run summary
counts it under `blocked`, never under `passed`.
