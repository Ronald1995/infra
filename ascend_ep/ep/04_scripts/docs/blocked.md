# Blocked capabilities

This is the document a reviewer reads to know what is **not** done. Every row separates
three things: what is implemented in this tree, what is missing, and what would have to be
attached to a run before the item may be called verified.

Ground truth for this host: `python run_benchmark.py doctor --profile cpu` reports
`CANN not installed`, `SoC unknown (devices: 0)`, `MPI not installed`, `RDMA devices: 0`.
There is no NPU and no CANN installation, so **no NPU-dependent verification has happened**
and none of the items below may be described as working.

## A. Device kernels

| Capability | Implemented here | Missing environment / interface | Verification condition |
|---|---|---|---|
| AIV `write_nbi` device kernel | host launcher, ctypes ABI (`kernels/asc_comm/common/abi.h`), Python mirror (`dataplane.AscCommTransfer`), build system, probe | `kernels/asc_comm/aiv/write_nbi/kernel.cpp` does not exist; the Ascend C / HCOMM device headers and the `ccec` driver of the installed toolkit are not available on this host | `kernels/asc_comm/build.sh --soc <soc> --strict` succeeds on the target; `build/manifest.json` has an `AIV` entry exporting `asc_comm_aiv_write_nbi` and `asc_comm_aiv_drain`; `LINK-002` runs with `completion.level == remote_visible`; msprof shows `asc_comm_aiv_write_nbi*` |
| CCU DSL kernel | same host side; `CCUDataPlane`, `ccu_completion` contract | `kernels/asc_comm/ccu/dsl/kernel.cpp` does not exist; the CCU DSL frontend, its launch/bind API and its completion query are unavailable here | the CCU build entry exports `asc_comm_ccu_submit` / `asc_comm_ccu_wait`; `LINK-008` acquires `CcuUrmaChannel`, reports `remote_visible`, and msprof shows a CCU kernel record |
| Device-ABI buffer registration | `_DeviceDataPlane.initialize` resolves the ABI symbols | the case-level registration of benchmark buffers against that ABI has never been executed; `_DeviceDataPlane.submit`/`wait` raise `not_implemented` | a `LINK-001`..`004`/`008` run reaches `execution_verified` with a validated payload on both peers |
| Kernel build artifact | `kernel_artifacts.probe` distinguishes source / toolchain / library / symbols | no `kernels/asc_comm/build/manifest.json`, no `ASCEND_HOME_PATH`, no CANN | `python run_benchmark.py capabilities --json` reports `kernel_available` (not `dependency_missing`) for the AIV/CCU channels |

## B. Hardware execution

| Capability | Implemented here | Missing environment / interface | Verification condition |
|---|---|---|---|
| Any device execution | the full runner, timer, validation and reporting path, exercised on CPU/Gloo | no NPU devices, no driver, no CANN runtime (`torch_npu` absent) | a run on an Ascend node produces `status: passed` records with `device: npu` |
| Channel Acquire on real hardware | `baselines/l0/transport/channel.py` calls `HcclChannelAcquire` per peer | the native bridge library is not built (needs CANN + HCCL); no device to acquire on | `capabilities.json` shows `supported: true` with `confidence: probed` for `LINK-001`..`013`, and a `LINK-*` result carries `channel_acquired` |
| Physical-link verification | `EvidenceCollector.resolve_link` and the ladder | needs a channel class from `HcclChannelAcquire` **and** an independent runtime source (rank-graph protocol or an msprof kernel) that agree | a record with `physical_link_confidence == "verified"` where the two sources agree; nothing weaker counts |
| `EXEC-001`..`006` collectives | `baselines/l0/collectives/native_hccl.py` parses the tool's own rows (`metrics.source = "native_hccl_tool"`, average only, `p95`/`p99` stay `None`) | the `hccl_test` binary (`EP_HCCL_TEST_BINARY` may point at it) plus a working HCCL runtime | the tool reports `check_result: success` for the swept sizes and `direct` metrics are recorded |
| `HW-001`..`006` framework collectives | `baselines/l0/collectives/torch_hccl.py` with the shared float32 payload | a device HCCL process group | element-exact payload check on a real HCCL group, per variant (`allgather` / `reducescatter` labelled correctly) |
| `LINK-012`/`LINK-013` SHMEM baselines (`HW-007`/`HW-008` aliases) | `baselines/l0/shmem/` with MTE/UDMA engines | `shmem` package (`aclshmem_init`, `aclshmemx_set_qp_num`), the UDMA device library (`benchmark.shmem_udma_library`) | `dependency_missing` disappears; both engines run with a byte-exact check |
| `MEM-002`/`MEM-003` local kernels | `baselines/l0/memory/local.py` (`l0_mem_copy`, MTE and UB modes) | the kernel build (`baselines/l0/memory/native/build_mem.sh`) and `benchmark.memory_library` | `dependency_missing` disappears; byte-exact local copy |
| EP64 / EP128 on hardware | planning, manifest and coverage for all three EP sizes | ≥ 64 / ≥ 128 Ascend devices, HCCL fabric, topology map, shared output root | a completed run per EP size with `completion_status: complete` and no `not_run` |
| Anti-early-return (delayed write) test | described in `docs/l0_data_plane.md` §6; the contracts it would falsify exist | two devices with UB links, the AIV kernel built, a two-rank launch with a peer spin read | rank 1's spin read matches before rank 0 reports completion; zero early returns |

## C. Profiling

| Capability | Implemented here | Missing environment / interface | Verification condition |
|---|---|---|---|
| NPU trace collection | `profiling/collect_profile.py` (torch / torch_npu profiler, `tensorboard_trace_handler`), `profile` command, `profile_manifest.json` with sha256 per artefact | `torch_npu` profiler; a device to trace | a `profile` attempt produces rank traces plus a parsed overlap/kernel analysis |
| msprof kernel evidence | `EvidenceCollector.msprof` and the `msprof_kernel` kind exist and are accepted by `path_verifier.STRONG_SOURCES` | `msprof` binary (ships inside CANN); **no production call site adds an `msprof_kernel` item** — only tests do | a `LINK-*` record whose evidence list contains an `msprof_kernel` item naming the executed kernel |
| `hccl_test` accelerator evidence | `baselines/l0/collectives/accelerator.py` probes the installed tool's modes | `hccl_test` binary (`EP_HCCL_TEST_BINARY` may point at it) | `capabilities.json` lists accelerator modes with `supported: true` |
| Overlap / device idle intervals | `profiling/parse_profile.py`, `reporting/report.py::overlap_analysis.md` | NPU traces | `overlap_analysis.md` shows non-null overlap/idle for a real run; today `overlap.internal_overlap_ratio` is recorded as `None` with a reason |

## D. Dependencies

| Dependency | State here | Missing | Verification condition |
|---|---|---|---|
| CANN toolkit, driver, HCCL/HCOMM, `npu-smi` | not installed (`doctor`: `CANN not installed`) | host image | `doctor --profile npu` has no `fail` |
| `torch_npu` (`==2.6.0.post3` in the manifest) | not installed | pip wheel matching the CANN release | `doctor` reports the pinned version as `pass` |
| Native transport bridge (`l0-native-bridges`) | not built | CANN + HCCL headers, a C++ toolchain on the target | `bridge.describe()` returns channels; `LINK-*` runs |
| DeepEP (`deep_ep`, `deep_ep_experimental`, v2), `fsdp_turbo`, `mega_moe`, `npu_patch`, `mindspeed`, `megatron`, `cann_ops_transformer`, `shmem` | not installed | source builds against the target CANN | the corresponding operators stop returning `dependency_missing` |
| `mc2_combine_add_rmsnorm_cann` | host side implemented (`cann_special.CombineRMSNormBackend`), registry status `integrated` | **upstream `ep/mc2_combine_add_rmsnorm_cann/demo.py` does not exist** in the pinned ascendep tree (21 adapters are vendored; this is not one of them), so `operators/ascendep.py::_import_demo` raises `source_unavailable` | either an upstream adapter is added and vendored, or the operator is re-based on a direct `torch_npu` call; until then `COMM-011` is planned but cannot run |
| Ascendep licence | vendored copy present, 23 files, commit `24ff84d3ab8c01e1e2071802465173e624e1f89d` | upstream declares **no licence file**; `operators/vendor/THIRD_PARTY_NOTICES.md` records this as a compliance item | an owner sign-off (or a licence grant) recorded next to the notices before any redistribution |
| MPI (`mpirun`) | not installed (`doctor`: `MPI not installed`) | Open MPI >= 4.1 or MPICH >= 3.4 | `doctor --profile multinode` has no `fail` for `mpi` |

## E. Multi-node

| Capability | Implemented here | Missing environment / interface | Verification condition |
|---|---|---|---|
| Multi-node launch | `--nnodes/--node-rank/--master-addr/--master-port/--session`, controller coordination files, `launchers/run_ep64.sh` | ≥ 2 nodes, a shared output root, a reachable rendezvous address, `HCCL_SOCKET_IFNAME` | a 2-node run completes with matching manifests on both node ranks |
| EP64/EP128 scaling | strong/weak scaling workload policy, topology composition and `verified` flag | real fabric and a deployment-owned rank-to-server map | `topology.verified: true` from the deployment, and per-EP `ep_size_comparison.md` rows from measured runs |
| `hccl_test`, DeepEP, SHMEM installs in a container | installer profiles `communication` / `profiling` / `multinode` exist and never install host drivers | the container must already carry CANN; those installs need the toolkit to build | `scripts/install_deps.sh --profile communication` ends with `doctor` exit 0 |

## F. What is *not* blocked

For the avoidance of doubt, these are implemented and verified here, on CPU, with tests and
the plan:

- the case catalog, suites, plan composition, manifest generation and EP-size parity
  (`check_completeness`, `check_parity`);
- capability scheduling per adapter (`status_for`, `check_constraints`) and the coverage
  reports;
- the float32 exact-integer collective payload and its reference arithmetic
  (`baselines/l0/payload.py`, `tests/test_payload.py`);
- the completion vocabulary and its construction-time guard
  (`tests/test_dataplane.py`);
- the evidence ladder and link resolution on synthetic evidence
  (`tests/test_evidence.py`);
- the dependency doctor and the installer's refusal to touch host state
  (`scripts/doctor.py`, `scripts/install_deps.sh`);
- the vendored ascendep provenance check (`scripts/vendor_ascendep.py --check`).

None of the above is a performance result. A CPU run validates framework behaviour; it says
nothing about Ascend performance.
