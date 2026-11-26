# L0 hardware baseline

L0 answers hardware questions, not operator questions: which communication
mechanism, on which execution resource, over which link, at what message size.
Every case records what it **requested** and what it could **observe**, and the
two are never merged.

## The four layers

| Layer | Values | Field |
| --- | --- | --- |
| Operation | put/get, p2p, alltoall, alltoallv, allgather, reducescatter, allreduce, copy | `operation` |
| Execution resource | `AIV`, `AICPU_TS`, `CCU`, `HOST_CPU` | `requested.execution_resource` |
| Mechanism | `ub_memory`, `urma`, `ub_ctp`, `ub_rtp`, `roce`, `hccs`, `uboe`, `shmem_mte`, `shmem_udma`, `auto` | `requested.transport_protocol` |
| Physical topology | HBM, local UB, UB interconnect, HCCS, Ethernet, supernode | `topology_graph`, `observed.physical_link` |

`HCCL_OP_EXPANSION_MODE` and the HCCL accelerator modes select an **execution
resource**, never a transport protocol. `UDMA` is a data-movement engine and is
not `URMA`. `UB_CTP`/`UB_RTP` are `CommProtocol` values, siblings of `UB_MEM`;
they are not URMA subtypes. `urma` in this suite names the URMA **channel
family** (`AivUrmaChannel`, `AicpuTsUrmaChannel`, `CcuUrmaChannel`,
`HostCpuUrmaChannel`), which HCOMM creates for `UB_CTP`/`UBC_TP` (and, on AIV,
`UB_RTP`). The evidence records the family and the protocol separately.

## Families

| Family | Cases | Implementation |
| --- | --- | --- |
| Local memory | `MEM-001..004` | `memory/` (torch copy, native MTE/UB kernels) |
| Native transport | `LINK-001..013` (legacy `HW-007`, `HW-008`) | `transport/`, `shmem/` |
| Communication execution | `EXEC-001..006` | `collectives/native_hccl.py` |
| Collective (framework) | `HW-001..006` | `collectives/torch_hccl.py` |
| Compute | `HW-010`, `HW-011` | `compute/torch_ops.py` |
| Overlap | `OVERLAP-001..009` (legacy `HW-012`) | `overlap/` |

Discovery: `topology/`. Path verification: `verification/`.

## Verification status

Each L0 record carries `verification.status`:

| Status | Meaning |
| --- | --- |
| `verified` | the required fields were established by strong evidence |
| `fallback` | the observed path differs from the requested one |
| `unverified` | no strong evidence; the number is real but not attributable |
| `unsupported` | the device/software cannot run the combination |
| `environment_unavailable` | a dependency or the topology is missing |
| `failed` | the case ran and failed |

Only `verified` rows may be compared across mechanisms. Strong evidence is a
native channel descriptor, an HCCL rank-graph query (`CommLink.linkAttr.linkProtocol`),
a SHMEM runtime query, or a configured structured log pattern. Free-text log
keywords and kernel names are recorded as *heuristic* and never promote a field.

## Running

```sh
python run_benchmark.py capabilities --config configs/hardware.yaml
python run_benchmark.py plan  --config configs/hardware.yaml --cases LINK-001,LINK-013
python run_benchmark.py run   --config configs/hardware.yaml --ep-size 8 --cases MEM-001,LINK-012,LINK-013
python run_benchmark.py run   --config configs/collectives.yaml --ep-size 8 --cases EXEC-001
python run_benchmark.py run   --config configs/overlap.yaml --ep-size 8 --cases OVERLAP-005
python run_benchmark.py report --run-id RUN_ID
```

Reports: `reports/l0_report.md`, `reports/l0_curves.csv`, `reports/l0_metrics.json`,
plus `capabilities.json` in the run root.

## Optional dependencies

Nothing here is mandatory for the rest of the benchmark:

| Dependency | Needed by | Absent result |
| --- | --- | --- |
| CANN SHMEM (`shmem` python package + rendezvous) | `LINK-012`, `LINK-013`, `HW-007`, `HW-008` | `dependency_missing` |
| `libep_l0_udma.so` | device-side UDMA (`LINK-013` device variant) | `dependency_missing` |
| Native transport bridge (`libl0_transport_bridge.so`) | `LINK-001..011` | `dependency_missing` |
| `libl0_mem.so` | `MEM-002`, `MEM-003` | `dependency_missing` |
| CANN oam-tools `hccl_test` | `EXEC-*` | `dependency_missing` |

## Measurement definitions

- `issue_latency` is the host time to submit; `completion_latency` is the wait
  that follows. The generic timer already records `host_issue_ns` and
  `host_wait_ns` separately, and the L0 cases never fold one into the other.
- `payload_bytes` counts logical application bytes. `per_rank_bytes` is one
  copy per destination peer; `global_bytes` sums every rank's fan-out.
  Physical link bytes are only reported when a counter is available.
- `p50/p95/p99` are rank-max percentiles for timer-based cases. The native HCCL
  test tool reports a single average, so its `p95`/`p99` stay `null`.
