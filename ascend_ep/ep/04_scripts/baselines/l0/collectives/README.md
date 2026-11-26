# Collective baselines

Two backends, one result schema.

| Backend | Cases | Measures |
| --- | --- | --- |
| `torch_hccl` | `HW-001..006` | what a framework actually gets from HCCL |
| `native_hccl` | `EXEC-001..006` | the native HCCL test tool, swept over accelerator modes |

The native backend runs CANN oam-tools `hccl_test` (source: `src/hccl_test`,
built on the target).  Binary per operation: `alltoall_test`, `alltoallv_test`,
`all_gather_test`, `reduce_scatter_test`, `all_reduce_test`.  There is **no P2P
test binary**, so `EXEC-006` reports `dependency_missing`.

Options used: `-b/-e` message size, `-i 1`, `-n` iterations, `-w` warmup,
`-c 1` (verify), `-p` ranks, `-a <accelerator>` for the requested accelerator.
Mode names are read from the installed binary's `--help`; only names it prints
are offered, so a plan cannot request a mode the local tool rejects.

Output parsing keys on the tool's table
(`data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:`).
The tool reports a single average per size, so `p95`/`p99` are `null`, and the
`check_result` column is the correctness check.

The accelerator selects an **execution resource**, not a transport protocol.  The
tool prints nothing about the path it took, so `observed.accelerator` stays
`unverified` unless an HCCL log provides evidence.  When
`HCCL_OP_EXPANSION_MODE` is unset, recent releases default the tool to
`ccu_sched` on new-generation SoCs; the ambient value is recorded in the
capability matrix and never assumed neutral.

Verification log support: set `benchmark.verification.hccl_log_paths`.  Keyword
hits are recorded as heuristic claims and cannot promote a field to `verified`.