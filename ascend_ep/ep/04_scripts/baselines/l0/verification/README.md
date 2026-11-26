# Path verification

`requested` records intent.  `observed` records what evidence established.
`path_verifier.verify()` folds evidence into the observation and returns one of
`verified`, `fallback`, `unverified`, `unsupported`, `environment_unavailable`,
`failed`.

Evidence sources:

| Source | Strength | Examples |
| --- | --- | --- |
| `native_descriptor` | strong | channel acquired with an explicit protocol descriptor |
| `rank_graph` | strong | `CommLink.linkAttr.linkProtocol` per peer |
| `shmem_query` / `runtime_api` | strong | engine/route queries |
| `hccl_runtime_log` / `hcomm_log` | strong only for configured structured patterns | `benchmark.verification.trust_descriptor` |
| `hccl_log_keyword` / `hcomm_log_keyword` / `profiler_kernel_name` | heuristic | free-text hits, kernel names |

A heuristic claim is recorded but never promotes a field.  Configuration is never
evidence.  A strong claim that contradicts the request produces `fallback`.

`profiler.collect()` reads `kernel_details.csv` from the existing profiling
pipeline and only reports kernels whose names name a mechanism.