# Topology and capability discovery

| Module | Responsibility |
| --- | --- |
| `device_info.py` | SoC, CANN and package versions, library presence, device count |
| `rank_graph.py` | configured rank/serving topology (explicitly *not* hardware discovery) |
| `capabilities.py` | capability matrix: channel, engine, accelerator, memory kernel |
| `discovery.py` | cached entry point used by the scheduler and the report |

Only `probed` entries are `supported`; `declared` entries are hypotheses.  A
probe that fails keeps its reason verbatim, and the scheduler turns that into an
`unsupported` task.

`rank_graph.build()` classifies rank pairs from `topology.server_ids` /
`supernode_ids` and the per-rank host mapping, and labels the result
`source: configured_topology`.  The real link type comes from the HCCL rank graph
at run time and is recorded separately as evidence.

An explicit matrix can replace probing:

```sh
export EP_L0_CAPABILITIES=/path/to/matrix.json
python run_benchmark.py capabilities --matrix /path/to/matrix.json --json
```

Overrides are marked `source: override` everywhere, so a mocked run cannot be
mistaken for a probe.