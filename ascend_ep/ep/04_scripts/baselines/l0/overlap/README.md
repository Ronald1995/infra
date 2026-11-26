# Communication + compute overlap

`OVERLAP-001..009` measure three phases in one preparation:

1. communication only,
2. compute only,
3. both submitted on two NPU streams.

`HW-012` keeps the original AllToAll-on-a-second-stream behaviour.

Overlap is not inferred from "both tasks were submitted".  The concurrent phase
records per-stream device events and computes the intersection of the two
windows; `overlap_verified` is true only when the events worked and the
intersection exceeds 5 % of the shorter task.  Validation requires both tasks to
produce correct data.

Reported metrics: communication-only, compute-only and concurrent latency,
communication/compute slowdown, overlap ratio and overlap evidence
(`device_events` or `host_timestamps`).

The communication task is a framework AllToAll today.  The transport-specific
variants (`OVERLAP-001..007`) are declared with their mechanism and execution
resource so a native channel can be substituted without changing the case ID;
until that substitution is wired they carry `transport_protocol: auto` and stay
`unverified`.