# Compute baselines

`HW-010` (GMM) and `HW-011` (SwiGLU) measure one kernel group at a time:

- variants `gmm1` and `gmm2` are separate cases,
- `gmm1_dgrad`/`gmm1_wgrad`/`gmm2_dgrad`/`gmm2_wgrad` are separate cases,
- the router, dispatch and combine are excluded: the expert input comes from a
  local dispatch of the workload, and its cost is not part of the timed section.

Inputs are the workload's token counts, experts, hidden size and dtype, so the
same case can be swept by configuration.