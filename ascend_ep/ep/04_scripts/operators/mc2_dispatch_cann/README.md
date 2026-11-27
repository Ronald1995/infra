# mc2_dispatch_cann

Native API: `npu_moe_distribute_dispatch_v2 / npu_moe_distribute_combine_v2`. Source commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`. [Source](https://github.com/penn513/ascendep)

- **Adapter** · `historical_verified`. V2 public bindings, including per-token INT8 dispatch (COMM-009). Device kernels may have different names. One L1 invocation is limited to 512 tokens; the upstream L3 adapter chunks larger inputs.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `moe_forward`, `moe_backward`, `router_weight_grad`, `expert_weight_grad`, `dispatch`, `combine`, `quantized_dispatch`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_dispatch_cann.benchmark --config operators/mc2_dispatch_cann/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L2 串行组合 FUSE-001, FUSE-003; L3 完整 MoE MOE-001..003, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: Native resized workspace lifecycle is pending; a single 950DT V2 invocation supports at most 512 tokens per rank; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires chunk_pipeline; operation requires comm_resources; operation requires concurrent ….

- **Correctness contract**: dispatch layout via `validate_dispatch`; COMM-009 uses `MC2V2Backend.validate_quantized_dispatch` (INT8 rows + one scale per route, dequantised before comparison, tolerance 0.015); L3 uses `validate_moe`.
- **Timing contract**: L1 handlers submit and wait inside the timed block and record `local_complete`. A single native invocation is limited to 512 tokens per rank: `MC2V2Backend.exchange` raises `unsupported` above that, and `adapter_token_limit` marks the L1 points `unsupported` from the configuration alone.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`.
- **Source**: vendored `operators/vendor/ascendep/ep/mc2_dispatch_cann/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; `torch_npu` V2 bindings are native.
