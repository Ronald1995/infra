# dispatch_ffn_combine_vllm_ascend

Native API: `torch.ops._C_ascend.dispatch_ffn_combine`. Source commit `0f63cfa1a52225f301ffd4a335b8daa3028c283f`. [Source](https://github.com/vllm-project/vllm-ascend)

- **Adapter** · `integrated`. The pinned BF16 custom op is built for Atlas A3. It consumes NZ weights. Validation checks that the selected workload stays within the shared SwiGLU clipping limit.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L3. `moe_forward`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.dispatch_ffn_combine_vllm_ascend.benchmark --config operators/dispatch_ffn_combine_vllm_ascend/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L3 完整 MoE MOE-001, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-002, MOE-003, MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline; operation requires combine, dispatch ….

- **Correctness contract**: `validate_moe` over the forward output; the registry constrains `hardware=[Atlas A3 (ascend910_93)]`, so on any other SoC the point is `unsupported` and no correctness claim is made.
- **Timing contract**: `moe_case`; the fused custom op runs inside one timed call and the sample closes after `ctx.synchronize()`.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `min_tokens=1`.
- **Source**: upstream vllm-ascend at commit `0f63cfa1a52225f301ffd4a335b8daa3028c283f`; not ascendep-vendored, so the torch extension must come from `dependencies/manifest.yaml`.
