# alltoall_fsdpturbo

原生入口为 `all2all_experts_forward_fn (fused=False)`。参考提交为 `24ff84d3ab8c01e1e2071802465173e624e1f89d`。 [源码](https://github.com/penn513/ascendep/tree/24ff84d3ab8c01e1e2071802465173e624e1f89d)

- **接入与验证** · `historical_verified`。参考报告覆盖 950DT EP8。当前框架运行时重新验证正确性和性能。
- **布局和范围** · 参考适配配置为 BF16、EP2/EP8、24 个专家、Top-6、各 rank 等长。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L3。`moe_forward`、`moe_backward`、`router_weight_grad`、`expert_weight_grad`。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.alltoall_fsdpturbo.benchmark --config operators/alltoall_fsdpturbo/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L3 完整 MoE MOE-001..003, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: Native resized workspace lifecycle is pending; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline ….

- **Correctness contract**: `validate_moe` over the forward output and (for a training case) the input/expert weight gradients; below 2M elements the reference is the CPU FP32 oracle.
- **Timing contract**: `moe_case`; one timed call, `wait` is the identity, the sample closes after `ctx.synchronize()`.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`; EP64/EP128 points are `unsupported`.
- **Source**: vendored `operators/vendor/ascendep/ep/alltoall_fsdpturbo/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; no external checkout, `EP_ASCENDEP_ROOT` is an optional override.
