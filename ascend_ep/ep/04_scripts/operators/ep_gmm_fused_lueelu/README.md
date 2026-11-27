# ep_gmm_fused_lueelu

原生入口为 `deep_ep_experimental.ep_gmm_fused`。参考提交为 `24ff84d3ab8c01e1e2071802465173e624e1f89d`。 [源码](https://github.com/penn513/ascendep/tree/24ff84d3ab8c01e1e2071802465173e624e1f89d)

- **接入与验证** · `historical_verified`。参考报告覆盖 950DT EP8。当前框架运行时重新验证正确性和性能。
- **布局和范围** · 参考适配配置为 BF16、EP2/EP8、24 个专家、Top-6、各 rank 等长。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L2/L3。FUSE-006 通过原生 `tiling` 参数变更 `m0/n0/k0`，FUSE-007 变更 `self_copy_cores`。扫描使用 [Lueelu 固定源码](https://github.com/lueelu/DeepEP/tree/83252a54d2363734a706a7dd8368bebc3094af5a/experiments/ep_gmm_fused)。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · 按上游模块 README 安装依赖并设置 `EP_ASCENDEP_ROOT`。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.ep_gmm_fused_lueelu.benchmark --config operators/ep_gmm_fused_lueelu/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L2 原生融合 FUSE-002, FUSE-004, FUSE-006, FUSE-007, FUSE-009; L3 完整 MoE MOE-001..003, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-005, FUSE-008; L3 完整 MoE MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: Native resized workspace lifecycle is pending; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline ….

- **Correctness contract**: FUSE-006/FUSE-007 compare the tuned dispatch+GMM against the reference GMM-1 (relative L2); FUSE-002/004 use the fused GMM pair, FUSE-009 the fused backward; L3 uses `validate_moe`. A tiling the native `validate_tiling` rejects becomes `unsupported`, not a silent fallback.
- **Timing contract**: FUSE-006/FUSE-007 build the tiling outside the timer and run one dispatch inside it (`benchmarks/cases/l2_fusion.py::_tuned`); the tiling variant is the measurement, so the variants are not repeats.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `top_k=6`, `num_experts=24`, `equal_tokens`, `contiguous_placement`, `swiglu_limit=10`.
- **Source**: vendored `operators/vendor/ascendep/ep/ep_gmm_fused_lueelu/demo.py` at commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`; `deep_ep_experimental` (including `_tiling`) is a native dependency.
