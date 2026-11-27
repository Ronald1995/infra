# megamoe_cann

原生入口为 `cann_ops_transformer.ops.get_symm_buffer_for_mega_moe` 和 `mega_moe`。参考提交为 `68a67e130417593f37ea3cece1ebfb4be366b95f`。 [CANN A16W16 文档](https://gitcode.com/cann/ops-transformer/blob/68a67e130417593f37ea3cece1ebfb4be366b95f/mc2/mega_moe/README.md)

- 950DT A16W16 路径使用 MTE 拓扑、BF16 和不量化的 Combine。更早的 EP8 参考环境曾拒绝 quant mode 0。该运行时限制会保存为状态和原始日志。
- L3 前向测量整个融合调用及每次必须的路由格式转换。分片堆叠权重布局 `[E_local,2I,H]` 和 `[E_local,H,I]` 在计时外准备。对照使用共享限幅 SwiGLU 和同一逻辑路由。
- 需要固定源码兼容的 CANN、torch_npu 和算子扩展。前向有效 FLOPs 为 `6 × routes × H × I`。内核内部通信和重叠缺少拆分证据时保留空值。
- [注册表](../registry.yaml) · [EP8 历史归档](../../../05_results/ep_benchmark_ep8.md) · [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.megamoe_cann.benchmark --config operators/megamoe_cann/config.yaml --cases MOE-001
```

## Benchmark contract

- **Cases it can run** (capability-level): L3 完整 MoE MOE-001, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-002, MOE-003, MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: Symmetric window resize and reuse are pending; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline ….

- **Correctness contract**: `validate_moe` over the forward output; the registry declares only `moe_forward`, so no gradient case is claimed for this adapter.
- **Timing contract**: `moe_case`; one fused forward inside the timed block, the stacked `[E_local,2I,H]` / `[E_local,H,I]` weight preparation stays outside it.
- **Registry constraints**: `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `min_tokens=1`; no `ep_sizes`/`top_k` pin, and `dynamic_shapes` is recorded as unadapted.
- **Source**: CANN ops-transformer at commit `68a67e130417593f37ea3cece1ebfb4be366b95f`; not ascendep-vendored, and no device execution record exists.
