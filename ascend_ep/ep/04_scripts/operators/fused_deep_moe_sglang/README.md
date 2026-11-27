# fused_deep_moe_sglang

原生入口为 sgl-kernel-npu `deep_ep.Buffer.fused_deep_moe`，选择 `FuseMode.FUSED_DEEP_MOE`。参考提交 `458e72caf6cdfb5b9c06711e6f84d134d15e3f97`。 [A5 原生测试](https://github.com/sgl-project/sgl-kernel-npu/blob/458e72caf6cdfb5b9c06711e6f84d134d15e3f97/tests/python/deepep/test_fused_deep_moe_a5.py)

- Ascend 950 L3 MXFP8 推理前向。输入 BF16 `[T,H]` 和 FP32 路由概率。已接入专家分别持有 MXFP8 权重 `[E_local,H,2I]`、`[E_local,I,H]` 与 E8M0 scales。
- 静态权重量化在计时外。前向包含路由转换、动态量化和完整融合算子。参考按 MX 分组反量化并执行独立专家数学运算。有效 FLOPs 按 `6 × routes × H × I` 统计。
- 安装固定版本的 sgl-kernel-npu DeepEP 扩展、具有 E8M0 类型支持的 PyTorch 和匹配 torch_npu。框架尚无设备验证结果。
- [注册表](../registry.yaml) · [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.fused_deep_moe_sglang.benchmark --config configs/prefill.yaml --cases MOE-001
```

## Benchmark contract

- **Cases it can run** (capability-level): L3 完整 MoE MOE-001, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-002, MOE-003, MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline; operation requires combine, dispatch ….

- **Correctness contract**: `validate_moe` over the forward output only; the registry records `numeric_contract: mxfp8_activations_and_weights` and `hardware=[Ascend 950]`, so a non-MXFP8 or non-950 comparison is not valid.
- **Timing contract**: `moe_case`; the fused `FuseMode.FUSED_DEEP_MOE` call is one timed block, static weight quantisation stays outside it.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `min_tokens=1`.
- **Source**: upstream sgl-kernel-npu at commit `458e72caf6cdfb5b9c06711e6f84d134d15e3f97`; not ascendep-vendored, and no device validation exists in this tree.
