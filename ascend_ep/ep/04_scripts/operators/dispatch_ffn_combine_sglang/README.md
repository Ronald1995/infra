# dispatch_ffn_combine_sglang

原生入口为 sgl-kernel-npu `deep_ep.Buffer.fused_deep_moe`，选择 `FuseMode.DISPATCH_FFN_COMBINE` 和 W8A8。参考提交 `458e72caf6cdfb5b9c06711e6f84d134d15e3f97`。 [原生测试](https://github.com/sgl-project/sgl-kernel-npu/blob/458e72caf6cdfb5b9c06711e6f84d134d15e3f97/tests/python/deepep/test_dispatch_ffn_combine.py)

- Atlas A3 L3 推理前向。输入 BF16 `[T,H]`、Top-K 路由和 FP32 概率。各 rank 等长，EP2/EP8 配置。INT8 权重按输出通道量化到 NZ 格式，FP32 scale 使用 INT64 编码传入。
- 静态权重量化在计时外。测量归并完成的融合前向，包含必要的路由转换和动态激活量化。参考路径使用对应 INT8 量化及 FP32 矩阵乘。有效专家前向 FLOPs 为 `6 × routes × H × I`。
- 安装固定版本的 sgl-kernel-npu DeepEP 扩展和匹配 torch_npu。框架尚无设备验证结果。
- [注册表](../registry.yaml) · [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.dispatch_ffn_combine_sglang.benchmark --config configs/decode.yaml --cases MOE-001
```

## Benchmark contract

- **Cases it can run** (capability-level): L3 完整 MoE MOE-001, MOE-005, MOE-006, MOE-008.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-002, MOE-003, MOE-004, MOE-007, MOE-009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine, combine, dispatch; operation requires async_dispatch, dispatch; operation requires cached_dispatch, dispatch; operation requires chunk_pipeline; operation requires combine, dispatch ….

- **Correctness contract**: `validate_moe` over the forward output only; the registry records `numeric_contract: w8a8_integer`, so the result is not comparable with a BF16 case.
- **Timing contract**: `moe_case`; the fused `FuseMode.DISPATCH_FFN_COMBINE` call is one timed block, weight quantisation stays outside it.
- **Registry constraints**: `ep_sizes=[2,8]`, `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `hardware=[Atlas A3]`.
- **Source**: upstream sgl-kernel-npu at commit `458e72caf6cdfb5b9c06711e6f84d134d15e3f97`; not ascendep-vendored.
