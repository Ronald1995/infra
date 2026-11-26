# dispatch_ffn_combine_sglang

原生入口为 sgl-kernel-npu `deep_ep.Buffer.fused_deep_moe`，选择 `FuseMode.DISPATCH_FFN_COMBINE` 和 W8A8。参考提交 `458e72caf6cdfb5b9c06711e6f84d134d15e3f97`。 [原生测试](https://github.com/sgl-project/sgl-kernel-npu/blob/458e72caf6cdfb5b9c06711e6f84d134d15e3f97/tests/python/deepep/test_dispatch_ffn_combine.py)

- Atlas A3 L3 推理前向。输入 BF16 `[T,H]`、Top-K 路由和 FP32 概率。各 rank 等长，EP2/EP8 配置。INT8 权重按输出通道量化到 NZ 格式，FP32 scale 使用 INT64 编码传入。
- 静态权重量化在计时外。测量归并完成的融合前向，包含必要的路由转换和动态激活量化。参考路径使用对应 INT8 量化及 FP32 矩阵乘。有效专家前向 FLOPs 为 `6 × routes × H × I`。
- 安装固定版本的 sgl-kernel-npu DeepEP 扩展和匹配 torch_npu。框架尚无设备验证结果。
- [注册表](../registry.yaml) · [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.dispatch_ffn_combine_sglang.benchmark --config configs/decode.yaml --cases MOE-001
```
