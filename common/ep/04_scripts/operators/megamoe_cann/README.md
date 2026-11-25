# megamoe_cann

原生入口为 `cann_ops_transformer.ops.get_symm_buffer_for_mega_moe` 和 `mega_moe`。参考提交为 `68a67e130417593f37ea3cece1ebfb4be366b95f`。 [CANN A16W16 文档](https://gitcode.com/cann/ops-transformer/blob/68a67e130417593f37ea3cece1ebfb4be366b95f/mc2/mega_moe/README.md)

- 950DT A16W16 路径使用 MTE 拓扑、BF16 和不量化的 Combine。更早的 EP8 参考环境曾拒绝 quant mode 0。该运行时限制会保存为状态和原始日志。
- L3 前向测量整个融合调用及每次必须的路由格式转换。分片堆叠权重布局 `[E_local,2I,H]` 和 `[E_local,H,I]` 在计时外准备。对照使用共享限幅 SwiGLU 和同一逻辑路由。
- 需要固定源码兼容的 CANN、torch_npu 和算子扩展。前向有效 FLOPs 为 `6 × routes × H × I`。内核内部通信和重叠缺少拆分证据时保留空值。
- [注册表](../registry.yaml) · [EP8 历史归档](../../../05_results/ep_benchmark_ep8.md) · [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.megamoe_cann.benchmark --config operators/megamoe_cann/config.yaml --cases MOE-001
```
