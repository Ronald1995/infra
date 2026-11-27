# allgather_hccl

原生入口为 `torch.distributed.all_gather / reduce_scatter_tensor`。使用环境中实际安装的源码或二进制版本。 [源码](https://github.com/Ascend/pytorch)

- **接入与验证** · `integrated`。支持 NPU/HCCL。CPU/Gloo 路径用于验证公共逻辑。
- **布局和范围** · 支持非等长源 token 和指定专家归属。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L1/L2 serial/L3/L4。`moe_forward`、`moe_backward`、`router_weight_grad`、`expert_weight_grad`、`dispatch`、`combine`、`router_layer`、`recompute`。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · PyTorch。NPU 另需 torch_npu 和 HCCL。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.allgather_hccl.benchmark --config operators/allgather_hccl/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 独立通信 COMM-001, COMM-003, COMM-004, COMM-007, COMM-008; L2 串行组合 FUSE-001, FUSE-003; L3 完整 MoE MOE-001..009; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-009.
- **Conditional cases** (runnable only under a matching configuration): LAYER-007 (L3 完整 MoE): only when workload={'routing': 'real_trace_replay'}.
- **Cases it cannot run**: L1 独立通信 COMM-002, COMM-005, COMM-006, COMM-009, COMM-010, COMM-011, COMM-012; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L4 路由器/规划/去重 LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008.
- **Why not**: deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; operation requires async_combine; operation requires async_dispatch; operation requires cached_dispatch; operation requires chunk_pipeline; operation requires combine_add_rmsnorm ….

- **Correctness contract**: dispatch layout via `validate_dispatch` (expert counts, expert ids, inverse route slots against the reference collective); combine output and MoE output/gradients via relative L2 (`common/validation.py`).
- **Timing contract**: L1 handlers submit and wait inside the timed block (`benchmarks/cases/common.py::dispatch`/`combine`); `result["completion"]["level"]` is `remote_visible` from `CollectiveBackend.COMPLETION` ("the collective returns after the operation completed on this rank"), which still requires the payload check — it is not a wire-level proof.
- **Traffic model**: `collective_allgather`; L1 bandwidth is a routing-trace model (`routing_trace_model`, `declared`), never a wire measurement.
- **Registry constraints**: `devices=[npu,cpu]` only; the CPU/Gloo path is a functional stand-in and is recorded as a fallback by `common/runner.py::_fallback_reason`.
