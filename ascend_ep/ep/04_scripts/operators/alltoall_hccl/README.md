# alltoall_hccl

原生入口为 `torch.distributed.all_to_all_single`。使用环境中实际安装的源码或二进制版本。 [源码](https://github.com/Ascend/pytorch)

- **接入与验证** · `integrated`。支持 NPU/HCCL。CPU/Gloo 路径用于验证公共逻辑。
- **布局和范围** · 支持非等长源 token 和指定专家归属。输入含 `[T,H]`、`[T,K]` 和本地专家权重，专家前向输出为 `[T,H]`。
- **层次与能力** · L1/L2 serial/L3/L4。`moe_forward`、`moe_backward`、`router_weight_grad`、`expert_weight_grad`、`dispatch`、`combine`、`router_layer`、`recompute`、`cached_dispatch`。 [注册表](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **依赖** · PyTorch。NPU 另需 torch_npu 和 HCCL。
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.alltoall_hccl.benchmark --config operators/alltoall_hccl/config.yaml
```

## Benchmark contract

- **Cases it can run**: L1 COMM-001..COMM-008, COMM-010; L2 FUSE-001, FUSE-003; L3 LAYER-007, MOE-001..MOE-009; L4 LAYER-001, LAYER-002, LAYER-009.
- **Cases it cannot run**: COMM-009 (`quantized_dispatch`), COMM-011 (`combine_add_rmsnorm`), COMM-012 (`planned_dispatch`); FUSE-002/004/005/006/007/008/009 (`dispatch_gmm`/`gmm_combine`/`chunk_pipeline`/`tile_schedule`/`comm_resources`/`quantized_fusion`/`fused_backward`); LAYER-003/004/008 (`dynamic_planner`), LAYER-005 (`rank_dedup`+`dedup_training`), LAYER-006 (`server_dedup`+`dedup_training`). They remain in the manifest as `unsupported` with the missing token as the reason.
- **Fixed owner**: `LAYER-003`, `LAYER-005` and `LAYER-006` name this adapter as their owner (`common/case_spec.py::_CASE_OWNER`). Those points run with a single `fixed_owner` parameter set here instead of the case's own variants, measured against the relaxed plain-MoE contract (`common/scheduler.py::_contract`).
- **Correctness contract**: dispatch layout via `validate_dispatch`; combine and MoE output/gradients via relative L2 (`common/validation.py`).
- **Timing contract**: L1 handlers submit and wait inside the timed block; `result["completion"]["level"]` is `remote_visible` from `CollectiveBackend.COMPLETION` ("the collective returns after the operation completed on this rank"). Traffic model `collective_alltoallv`; L1 bandwidth is a routing-trace model.
- **Registry constraints**: `devices=[npu,cpu]` only; the CPU/Gloo path is a functional stand-in recorded by `common/runner.py::_fallback_reason`.
