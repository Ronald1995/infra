# EP 算子能力审计（自动生成）

由 `python scripts/capability_audit.py` 生成。数据来自 `operators/registry.yaml` 的能力声明、`common/case_spec.py` 的 Case 定义，以及 `common/scheduler.py` 在 `configs/ep8_bf16.yaml` 下的实际判定。

生成说明：`conditional` 表示参考配置未选中该场景、但换成匹配配置即可运行；`unsupported` 表示能力或约束不满足；`not_implemented` 表示已识别但未实现。

## 1. 能力声明与层次覆盖

| Operator | 声明能力 | L1 | L2 Serial | L2 Native | L3 | L4 | L0 底层映射 |
|---|---|---|---|---|---|---|---|
| `allgather_fsdpturbo` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `allgather_hccl` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch, combine, router_layer, recompute, dynamic_shapes, concurrent | COMM-001, COMM-003, COMM-004, COMM-007, COMM-008 | FUSE-001, FUSE-003 | — | MOE-001..009（条件 LAYER-007） | LAYER-001, LAYER-002, LAYER-009 | unknown（registry 未声明底层原语，需按源码补齐） |
| `allgather_megatron` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `allgather_overlap_mindspeed` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `alltoall_fsdpturbo` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `alltoall_hccl` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch, combine, router_layer, recompute, cached_dispatch, async_dispatch, async_combine, dynamic_shapes, concurrent | COMM-001..008 | FUSE-001, FUSE-003 | — | MOE-001..009（条件 LAYER-007） | LAYER-001, LAYER-002, LAYER-009（条件 LAYER-003, LAYER-005, LAYER-006） | unknown（registry 未声明底层原语，需按源码补齐） |
| `alltoall_megatron` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `alltoallv_gmm_cann` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch_gmm, gmm_combine, fused_backward | — | — | FUSE-002, FUSE-004, FUSE-009 | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `alltoallv_quant_gmm_cann` | quantized_fusion | — | — | FUSE-008 | — | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `chunkmoe_fsdpturbo` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, chunk_pipeline | — | — | FUSE-005 | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `deepep_ascend` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch, combine, cached_dispatch | COMM-001..004, COMM-007, COMM-008 | FUSE-001, FUSE-003 | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `deepep_cann` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-006, MOE-008 | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `deepep_direct_lueelu` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch, combine, cached_dispatch, async_dispatch, async_combine | COMM-001..008 | FUSE-001, FUSE-003 | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `deepep_lueelu` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch, combine, cached_dispatch, async_dispatch, async_combine | COMM-001..008 | FUSE-001, FUSE-003 | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `deepep_sglang` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `deepep_v2_ascend` | dispatch, combine, cached_dispatch, async_dispatch, async_combine, moe_forward | — | — | — | — | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `dispatch_ffn_combine_sglang` | moe_forward | — | — | — | MOE-001, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `dispatch_ffn_combine_vllm_ascend` | moe_forward | — | — | — | MOE-001, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `domino_fsdpturbo` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `ep_gmm_fused_lueelu` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch_gmm, gmm_combine, fused_backward, tile_schedule, comm_resources | — | — | FUSE-002, FUSE-004, FUSE-006, FUSE-007, FUSE-009 | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `fused_deep_moe_sglang` | moe_forward | — | — | — | MOE-001, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `fused_fsdpturbo` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `hierarchical_alltoallv_mindspeed` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, router_layer | — | — | — | — | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `mc2_combine_add_rmsnorm_cann` | dispatch, combine_add_rmsnorm | — | — | — | — | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `mc2_dispatch_cann` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad, dispatch, combine, quantized_dispatch | — | FUSE-001, FUSE-003 | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `mc2_dispatch_cann_v1` | dispatch, combine, moe_forward | — | — | — | — | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `mc2_dispatch_cann_v3` | dispatch, combine, moe_forward, quantized_dispatch | — | — | — | — | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `mc2_fsdpturbo` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `megamoe_cann` | moe_forward | — | — | — | MOE-001, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `megamoe_deepgemm_ascend` | moe_forward | — | — | — | MOE-001, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `megamoe_fsdpturbo` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-005, MOE-006, MOE-008（条件 LAYER-007） | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `megamoe_lueelu` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-006, MOE-008 | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `megamoe_td` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-006, MOE-008 | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `moonep` | moe_forward, moe_backward, router_weight_grad, expert_weight_grad | — | — | — | MOE-001..003, MOE-006, MOE-008 | — | unknown（registry 未声明底层原语，需按源码补齐） |
| `moonep_dynamic_planner` | planned_dispatch, dynamic_planner, router_weight_grad, expert_weight_grad | （条件 COMM-012） | — | — | — | （条件 LAYER-003, LAYER-004, LAYER-008） | unknown（registry 未声明底层原语，需按源码补齐） |
| `moonep_server_dedup` | rank_dedup, server_dedup, dedup_training | （条件 COMM-010） | — | — | — | （条件 LAYER-005, LAYER-006） | unknown（registry 未声明底层原语，需按源码补齐） |

## 2. 逐 Case 判定

| Operator | Case | Level | Requires | Setup | Conditional | Support |
|---|---|---|---|---|---|---|
| `allgather_fsdpturbo` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_fsdpturbo` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `allgather_fsdpturbo` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_fsdpturbo` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_fsdpturbo` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `allgather_fsdpturbo` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `allgather_fsdpturbo` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_fsdpturbo` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_fsdpturbo` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `allgather_fsdpturbo` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `allgather_fsdpturbo` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `allgather_fsdpturbo` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_fsdpturbo` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_fsdpturbo` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `allgather_fsdpturbo` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_fsdpturbo` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `allgather_fsdpturbo` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `allgather_fsdpturbo` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `allgather_fsdpturbo` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `allgather_fsdpturbo` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `allgather_fsdpturbo` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `allgather_fsdpturbo` | MOE-001 | L3 | moe_forward | — | — | yes |
| `allgather_fsdpturbo` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `allgather_fsdpturbo` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `allgather_fsdpturbo` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `allgather_fsdpturbo` | MOE-005 | L3 | moe_forward | — | — | yes |
| `allgather_fsdpturbo` | MOE-006 | L3 | moe_forward | — | — | yes |
| `allgather_fsdpturbo` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `allgather_fsdpturbo` | MOE-008 | L3 | moe_forward | — | — | yes |
| `allgather_fsdpturbo` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `allgather_fsdpturbo` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `allgather_fsdpturbo` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `allgather_fsdpturbo` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_fsdpturbo` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_fsdpturbo` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `allgather_fsdpturbo` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `allgather_fsdpturbo` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `allgather_fsdpturbo` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_fsdpturbo` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `allgather_hccl` | COMM-001 | L1 | dispatch | — | — | yes |
| `allgather_hccl` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch） |
| `allgather_hccl` | COMM-003 | L1 | dispatch, combine | — | — | yes |
| `allgather_hccl` | COMM-004 | L1 | dispatch, combine | — | — | yes |
| `allgather_hccl` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch） |
| `allgather_hccl` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine） |
| `allgather_hccl` | COMM-007 | L1 | dispatch | — | — | yes |
| `allgather_hccl` | COMM-008 | L1 | dispatch | — | — | yes |
| `allgather_hccl` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires quantized_dispatch） |
| `allgather_hccl` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `allgather_hccl` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `allgather_hccl` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_hccl` | FUSE-001 | L2 | dispatch | — | — | yes |
| `allgather_hccl` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `allgather_hccl` | FUSE-003 | L2 | dispatch, combine | — | — | yes |
| `allgather_hccl` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `allgather_hccl` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `allgather_hccl` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `allgather_hccl` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `allgather_hccl` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `allgather_hccl` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `allgather_hccl` | MOE-001 | L3 | moe_forward | — | — | yes |
| `allgather_hccl` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `allgather_hccl` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `allgather_hccl` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | yes |
| `allgather_hccl` | MOE-005 | L3 | moe_forward | — | — | yes |
| `allgather_hccl` | MOE-006 | L3 | moe_forward | — | — | yes |
| `allgather_hccl` | MOE-007 | L3 | concurrent | — | — | yes |
| `allgather_hccl` | MOE-008 | L3 | moe_forward | — | — | yes |
| `allgather_hccl` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | yes |
| `allgather_hccl` | LAYER-001 | L4 | router_layer | — | — | yes |
| `allgather_hccl` | LAYER-002 | L4 | router_layer, moe_backward | — | — | yes |
| `allgather_hccl` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_hccl` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_hccl` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `allgather_hccl` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `allgather_hccl` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `allgather_hccl` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_hccl` | LAYER-009 | L4 | router_layer, moe_backward | — | — | yes |
| `allgather_megatron` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_megatron` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `allgather_megatron` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_megatron` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_megatron` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `allgather_megatron` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `allgather_megatron` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_megatron` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_megatron` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `allgather_megatron` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `allgather_megatron` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `allgather_megatron` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_megatron` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_megatron` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `allgather_megatron` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_megatron` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `allgather_megatron` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `allgather_megatron` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `allgather_megatron` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `allgather_megatron` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `allgather_megatron` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `allgather_megatron` | MOE-001 | L3 | moe_forward | — | — | yes |
| `allgather_megatron` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `allgather_megatron` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `allgather_megatron` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `allgather_megatron` | MOE-005 | L3 | moe_forward | — | — | yes |
| `allgather_megatron` | MOE-006 | L3 | moe_forward | — | — | yes |
| `allgather_megatron` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `allgather_megatron` | MOE-008 | L3 | moe_forward | — | — | yes |
| `allgather_megatron` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `allgather_megatron` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `allgather_megatron` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `allgather_megatron` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_megatron` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_megatron` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `allgather_megatron` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `allgather_megatron` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `allgather_megatron` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_megatron` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `allgather_overlap_mindspeed` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_overlap_mindspeed` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `allgather_overlap_mindspeed` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_overlap_mindspeed` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_overlap_mindspeed` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `allgather_overlap_mindspeed` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `allgather_overlap_mindspeed` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_overlap_mindspeed` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_overlap_mindspeed` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `allgather_overlap_mindspeed` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `allgather_overlap_mindspeed` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `allgather_overlap_mindspeed` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_overlap_mindspeed` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `allgather_overlap_mindspeed` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `allgather_overlap_mindspeed` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `allgather_overlap_mindspeed` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `allgather_overlap_mindspeed` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `allgather_overlap_mindspeed` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `allgather_overlap_mindspeed` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `allgather_overlap_mindspeed` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `allgather_overlap_mindspeed` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `allgather_overlap_mindspeed` | MOE-001 | L3 | moe_forward | — | — | yes |
| `allgather_overlap_mindspeed` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `allgather_overlap_mindspeed` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `allgather_overlap_mindspeed` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `allgather_overlap_mindspeed` | MOE-005 | L3 | moe_forward | — | — | yes |
| `allgather_overlap_mindspeed` | MOE-006 | L3 | moe_forward | — | — | yes |
| `allgather_overlap_mindspeed` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `allgather_overlap_mindspeed` | MOE-008 | L3 | moe_forward | — | — | yes |
| `allgather_overlap_mindspeed` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `allgather_overlap_mindspeed` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `allgather_overlap_mindspeed` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `allgather_overlap_mindspeed` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_overlap_mindspeed` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_overlap_mindspeed` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `allgather_overlap_mindspeed` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `allgather_overlap_mindspeed` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `allgather_overlap_mindspeed` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `allgather_overlap_mindspeed` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `alltoall_fsdpturbo` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_fsdpturbo` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `alltoall_fsdpturbo` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoall_fsdpturbo` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoall_fsdpturbo` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `alltoall_fsdpturbo` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `alltoall_fsdpturbo` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_fsdpturbo` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_fsdpturbo` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `alltoall_fsdpturbo` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `alltoall_fsdpturbo` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `alltoall_fsdpturbo` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_fsdpturbo` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_fsdpturbo` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `alltoall_fsdpturbo` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoall_fsdpturbo` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `alltoall_fsdpturbo` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `alltoall_fsdpturbo` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `alltoall_fsdpturbo` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `alltoall_fsdpturbo` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `alltoall_fsdpturbo` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `alltoall_fsdpturbo` | MOE-001 | L3 | moe_forward | — | — | yes |
| `alltoall_fsdpturbo` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `alltoall_fsdpturbo` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `alltoall_fsdpturbo` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `alltoall_fsdpturbo` | MOE-005 | L3 | moe_forward | — | — | yes |
| `alltoall_fsdpturbo` | MOE-006 | L3 | moe_forward | — | — | yes |
| `alltoall_fsdpturbo` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `alltoall_fsdpturbo` | MOE-008 | L3 | moe_forward | — | — | yes |
| `alltoall_fsdpturbo` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `alltoall_fsdpturbo` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `alltoall_fsdpturbo` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `alltoall_fsdpturbo` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_fsdpturbo` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_fsdpturbo` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `alltoall_fsdpturbo` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `alltoall_fsdpturbo` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `alltoall_fsdpturbo` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_fsdpturbo` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `alltoall_hccl` | COMM-001 | L1 | dispatch | — | — | yes |
| `alltoall_hccl` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | yes |
| `alltoall_hccl` | COMM-003 | L1 | dispatch, combine | — | — | yes |
| `alltoall_hccl` | COMM-004 | L1 | dispatch, combine | — | — | yes |
| `alltoall_hccl` | COMM-005 | L1 | dispatch, async_dispatch | — | — | yes |
| `alltoall_hccl` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | yes |
| `alltoall_hccl` | COMM-007 | L1 | dispatch | — | — | yes |
| `alltoall_hccl` | COMM-008 | L1 | dispatch | — | — | yes |
| `alltoall_hccl` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires quantized_dispatch） |
| `alltoall_hccl` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `alltoall_hccl` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `alltoall_hccl` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_hccl` | FUSE-001 | L2 | dispatch | — | — | yes |
| `alltoall_hccl` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `alltoall_hccl` | FUSE-003 | L2 | dispatch, combine | — | — | yes |
| `alltoall_hccl` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `alltoall_hccl` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `alltoall_hccl` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `alltoall_hccl` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `alltoall_hccl` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `alltoall_hccl` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `alltoall_hccl` | MOE-001 | L3 | moe_forward | — | — | yes |
| `alltoall_hccl` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `alltoall_hccl` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `alltoall_hccl` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | yes |
| `alltoall_hccl` | MOE-005 | L3 | moe_forward | — | — | yes |
| `alltoall_hccl` | MOE-006 | L3 | moe_forward | — | — | yes |
| `alltoall_hccl` | MOE-007 | L3 | concurrent | — | — | yes |
| `alltoall_hccl` | MOE-008 | L3 | moe_forward | — | — | yes |
| `alltoall_hccl` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | yes |
| `alltoall_hccl` | LAYER-001 | L4 | router_layer | — | — | yes |
| `alltoall_hccl` | LAYER-002 | L4 | router_layer, moe_backward | — | — | yes |
| `alltoall_hccl` | LAYER-003 | L4 | dynamic_planner | — | — | conditional（only when load_balance={'enabled': True}） |
| `alltoall_hccl` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_hccl` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | conditional（only when dedup={'mode': 'rank'}） |
| `alltoall_hccl` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | conditional（only when dedup={'mode': 'server'}） |
| `alltoall_hccl` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `alltoall_hccl` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_hccl` | LAYER-009 | L4 | router_layer, moe_backward | — | — | yes |
| `alltoall_megatron` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_megatron` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `alltoall_megatron` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoall_megatron` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoall_megatron` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `alltoall_megatron` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `alltoall_megatron` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_megatron` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_megatron` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `alltoall_megatron` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `alltoall_megatron` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `alltoall_megatron` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_megatron` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `alltoall_megatron` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `alltoall_megatron` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoall_megatron` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `alltoall_megatron` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `alltoall_megatron` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `alltoall_megatron` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `alltoall_megatron` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `alltoall_megatron` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `alltoall_megatron` | MOE-001 | L3 | moe_forward | — | — | yes |
| `alltoall_megatron` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `alltoall_megatron` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `alltoall_megatron` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `alltoall_megatron` | MOE-005 | L3 | moe_forward | — | — | yes |
| `alltoall_megatron` | MOE-006 | L3 | moe_forward | — | — | yes |
| `alltoall_megatron` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `alltoall_megatron` | MOE-008 | L3 | moe_forward | — | — | yes |
| `alltoall_megatron` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `alltoall_megatron` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `alltoall_megatron` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `alltoall_megatron` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_megatron` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_megatron` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `alltoall_megatron` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `alltoall_megatron` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `alltoall_megatron` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoall_megatron` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `alltoallv_gmm_cann` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_gmm_cann` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `alltoallv_gmm_cann` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoallv_gmm_cann` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoallv_gmm_cann` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `alltoallv_gmm_cann` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `alltoallv_gmm_cann` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_gmm_cann` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_gmm_cann` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `alltoallv_gmm_cann` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `alltoallv_gmm_cann` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `alltoallv_gmm_cann` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_gmm_cann` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_gmm_cann` | FUSE-002 | L2 | dispatch_gmm | — | — | yes |
| `alltoallv_gmm_cann` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoallv_gmm_cann` | FUSE-004 | L2 | gmm_combine | — | — | yes |
| `alltoallv_gmm_cann` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `alltoallv_gmm_cann` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `alltoallv_gmm_cann` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `alltoallv_gmm_cann` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `alltoallv_gmm_cann` | FUSE-009 | L2 | fused_backward | — | — | yes |
| `alltoallv_gmm_cann` | MOE-001 | L3 | moe_forward | — | — | yes |
| `alltoallv_gmm_cann` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `alltoallv_gmm_cann` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `alltoallv_gmm_cann` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `alltoallv_gmm_cann` | MOE-005 | L3 | moe_forward | — | — | yes |
| `alltoallv_gmm_cann` | MOE-006 | L3 | moe_forward | — | — | yes |
| `alltoallv_gmm_cann` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `alltoallv_gmm_cann` | MOE-008 | L3 | moe_forward | — | — | yes |
| `alltoallv_gmm_cann` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `alltoallv_gmm_cann` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `alltoallv_gmm_cann` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `alltoallv_gmm_cann` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_gmm_cann` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_gmm_cann` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `alltoallv_gmm_cann` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `alltoallv_gmm_cann` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `alltoallv_gmm_cann` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_gmm_cann` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `alltoallv_quant_gmm_cann` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `alltoallv_quant_gmm_cann` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `alltoallv_quant_gmm_cann` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `alltoallv_quant_gmm_cann` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_quant_gmm_cann` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `alltoallv_quant_gmm_cann` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `alltoallv_quant_gmm_cann` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `alltoallv_quant_gmm_cann` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `alltoallv_quant_gmm_cann` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `alltoallv_quant_gmm_cann` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `alltoallv_quant_gmm_cann` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `alltoallv_quant_gmm_cann` | FUSE-008 | L2 | quantized_fusion | — | — | yes |
| `alltoallv_quant_gmm_cann` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `alltoallv_quant_gmm_cann` | MOE-001 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `alltoallv_quant_gmm_cann` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `alltoallv_quant_gmm_cann` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward, moe_forward） |
| `alltoallv_quant_gmm_cann` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `alltoallv_quant_gmm_cann` | MOE-005 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `alltoallv_quant_gmm_cann` | MOE-006 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `alltoallv_quant_gmm_cann` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `alltoallv_quant_gmm_cann` | MOE-008 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `alltoallv_quant_gmm_cann` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（operation requires dynamic_shapes, moe_forward） |
| `alltoallv_quant_gmm_cann` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `alltoallv_quant_gmm_cann` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `alltoallv_quant_gmm_cann` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_quant_gmm_cann` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_quant_gmm_cann` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `alltoallv_quant_gmm_cann` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `alltoallv_quant_gmm_cann` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `alltoallv_quant_gmm_cann` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `alltoallv_quant_gmm_cann` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `chunkmoe_fsdpturbo` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `chunkmoe_fsdpturbo` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `chunkmoe_fsdpturbo` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `chunkmoe_fsdpturbo` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `chunkmoe_fsdpturbo` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `chunkmoe_fsdpturbo` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `chunkmoe_fsdpturbo` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `chunkmoe_fsdpturbo` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `chunkmoe_fsdpturbo` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `chunkmoe_fsdpturbo` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `chunkmoe_fsdpturbo` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `chunkmoe_fsdpturbo` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `chunkmoe_fsdpturbo` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `chunkmoe_fsdpturbo` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `chunkmoe_fsdpturbo` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `chunkmoe_fsdpturbo` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `chunkmoe_fsdpturbo` | FUSE-005 | L2 | chunk_pipeline | — | — | yes |
| `chunkmoe_fsdpturbo` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `chunkmoe_fsdpturbo` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `chunkmoe_fsdpturbo` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `chunkmoe_fsdpturbo` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `chunkmoe_fsdpturbo` | MOE-001 | L3 | moe_forward | — | — | yes |
| `chunkmoe_fsdpturbo` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `chunkmoe_fsdpturbo` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `chunkmoe_fsdpturbo` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `chunkmoe_fsdpturbo` | MOE-005 | L3 | moe_forward | — | — | yes |
| `chunkmoe_fsdpturbo` | MOE-006 | L3 | moe_forward | — | — | yes |
| `chunkmoe_fsdpturbo` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `chunkmoe_fsdpturbo` | MOE-008 | L3 | moe_forward | — | — | yes |
| `chunkmoe_fsdpturbo` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `chunkmoe_fsdpturbo` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `chunkmoe_fsdpturbo` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `chunkmoe_fsdpturbo` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `chunkmoe_fsdpturbo` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `chunkmoe_fsdpturbo` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `chunkmoe_fsdpturbo` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `chunkmoe_fsdpturbo` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `chunkmoe_fsdpturbo` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `chunkmoe_fsdpturbo` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_ascend` | COMM-001 | L1 | dispatch | — | — | yes |
| `deepep_ascend` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | yes |
| `deepep_ascend` | COMM-003 | L1 | dispatch, combine | — | — | yes |
| `deepep_ascend` | COMM-004 | L1 | dispatch, combine | — | — | yes |
| `deepep_ascend` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch） |
| `deepep_ascend` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine） |
| `deepep_ascend` | COMM-007 | L1 | dispatch | — | — | yes |
| `deepep_ascend` | COMM-008 | L1 | dispatch | — | — | yes |
| `deepep_ascend` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires quantized_dispatch） |
| `deepep_ascend` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `deepep_ascend` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `deepep_ascend` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_ascend` | FUSE-001 | L2 | dispatch | — | — | yes |
| `deepep_ascend` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `deepep_ascend` | FUSE-003 | L2 | dispatch, combine | — | — | yes |
| `deepep_ascend` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `deepep_ascend` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `deepep_ascend` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `deepep_ascend` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `deepep_ascend` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `deepep_ascend` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `deepep_ascend` | MOE-001 | L3 | moe_forward | — | — | yes |
| `deepep_ascend` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `deepep_ascend` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `deepep_ascend` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `deepep_ascend` | MOE-005 | L3 | moe_forward | — | — | yes |
| `deepep_ascend` | MOE-006 | L3 | moe_forward | — | — | yes |
| `deepep_ascend` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `deepep_ascend` | MOE-008 | L3 | moe_forward | — | — | yes |
| `deepep_ascend` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `deepep_ascend` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `deepep_ascend` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_ascend` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_ascend` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_ascend` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `deepep_ascend` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `deepep_ascend` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `deepep_ascend` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_ascend` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_cann` | COMM-001 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-003 | L1 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-004 | L1 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-005 | L1 | dispatch, async_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-007 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-008 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `deepep_cann` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `deepep_cann` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_cann` | FUSE-001 | L2 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `deepep_cann` | FUSE-003 | L2 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_cann` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `deepep_cann` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `deepep_cann` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `deepep_cann` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `deepep_cann` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `deepep_cann` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `deepep_cann` | MOE-001 | L3 | moe_forward | — | — | yes |
| `deepep_cann` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `deepep_cann` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `deepep_cann` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `deepep_cann` | MOE-005 | L3 | moe_forward | — | — | no（adapter requires routing from ['balanced']） |
| `deepep_cann` | MOE-006 | L3 | moe_forward | — | — | yes |
| `deepep_cann` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `deepep_cann` | MOE-008 | L3 | moe_forward | — | — | yes |
| `deepep_cann` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `deepep_cann` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `deepep_cann` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_cann` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_cann` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_cann` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `deepep_cann` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `deepep_cann` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `deepep_cann` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_cann` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_direct_lueelu` | COMM-001 | L1 | dispatch | — | — | yes |
| `deepep_direct_lueelu` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | yes |
| `deepep_direct_lueelu` | COMM-003 | L1 | dispatch, combine | — | — | yes |
| `deepep_direct_lueelu` | COMM-004 | L1 | dispatch, combine | — | — | yes |
| `deepep_direct_lueelu` | COMM-005 | L1 | dispatch, async_dispatch | — | — | yes |
| `deepep_direct_lueelu` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | yes |
| `deepep_direct_lueelu` | COMM-007 | L1 | dispatch | — | — | yes |
| `deepep_direct_lueelu` | COMM-008 | L1 | dispatch | — | — | yes |
| `deepep_direct_lueelu` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires quantized_dispatch） |
| `deepep_direct_lueelu` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `deepep_direct_lueelu` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `deepep_direct_lueelu` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_direct_lueelu` | FUSE-001 | L2 | dispatch | — | — | yes |
| `deepep_direct_lueelu` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `deepep_direct_lueelu` | FUSE-003 | L2 | dispatch, combine | — | — | yes |
| `deepep_direct_lueelu` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `deepep_direct_lueelu` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `deepep_direct_lueelu` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `deepep_direct_lueelu` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `deepep_direct_lueelu` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `deepep_direct_lueelu` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `deepep_direct_lueelu` | MOE-001 | L3 | moe_forward | — | — | yes |
| `deepep_direct_lueelu` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `deepep_direct_lueelu` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `deepep_direct_lueelu` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `deepep_direct_lueelu` | MOE-005 | L3 | moe_forward | — | — | yes |
| `deepep_direct_lueelu` | MOE-006 | L3 | moe_forward | — | — | yes |
| `deepep_direct_lueelu` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `deepep_direct_lueelu` | MOE-008 | L3 | moe_forward | — | — | yes |
| `deepep_direct_lueelu` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `deepep_direct_lueelu` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `deepep_direct_lueelu` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_direct_lueelu` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_direct_lueelu` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_direct_lueelu` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `deepep_direct_lueelu` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `deepep_direct_lueelu` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `deepep_direct_lueelu` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_direct_lueelu` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_lueelu` | COMM-001 | L1 | dispatch | — | — | yes |
| `deepep_lueelu` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | yes |
| `deepep_lueelu` | COMM-003 | L1 | dispatch, combine | — | — | yes |
| `deepep_lueelu` | COMM-004 | L1 | dispatch, combine | — | — | yes |
| `deepep_lueelu` | COMM-005 | L1 | dispatch, async_dispatch | — | — | yes |
| `deepep_lueelu` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | yes |
| `deepep_lueelu` | COMM-007 | L1 | dispatch | — | — | yes |
| `deepep_lueelu` | COMM-008 | L1 | dispatch | — | — | yes |
| `deepep_lueelu` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires quantized_dispatch） |
| `deepep_lueelu` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `deepep_lueelu` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `deepep_lueelu` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_lueelu` | FUSE-001 | L2 | dispatch | — | — | yes |
| `deepep_lueelu` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `deepep_lueelu` | FUSE-003 | L2 | dispatch, combine | — | — | yes |
| `deepep_lueelu` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `deepep_lueelu` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `deepep_lueelu` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `deepep_lueelu` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `deepep_lueelu` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `deepep_lueelu` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `deepep_lueelu` | MOE-001 | L3 | moe_forward | — | — | yes |
| `deepep_lueelu` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `deepep_lueelu` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `deepep_lueelu` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `deepep_lueelu` | MOE-005 | L3 | moe_forward | — | — | yes |
| `deepep_lueelu` | MOE-006 | L3 | moe_forward | — | — | yes |
| `deepep_lueelu` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `deepep_lueelu` | MOE-008 | L3 | moe_forward | — | — | yes |
| `deepep_lueelu` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `deepep_lueelu` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `deepep_lueelu` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_lueelu` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_lueelu` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_lueelu` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `deepep_lueelu` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `deepep_lueelu` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `deepep_lueelu` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_lueelu` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_sglang` | COMM-001 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-003 | L1 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-004 | L1 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-005 | L1 | dispatch, async_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-007 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-008 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `deepep_sglang` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `deepep_sglang` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_sglang` | FUSE-001 | L2 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `deepep_sglang` | FUSE-003 | L2 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `deepep_sglang` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `deepep_sglang` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `deepep_sglang` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `deepep_sglang` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `deepep_sglang` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `deepep_sglang` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `deepep_sglang` | MOE-001 | L3 | moe_forward | — | — | yes |
| `deepep_sglang` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `deepep_sglang` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `deepep_sglang` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `deepep_sglang` | MOE-005 | L3 | moe_forward | — | — | yes |
| `deepep_sglang` | MOE-006 | L3 | moe_forward | — | — | yes |
| `deepep_sglang` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `deepep_sglang` | MOE-008 | L3 | moe_forward | — | — | yes |
| `deepep_sglang` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `deepep_sglang` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `deepep_sglang` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_sglang` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_sglang` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_sglang` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `deepep_sglang` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `deepep_sglang` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `deepep_sglang` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_sglang` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `deepep_v2_ascend` | COMM-001 | L1 | dispatch | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-003 | L1 | dispatch, combine | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-004 | L1 | dispatch, combine | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-007 | L1 | dispatch | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-008 | L1 | dispatch | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `deepep_v2_ascend` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_v2_ascend` | FUSE-001 | L2 | dispatch | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-002 | L2 | dispatch_gmm | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-003 | L2 | dispatch, combine | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-004 | L2 | gmm_combine | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-005 | L2 | chunk_pipeline | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-006 | L2 | tile_schedule | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-007 | L2 | comm_resources | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-008 | L2 | quantized_fusion | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | FUSE-009 | L2 | fused_backward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-001 | L3 | moe_forward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-002 | L3 | moe_backward | moe_forward | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-005 | L3 | moe_forward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-006 | L3 | moe_forward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-007 | L3 | concurrent | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-008 | L3 | moe_forward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | LAYER-001 | L4 | router_layer | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `deepep_v2_ascend` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_v2_ascend` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_v2_ascend` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `deepep_v2_ascend` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `deepep_v2_ascend` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `deepep_v2_ascend` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `deepep_v2_ascend` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（EP8 outside adapter's supported EP sizes [32, 64, 128]） |
| `dispatch_ffn_combine_sglang` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `dispatch_ffn_combine_sglang` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `dispatch_ffn_combine_sglang` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `dispatch_ffn_combine_sglang` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_sglang` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_sglang` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `dispatch_ffn_combine_sglang` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `dispatch_ffn_combine_sglang` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `dispatch_ffn_combine_sglang` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `dispatch_ffn_combine_sglang` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `dispatch_ffn_combine_sglang` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `dispatch_ffn_combine_sglang` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `dispatch_ffn_combine_sglang` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `dispatch_ffn_combine_sglang` | MOE-001 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_sglang` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `dispatch_ffn_combine_sglang` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward） |
| `dispatch_ffn_combine_sglang` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `dispatch_ffn_combine_sglang` | MOE-005 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_sglang` | MOE-006 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_sglang` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `dispatch_ffn_combine_sglang` | MOE-008 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_sglang` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（operation requires dynamic_shapes） |
| `dispatch_ffn_combine_sglang` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `dispatch_ffn_combine_sglang` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `dispatch_ffn_combine_sglang` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_sglang` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_sglang` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `dispatch_ffn_combine_sglang` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `dispatch_ffn_combine_sglang` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `dispatch_ffn_combine_sglang` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_sglang` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `dispatch_ffn_combine_vllm_ascend` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `dispatch_ffn_combine_vllm_ascend` | MOE-001 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_vllm_ascend` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `dispatch_ffn_combine_vllm_ascend` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward） |
| `dispatch_ffn_combine_vllm_ascend` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `dispatch_ffn_combine_vllm_ascend` | MOE-005 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_vllm_ascend` | MOE-006 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_vllm_ascend` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `dispatch_ffn_combine_vllm_ascend` | MOE-008 | L3 | moe_forward | — | — | yes |
| `dispatch_ffn_combine_vllm_ascend` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（operation requires dynamic_shapes） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `domino_fsdpturbo` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `domino_fsdpturbo` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `domino_fsdpturbo` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `domino_fsdpturbo` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `domino_fsdpturbo` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `domino_fsdpturbo` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `domino_fsdpturbo` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `domino_fsdpturbo` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `domino_fsdpturbo` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `domino_fsdpturbo` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `domino_fsdpturbo` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `domino_fsdpturbo` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `domino_fsdpturbo` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `domino_fsdpturbo` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `domino_fsdpturbo` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `domino_fsdpturbo` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `domino_fsdpturbo` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `domino_fsdpturbo` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `domino_fsdpturbo` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `domino_fsdpturbo` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `domino_fsdpturbo` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `domino_fsdpturbo` | MOE-001 | L3 | moe_forward | — | — | yes |
| `domino_fsdpturbo` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `domino_fsdpturbo` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `domino_fsdpturbo` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `domino_fsdpturbo` | MOE-005 | L3 | moe_forward | — | — | yes |
| `domino_fsdpturbo` | MOE-006 | L3 | moe_forward | — | — | yes |
| `domino_fsdpturbo` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `domino_fsdpturbo` | MOE-008 | L3 | moe_forward | — | — | yes |
| `domino_fsdpturbo` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `domino_fsdpturbo` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `domino_fsdpturbo` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `domino_fsdpturbo` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `domino_fsdpturbo` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `domino_fsdpturbo` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `domino_fsdpturbo` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `domino_fsdpturbo` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `domino_fsdpturbo` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `domino_fsdpturbo` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `ep_gmm_fused_lueelu` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `ep_gmm_fused_lueelu` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `ep_gmm_fused_lueelu` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `ep_gmm_fused_lueelu` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `ep_gmm_fused_lueelu` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `ep_gmm_fused_lueelu` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `ep_gmm_fused_lueelu` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `ep_gmm_fused_lueelu` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `ep_gmm_fused_lueelu` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `ep_gmm_fused_lueelu` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `ep_gmm_fused_lueelu` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `ep_gmm_fused_lueelu` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `ep_gmm_fused_lueelu` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `ep_gmm_fused_lueelu` | FUSE-002 | L2 | dispatch_gmm | — | — | yes |
| `ep_gmm_fused_lueelu` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `ep_gmm_fused_lueelu` | FUSE-004 | L2 | gmm_combine | — | — | yes |
| `ep_gmm_fused_lueelu` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `ep_gmm_fused_lueelu` | FUSE-006 | L2 | tile_schedule | — | — | yes |
| `ep_gmm_fused_lueelu` | FUSE-007 | L2 | comm_resources | — | — | yes |
| `ep_gmm_fused_lueelu` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `ep_gmm_fused_lueelu` | FUSE-009 | L2 | fused_backward | — | — | yes |
| `ep_gmm_fused_lueelu` | MOE-001 | L3 | moe_forward | — | — | yes |
| `ep_gmm_fused_lueelu` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `ep_gmm_fused_lueelu` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `ep_gmm_fused_lueelu` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `ep_gmm_fused_lueelu` | MOE-005 | L3 | moe_forward | — | — | yes |
| `ep_gmm_fused_lueelu` | MOE-006 | L3 | moe_forward | — | — | yes |
| `ep_gmm_fused_lueelu` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `ep_gmm_fused_lueelu` | MOE-008 | L3 | moe_forward | — | — | yes |
| `ep_gmm_fused_lueelu` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `ep_gmm_fused_lueelu` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `ep_gmm_fused_lueelu` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `ep_gmm_fused_lueelu` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `ep_gmm_fused_lueelu` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `ep_gmm_fused_lueelu` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `ep_gmm_fused_lueelu` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `ep_gmm_fused_lueelu` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `ep_gmm_fused_lueelu` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `ep_gmm_fused_lueelu` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `fused_deep_moe_sglang` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `fused_deep_moe_sglang` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `fused_deep_moe_sglang` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `fused_deep_moe_sglang` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `fused_deep_moe_sglang` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `fused_deep_moe_sglang` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `fused_deep_moe_sglang` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `fused_deep_moe_sglang` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `fused_deep_moe_sglang` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `fused_deep_moe_sglang` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `fused_deep_moe_sglang` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `fused_deep_moe_sglang` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_deep_moe_sglang` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `fused_deep_moe_sglang` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `fused_deep_moe_sglang` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `fused_deep_moe_sglang` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `fused_deep_moe_sglang` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `fused_deep_moe_sglang` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `fused_deep_moe_sglang` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `fused_deep_moe_sglang` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `fused_deep_moe_sglang` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `fused_deep_moe_sglang` | MOE-001 | L3 | moe_forward | — | — | yes |
| `fused_deep_moe_sglang` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `fused_deep_moe_sglang` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward） |
| `fused_deep_moe_sglang` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `fused_deep_moe_sglang` | MOE-005 | L3 | moe_forward | — | — | yes |
| `fused_deep_moe_sglang` | MOE-006 | L3 | moe_forward | — | — | yes |
| `fused_deep_moe_sglang` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `fused_deep_moe_sglang` | MOE-008 | L3 | moe_forward | — | — | yes |
| `fused_deep_moe_sglang` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（operation requires dynamic_shapes） |
| `fused_deep_moe_sglang` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `fused_deep_moe_sglang` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `fused_deep_moe_sglang` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_deep_moe_sglang` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_deep_moe_sglang` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `fused_deep_moe_sglang` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `fused_deep_moe_sglang` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `fused_deep_moe_sglang` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_deep_moe_sglang` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `fused_fsdpturbo` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `fused_fsdpturbo` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `fused_fsdpturbo` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `fused_fsdpturbo` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `fused_fsdpturbo` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `fused_fsdpturbo` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `fused_fsdpturbo` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `fused_fsdpturbo` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `fused_fsdpturbo` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `fused_fsdpturbo` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `fused_fsdpturbo` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `fused_fsdpturbo` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_fsdpturbo` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `fused_fsdpturbo` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `fused_fsdpturbo` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `fused_fsdpturbo` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `fused_fsdpturbo` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `fused_fsdpturbo` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `fused_fsdpturbo` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `fused_fsdpturbo` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `fused_fsdpturbo` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `fused_fsdpturbo` | MOE-001 | L3 | moe_forward | — | — | yes |
| `fused_fsdpturbo` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `fused_fsdpturbo` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `fused_fsdpturbo` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `fused_fsdpturbo` | MOE-005 | L3 | moe_forward | — | — | yes |
| `fused_fsdpturbo` | MOE-006 | L3 | moe_forward | — | — | yes |
| `fused_fsdpturbo` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `fused_fsdpturbo` | MOE-008 | L3 | moe_forward | — | — | yes |
| `fused_fsdpturbo` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `fused_fsdpturbo` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `fused_fsdpturbo` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `fused_fsdpturbo` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_fsdpturbo` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_fsdpturbo` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `fused_fsdpturbo` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `fused_fsdpturbo` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `fused_fsdpturbo` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `fused_fsdpturbo` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `hierarchical_alltoallv_mindspeed` | COMM-001 | L1 | dispatch | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-003 | L1 | dispatch, combine | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-004 | L1 | dispatch, combine | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-007 | L1 | dispatch | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-008 | L1 | dispatch | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `hierarchical_alltoallv_mindspeed` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `hierarchical_alltoallv_mindspeed` | FUSE-001 | L2 | dispatch | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-002 | L2 | dispatch_gmm | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-003 | L2 | dispatch, combine | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-004 | L2 | gmm_combine | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-005 | L2 | chunk_pipeline | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-006 | L2 | tile_schedule | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-007 | L2 | comm_resources | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-008 | L2 | quantized_fusion | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | FUSE-009 | L2 | fused_backward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-001 | L3 | moe_forward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-002 | L3 | moe_backward | moe_forward | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-005 | L3 | moe_forward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-006 | L3 | moe_forward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-007 | L3 | concurrent | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-008 | L3 | moe_forward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | LAYER-001 | L4 | router_layer | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `hierarchical_alltoallv_mindspeed` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `hierarchical_alltoallv_mindspeed` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `hierarchical_alltoallv_mindspeed` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `hierarchical_alltoallv_mindspeed` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `hierarchical_alltoallv_mindspeed` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `hierarchical_alltoallv_mindspeed` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `hierarchical_alltoallv_mindspeed` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（native hierarchy requires at least two outer ranks, each with at least two inner ranks） |
| `mc2_combine_add_rmsnorm_cann` | COMM-001 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-003 | L1 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-004 | L1 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-007 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-008 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `mc2_combine_add_rmsnorm_cann` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-001 | L2 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-002 | L2 | dispatch_gmm | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-003 | L2 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-004 | L2 | gmm_combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-005 | L2 | chunk_pipeline | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-006 | L2 | tile_schedule | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-007 | L2 | comm_resources | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-008 | L2 | quantized_fusion | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | FUSE-009 | L2 | fused_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-001 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-002 | L3 | moe_backward | moe_forward | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-005 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-006 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-007 | L3 | concurrent | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-008 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-001 | L4 | router_layer | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_combine_add_rmsnorm_cann` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann` | COMM-001 | L1 | dispatch | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-003 | L1 | dispatch, combine | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-004 | L1 | dispatch, combine | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-007 | L1 | dispatch | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-008 | L1 | dispatch | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `mc2_dispatch_cann` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（a single 950DT V2 invocation supports at most 512 tokens per rank） |
| `mc2_dispatch_cann` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann` | FUSE-001 | L2 | dispatch | — | — | yes |
| `mc2_dispatch_cann` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `mc2_dispatch_cann` | FUSE-003 | L2 | dispatch, combine | — | — | yes |
| `mc2_dispatch_cann` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `mc2_dispatch_cann` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `mc2_dispatch_cann` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `mc2_dispatch_cann` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `mc2_dispatch_cann` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `mc2_dispatch_cann` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `mc2_dispatch_cann` | MOE-001 | L3 | moe_forward | — | — | yes |
| `mc2_dispatch_cann` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `mc2_dispatch_cann` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `mc2_dispatch_cann` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `mc2_dispatch_cann` | MOE-005 | L3 | moe_forward | — | — | yes |
| `mc2_dispatch_cann` | MOE-006 | L3 | moe_forward | — | — | yes |
| `mc2_dispatch_cann` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `mc2_dispatch_cann` | MOE-008 | L3 | moe_forward | — | — | yes |
| `mc2_dispatch_cann` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `mc2_dispatch_cann` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `mc2_dispatch_cann` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `mc2_dispatch_cann` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `mc2_dispatch_cann` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `mc2_dispatch_cann` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `mc2_dispatch_cann` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `mc2_dispatch_cann_v1` | COMM-001 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-003 | L1 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-004 | L1 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-007 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-008 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `mc2_dispatch_cann_v1` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v1` | FUSE-001 | L2 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-002 | L2 | dispatch_gmm | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-003 | L2 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-004 | L2 | gmm_combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-005 | L2 | chunk_pipeline | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-006 | L2 | tile_schedule | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-007 | L2 | comm_resources | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-008 | L2 | quantized_fusion | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | FUSE-009 | L2 | fused_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-001 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-002 | L3 | moe_backward | moe_forward | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-005 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-006 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-007 | L3 | concurrent | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-008 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | LAYER-001 | L4 | router_layer | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v1` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v1` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v1` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `mc2_dispatch_cann_v1` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `mc2_dispatch_cann_v1` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `mc2_dispatch_cann_v1` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v1` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-001 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-003 | L1 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-004 | L1 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-007 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-008 | L1 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `mc2_dispatch_cann_v3` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v3` | FUSE-001 | L2 | dispatch | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-002 | L2 | dispatch_gmm | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-003 | L2 | dispatch, combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-004 | L2 | gmm_combine | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-005 | L2 | chunk_pipeline | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-006 | L2 | tile_schedule | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-007 | L2 | comm_resources | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-008 | L2 | quantized_fusion | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | FUSE-009 | L2 | fused_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-001 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-002 | L3 | moe_backward | moe_forward | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-005 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-006 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-007 | L3 | concurrent | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-008 | L3 | moe_forward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | LAYER-001 | L4 | router_layer | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_dispatch_cann_v3` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v3` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v3` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `mc2_dispatch_cann_v3` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `mc2_dispatch_cann_v3` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `mc2_dispatch_cann_v3` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_dispatch_cann_v3` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（tokens per rank exceeds adapter capacity 512） |
| `mc2_fsdpturbo` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `mc2_fsdpturbo` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `mc2_fsdpturbo` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `mc2_fsdpturbo` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `mc2_fsdpturbo` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `mc2_fsdpturbo` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `mc2_fsdpturbo` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `mc2_fsdpturbo` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `mc2_fsdpturbo` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `mc2_fsdpturbo` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `mc2_fsdpturbo` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `mc2_fsdpturbo` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_fsdpturbo` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `mc2_fsdpturbo` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `mc2_fsdpturbo` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `mc2_fsdpturbo` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `mc2_fsdpturbo` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `mc2_fsdpturbo` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `mc2_fsdpturbo` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `mc2_fsdpturbo` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `mc2_fsdpturbo` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `mc2_fsdpturbo` | MOE-001 | L3 | moe_forward | — | — | yes |
| `mc2_fsdpturbo` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `mc2_fsdpturbo` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `mc2_fsdpturbo` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `mc2_fsdpturbo` | MOE-005 | L3 | moe_forward | — | — | yes |
| `mc2_fsdpturbo` | MOE-006 | L3 | moe_forward | — | — | yes |
| `mc2_fsdpturbo` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `mc2_fsdpturbo` | MOE-008 | L3 | moe_forward | — | — | yes |
| `mc2_fsdpturbo` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `mc2_fsdpturbo` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `mc2_fsdpturbo` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `mc2_fsdpturbo` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_fsdpturbo` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_fsdpturbo` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `mc2_fsdpturbo` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `mc2_fsdpturbo` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `mc2_fsdpturbo` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `mc2_fsdpturbo` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `megamoe_cann` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_cann` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `megamoe_cann` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_cann` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_cann` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `megamoe_cann` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `megamoe_cann` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_cann` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_cann` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `megamoe_cann` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `megamoe_cann` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `megamoe_cann` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_cann` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_cann` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `megamoe_cann` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_cann` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `megamoe_cann` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `megamoe_cann` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `megamoe_cann` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `megamoe_cann` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `megamoe_cann` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `megamoe_cann` | MOE-001 | L3 | moe_forward | — | — | yes |
| `megamoe_cann` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `megamoe_cann` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward） |
| `megamoe_cann` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `megamoe_cann` | MOE-005 | L3 | moe_forward | — | — | yes |
| `megamoe_cann` | MOE-006 | L3 | moe_forward | — | — | yes |
| `megamoe_cann` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `megamoe_cann` | MOE-008 | L3 | moe_forward | — | — | yes |
| `megamoe_cann` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Symmetric window resize and reuse are pending） |
| `megamoe_cann` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `megamoe_cann` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `megamoe_cann` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_cann` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_cann` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `megamoe_cann` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `megamoe_cann` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `megamoe_cann` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_cann` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `megamoe_deepgemm_ascend` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_deepgemm_ascend` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `megamoe_deepgemm_ascend` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_deepgemm_ascend` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_deepgemm_ascend` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `megamoe_deepgemm_ascend` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `megamoe_deepgemm_ascend` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_deepgemm_ascend` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_deepgemm_ascend` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `megamoe_deepgemm_ascend` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `megamoe_deepgemm_ascend` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `megamoe_deepgemm_ascend` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_deepgemm_ascend` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_deepgemm_ascend` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `megamoe_deepgemm_ascend` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_deepgemm_ascend` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `megamoe_deepgemm_ascend` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `megamoe_deepgemm_ascend` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `megamoe_deepgemm_ascend` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `megamoe_deepgemm_ascend` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `megamoe_deepgemm_ascend` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `megamoe_deepgemm_ascend` | MOE-001 | L3 | moe_forward | — | — | yes |
| `megamoe_deepgemm_ascend` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `megamoe_deepgemm_ascend` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward） |
| `megamoe_deepgemm_ascend` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `megamoe_deepgemm_ascend` | MOE-005 | L3 | moe_forward | — | — | yes |
| `megamoe_deepgemm_ascend` | MOE-006 | L3 | moe_forward | — | — | yes |
| `megamoe_deepgemm_ascend` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `megamoe_deepgemm_ascend` | MOE-008 | L3 | moe_forward | — | — | yes |
| `megamoe_deepgemm_ascend` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（operation requires dynamic_shapes） |
| `megamoe_deepgemm_ascend` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `megamoe_deepgemm_ascend` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `megamoe_deepgemm_ascend` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_deepgemm_ascend` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_deepgemm_ascend` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `megamoe_deepgemm_ascend` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `megamoe_deepgemm_ascend` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `megamoe_deepgemm_ascend` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_deepgemm_ascend` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `megamoe_fsdpturbo` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_fsdpturbo` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `megamoe_fsdpturbo` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_fsdpturbo` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_fsdpturbo` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `megamoe_fsdpturbo` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `megamoe_fsdpturbo` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_fsdpturbo` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_fsdpturbo` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `megamoe_fsdpturbo` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `megamoe_fsdpturbo` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `megamoe_fsdpturbo` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_fsdpturbo` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_fsdpturbo` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `megamoe_fsdpturbo` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_fsdpturbo` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `megamoe_fsdpturbo` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `megamoe_fsdpturbo` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `megamoe_fsdpturbo` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `megamoe_fsdpturbo` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `megamoe_fsdpturbo` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `megamoe_fsdpturbo` | MOE-001 | L3 | moe_forward | — | — | yes |
| `megamoe_fsdpturbo` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `megamoe_fsdpturbo` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `megamoe_fsdpturbo` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `megamoe_fsdpturbo` | MOE-005 | L3 | moe_forward | — | — | yes |
| `megamoe_fsdpturbo` | MOE-006 | L3 | moe_forward | — | — | yes |
| `megamoe_fsdpturbo` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `megamoe_fsdpturbo` | MOE-008 | L3 | moe_forward | — | — | yes |
| `megamoe_fsdpturbo` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `megamoe_fsdpturbo` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `megamoe_fsdpturbo` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `megamoe_fsdpturbo` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_fsdpturbo` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_fsdpturbo` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `megamoe_fsdpturbo` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `megamoe_fsdpturbo` | LAYER-007 | L3 | moe_forward | — | — | conditional（only when workload={'routing': 'real_trace_replay'}） |
| `megamoe_fsdpturbo` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_fsdpturbo` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `megamoe_lueelu` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_lueelu` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `megamoe_lueelu` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_lueelu` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_lueelu` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `megamoe_lueelu` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `megamoe_lueelu` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_lueelu` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_lueelu` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `megamoe_lueelu` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `megamoe_lueelu` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `megamoe_lueelu` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_lueelu` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_lueelu` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `megamoe_lueelu` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_lueelu` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `megamoe_lueelu` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `megamoe_lueelu` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `megamoe_lueelu` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `megamoe_lueelu` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `megamoe_lueelu` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `megamoe_lueelu` | MOE-001 | L3 | moe_forward | — | — | yes |
| `megamoe_lueelu` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `megamoe_lueelu` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `megamoe_lueelu` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `megamoe_lueelu` | MOE-005 | L3 | moe_forward | — | — | no（adapter requires routing from ['balanced']） |
| `megamoe_lueelu` | MOE-006 | L3 | moe_forward | — | — | yes |
| `megamoe_lueelu` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `megamoe_lueelu` | MOE-008 | L3 | moe_forward | — | — | yes |
| `megamoe_lueelu` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `megamoe_lueelu` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `megamoe_lueelu` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `megamoe_lueelu` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_lueelu` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_lueelu` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `megamoe_lueelu` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `megamoe_lueelu` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `megamoe_lueelu` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_lueelu` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `megamoe_td` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_td` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `megamoe_td` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_td` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_td` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `megamoe_td` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `megamoe_td` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_td` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_td` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `megamoe_td` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `megamoe_td` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `megamoe_td` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_td` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `megamoe_td` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `megamoe_td` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `megamoe_td` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `megamoe_td` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `megamoe_td` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `megamoe_td` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `megamoe_td` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `megamoe_td` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `megamoe_td` | MOE-001 | L3 | moe_forward | — | — | yes |
| `megamoe_td` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `megamoe_td` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `megamoe_td` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `megamoe_td` | MOE-005 | L3 | moe_forward | — | — | no（adapter requires routing from ['balanced']） |
| `megamoe_td` | MOE-006 | L3 | moe_forward | — | — | yes |
| `megamoe_td` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `megamoe_td` | MOE-008 | L3 | moe_forward | — | — | yes |
| `megamoe_td` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `megamoe_td` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `megamoe_td` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `megamoe_td` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_td` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_td` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `megamoe_td` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `megamoe_td` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `megamoe_td` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `megamoe_td` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `moonep` | COMM-001 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-003 | L1 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-004 | L1 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-005 | L1 | dispatch, async_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-007 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-008 | L1 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `moonep` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `moonep` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep` | FUSE-001 | L2 | dispatch | — | — | not_implemented（Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `moonep` | FUSE-003 | L2 | dispatch, combine | — | — | not_implemented（Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending.） |
| `moonep` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `moonep` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `moonep` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `moonep` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `moonep` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `moonep` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `moonep` | MOE-001 | L3 | moe_forward | — | — | yes |
| `moonep` | MOE-002 | L3 | moe_backward | moe_forward | — | yes |
| `moonep` | MOE-003 | L3 | moe_forward, moe_backward | — | — | yes |
| `moonep` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires recompute） |
| `moonep` | MOE-005 | L3 | moe_forward | — | — | no（adapter requires routing from ['balanced']） |
| `moonep` | MOE-006 | L3 | moe_forward | — | — | yes |
| `moonep` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `moonep` | MOE-008 | L3 | moe_forward | — | — | yes |
| `moonep` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | not_implemented（Native resized workspace lifecycle is pending） |
| `moonep` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `moonep` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `moonep` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `moonep` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `moonep` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `moonep` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires router_layer） |
| `moonep_dynamic_planner` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_dynamic_planner` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `moonep_dynamic_planner` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `moonep_dynamic_planner` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `moonep_dynamic_planner` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `moonep_dynamic_planner` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `moonep_dynamic_planner` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_dynamic_planner` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_dynamic_planner` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `moonep_dynamic_planner` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | not_applicable（deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none） |
| `moonep_dynamic_planner` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `moonep_dynamic_planner` | COMM-012 | L1 | planned_dispatch | — | — | conditional（only when load_balance={'enabled': True}） |
| `moonep_dynamic_planner` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_dynamic_planner` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `moonep_dynamic_planner` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `moonep_dynamic_planner` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `moonep_dynamic_planner` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `moonep_dynamic_planner` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `moonep_dynamic_planner` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `moonep_dynamic_planner` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `moonep_dynamic_planner` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `moonep_dynamic_planner` | MOE-001 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_dynamic_planner` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `moonep_dynamic_planner` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward, moe_forward） |
| `moonep_dynamic_planner` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `moonep_dynamic_planner` | MOE-005 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_dynamic_planner` | MOE-006 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_dynamic_planner` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `moonep_dynamic_planner` | MOE-008 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_dynamic_planner` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（operation requires dynamic_shapes, moe_forward） |
| `moonep_dynamic_planner` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `moonep_dynamic_planner` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `moonep_dynamic_planner` | LAYER-003 | L4 | dynamic_planner | — | — | conditional（only when load_balance={'enabled': True}） |
| `moonep_dynamic_planner` | LAYER-004 | L4 | dynamic_planner | — | — | conditional（only when load_balance={'enabled': True}） |
| `moonep_dynamic_planner` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | not_applicable（this case measures rank dedup; the plan selects dedup.mode=none） |
| `moonep_dynamic_planner` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | not_applicable（this case measures server dedup; the plan selects dedup.mode=none） |
| `moonep_dynamic_planner` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `moonep_dynamic_planner` | LAYER-008 | L4 | dynamic_planner | — | — | conditional（only when load_balance={'enabled': True}） |
| `moonep_dynamic_planner` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `moonep_server_dedup` | COMM-001 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_server_dedup` | COMM-002 | L1 | dispatch, cached_dispatch | — | — | no（operation requires cached_dispatch, dispatch） |
| `moonep_server_dedup` | COMM-003 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `moonep_server_dedup` | COMM-004 | L1 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `moonep_server_dedup` | COMM-005 | L1 | dispatch, async_dispatch | — | — | no（operation requires async_dispatch, dispatch） |
| `moonep_server_dedup` | COMM-006 | L1 | dispatch, combine, async_combine | — | — | no（operation requires async_combine, combine, dispatch） |
| `moonep_server_dedup` | COMM-007 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_server_dedup` | COMM-008 | L1 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_server_dedup` | COMM-009 | L1 | dispatch, quantized_dispatch | — | — | no（operation requires dispatch, quantized_dispatch） |
| `moonep_server_dedup` | COMM-010 | L1 | — | — | dedup.mode=rank → rank_dedup; dedup.mode=server → server_dedup | conditional（only when dedup={'mode': 'server'}） |
| `moonep_server_dedup` | COMM-011 | L1 | combine_add_rmsnorm | dispatch | — | no（operation requires combine_add_rmsnorm） |
| `moonep_server_dedup` | COMM-012 | L1 | planned_dispatch | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep_server_dedup` | FUSE-001 | L2 | dispatch | — | — | no（operation requires dispatch） |
| `moonep_server_dedup` | FUSE-002 | L2 | dispatch_gmm | — | — | no（operation requires dispatch_gmm） |
| `moonep_server_dedup` | FUSE-003 | L2 | dispatch, combine | — | — | no（operation requires combine, dispatch） |
| `moonep_server_dedup` | FUSE-004 | L2 | gmm_combine | — | — | no（operation requires gmm_combine） |
| `moonep_server_dedup` | FUSE-005 | L2 | chunk_pipeline | — | — | no（operation requires chunk_pipeline） |
| `moonep_server_dedup` | FUSE-006 | L2 | tile_schedule | — | — | no（operation requires tile_schedule） |
| `moonep_server_dedup` | FUSE-007 | L2 | comm_resources | — | — | no（operation requires comm_resources） |
| `moonep_server_dedup` | FUSE-008 | L2 | quantized_fusion | — | — | no（operation requires quantized_fusion） |
| `moonep_server_dedup` | FUSE-009 | L2 | fused_backward | — | — | no（operation requires fused_backward） |
| `moonep_server_dedup` | MOE-001 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_server_dedup` | MOE-002 | L3 | moe_backward | moe_forward | — | no（operation requires moe_backward） |
| `moonep_server_dedup` | MOE-003 | L3 | moe_forward, moe_backward | — | — | no（operation requires moe_backward, moe_forward） |
| `moonep_server_dedup` | MOE-004 | L3 | moe_backward, recompute | moe_forward, moe_backward | — | no（operation requires moe_backward, recompute） |
| `moonep_server_dedup` | MOE-005 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_server_dedup` | MOE-006 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_server_dedup` | MOE-007 | L3 | concurrent | — | — | no（operation requires concurrent） |
| `moonep_server_dedup` | MOE-008 | L3 | moe_forward | — | — | no（operation requires moe_forward） |
| `moonep_server_dedup` | MOE-009 | L3 | moe_forward, dynamic_shapes | — | — | no（operation requires dynamic_shapes, moe_forward） |
| `moonep_server_dedup` | LAYER-001 | L4 | router_layer | — | — | no（operation requires router_layer） |
| `moonep_server_dedup` | LAYER-002 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |
| `moonep_server_dedup` | LAYER-003 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep_server_dedup` | LAYER-004 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep_server_dedup` | LAYER-005 | L4 | rank_dedup, dedup_training | — | — | conditional（only when dedup={'mode': 'rank'}） |
| `moonep_server_dedup` | LAYER-006 | L4 | server_dedup, dedup_training | — | — | conditional（only when dedup={'mode': 'server'}） |
| `moonep_server_dedup` | LAYER-007 | L3 | moe_forward | — | — | not_applicable（L3 replay needs a real_trace_replay workload; this configuration does not select one） |
| `moonep_server_dedup` | LAYER-008 | L4 | dynamic_planner | — | — | not_applicable（the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true） |
| `moonep_server_dedup` | LAYER-009 | L4 | router_layer, moe_backward | — | — | no（operation requires moe_backward, router_layer） |

## 3. 不可运行原因汇总

| Operator | Case | Level | Reason |
|---|---|---|---|
| `allgather_fsdpturbo` | COMM-001 | L1 | operation requires dispatch |
| `allgather_fsdpturbo` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `allgather_fsdpturbo` | COMM-003 | L1 | operation requires combine, dispatch |
| `allgather_fsdpturbo` | COMM-004 | L1 | operation requires combine, dispatch |
| `allgather_fsdpturbo` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `allgather_fsdpturbo` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `allgather_fsdpturbo` | COMM-007 | L1 | operation requires dispatch |
| `allgather_fsdpturbo` | COMM-008 | L1 | operation requires dispatch |
| `allgather_fsdpturbo` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `allgather_fsdpturbo` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `allgather_fsdpturbo` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `allgather_fsdpturbo` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_fsdpturbo` | FUSE-001 | L2 | operation requires dispatch |
| `allgather_fsdpturbo` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `allgather_fsdpturbo` | FUSE-003 | L2 | operation requires combine, dispatch |
| `allgather_fsdpturbo` | FUSE-004 | L2 | operation requires gmm_combine |
| `allgather_fsdpturbo` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `allgather_fsdpturbo` | FUSE-006 | L2 | operation requires tile_schedule |
| `allgather_fsdpturbo` | FUSE-007 | L2 | operation requires comm_resources |
| `allgather_fsdpturbo` | FUSE-008 | L2 | operation requires quantized_fusion |
| `allgather_fsdpturbo` | FUSE-009 | L2 | operation requires fused_backward |
| `allgather_fsdpturbo` | MOE-004 | L3 | operation requires recompute |
| `allgather_fsdpturbo` | MOE-007 | L3 | operation requires concurrent |
| `allgather_fsdpturbo` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `allgather_fsdpturbo` | LAYER-001 | L4 | operation requires router_layer |
| `allgather_fsdpturbo` | LAYER-002 | L4 | operation requires router_layer |
| `allgather_fsdpturbo` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_fsdpturbo` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_fsdpturbo` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `allgather_fsdpturbo` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `allgather_fsdpturbo` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_fsdpturbo` | LAYER-009 | L4 | operation requires router_layer |
| `allgather_hccl` | COMM-002 | L1 | operation requires cached_dispatch |
| `allgather_hccl` | COMM-005 | L1 | operation requires async_dispatch |
| `allgather_hccl` | COMM-006 | L1 | operation requires async_combine |
| `allgather_hccl` | COMM-009 | L1 | operation requires quantized_dispatch |
| `allgather_hccl` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `allgather_hccl` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `allgather_hccl` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_hccl` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `allgather_hccl` | FUSE-004 | L2 | operation requires gmm_combine |
| `allgather_hccl` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `allgather_hccl` | FUSE-006 | L2 | operation requires tile_schedule |
| `allgather_hccl` | FUSE-007 | L2 | operation requires comm_resources |
| `allgather_hccl` | FUSE-008 | L2 | operation requires quantized_fusion |
| `allgather_hccl` | FUSE-009 | L2 | operation requires fused_backward |
| `allgather_hccl` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_hccl` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_hccl` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `allgather_hccl` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `allgather_hccl` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_megatron` | COMM-001 | L1 | operation requires dispatch |
| `allgather_megatron` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `allgather_megatron` | COMM-003 | L1 | operation requires combine, dispatch |
| `allgather_megatron` | COMM-004 | L1 | operation requires combine, dispatch |
| `allgather_megatron` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `allgather_megatron` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `allgather_megatron` | COMM-007 | L1 | operation requires dispatch |
| `allgather_megatron` | COMM-008 | L1 | operation requires dispatch |
| `allgather_megatron` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `allgather_megatron` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `allgather_megatron` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `allgather_megatron` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_megatron` | FUSE-001 | L2 | operation requires dispatch |
| `allgather_megatron` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `allgather_megatron` | FUSE-003 | L2 | operation requires combine, dispatch |
| `allgather_megatron` | FUSE-004 | L2 | operation requires gmm_combine |
| `allgather_megatron` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `allgather_megatron` | FUSE-006 | L2 | operation requires tile_schedule |
| `allgather_megatron` | FUSE-007 | L2 | operation requires comm_resources |
| `allgather_megatron` | FUSE-008 | L2 | operation requires quantized_fusion |
| `allgather_megatron` | FUSE-009 | L2 | operation requires fused_backward |
| `allgather_megatron` | MOE-004 | L3 | operation requires recompute |
| `allgather_megatron` | MOE-007 | L3 | operation requires concurrent |
| `allgather_megatron` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `allgather_megatron` | LAYER-001 | L4 | operation requires router_layer |
| `allgather_megatron` | LAYER-002 | L4 | operation requires router_layer |
| `allgather_megatron` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_megatron` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_megatron` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `allgather_megatron` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `allgather_megatron` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_megatron` | LAYER-009 | L4 | operation requires router_layer |
| `allgather_overlap_mindspeed` | COMM-001 | L1 | operation requires dispatch |
| `allgather_overlap_mindspeed` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `allgather_overlap_mindspeed` | COMM-003 | L1 | operation requires combine, dispatch |
| `allgather_overlap_mindspeed` | COMM-004 | L1 | operation requires combine, dispatch |
| `allgather_overlap_mindspeed` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `allgather_overlap_mindspeed` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `allgather_overlap_mindspeed` | COMM-007 | L1 | operation requires dispatch |
| `allgather_overlap_mindspeed` | COMM-008 | L1 | operation requires dispatch |
| `allgather_overlap_mindspeed` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `allgather_overlap_mindspeed` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `allgather_overlap_mindspeed` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `allgather_overlap_mindspeed` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_overlap_mindspeed` | FUSE-001 | L2 | operation requires dispatch |
| `allgather_overlap_mindspeed` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `allgather_overlap_mindspeed` | FUSE-003 | L2 | operation requires combine, dispatch |
| `allgather_overlap_mindspeed` | FUSE-004 | L2 | operation requires gmm_combine |
| `allgather_overlap_mindspeed` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `allgather_overlap_mindspeed` | FUSE-006 | L2 | operation requires tile_schedule |
| `allgather_overlap_mindspeed` | FUSE-007 | L2 | operation requires comm_resources |
| `allgather_overlap_mindspeed` | FUSE-008 | L2 | operation requires quantized_fusion |
| `allgather_overlap_mindspeed` | FUSE-009 | L2 | operation requires fused_backward |
| `allgather_overlap_mindspeed` | MOE-004 | L3 | operation requires recompute |
| `allgather_overlap_mindspeed` | MOE-007 | L3 | operation requires concurrent |
| `allgather_overlap_mindspeed` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `allgather_overlap_mindspeed` | LAYER-001 | L4 | operation requires router_layer |
| `allgather_overlap_mindspeed` | LAYER-002 | L4 | operation requires router_layer |
| `allgather_overlap_mindspeed` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_overlap_mindspeed` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_overlap_mindspeed` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `allgather_overlap_mindspeed` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `allgather_overlap_mindspeed` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `allgather_overlap_mindspeed` | LAYER-009 | L4 | operation requires router_layer |
| `alltoall_fsdpturbo` | COMM-001 | L1 | operation requires dispatch |
| `alltoall_fsdpturbo` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `alltoall_fsdpturbo` | COMM-003 | L1 | operation requires combine, dispatch |
| `alltoall_fsdpturbo` | COMM-004 | L1 | operation requires combine, dispatch |
| `alltoall_fsdpturbo` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `alltoall_fsdpturbo` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `alltoall_fsdpturbo` | COMM-007 | L1 | operation requires dispatch |
| `alltoall_fsdpturbo` | COMM-008 | L1 | operation requires dispatch |
| `alltoall_fsdpturbo` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `alltoall_fsdpturbo` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `alltoall_fsdpturbo` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `alltoall_fsdpturbo` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_fsdpturbo` | FUSE-001 | L2 | operation requires dispatch |
| `alltoall_fsdpturbo` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `alltoall_fsdpturbo` | FUSE-003 | L2 | operation requires combine, dispatch |
| `alltoall_fsdpturbo` | FUSE-004 | L2 | operation requires gmm_combine |
| `alltoall_fsdpturbo` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `alltoall_fsdpturbo` | FUSE-006 | L2 | operation requires tile_schedule |
| `alltoall_fsdpturbo` | FUSE-007 | L2 | operation requires comm_resources |
| `alltoall_fsdpturbo` | FUSE-008 | L2 | operation requires quantized_fusion |
| `alltoall_fsdpturbo` | FUSE-009 | L2 | operation requires fused_backward |
| `alltoall_fsdpturbo` | MOE-004 | L3 | operation requires recompute |
| `alltoall_fsdpturbo` | MOE-007 | L3 | operation requires concurrent |
| `alltoall_fsdpturbo` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `alltoall_fsdpturbo` | LAYER-001 | L4 | operation requires router_layer |
| `alltoall_fsdpturbo` | LAYER-002 | L4 | operation requires router_layer |
| `alltoall_fsdpturbo` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_fsdpturbo` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_fsdpturbo` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `alltoall_fsdpturbo` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `alltoall_fsdpturbo` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_fsdpturbo` | LAYER-009 | L4 | operation requires router_layer |
| `alltoall_hccl` | COMM-009 | L1 | operation requires quantized_dispatch |
| `alltoall_hccl` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `alltoall_hccl` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `alltoall_hccl` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_hccl` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `alltoall_hccl` | FUSE-004 | L2 | operation requires gmm_combine |
| `alltoall_hccl` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `alltoall_hccl` | FUSE-006 | L2 | operation requires tile_schedule |
| `alltoall_hccl` | FUSE-007 | L2 | operation requires comm_resources |
| `alltoall_hccl` | FUSE-008 | L2 | operation requires quantized_fusion |
| `alltoall_hccl` | FUSE-009 | L2 | operation requires fused_backward |
| `alltoall_hccl` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_hccl` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_megatron` | COMM-001 | L1 | operation requires dispatch |
| `alltoall_megatron` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `alltoall_megatron` | COMM-003 | L1 | operation requires combine, dispatch |
| `alltoall_megatron` | COMM-004 | L1 | operation requires combine, dispatch |
| `alltoall_megatron` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `alltoall_megatron` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `alltoall_megatron` | COMM-007 | L1 | operation requires dispatch |
| `alltoall_megatron` | COMM-008 | L1 | operation requires dispatch |
| `alltoall_megatron` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `alltoall_megatron` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `alltoall_megatron` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `alltoall_megatron` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_megatron` | FUSE-001 | L2 | operation requires dispatch |
| `alltoall_megatron` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `alltoall_megatron` | FUSE-003 | L2 | operation requires combine, dispatch |
| `alltoall_megatron` | FUSE-004 | L2 | operation requires gmm_combine |
| `alltoall_megatron` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `alltoall_megatron` | FUSE-006 | L2 | operation requires tile_schedule |
| `alltoall_megatron` | FUSE-007 | L2 | operation requires comm_resources |
| `alltoall_megatron` | FUSE-008 | L2 | operation requires quantized_fusion |
| `alltoall_megatron` | FUSE-009 | L2 | operation requires fused_backward |
| `alltoall_megatron` | MOE-004 | L3 | operation requires recompute |
| `alltoall_megatron` | MOE-007 | L3 | operation requires concurrent |
| `alltoall_megatron` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `alltoall_megatron` | LAYER-001 | L4 | operation requires router_layer |
| `alltoall_megatron` | LAYER-002 | L4 | operation requires router_layer |
| `alltoall_megatron` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_megatron` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_megatron` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `alltoall_megatron` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `alltoall_megatron` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoall_megatron` | LAYER-009 | L4 | operation requires router_layer |
| `alltoallv_gmm_cann` | COMM-001 | L1 | operation requires dispatch |
| `alltoallv_gmm_cann` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `alltoallv_gmm_cann` | COMM-003 | L1 | operation requires combine, dispatch |
| `alltoallv_gmm_cann` | COMM-004 | L1 | operation requires combine, dispatch |
| `alltoallv_gmm_cann` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `alltoallv_gmm_cann` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `alltoallv_gmm_cann` | COMM-007 | L1 | operation requires dispatch |
| `alltoallv_gmm_cann` | COMM-008 | L1 | operation requires dispatch |
| `alltoallv_gmm_cann` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `alltoallv_gmm_cann` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `alltoallv_gmm_cann` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `alltoallv_gmm_cann` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_gmm_cann` | FUSE-001 | L2 | operation requires dispatch |
| `alltoallv_gmm_cann` | FUSE-003 | L2 | operation requires combine, dispatch |
| `alltoallv_gmm_cann` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `alltoallv_gmm_cann` | FUSE-006 | L2 | operation requires tile_schedule |
| `alltoallv_gmm_cann` | FUSE-007 | L2 | operation requires comm_resources |
| `alltoallv_gmm_cann` | FUSE-008 | L2 | operation requires quantized_fusion |
| `alltoallv_gmm_cann` | MOE-004 | L3 | operation requires recompute |
| `alltoallv_gmm_cann` | MOE-007 | L3 | operation requires concurrent |
| `alltoallv_gmm_cann` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `alltoallv_gmm_cann` | LAYER-001 | L4 | operation requires router_layer |
| `alltoallv_gmm_cann` | LAYER-002 | L4 | operation requires router_layer |
| `alltoallv_gmm_cann` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_gmm_cann` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_gmm_cann` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `alltoallv_gmm_cann` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `alltoallv_gmm_cann` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_gmm_cann` | LAYER-009 | L4 | operation requires router_layer |
| `alltoallv_quant_gmm_cann` | COMM-001 | L1 | operation requires dispatch |
| `alltoallv_quant_gmm_cann` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `alltoallv_quant_gmm_cann` | COMM-003 | L1 | operation requires combine, dispatch |
| `alltoallv_quant_gmm_cann` | COMM-004 | L1 | operation requires combine, dispatch |
| `alltoallv_quant_gmm_cann` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `alltoallv_quant_gmm_cann` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `alltoallv_quant_gmm_cann` | COMM-007 | L1 | operation requires dispatch |
| `alltoallv_quant_gmm_cann` | COMM-008 | L1 | operation requires dispatch |
| `alltoallv_quant_gmm_cann` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `alltoallv_quant_gmm_cann` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `alltoallv_quant_gmm_cann` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `alltoallv_quant_gmm_cann` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_quant_gmm_cann` | FUSE-001 | L2 | operation requires dispatch |
| `alltoallv_quant_gmm_cann` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `alltoallv_quant_gmm_cann` | FUSE-003 | L2 | operation requires combine, dispatch |
| `alltoallv_quant_gmm_cann` | FUSE-004 | L2 | operation requires gmm_combine |
| `alltoallv_quant_gmm_cann` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `alltoallv_quant_gmm_cann` | FUSE-006 | L2 | operation requires tile_schedule |
| `alltoallv_quant_gmm_cann` | FUSE-007 | L2 | operation requires comm_resources |
| `alltoallv_quant_gmm_cann` | FUSE-009 | L2 | operation requires fused_backward |
| `alltoallv_quant_gmm_cann` | MOE-001 | L3 | operation requires moe_forward |
| `alltoallv_quant_gmm_cann` | MOE-002 | L3 | operation requires moe_backward |
| `alltoallv_quant_gmm_cann` | MOE-003 | L3 | operation requires moe_backward, moe_forward |
| `alltoallv_quant_gmm_cann` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `alltoallv_quant_gmm_cann` | MOE-005 | L3 | operation requires moe_forward |
| `alltoallv_quant_gmm_cann` | MOE-006 | L3 | operation requires moe_forward |
| `alltoallv_quant_gmm_cann` | MOE-007 | L3 | operation requires concurrent |
| `alltoallv_quant_gmm_cann` | MOE-008 | L3 | operation requires moe_forward |
| `alltoallv_quant_gmm_cann` | MOE-009 | L3 | operation requires dynamic_shapes, moe_forward |
| `alltoallv_quant_gmm_cann` | LAYER-001 | L4 | operation requires router_layer |
| `alltoallv_quant_gmm_cann` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `alltoallv_quant_gmm_cann` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_quant_gmm_cann` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_quant_gmm_cann` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `alltoallv_quant_gmm_cann` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `alltoallv_quant_gmm_cann` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `alltoallv_quant_gmm_cann` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `alltoallv_quant_gmm_cann` | LAYER-009 | L4 | operation requires moe_backward, router_layer |
| `chunkmoe_fsdpturbo` | COMM-001 | L1 | operation requires dispatch |
| `chunkmoe_fsdpturbo` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `chunkmoe_fsdpturbo` | COMM-003 | L1 | operation requires combine, dispatch |
| `chunkmoe_fsdpturbo` | COMM-004 | L1 | operation requires combine, dispatch |
| `chunkmoe_fsdpturbo` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `chunkmoe_fsdpturbo` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `chunkmoe_fsdpturbo` | COMM-007 | L1 | operation requires dispatch |
| `chunkmoe_fsdpturbo` | COMM-008 | L1 | operation requires dispatch |
| `chunkmoe_fsdpturbo` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `chunkmoe_fsdpturbo` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `chunkmoe_fsdpturbo` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `chunkmoe_fsdpturbo` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `chunkmoe_fsdpturbo` | FUSE-001 | L2 | operation requires dispatch |
| `chunkmoe_fsdpturbo` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `chunkmoe_fsdpturbo` | FUSE-003 | L2 | operation requires combine, dispatch |
| `chunkmoe_fsdpturbo` | FUSE-004 | L2 | operation requires gmm_combine |
| `chunkmoe_fsdpturbo` | FUSE-006 | L2 | operation requires tile_schedule |
| `chunkmoe_fsdpturbo` | FUSE-007 | L2 | operation requires comm_resources |
| `chunkmoe_fsdpturbo` | FUSE-008 | L2 | operation requires quantized_fusion |
| `chunkmoe_fsdpturbo` | FUSE-009 | L2 | operation requires fused_backward |
| `chunkmoe_fsdpturbo` | MOE-004 | L3 | operation requires recompute |
| `chunkmoe_fsdpturbo` | MOE-007 | L3 | operation requires concurrent |
| `chunkmoe_fsdpturbo` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `chunkmoe_fsdpturbo` | LAYER-001 | L4 | operation requires router_layer |
| `chunkmoe_fsdpturbo` | LAYER-002 | L4 | operation requires router_layer |
| `chunkmoe_fsdpturbo` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `chunkmoe_fsdpturbo` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `chunkmoe_fsdpturbo` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `chunkmoe_fsdpturbo` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `chunkmoe_fsdpturbo` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `chunkmoe_fsdpturbo` | LAYER-009 | L4 | operation requires router_layer |
| `deepep_ascend` | COMM-005 | L1 | operation requires async_dispatch |
| `deepep_ascend` | COMM-006 | L1 | operation requires async_combine |
| `deepep_ascend` | COMM-009 | L1 | operation requires quantized_dispatch |
| `deepep_ascend` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `deepep_ascend` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `deepep_ascend` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_ascend` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `deepep_ascend` | FUSE-004 | L2 | operation requires gmm_combine |
| `deepep_ascend` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `deepep_ascend` | FUSE-006 | L2 | operation requires tile_schedule |
| `deepep_ascend` | FUSE-007 | L2 | operation requires comm_resources |
| `deepep_ascend` | FUSE-008 | L2 | operation requires quantized_fusion |
| `deepep_ascend` | FUSE-009 | L2 | operation requires fused_backward |
| `deepep_ascend` | MOE-004 | L3 | operation requires recompute |
| `deepep_ascend` | MOE-007 | L3 | operation requires concurrent |
| `deepep_ascend` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `deepep_ascend` | LAYER-001 | L4 | operation requires router_layer |
| `deepep_ascend` | LAYER-002 | L4 | operation requires router_layer |
| `deepep_ascend` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_ascend` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_ascend` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `deepep_ascend` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `deepep_ascend` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_ascend` | LAYER-009 | L4 | operation requires router_layer |
| `deepep_cann` | COMM-001 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-002 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-003 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-004 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-005 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-006 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-007 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-008 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-009 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `deepep_cann` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `deepep_cann` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_cann` | FUSE-001 | L2 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `deepep_cann` | FUSE-003 | L2 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_cann` | FUSE-004 | L2 | operation requires gmm_combine |
| `deepep_cann` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `deepep_cann` | FUSE-006 | L2 | operation requires tile_schedule |
| `deepep_cann` | FUSE-007 | L2 | operation requires comm_resources |
| `deepep_cann` | FUSE-008 | L2 | operation requires quantized_fusion |
| `deepep_cann` | FUSE-009 | L2 | operation requires fused_backward |
| `deepep_cann` | MOE-004 | L3 | operation requires recompute |
| `deepep_cann` | MOE-005 | L3 | adapter requires routing from ['balanced'] |
| `deepep_cann` | MOE-007 | L3 | operation requires concurrent |
| `deepep_cann` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `deepep_cann` | LAYER-001 | L4 | operation requires router_layer |
| `deepep_cann` | LAYER-002 | L4 | operation requires router_layer |
| `deepep_cann` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_cann` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_cann` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `deepep_cann` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `deepep_cann` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `deepep_cann` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_cann` | LAYER-009 | L4 | operation requires router_layer |
| `deepep_direct_lueelu` | COMM-009 | L1 | operation requires quantized_dispatch |
| `deepep_direct_lueelu` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `deepep_direct_lueelu` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `deepep_direct_lueelu` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_direct_lueelu` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `deepep_direct_lueelu` | FUSE-004 | L2 | operation requires gmm_combine |
| `deepep_direct_lueelu` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `deepep_direct_lueelu` | FUSE-006 | L2 | operation requires tile_schedule |
| `deepep_direct_lueelu` | FUSE-007 | L2 | operation requires comm_resources |
| `deepep_direct_lueelu` | FUSE-008 | L2 | operation requires quantized_fusion |
| `deepep_direct_lueelu` | FUSE-009 | L2 | operation requires fused_backward |
| `deepep_direct_lueelu` | MOE-004 | L3 | operation requires recompute |
| `deepep_direct_lueelu` | MOE-007 | L3 | operation requires concurrent |
| `deepep_direct_lueelu` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `deepep_direct_lueelu` | LAYER-001 | L4 | operation requires router_layer |
| `deepep_direct_lueelu` | LAYER-002 | L4 | operation requires router_layer |
| `deepep_direct_lueelu` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_direct_lueelu` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_direct_lueelu` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `deepep_direct_lueelu` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `deepep_direct_lueelu` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_direct_lueelu` | LAYER-009 | L4 | operation requires router_layer |
| `deepep_lueelu` | COMM-009 | L1 | operation requires quantized_dispatch |
| `deepep_lueelu` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `deepep_lueelu` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `deepep_lueelu` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_lueelu` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `deepep_lueelu` | FUSE-004 | L2 | operation requires gmm_combine |
| `deepep_lueelu` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `deepep_lueelu` | FUSE-006 | L2 | operation requires tile_schedule |
| `deepep_lueelu` | FUSE-007 | L2 | operation requires comm_resources |
| `deepep_lueelu` | FUSE-008 | L2 | operation requires quantized_fusion |
| `deepep_lueelu` | FUSE-009 | L2 | operation requires fused_backward |
| `deepep_lueelu` | MOE-004 | L3 | operation requires recompute |
| `deepep_lueelu` | MOE-007 | L3 | operation requires concurrent |
| `deepep_lueelu` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `deepep_lueelu` | LAYER-001 | L4 | operation requires router_layer |
| `deepep_lueelu` | LAYER-002 | L4 | operation requires router_layer |
| `deepep_lueelu` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_lueelu` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_lueelu` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `deepep_lueelu` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `deepep_lueelu` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_lueelu` | LAYER-009 | L4 | operation requires router_layer |
| `deepep_sglang` | COMM-001 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-002 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-003 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-004 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-005 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-006 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-007 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-008 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-009 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `deepep_sglang` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `deepep_sglang` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_sglang` | FUSE-001 | L2 | Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `deepep_sglang` | FUSE-003 | L2 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `deepep_sglang` | FUSE-004 | L2 | operation requires gmm_combine |
| `deepep_sglang` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `deepep_sglang` | FUSE-006 | L2 | operation requires tile_schedule |
| `deepep_sglang` | FUSE-007 | L2 | operation requires comm_resources |
| `deepep_sglang` | FUSE-008 | L2 | operation requires quantized_fusion |
| `deepep_sglang` | FUSE-009 | L2 | operation requires fused_backward |
| `deepep_sglang` | MOE-004 | L3 | operation requires recompute |
| `deepep_sglang` | MOE-007 | L3 | operation requires concurrent |
| `deepep_sglang` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `deepep_sglang` | LAYER-001 | L4 | operation requires router_layer |
| `deepep_sglang` | LAYER-002 | L4 | operation requires router_layer |
| `deepep_sglang` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_sglang` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_sglang` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `deepep_sglang` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `deepep_sglang` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_sglang` | LAYER-009 | L4 | operation requires router_layer |
| `deepep_v2_ascend` | COMM-001 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-002 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-003 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-004 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-005 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-006 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-007 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-008 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-009 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `deepep_v2_ascend` | COMM-011 | L1 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_v2_ascend` | FUSE-001 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-002 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-003 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-004 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-005 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-006 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-007 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-008 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | FUSE-009 | L2 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-001 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-002 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-003 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-004 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-005 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-006 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-007 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-008 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | MOE-009 | L3 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | LAYER-001 | L4 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | LAYER-002 | L4 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `deepep_v2_ascend` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_v2_ascend` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_v2_ascend` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `deepep_v2_ascend` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `deepep_v2_ascend` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `deepep_v2_ascend` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `deepep_v2_ascend` | LAYER-009 | L4 | EP8 outside adapter's supported EP sizes [32, 64, 128] |
| `dispatch_ffn_combine_sglang` | COMM-001 | L1 | operation requires dispatch |
| `dispatch_ffn_combine_sglang` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `dispatch_ffn_combine_sglang` | COMM-003 | L1 | operation requires combine, dispatch |
| `dispatch_ffn_combine_sglang` | COMM-004 | L1 | operation requires combine, dispatch |
| `dispatch_ffn_combine_sglang` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `dispatch_ffn_combine_sglang` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `dispatch_ffn_combine_sglang` | COMM-007 | L1 | operation requires dispatch |
| `dispatch_ffn_combine_sglang` | COMM-008 | L1 | operation requires dispatch |
| `dispatch_ffn_combine_sglang` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `dispatch_ffn_combine_sglang` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `dispatch_ffn_combine_sglang` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `dispatch_ffn_combine_sglang` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_sglang` | FUSE-001 | L2 | operation requires dispatch |
| `dispatch_ffn_combine_sglang` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `dispatch_ffn_combine_sglang` | FUSE-003 | L2 | operation requires combine, dispatch |
| `dispatch_ffn_combine_sglang` | FUSE-004 | L2 | operation requires gmm_combine |
| `dispatch_ffn_combine_sglang` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `dispatch_ffn_combine_sglang` | FUSE-006 | L2 | operation requires tile_schedule |
| `dispatch_ffn_combine_sglang` | FUSE-007 | L2 | operation requires comm_resources |
| `dispatch_ffn_combine_sglang` | FUSE-008 | L2 | operation requires quantized_fusion |
| `dispatch_ffn_combine_sglang` | FUSE-009 | L2 | operation requires fused_backward |
| `dispatch_ffn_combine_sglang` | MOE-002 | L3 | operation requires moe_backward |
| `dispatch_ffn_combine_sglang` | MOE-003 | L3 | operation requires moe_backward |
| `dispatch_ffn_combine_sglang` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `dispatch_ffn_combine_sglang` | MOE-007 | L3 | operation requires concurrent |
| `dispatch_ffn_combine_sglang` | MOE-009 | L3 | operation requires dynamic_shapes |
| `dispatch_ffn_combine_sglang` | LAYER-001 | L4 | operation requires router_layer |
| `dispatch_ffn_combine_sglang` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `dispatch_ffn_combine_sglang` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_sglang` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_sglang` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `dispatch_ffn_combine_sglang` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `dispatch_ffn_combine_sglang` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_sglang` | LAYER-009 | L4 | operation requires moe_backward, router_layer |
| `dispatch_ffn_combine_vllm_ascend` | COMM-001 | L1 | operation requires dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-003 | L1 | operation requires combine, dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-004 | L1 | operation requires combine, dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-007 | L1 | operation requires dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-008 | L1 | operation requires dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `dispatch_ffn_combine_vllm_ascend` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `dispatch_ffn_combine_vllm_ascend` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `dispatch_ffn_combine_vllm_ascend` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-001 | L2 | operation requires dispatch |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-003 | L2 | operation requires combine, dispatch |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-004 | L2 | operation requires gmm_combine |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-006 | L2 | operation requires tile_schedule |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-007 | L2 | operation requires comm_resources |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-008 | L2 | operation requires quantized_fusion |
| `dispatch_ffn_combine_vllm_ascend` | FUSE-009 | L2 | operation requires fused_backward |
| `dispatch_ffn_combine_vllm_ascend` | MOE-002 | L3 | operation requires moe_backward |
| `dispatch_ffn_combine_vllm_ascend` | MOE-003 | L3 | operation requires moe_backward |
| `dispatch_ffn_combine_vllm_ascend` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `dispatch_ffn_combine_vllm_ascend` | MOE-007 | L3 | operation requires concurrent |
| `dispatch_ffn_combine_vllm_ascend` | MOE-009 | L3 | operation requires dynamic_shapes |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-001 | L4 | operation requires router_layer |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `dispatch_ffn_combine_vllm_ascend` | LAYER-009 | L4 | operation requires moe_backward, router_layer |
| `domino_fsdpturbo` | COMM-001 | L1 | operation requires dispatch |
| `domino_fsdpturbo` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `domino_fsdpturbo` | COMM-003 | L1 | operation requires combine, dispatch |
| `domino_fsdpturbo` | COMM-004 | L1 | operation requires combine, dispatch |
| `domino_fsdpturbo` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `domino_fsdpturbo` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `domino_fsdpturbo` | COMM-007 | L1 | operation requires dispatch |
| `domino_fsdpturbo` | COMM-008 | L1 | operation requires dispatch |
| `domino_fsdpturbo` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `domino_fsdpturbo` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `domino_fsdpturbo` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `domino_fsdpturbo` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `domino_fsdpturbo` | FUSE-001 | L2 | operation requires dispatch |
| `domino_fsdpturbo` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `domino_fsdpturbo` | FUSE-003 | L2 | operation requires combine, dispatch |
| `domino_fsdpturbo` | FUSE-004 | L2 | operation requires gmm_combine |
| `domino_fsdpturbo` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `domino_fsdpturbo` | FUSE-006 | L2 | operation requires tile_schedule |
| `domino_fsdpturbo` | FUSE-007 | L2 | operation requires comm_resources |
| `domino_fsdpturbo` | FUSE-008 | L2 | operation requires quantized_fusion |
| `domino_fsdpturbo` | FUSE-009 | L2 | operation requires fused_backward |
| `domino_fsdpturbo` | MOE-004 | L3 | operation requires recompute |
| `domino_fsdpturbo` | MOE-007 | L3 | operation requires concurrent |
| `domino_fsdpturbo` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `domino_fsdpturbo` | LAYER-001 | L4 | operation requires router_layer |
| `domino_fsdpturbo` | LAYER-002 | L4 | operation requires router_layer |
| `domino_fsdpturbo` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `domino_fsdpturbo` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `domino_fsdpturbo` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `domino_fsdpturbo` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `domino_fsdpturbo` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `domino_fsdpturbo` | LAYER-009 | L4 | operation requires router_layer |
| `ep_gmm_fused_lueelu` | COMM-001 | L1 | operation requires dispatch |
| `ep_gmm_fused_lueelu` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `ep_gmm_fused_lueelu` | COMM-003 | L1 | operation requires combine, dispatch |
| `ep_gmm_fused_lueelu` | COMM-004 | L1 | operation requires combine, dispatch |
| `ep_gmm_fused_lueelu` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `ep_gmm_fused_lueelu` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `ep_gmm_fused_lueelu` | COMM-007 | L1 | operation requires dispatch |
| `ep_gmm_fused_lueelu` | COMM-008 | L1 | operation requires dispatch |
| `ep_gmm_fused_lueelu` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `ep_gmm_fused_lueelu` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `ep_gmm_fused_lueelu` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `ep_gmm_fused_lueelu` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `ep_gmm_fused_lueelu` | FUSE-001 | L2 | operation requires dispatch |
| `ep_gmm_fused_lueelu` | FUSE-003 | L2 | operation requires combine, dispatch |
| `ep_gmm_fused_lueelu` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `ep_gmm_fused_lueelu` | FUSE-008 | L2 | operation requires quantized_fusion |
| `ep_gmm_fused_lueelu` | MOE-004 | L3 | operation requires recompute |
| `ep_gmm_fused_lueelu` | MOE-007 | L3 | operation requires concurrent |
| `ep_gmm_fused_lueelu` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `ep_gmm_fused_lueelu` | LAYER-001 | L4 | operation requires router_layer |
| `ep_gmm_fused_lueelu` | LAYER-002 | L4 | operation requires router_layer |
| `ep_gmm_fused_lueelu` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `ep_gmm_fused_lueelu` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `ep_gmm_fused_lueelu` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `ep_gmm_fused_lueelu` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `ep_gmm_fused_lueelu` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `ep_gmm_fused_lueelu` | LAYER-009 | L4 | operation requires router_layer |
| `fused_deep_moe_sglang` | COMM-001 | L1 | operation requires dispatch |
| `fused_deep_moe_sglang` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `fused_deep_moe_sglang` | COMM-003 | L1 | operation requires combine, dispatch |
| `fused_deep_moe_sglang` | COMM-004 | L1 | operation requires combine, dispatch |
| `fused_deep_moe_sglang` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `fused_deep_moe_sglang` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `fused_deep_moe_sglang` | COMM-007 | L1 | operation requires dispatch |
| `fused_deep_moe_sglang` | COMM-008 | L1 | operation requires dispatch |
| `fused_deep_moe_sglang` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `fused_deep_moe_sglang` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `fused_deep_moe_sglang` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `fused_deep_moe_sglang` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_deep_moe_sglang` | FUSE-001 | L2 | operation requires dispatch |
| `fused_deep_moe_sglang` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `fused_deep_moe_sglang` | FUSE-003 | L2 | operation requires combine, dispatch |
| `fused_deep_moe_sglang` | FUSE-004 | L2 | operation requires gmm_combine |
| `fused_deep_moe_sglang` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `fused_deep_moe_sglang` | FUSE-006 | L2 | operation requires tile_schedule |
| `fused_deep_moe_sglang` | FUSE-007 | L2 | operation requires comm_resources |
| `fused_deep_moe_sglang` | FUSE-008 | L2 | operation requires quantized_fusion |
| `fused_deep_moe_sglang` | FUSE-009 | L2 | operation requires fused_backward |
| `fused_deep_moe_sglang` | MOE-002 | L3 | operation requires moe_backward |
| `fused_deep_moe_sglang` | MOE-003 | L3 | operation requires moe_backward |
| `fused_deep_moe_sglang` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `fused_deep_moe_sglang` | MOE-007 | L3 | operation requires concurrent |
| `fused_deep_moe_sglang` | MOE-009 | L3 | operation requires dynamic_shapes |
| `fused_deep_moe_sglang` | LAYER-001 | L4 | operation requires router_layer |
| `fused_deep_moe_sglang` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `fused_deep_moe_sglang` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_deep_moe_sglang` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_deep_moe_sglang` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `fused_deep_moe_sglang` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `fused_deep_moe_sglang` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_deep_moe_sglang` | LAYER-009 | L4 | operation requires moe_backward, router_layer |
| `fused_fsdpturbo` | COMM-001 | L1 | operation requires dispatch |
| `fused_fsdpturbo` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `fused_fsdpturbo` | COMM-003 | L1 | operation requires combine, dispatch |
| `fused_fsdpturbo` | COMM-004 | L1 | operation requires combine, dispatch |
| `fused_fsdpturbo` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `fused_fsdpturbo` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `fused_fsdpturbo` | COMM-007 | L1 | operation requires dispatch |
| `fused_fsdpturbo` | COMM-008 | L1 | operation requires dispatch |
| `fused_fsdpturbo` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `fused_fsdpturbo` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `fused_fsdpturbo` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `fused_fsdpturbo` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_fsdpturbo` | FUSE-001 | L2 | operation requires dispatch |
| `fused_fsdpturbo` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `fused_fsdpturbo` | FUSE-003 | L2 | operation requires combine, dispatch |
| `fused_fsdpturbo` | FUSE-004 | L2 | operation requires gmm_combine |
| `fused_fsdpturbo` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `fused_fsdpturbo` | FUSE-006 | L2 | operation requires tile_schedule |
| `fused_fsdpturbo` | FUSE-007 | L2 | operation requires comm_resources |
| `fused_fsdpturbo` | FUSE-008 | L2 | operation requires quantized_fusion |
| `fused_fsdpturbo` | FUSE-009 | L2 | operation requires fused_backward |
| `fused_fsdpturbo` | MOE-004 | L3 | operation requires recompute |
| `fused_fsdpturbo` | MOE-007 | L3 | operation requires concurrent |
| `fused_fsdpturbo` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `fused_fsdpturbo` | LAYER-001 | L4 | operation requires router_layer |
| `fused_fsdpturbo` | LAYER-002 | L4 | operation requires router_layer |
| `fused_fsdpturbo` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_fsdpturbo` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_fsdpturbo` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `fused_fsdpturbo` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `fused_fsdpturbo` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `fused_fsdpturbo` | LAYER-009 | L4 | operation requires router_layer |
| `hierarchical_alltoallv_mindspeed` | COMM-001 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-002 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-003 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-004 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-005 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-006 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-007 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-008 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-009 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `hierarchical_alltoallv_mindspeed` | COMM-011 | L1 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `hierarchical_alltoallv_mindspeed` | FUSE-001 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-002 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-003 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-004 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-005 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-006 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-007 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-008 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | FUSE-009 | L2 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-001 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-002 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-003 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-004 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-005 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-006 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-007 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-008 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | MOE-009 | L3 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | LAYER-001 | L4 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | LAYER-002 | L4 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `hierarchical_alltoallv_mindspeed` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `hierarchical_alltoallv_mindspeed` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `hierarchical_alltoallv_mindspeed` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `hierarchical_alltoallv_mindspeed` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `hierarchical_alltoallv_mindspeed` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `hierarchical_alltoallv_mindspeed` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `hierarchical_alltoallv_mindspeed` | LAYER-009 | L4 | native hierarchy requires at least two outer ranks, each with at least two inner ranks |
| `mc2_combine_add_rmsnorm_cann` | COMM-001 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-002 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-003 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-004 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-005 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-006 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-007 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-008 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-009 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `mc2_combine_add_rmsnorm_cann` | COMM-011 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_combine_add_rmsnorm_cann` | FUSE-001 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-002 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-003 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-004 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-005 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-006 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-007 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-008 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | FUSE-009 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-001 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-002 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-003 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-004 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-005 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-006 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-007 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-008 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | MOE-009 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | LAYER-001 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | LAYER-002 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_combine_add_rmsnorm_cann` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_combine_add_rmsnorm_cann` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_combine_add_rmsnorm_cann` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `mc2_combine_add_rmsnorm_cann` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `mc2_combine_add_rmsnorm_cann` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `mc2_combine_add_rmsnorm_cann` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_combine_add_rmsnorm_cann` | LAYER-009 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann` | COMM-001 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-002 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-003 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-004 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-005 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-006 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-007 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-008 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-009 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `mc2_dispatch_cann` | COMM-011 | L1 | a single 950DT V2 invocation supports at most 512 tokens per rank |
| `mc2_dispatch_cann` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `mc2_dispatch_cann` | FUSE-004 | L2 | operation requires gmm_combine |
| `mc2_dispatch_cann` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `mc2_dispatch_cann` | FUSE-006 | L2 | operation requires tile_schedule |
| `mc2_dispatch_cann` | FUSE-007 | L2 | operation requires comm_resources |
| `mc2_dispatch_cann` | FUSE-008 | L2 | operation requires quantized_fusion |
| `mc2_dispatch_cann` | FUSE-009 | L2 | operation requires fused_backward |
| `mc2_dispatch_cann` | MOE-004 | L3 | operation requires recompute |
| `mc2_dispatch_cann` | MOE-007 | L3 | operation requires concurrent |
| `mc2_dispatch_cann` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `mc2_dispatch_cann` | LAYER-001 | L4 | operation requires router_layer |
| `mc2_dispatch_cann` | LAYER-002 | L4 | operation requires router_layer |
| `mc2_dispatch_cann` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `mc2_dispatch_cann` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `mc2_dispatch_cann` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann` | LAYER-009 | L4 | operation requires router_layer |
| `mc2_dispatch_cann_v1` | COMM-001 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-002 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-003 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-004 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-005 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-006 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-007 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-008 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-009 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `mc2_dispatch_cann_v1` | COMM-011 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v1` | FUSE-001 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-002 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-003 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-004 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-005 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-006 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-007 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-008 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | FUSE-009 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-001 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-002 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-003 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-004 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-005 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-006 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-007 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-008 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | MOE-009 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | LAYER-001 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | LAYER-002 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v1` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v1` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v1` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `mc2_dispatch_cann_v1` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `mc2_dispatch_cann_v1` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `mc2_dispatch_cann_v1` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v1` | LAYER-009 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-001 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-002 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-003 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-004 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-005 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-006 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-007 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-008 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-009 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `mc2_dispatch_cann_v3` | COMM-011 | L1 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v3` | FUSE-001 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-002 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-003 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-004 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-005 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-006 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-007 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-008 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | FUSE-009 | L2 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-001 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-002 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-003 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-004 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-005 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-006 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-007 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-008 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | MOE-009 | L3 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | LAYER-001 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | LAYER-002 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_dispatch_cann_v3` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v3` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v3` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `mc2_dispatch_cann_v3` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `mc2_dispatch_cann_v3` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `mc2_dispatch_cann_v3` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_dispatch_cann_v3` | LAYER-009 | L4 | tokens per rank exceeds adapter capacity 512 |
| `mc2_fsdpturbo` | COMM-001 | L1 | operation requires dispatch |
| `mc2_fsdpturbo` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `mc2_fsdpturbo` | COMM-003 | L1 | operation requires combine, dispatch |
| `mc2_fsdpturbo` | COMM-004 | L1 | operation requires combine, dispatch |
| `mc2_fsdpturbo` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `mc2_fsdpturbo` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `mc2_fsdpturbo` | COMM-007 | L1 | operation requires dispatch |
| `mc2_fsdpturbo` | COMM-008 | L1 | operation requires dispatch |
| `mc2_fsdpturbo` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `mc2_fsdpturbo` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `mc2_fsdpturbo` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `mc2_fsdpturbo` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_fsdpturbo` | FUSE-001 | L2 | operation requires dispatch |
| `mc2_fsdpturbo` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `mc2_fsdpturbo` | FUSE-003 | L2 | operation requires combine, dispatch |
| `mc2_fsdpturbo` | FUSE-004 | L2 | operation requires gmm_combine |
| `mc2_fsdpturbo` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `mc2_fsdpturbo` | FUSE-006 | L2 | operation requires tile_schedule |
| `mc2_fsdpturbo` | FUSE-007 | L2 | operation requires comm_resources |
| `mc2_fsdpturbo` | FUSE-008 | L2 | operation requires quantized_fusion |
| `mc2_fsdpturbo` | FUSE-009 | L2 | operation requires fused_backward |
| `mc2_fsdpturbo` | MOE-004 | L3 | operation requires recompute |
| `mc2_fsdpturbo` | MOE-007 | L3 | operation requires concurrent |
| `mc2_fsdpturbo` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `mc2_fsdpturbo` | LAYER-001 | L4 | operation requires router_layer |
| `mc2_fsdpturbo` | LAYER-002 | L4 | operation requires router_layer |
| `mc2_fsdpturbo` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_fsdpturbo` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_fsdpturbo` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `mc2_fsdpturbo` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `mc2_fsdpturbo` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `mc2_fsdpturbo` | LAYER-009 | L4 | operation requires router_layer |
| `megamoe_cann` | COMM-001 | L1 | operation requires dispatch |
| `megamoe_cann` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `megamoe_cann` | COMM-003 | L1 | operation requires combine, dispatch |
| `megamoe_cann` | COMM-004 | L1 | operation requires combine, dispatch |
| `megamoe_cann` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `megamoe_cann` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `megamoe_cann` | COMM-007 | L1 | operation requires dispatch |
| `megamoe_cann` | COMM-008 | L1 | operation requires dispatch |
| `megamoe_cann` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `megamoe_cann` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `megamoe_cann` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `megamoe_cann` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_cann` | FUSE-001 | L2 | operation requires dispatch |
| `megamoe_cann` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `megamoe_cann` | FUSE-003 | L2 | operation requires combine, dispatch |
| `megamoe_cann` | FUSE-004 | L2 | operation requires gmm_combine |
| `megamoe_cann` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `megamoe_cann` | FUSE-006 | L2 | operation requires tile_schedule |
| `megamoe_cann` | FUSE-007 | L2 | operation requires comm_resources |
| `megamoe_cann` | FUSE-008 | L2 | operation requires quantized_fusion |
| `megamoe_cann` | FUSE-009 | L2 | operation requires fused_backward |
| `megamoe_cann` | MOE-002 | L3 | operation requires moe_backward |
| `megamoe_cann` | MOE-003 | L3 | operation requires moe_backward |
| `megamoe_cann` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `megamoe_cann` | MOE-007 | L3 | operation requires concurrent |
| `megamoe_cann` | MOE-009 | L3 | Symmetric window resize and reuse are pending |
| `megamoe_cann` | LAYER-001 | L4 | operation requires router_layer |
| `megamoe_cann` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `megamoe_cann` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_cann` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_cann` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `megamoe_cann` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `megamoe_cann` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_cann` | LAYER-009 | L4 | operation requires moe_backward, router_layer |
| `megamoe_deepgemm_ascend` | COMM-001 | L1 | operation requires dispatch |
| `megamoe_deepgemm_ascend` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `megamoe_deepgemm_ascend` | COMM-003 | L1 | operation requires combine, dispatch |
| `megamoe_deepgemm_ascend` | COMM-004 | L1 | operation requires combine, dispatch |
| `megamoe_deepgemm_ascend` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `megamoe_deepgemm_ascend` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `megamoe_deepgemm_ascend` | COMM-007 | L1 | operation requires dispatch |
| `megamoe_deepgemm_ascend` | COMM-008 | L1 | operation requires dispatch |
| `megamoe_deepgemm_ascend` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `megamoe_deepgemm_ascend` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `megamoe_deepgemm_ascend` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `megamoe_deepgemm_ascend` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_deepgemm_ascend` | FUSE-001 | L2 | operation requires dispatch |
| `megamoe_deepgemm_ascend` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `megamoe_deepgemm_ascend` | FUSE-003 | L2 | operation requires combine, dispatch |
| `megamoe_deepgemm_ascend` | FUSE-004 | L2 | operation requires gmm_combine |
| `megamoe_deepgemm_ascend` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `megamoe_deepgemm_ascend` | FUSE-006 | L2 | operation requires tile_schedule |
| `megamoe_deepgemm_ascend` | FUSE-007 | L2 | operation requires comm_resources |
| `megamoe_deepgemm_ascend` | FUSE-008 | L2 | operation requires quantized_fusion |
| `megamoe_deepgemm_ascend` | FUSE-009 | L2 | operation requires fused_backward |
| `megamoe_deepgemm_ascend` | MOE-002 | L3 | operation requires moe_backward |
| `megamoe_deepgemm_ascend` | MOE-003 | L3 | operation requires moe_backward |
| `megamoe_deepgemm_ascend` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `megamoe_deepgemm_ascend` | MOE-007 | L3 | operation requires concurrent |
| `megamoe_deepgemm_ascend` | MOE-009 | L3 | operation requires dynamic_shapes |
| `megamoe_deepgemm_ascend` | LAYER-001 | L4 | operation requires router_layer |
| `megamoe_deepgemm_ascend` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `megamoe_deepgemm_ascend` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_deepgemm_ascend` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_deepgemm_ascend` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `megamoe_deepgemm_ascend` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `megamoe_deepgemm_ascend` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_deepgemm_ascend` | LAYER-009 | L4 | operation requires moe_backward, router_layer |
| `megamoe_fsdpturbo` | COMM-001 | L1 | operation requires dispatch |
| `megamoe_fsdpturbo` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `megamoe_fsdpturbo` | COMM-003 | L1 | operation requires combine, dispatch |
| `megamoe_fsdpturbo` | COMM-004 | L1 | operation requires combine, dispatch |
| `megamoe_fsdpturbo` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `megamoe_fsdpturbo` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `megamoe_fsdpturbo` | COMM-007 | L1 | operation requires dispatch |
| `megamoe_fsdpturbo` | COMM-008 | L1 | operation requires dispatch |
| `megamoe_fsdpturbo` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `megamoe_fsdpturbo` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `megamoe_fsdpturbo` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `megamoe_fsdpturbo` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_fsdpturbo` | FUSE-001 | L2 | operation requires dispatch |
| `megamoe_fsdpturbo` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `megamoe_fsdpturbo` | FUSE-003 | L2 | operation requires combine, dispatch |
| `megamoe_fsdpturbo` | FUSE-004 | L2 | operation requires gmm_combine |
| `megamoe_fsdpturbo` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `megamoe_fsdpturbo` | FUSE-006 | L2 | operation requires tile_schedule |
| `megamoe_fsdpturbo` | FUSE-007 | L2 | operation requires comm_resources |
| `megamoe_fsdpturbo` | FUSE-008 | L2 | operation requires quantized_fusion |
| `megamoe_fsdpturbo` | FUSE-009 | L2 | operation requires fused_backward |
| `megamoe_fsdpturbo` | MOE-004 | L3 | operation requires recompute |
| `megamoe_fsdpturbo` | MOE-007 | L3 | operation requires concurrent |
| `megamoe_fsdpturbo` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `megamoe_fsdpturbo` | LAYER-001 | L4 | operation requires router_layer |
| `megamoe_fsdpturbo` | LAYER-002 | L4 | operation requires router_layer |
| `megamoe_fsdpturbo` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_fsdpturbo` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_fsdpturbo` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `megamoe_fsdpturbo` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `megamoe_fsdpturbo` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_fsdpturbo` | LAYER-009 | L4 | operation requires router_layer |
| `megamoe_lueelu` | COMM-001 | L1 | operation requires dispatch |
| `megamoe_lueelu` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `megamoe_lueelu` | COMM-003 | L1 | operation requires combine, dispatch |
| `megamoe_lueelu` | COMM-004 | L1 | operation requires combine, dispatch |
| `megamoe_lueelu` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `megamoe_lueelu` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `megamoe_lueelu` | COMM-007 | L1 | operation requires dispatch |
| `megamoe_lueelu` | COMM-008 | L1 | operation requires dispatch |
| `megamoe_lueelu` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `megamoe_lueelu` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `megamoe_lueelu` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `megamoe_lueelu` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_lueelu` | FUSE-001 | L2 | operation requires dispatch |
| `megamoe_lueelu` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `megamoe_lueelu` | FUSE-003 | L2 | operation requires combine, dispatch |
| `megamoe_lueelu` | FUSE-004 | L2 | operation requires gmm_combine |
| `megamoe_lueelu` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `megamoe_lueelu` | FUSE-006 | L2 | operation requires tile_schedule |
| `megamoe_lueelu` | FUSE-007 | L2 | operation requires comm_resources |
| `megamoe_lueelu` | FUSE-008 | L2 | operation requires quantized_fusion |
| `megamoe_lueelu` | FUSE-009 | L2 | operation requires fused_backward |
| `megamoe_lueelu` | MOE-004 | L3 | operation requires recompute |
| `megamoe_lueelu` | MOE-005 | L3 | adapter requires routing from ['balanced'] |
| `megamoe_lueelu` | MOE-007 | L3 | operation requires concurrent |
| `megamoe_lueelu` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `megamoe_lueelu` | LAYER-001 | L4 | operation requires router_layer |
| `megamoe_lueelu` | LAYER-002 | L4 | operation requires router_layer |
| `megamoe_lueelu` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_lueelu` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_lueelu` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `megamoe_lueelu` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `megamoe_lueelu` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `megamoe_lueelu` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_lueelu` | LAYER-009 | L4 | operation requires router_layer |
| `megamoe_td` | COMM-001 | L1 | operation requires dispatch |
| `megamoe_td` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `megamoe_td` | COMM-003 | L1 | operation requires combine, dispatch |
| `megamoe_td` | COMM-004 | L1 | operation requires combine, dispatch |
| `megamoe_td` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `megamoe_td` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `megamoe_td` | COMM-007 | L1 | operation requires dispatch |
| `megamoe_td` | COMM-008 | L1 | operation requires dispatch |
| `megamoe_td` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `megamoe_td` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `megamoe_td` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `megamoe_td` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_td` | FUSE-001 | L2 | operation requires dispatch |
| `megamoe_td` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `megamoe_td` | FUSE-003 | L2 | operation requires combine, dispatch |
| `megamoe_td` | FUSE-004 | L2 | operation requires gmm_combine |
| `megamoe_td` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `megamoe_td` | FUSE-006 | L2 | operation requires tile_schedule |
| `megamoe_td` | FUSE-007 | L2 | operation requires comm_resources |
| `megamoe_td` | FUSE-008 | L2 | operation requires quantized_fusion |
| `megamoe_td` | FUSE-009 | L2 | operation requires fused_backward |
| `megamoe_td` | MOE-004 | L3 | operation requires recompute |
| `megamoe_td` | MOE-005 | L3 | adapter requires routing from ['balanced'] |
| `megamoe_td` | MOE-007 | L3 | operation requires concurrent |
| `megamoe_td` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `megamoe_td` | LAYER-001 | L4 | operation requires router_layer |
| `megamoe_td` | LAYER-002 | L4 | operation requires router_layer |
| `megamoe_td` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_td` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_td` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `megamoe_td` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `megamoe_td` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `megamoe_td` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `megamoe_td` | LAYER-009 | L4 | operation requires router_layer |
| `moonep` | COMM-001 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-002 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-003 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-004 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-005 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-006 | L1 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-007 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-008 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-009 | L1 | Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `moonep` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `moonep` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep` | FUSE-001 | L2 | Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `moonep` | FUSE-003 | L2 | Native combine exists. Standalone layout validation is pending.; Native dispatch exists. Standalone layout validation is pending. |
| `moonep` | FUSE-004 | L2 | operation requires gmm_combine |
| `moonep` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `moonep` | FUSE-006 | L2 | operation requires tile_schedule |
| `moonep` | FUSE-007 | L2 | operation requires comm_resources |
| `moonep` | FUSE-008 | L2 | operation requires quantized_fusion |
| `moonep` | FUSE-009 | L2 | operation requires fused_backward |
| `moonep` | MOE-004 | L3 | operation requires recompute |
| `moonep` | MOE-005 | L3 | adapter requires routing from ['balanced'] |
| `moonep` | MOE-007 | L3 | operation requires concurrent |
| `moonep` | MOE-009 | L3 | Native resized workspace lifecycle is pending |
| `moonep` | LAYER-001 | L4 | operation requires router_layer |
| `moonep` | LAYER-002 | L4 | operation requires router_layer |
| `moonep` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `moonep` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `moonep` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `moonep` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep` | LAYER-009 | L4 | operation requires router_layer |
| `moonep_dynamic_planner` | COMM-001 | L1 | operation requires dispatch |
| `moonep_dynamic_planner` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `moonep_dynamic_planner` | COMM-003 | L1 | operation requires combine, dispatch |
| `moonep_dynamic_planner` | COMM-004 | L1 | operation requires combine, dispatch |
| `moonep_dynamic_planner` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `moonep_dynamic_planner` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `moonep_dynamic_planner` | COMM-007 | L1 | operation requires dispatch |
| `moonep_dynamic_planner` | COMM-008 | L1 | operation requires dispatch |
| `moonep_dynamic_planner` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `moonep_dynamic_planner` | COMM-010 | L1 | deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none |
| `moonep_dynamic_planner` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `moonep_dynamic_planner` | FUSE-001 | L2 | operation requires dispatch |
| `moonep_dynamic_planner` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `moonep_dynamic_planner` | FUSE-003 | L2 | operation requires combine, dispatch |
| `moonep_dynamic_planner` | FUSE-004 | L2 | operation requires gmm_combine |
| `moonep_dynamic_planner` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `moonep_dynamic_planner` | FUSE-006 | L2 | operation requires tile_schedule |
| `moonep_dynamic_planner` | FUSE-007 | L2 | operation requires comm_resources |
| `moonep_dynamic_planner` | FUSE-008 | L2 | operation requires quantized_fusion |
| `moonep_dynamic_planner` | FUSE-009 | L2 | operation requires fused_backward |
| `moonep_dynamic_planner` | MOE-001 | L3 | operation requires moe_forward |
| `moonep_dynamic_planner` | MOE-002 | L3 | operation requires moe_backward |
| `moonep_dynamic_planner` | MOE-003 | L3 | operation requires moe_backward, moe_forward |
| `moonep_dynamic_planner` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `moonep_dynamic_planner` | MOE-005 | L3 | operation requires moe_forward |
| `moonep_dynamic_planner` | MOE-006 | L3 | operation requires moe_forward |
| `moonep_dynamic_planner` | MOE-007 | L3 | operation requires concurrent |
| `moonep_dynamic_planner` | MOE-008 | L3 | operation requires moe_forward |
| `moonep_dynamic_planner` | MOE-009 | L3 | operation requires dynamic_shapes, moe_forward |
| `moonep_dynamic_planner` | LAYER-001 | L4 | operation requires router_layer |
| `moonep_dynamic_planner` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `moonep_dynamic_planner` | LAYER-005 | L4 | this case measures rank dedup; the plan selects dedup.mode=none |
| `moonep_dynamic_planner` | LAYER-006 | L4 | this case measures server dedup; the plan selects dedup.mode=none |
| `moonep_dynamic_planner` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `moonep_dynamic_planner` | LAYER-009 | L4 | operation requires moe_backward, router_layer |
| `moonep_server_dedup` | COMM-001 | L1 | operation requires dispatch |
| `moonep_server_dedup` | COMM-002 | L1 | operation requires cached_dispatch, dispatch |
| `moonep_server_dedup` | COMM-003 | L1 | operation requires combine, dispatch |
| `moonep_server_dedup` | COMM-004 | L1 | operation requires combine, dispatch |
| `moonep_server_dedup` | COMM-005 | L1 | operation requires async_dispatch, dispatch |
| `moonep_server_dedup` | COMM-006 | L1 | operation requires async_combine, combine, dispatch |
| `moonep_server_dedup` | COMM-007 | L1 | operation requires dispatch |
| `moonep_server_dedup` | COMM-008 | L1 | operation requires dispatch |
| `moonep_server_dedup` | COMM-009 | L1 | operation requires dispatch, quantized_dispatch |
| `moonep_server_dedup` | COMM-011 | L1 | operation requires combine_add_rmsnorm |
| `moonep_server_dedup` | COMM-012 | L1 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep_server_dedup` | FUSE-001 | L2 | operation requires dispatch |
| `moonep_server_dedup` | FUSE-002 | L2 | operation requires dispatch_gmm |
| `moonep_server_dedup` | FUSE-003 | L2 | operation requires combine, dispatch |
| `moonep_server_dedup` | FUSE-004 | L2 | operation requires gmm_combine |
| `moonep_server_dedup` | FUSE-005 | L2 | operation requires chunk_pipeline |
| `moonep_server_dedup` | FUSE-006 | L2 | operation requires tile_schedule |
| `moonep_server_dedup` | FUSE-007 | L2 | operation requires comm_resources |
| `moonep_server_dedup` | FUSE-008 | L2 | operation requires quantized_fusion |
| `moonep_server_dedup` | FUSE-009 | L2 | operation requires fused_backward |
| `moonep_server_dedup` | MOE-001 | L3 | operation requires moe_forward |
| `moonep_server_dedup` | MOE-002 | L3 | operation requires moe_backward |
| `moonep_server_dedup` | MOE-003 | L3 | operation requires moe_backward, moe_forward |
| `moonep_server_dedup` | MOE-004 | L3 | operation requires moe_backward, recompute |
| `moonep_server_dedup` | MOE-005 | L3 | operation requires moe_forward |
| `moonep_server_dedup` | MOE-006 | L3 | operation requires moe_forward |
| `moonep_server_dedup` | MOE-007 | L3 | operation requires concurrent |
| `moonep_server_dedup` | MOE-008 | L3 | operation requires moe_forward |
| `moonep_server_dedup` | MOE-009 | L3 | operation requires dynamic_shapes, moe_forward |
| `moonep_server_dedup` | LAYER-001 | L4 | operation requires router_layer |
| `moonep_server_dedup` | LAYER-002 | L4 | operation requires moe_backward, router_layer |
| `moonep_server_dedup` | LAYER-003 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep_server_dedup` | LAYER-004 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep_server_dedup` | LAYER-007 | L3 | L3 replay needs a real_trace_replay workload; this configuration does not select one |
| `moonep_server_dedup` | LAYER-008 | L4 | the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true |
| `moonep_server_dedup` | LAYER-009 | L4 | operation requires moe_backward, router_layer |

## 4. 层次定义对照

| 层次 | 测试对象 | 入口 |
|---|---|---|
| L0 | 硬件与基础能力（HBM/P2P/HCCL/SHMEM/GMM/SwiGLU/并发），与具体 EP 算子解耦 | `benchmarks/l0_hardware.py` → `baselines/l0/` |
| L1 | 独立通信接口（Dispatch/Combine/缓存/异步/量化/去重/规划） | `benchmarks/cases/l1_communication.py` |
| L2 | 通信与计算融合：串行组合（FUSE-001/003）与原生融合（其余 FUSE） | `benchmarks/cases/l2_fusion.py` |
| L3 | 完整 MoE 执行（前向/反向/重算/并发/多层/动态 token/trace replay） | `benchmarks/cases/l3_moe.py` |
| L4 | 路由器、动态规划与去重层次 | `benchmarks/cases/l4_router_planner.py` |

## 5. 本次审计发现的结构性问题

- `EXEC-001` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `EXEC-002` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `EXEC-003` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `EXEC-004` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `EXEC-005` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `EXEC-006` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-001` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-002` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-003` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-004` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-005` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-006` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-010` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `HW-011` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-001` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-002` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-003` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-004` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-005` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-006` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-007` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-008` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-009` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-010` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-011` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-012` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `LINK-013` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `MEM-001` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `MEM-002` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `MEM-003` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `MEM-004` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-001` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-002` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-003` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-004` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-005` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-006` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-007` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-008` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- `OVERLAP-009` 既无 `requires` 也无条件能力，静态分析会误判为全算子可运行。
- L0 与算子的底层映射尚未按源码逐项核实，`registry.yaml` 中的 `underlying` 字段保持 `unknown`，不按算子名称推断传输路径。
- COMM-011 的计时区间不包含准备阶段 Dispatch，已改为 `setup_requires` 表达。
- MOE-002/MOE-004 的前向图在 setup 构建，已改为 `setup_requires` 表达。

