# EP 算子接入矩阵

`historical_verified` 表示固定参考报告有实测证据。当前框架分别记录新运行状态。L0 由公共硬件基线执行。

| 算子 | 入口 | 已接入层次 | 接入状态 |
| --- | --- | --- | --- |
| [`allgather_hccl`](allgather_hccl/README.md) | torch.distributed.all_gather / reduce_scatter_tensor | L1/L2/L3/L4 | `integrated` |
| [`allgather_megatron`](allgather_megatron/README.md) | MoEAllGatherTokenDispatcher | L3 | `historical_verified` |
| [`allgather_fsdpturbo`](allgather_fsdpturbo/README.md) | all_gather_experts_forward_fn | L3 | `historical_verified` |
| [`allgather_overlap_mindspeed`](allgather_overlap_mindspeed/README.md) | MoELayerOverlapAllGather.apply | L3 | `historical_verified` |
| [`alltoall_hccl`](alltoall_hccl/README.md) | torch.distributed.all_to_all_single | L1/L2/L3/L4 | `integrated` |
| [`alltoall_megatron`](alltoall_megatron/README.md) | MoEAlltoAllTokenDispatcher | L3 | `historical_verified` |
| [`alltoall_fsdpturbo`](alltoall_fsdpturbo/README.md) | all2all_experts_forward_fn (fused=False) | L3 | `historical_verified` |
| [`fused_fsdpturbo`](fused_fsdpturbo/README.md) | all2all_experts_forward_fn (fused=True) | L3 | `historical_verified` |
| [`chunkmoe_fsdpturbo`](chunkmoe_fsdpturbo/README.md) | chunkmoe_experts_forward_fn | L2/L3 | `historical_verified` |
| [`domino_fsdpturbo`](domino_fsdpturbo/README.md) | get_domino_experts_forward_fn | L3 | `historical_verified` |
| [`hierarchical_alltoallv_mindspeed`](hierarchical_alltoallv_mindspeed/README.md) | --moe-hierarchical-alltoallv / --moe-tp-extend-ep | 待接入 | `not_implemented` |
| [`mc2_dispatch_cann_v1`](mc2_dispatch_cann_v1/README.md) | npu_moe_distribute_dispatch / npu_moe_distribute_combine | L1/L2/L3 | `integrated` |
| [`mc2_dispatch_cann`](mc2_dispatch_cann/README.md) | npu_moe_distribute_dispatch_v2 / npu_moe_distribute_combine_v2 | L1/L2/L3 | `historical_verified` |
| [`mc2_dispatch_cann_v3`](mc2_dispatch_cann_v3/README.md) | aclnnMoeDistributeDispatchV3 / aclnnMoeDistributeCombineV3 | 待接入 | `not_implemented` |
| [`mc2_combine_add_rmsnorm_cann`](mc2_combine_add_rmsnorm_cann/README.md) | npu_moe_distribute_combine_add_rms_norm | L1 | `integrated` |
| [`alltoallv_gmm_cann`](alltoallv_gmm_cann/README.md) | npu_alltoallv_gmm / npu_gmm_alltoallv | L2/L3 | `historical_verified` |
| [`alltoallv_quant_gmm_cann`](alltoallv_quant_gmm_cann/README.md) | npu_alltoallv_quant_gmm | L2 | `integrated` |
| [`mc2_fsdpturbo`](mc2_fsdpturbo/README.md) | dispatcher_mc2 | L3 | `historical_verified` |
| [`ep_gmm_fused_lueelu`](ep_gmm_fused_lueelu/README.md) | deep_ep_experimental.ep_gmm_fused | L2/L3 | `historical_verified` |
| [`deepep_cann`](deepep_cann/README.md) | cann_ops_transformer ElasticBuffer | L3 | `historical_verified` |
| [`deepep_ascend`](deepep_ascend/README.md) | deep_ep.EPBuffer | L1/L2/L3 | `historical_verified` |
| [`deepep_v2_ascend`](deepep_v2_ascend/README.md) | deep_ep.ElasticBuffer | L1/L2/L3 | `integrated` |
| [`deepep_lueelu`](deepep_lueelu/README.md) | deep_ep.ElasticBuffer (Mesh+Clos) | L1/L2/L3 | `historical_verified` |
| [`deepep_direct_lueelu`](deepep_direct_lueelu/README.md) | deep_ep.backends.direct.api.ElasticBuffer | L1/L2/L3 | `integrated` |
| [`deepep_sglang`](deepep_sglang/README.md) | CamMoeDispatchNormal / CamMoeCombineNormal | L3 | `historical_verified` |
| [`megamoe_cann`](megamoe_cann/README.md) | cann_ops_transformer.ops.mega_moe (A16W16) | L3 | `integrated` |
| [`megamoe_lueelu`](megamoe_lueelu/README.md) | bf16_mega_moe | L3 | `historical_verified` |
| [`megamoe_fsdpturbo`](megamoe_fsdpturbo/README.md) | get_mega_moe_forward_fn | L3 | `historical_verified` |
| [`megamoe_td`](megamoe_td/README.md) | MegaMoEFunction.apply | L3 | `historical_verified` |
| [`megamoe_deepgemm_ascend`](megamoe_deepgemm_ascend/README.md) | deep_gemm.fp8_fp4_mega_moe | L3 | `integrated` |
| [`moonep`](moonep/README.md) | npu_patch.Buffer.dispatch / combine (UDMA) | L3 | `historical_verified` |
| [`moonep_dynamic_planner`](moonep_dynamic_planner/README.md) | MoonEP dev_npu planner / replica refresh | L1 | `integrated` |
| [`moonep_server_dedup`](moonep_server_dedup/README.md) | MoonEP dev_server_dedup dispatch / combine | L1 | `integrated` |
| [`dispatch_ffn_combine_vllm_ascend`](dispatch_ffn_combine_vllm_ascend/README.md) | torch.ops._C_ascend.dispatch_ffn_combine | L3 | `integrated` |
| [`dispatch_ffn_combine_sglang`](dispatch_ffn_combine_sglang/README.md) | FuseMode.DISPATCH_FFN_COMBINE | L3 | `integrated` |
| [`fused_deep_moe_sglang`](fused_deep_moe_sglang/README.md) | FuseMode.FUSED_DEEP_MOE | L3 | `integrated` |
