"""MindSpeed's overlapped allgather layer with a fixed balanced router."""
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn import functional as F
import mindspeed.megatron_adaptor
from mindspeed.args_utils import get_full_args
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedMOEAllGatherOverLapDispatcherAdaptor
from mindspeed.core.transformer.moe.moe_feature.overlap.grouped_mlp_with_comp_and_comm_overlap_allgather import (
    grouped_mlp_with_comp_and_comm_overlap_allgather,
)
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_allgather import MoELayerOverlapAllGather

from ep.runner import run


def forward(layer, x, ids, weights):
    world = dist.get_world_size()
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=world)
    if not hasattr(layer, "dispatcher"):
        get_full_args().fp8 = None
        local = layer.num_global_experts // world
        start = local * dist.get_rank()
        cfg = TransformerConfig(num_layers=1, hidden_size=x.shape[1], num_attention_heads=1,
                                expert_model_parallel_size=world, num_moe_experts=layer.num_global_experts,
                                moe_router_topk=ids.shape[1], add_bias_linear=False, bf16=True,
                                moe_grouped_gemm=True, moe_token_dispatcher_type="allgather")
        cfg.n_shared_experts = None
        cfg.gemm_gradient_accumulation_fusion = False
        layer.dispatcher = MindSpeedMOEAllGatherOverLapDispatcherAdaptor(local, list(range(start, start+local)), cfg)
    cfg = layer.dispatcher.config

    def router(hidden):
        # Retain the zero hidden-state derivative expected by the native router backward.
        scores = weights + hidden[:, :1] * 0
        probs = weights.new_zeros((len(weights), cfg.num_moe_experts)).scatter(1, ids, scores)
        return probs, torch.zeros_like(probs, dtype=torch.bool).scatter_(1, ids, True)

    def activation(rows):
        gate, up = rows.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=layer.limit)) * up.clamp(-layer.limit, layer.limit)

    def experts(rows, counts, permuted_probs):
        w1 = layer.gate_up_proj.transpose(1, 2).contiguous()
        w2 = layer.down_proj.transpose(1, 2).contiguous()
        params = (layer.gate_up_proj, layer.down_proj, activation, counts.cumsum(0), None, cfg)
        return grouped_mlp_with_comp_and_comm_overlap_allgather(rows, w1, w2, params)

    native = SimpleNamespace(config=cfg, router=router, experts=experts, token_dispatcher=layer.dispatcher)
    out, _ = MoELayerOverlapAllGather.apply(x, cfg, native)
    return out


if __name__ == "__main__":
    run(forward, "allgather_overlap_mindspeed")
