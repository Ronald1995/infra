import torch
from torch.nn import functional as F


def dispatch(layer, x, ids, weights, dispatcher):
    from megatron.core import parallel_state as ps
    from megatron.core.transformer.transformer_config import TransformerConfig
    world = torch.distributed.get_world_size()
    local = 24 // world
    if not ps.model_parallel_is_initialized():
        ps.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=world)
    if not hasattr(layer, "dispatcher"):
        rank = ps.get_expert_model_parallel_rank()
        cfg = TransformerConfig(num_layers=1, hidden_size=x.shape[-1], num_attention_heads=1,
                                expert_model_parallel_size=world, num_moe_experts=24,
                                moe_router_topk=6, add_bias_linear=False,
                                moe_permute_fusion=False, bf16=True)
        layer.dispatcher = dispatcher(local, list(range(rank*local, (rank+1)*local)), cfg)
    probs = weights.new_zeros((len(x), 24)).scatter(1, ids.long(), weights)
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, ids.long(), True)
    packed, counts, p = layer.dispatcher.token_permutation(x[:, None], probs, mask)
    outputs = []
    for i, chunk in enumerate(packed.split(counts.tolist())):
        g, u = F.linear(chunk, layer.gate_up_proj[i]).chunk(2, -1)
        outputs.append(F.linear(F.silu(g.clamp(max=10))*u.clamp(-10,10), layer.down_proj[i]))
    result, _ = layer.dispatcher.token_unpermutation(torch.cat(outputs) * p.reshape(-1,1))
    return result.reshape_as(x)
