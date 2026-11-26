import torch
import torch.distributed as dist
from deep_ep_experimental.megamoe import SymmBuffer, bf16_mega_moe
from ep.runner import run


def forward(layer, x, ids, weights):
    if not hasattr(layer, "buffer"):
        layer.buffer = SymmBuffer(dist.group.WORLD, num_experts=24, num_max_tokens_per_rank=len(x),
                                  num_topk=6, hidden=x.shape[1], intermediate_hidden=layer.down_proj.shape[-1],
                                  mma_type="bf16xbf16", need_grad=True, gmm1_save_type="bf16",
                                  max_recv_tokens=len(x)*6)
        layer.release_buffer = layer.buffer.destroy
    buffer = layer.buffer
    buffer.x, buffer.topk_idx, buffer.topk_weights = x, ids.int(), weights
    y = torch.empty_like(x)
    bf16_mega_moe(y, layer.gate_up_proj, layer.down_proj, buffer, activation_clamp=10.)
    return y


if __name__ == "__main__":
    run(forward, "megamoe_lueelu")
