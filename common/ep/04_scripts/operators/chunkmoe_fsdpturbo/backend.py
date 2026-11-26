"""Native FSDPTurbo ChunkMoE with an explicit chunk-count benchmark input."""
from operators.ascendep import AscendEPBackend


class ChunkBackend(AscendEPBackend):
    def chunk_forward(self, work, chunks):
        ctx, torch = self.context, self.context.torch
        packed = torch.cat((work.w_gate, work.w_up), dim=1)
        settings = self.demo.EPPlanConfig(dispatcher="chunkmoe", moe_num_chunks=chunks)
        return self.demo.chunkmoe_experts_forward_fn(
            work.hidden_states, work.topk_ids.int(), work.topk_weights,
            (packed, work.w_down), len(work.route.placement), self.layer.expert_ids_per_ep_rank,
            ctx.ep_group, fused=True, act_limit=work.limit, ep_balance_strategy=None, ep_plan=settings,
        )


def create(spec):
    return ChunkBackend(spec)
