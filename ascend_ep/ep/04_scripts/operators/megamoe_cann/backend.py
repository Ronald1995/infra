"""CANN MegaMoE A16W16 forward using the versioned public symmetric-buffer API."""
from common.backend import BenchmarkUnavailable, EPBackend


class CANNMegaMoEBackend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

        self.allocate = get_symm_buffer_for_mega_moe
        self.native = mega_moe
        self.buffer = None

    def prepare(self, work):
        torch = self.context.torch
        _, hidden = work.hidden_states.shape
        intermediate = work.w_down.shape[-1]
        self.gate_up = [torch.cat((work.w_gate, work.w_up), dim=1).detach().contiguous()]
        self.down = [work.w_down.detach().contiguous()]
        try:
            self.buffer = self.allocate(
                self.context.ep_group, num_experts=len(work.route.placement),
                num_max_tokens_per_rank=max(work.route.tokens), num_topk=work.topk_ids.shape[1],
                hidden=hidden, intermediate_hidden=intermediate, dispatch_quant_mode=0,
                dispatch_quant_out_dtype=None, combine_quant_mode=0,
            )
        except ValueError as exc:
            if "dispatch_quant_mode" in str(exc):
                raise BenchmarkUnavailable("unsupported", f"installed CANN extension does not expose A16W16: {exc}") from exc
            raise

    def moe_forward(self, work):
        output, _ = self.native(
            x=work.hidden_states.detach().contiguous(), topk_ids=work.topk_ids.int().contiguous(),
            topk_weights=work.topk_weights.detach().float().contiguous(), l1_weights=self.gate_up,
            l2_weights=self.down, sym_buffer=self.buffer, activation="swiglu", activation_clamp=work.limit,
        )
        return output

    def finalize(self):
        if getattr(self, "buffer", None) is not None:
            self.buffer.destroy()


def create(spec):
    return CANNMegaMoEBackend(spec)
