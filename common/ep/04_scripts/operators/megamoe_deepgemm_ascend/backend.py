"""Native DeepGEMM-Ascend FP8×FP4 expert inference."""
from dataclasses import replace

from common.backend import BenchmarkUnavailable, EPBackend
from operators.mxfp import dequantize


class DeepGEMMBackend(EPBackend):
    def initialize(self, ctx):
        super().initialize(ctx)
        import deep_gemm
        import torch_npu

        self.native, self.npu = deep_gemm, torch_npu
        self.buffer = None
        if not all(hasattr(deep_gemm, name) for name in (
            "SymmBuffer", "fp8_fp4_mega_moe", "transform_weights_for_mega_moe"
        )):
            raise BenchmarkUnavailable("dependency_missing", "DeepGEMM Ascend MegaMoE symbols are unavailable")

    def quantize(self, value, fp4=False):
        torch = self.context.torch
        dtype = self.npu.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn
        payload, scale = self.npu.npu_dynamic_mx_quant(value.contiguous(), dst_type=dtype, block_size=32)
        packed = payload.view(torch.int8) if fp4 else payload.view(torch.float8_e4m3fn)
        return packed, scale.view(torch.uint8).view(torch.int16).squeeze(-1)

    def prepare(self, work):
        ctx, torch = self.context, self.context.torch
        h, i = work.hidden_states.shape[1], work.w_down.shape[-1]
        if h % 64 or i % 64:
            raise BenchmarkUnavailable("unsupported", "DeepGEMM MXFP requires H and I divisible by 64")
        gate_up = torch.cat((work.w_gate, work.w_up), dim=1).detach()
        self.quantized_weights = (self.quantize(gate_up, True), self.quantize(work.w_down.detach(), True))
        self.weights = self.native.transform_weights_for_mega_moe(*self.quantized_weights)
        self.buffer = self.native.SymmBuffer(ctx.ep_group, len(work.route.placement),
                                             len(work.hidden_states), work.topk_ids.shape[1], h, i)

    def moe_forward(self, work):
        buffer, torch = self.buffer, self.context.torch
        tokens = len(work.hidden_states)
        payload, scale = self.quantize(work.hidden_states.detach())
        buffer.x[:tokens].copy_(payload)
        buffer.x_sf[:tokens].copy_(scale)
        buffer.topk_idx[:tokens].copy_(work.topk_ids)
        buffer.topk_weights[:tokens].copy_(work.topk_weights)
        out = torch.empty_like(work.hidden_states)
        self.native.fp8_fp4_mega_moe(out, *self.weights, buffer, activation_clamp=work.limit)
        return out

    def validate_forward(self, work, output):
        from baselines.reference.collectives import CollectiveBackend, gather_rows
        from baselines.reference.experts import activate
        from common.validation import error

        ctx, torch = self.context, self.context.torch

        def decode(payload, scales, fp4=False):
            groups = scales.view(torch.uint8).reshape(*scales.shape, 2)
            return dequantize(torch, payload.cpu(), groups.cpu(), fp4=fp4).to(ctx.device)

        with torch.no_grad():
            x = decode(*self.quantize(work.hidden_states.detach()))
            gate_up, down = (decode(*pair, fp4=True) for pair in self.quantized_weights)
            reference = CollectiveBackend(None, "alltoall")
            reference.initialize(ctx)
            routed = reference.dispatch(replace(work, hidden_states=x))
            flattened = gather_rows(work.topk_weights.detach().float(), ctx).flatten()
            probabilities = flattened[routed.route_slots]
            counts = routed.expert_token_counts
            projected = torch.cat([rows @ weight.T for rows, weight in
                                   zip(routed.expert_input.split(counts), gate_up)]).bfloat16()
            hidden = activate(projected, work.limit) * probabilities[:, None]
            hidden = decode(*self.quantize(hidden.bfloat16()))
            rows = torch.cat([part @ weight.T for part, weight in zip(hidden.split(counts), down)])
            expected = reference.combine(replace(work, topk_weights=torch.ones_like(work.topk_weights)),
                                         routed, rows).output
        return {"scope": "mxfp8_fp4_reference_with_weighted_activation",
                "relative_l2": error(output, expected, .045)}

    def finalize(self):
        if getattr(self, "buffer", None) is not None:
            self.buffer.destroy()


def create(spec):
    return DeepGEMMBackend(spec)
