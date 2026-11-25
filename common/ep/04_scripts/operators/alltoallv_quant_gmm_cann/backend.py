"""CANN standalone AllToAllV-MXFP8-GMM, with dequantized FP32 reference."""
from dataclasses import replace

from common.backend import BenchmarkUnavailable, EPBackend
from operators.mxfp import dequantize


class QuantGMMBackend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        import torch_npu

        self.native = torch_npu
        self.scale_dtype = getattr(torch_npu, "float8_e8m0fnu", None)
        if self.scale_dtype is None or not all(hasattr(torch_npu, function) for function in
                                               ("npu_dynamic_mx_quant", "npu_alltoallv_quant_gmm")):
            raise BenchmarkUnavailable("dependency_missing", "MXFP8 bindings are missing from torch_npu")
        self.hcom = context.ep_group._get_backend(context.device).get_hccl_comm_name(context.rank)

    def prepare(self, work):
        torch = self.context.torch
        h = work.hidden_states.shape[1]
        if h % 64 or len(work.route.placement) > 256 or not 0 < len(work.local_experts) <= 32:
            raise BenchmarkUnavailable("unsupported", "MXFP8 requires H divisible by 64, E<=256 and local E in [1,32]")
        weights = torch.cat((work.w_gate, work.w_up), dim=1).detach().transpose(1, 2).contiguous()
        self.weight, self.weight_scale = self.quantize(weights, 1)

    def quantize(self, tensor, axis):
        return self.native.npu_dynamic_mx_quant(
            tensor, axis=axis, dst_type=self.context.torch.float8_e4m3fn, block_size=32
        )

    def quantized_fusion(self, work):
        from baselines.reference.collectives import exchange

        ctx, torch = self.context, self.context.torch
        ids = work.topk_ids.flatten()
        experts = len(work.route.placement)
        local_experts = len(work.local_experts)
        order = ids.argsort(stable=True)
        send = torch.bincount(ids, minlength=experts)
        receive = exchange(send, [local_experts] * ctx.size, [local_experts] * ctx.size, ctx)
        activation, scale = self.quantize(work.hidden_states.detach(), -1)
        tokens = order // work.topk_ids.shape[1]
        # Byte views preserve FP8/E8M0 encodings during the source-side gather.
        packed = activation.view(torch.uint8)[tokens].view(torch.float8_e4m3fn)
        packed_scale = scale.view(torch.uint8)[tokens]
        output, _, _ = self.native.npu_alltoallv_quant_gmm(
            packed, self.weight, packed_scale, self.weight_scale,
            self.hcom, ctx.size, send.tolist(), receive.tolist(), work.hidden_states.dtype,
            gmm_x_quant_mode=6, gmm_weight_quant_mode=6,
            gmm_x_scale_dtype=self.scale_dtype, gmm_weight_scale_dtype=self.scale_dtype,
        )
        return output

    def validate_quantized_fusion(self, work, output):
        from baselines.reference.collectives import CollectiveBackend
        from common.validation import error

        ctx = self.context
        qx, sx = self.quantize(work.hidden_states.detach(), -1)
        x = dequantize(ctx.torch, qx.cpu(), sx.cpu()).to(ctx.device)
        weights = dequantize(ctx.torch, self.weight.cpu(), self.weight_scale.cpu(), axis=1)
        ref = CollectiveBackend(None, "alltoall")
        ref.initialize(ctx)
        dispatched = ref.dispatch(replace(work, hidden_states=x))
        rows = dispatched.expert_input.cpu()
        pieces = [part @ weight for part, weight in zip(rows.split(dispatched.expert_token_counts), weights)]
        expected = ctx.torch.cat(pieces)
        return {"scope": "mxfp8_dequantized_fp32_gmm", "relative_l2": error(output, expected, 0.015)}


def create(spec):
    return QuantGMMBackend(spec)
