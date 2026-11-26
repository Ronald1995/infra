"""Pinned SGLang FuseEP inference paths and quantization-aware references."""
from dataclasses import replace

from common.backend import BenchmarkUnavailable, EPBackend
from operators.mxfp import dequantize


class SGLangFusedBackend(EPBackend):
    def __init__(self, spec, mode):
        super().__init__(spec)
        self.mode = mode

    def initialize(self, ctx):
        super().initialize(ctx)
        import torch_npu
        from deep_ep.buffer import Buffer, FuseMode

        self.torch_npu = torch_npu
        self.buffer = None
        self.group = None
        device = ctx.torch.npu.get_device_name(ctx.device).casefold()
        platform = ("950" in device or "a5" in device) if self.mode == "mxfp8" else (
            "910_93" in device or "atlas a3" in device
        )
        if not all(ctx.gather(platform)):
            expected = "Ascend 950" if self.mode == "mxfp8" else "Atlas A3"
            raise BenchmarkUnavailable("unsupported", f"the selected SGLang quantized recipe targets {expected}")
        self.fuse_mode = (FuseMode.FUSED_DEEP_MOE if self.mode == "mxfp8"
                          else FuseMode.DISPATCH_FFN_COMBINE)
        if not hasattr(Buffer, "fused_deep_moe"):
            raise BenchmarkUnavailable("dependency_missing", "install the pinned sgl-kernel-npu DeepEP extension")
        self.group = ctx.dist.new_group(backend="hccl")
        nvl_bytes = ctx.config["benchmark"].get("sglang_buffer_bytes", 2_000_000_000)
        self.buffer = Buffer(self.group, nvl_bytes, num_qps_per_rank=1)

    def prepare(self, work):
        ctx, torch = self.context, self.context.torch
        hidden = work.hidden_states.shape[-1]
        intermediate = work.w_down.shape[-1]
        if hidden % 64 or intermediate % 64:
            raise BenchmarkUnavailable("unsupported", "SGLang quantized recipes require H and I divisible by 64")
        gate_up = torch.cat((work.w_gate, work.w_up), dim=1).detach()
        first, second = gate_up.transpose(1, 2).contiguous(), work.w_down.detach().transpose(1, 2).contiguous()
        if self.mode == "mxfp8":
            self.w1, self.s1 = self.mxfp(first, 1)
            self.w2, self.s2 = self.mxfp(second, 1)
            scale_type = getattr(torch, "float8_e8m0fnu", None)
            if scale_type is None:
                raise BenchmarkUnavailable("dependency_missing", "SGLang MXFP8 requires E8M0 support in PyTorch")
            self.native_weights = (self.w1, self.s1.view(scale_type), self.w2, self.s2.view(scale_type))
        else:
            self.w1, self.s1 = self.int8_weights(first)
            self.w2, self.s2 = self.int8_weights(second)
            self.torch_npu.npu.config.allow_internal_format = True
            self.native_weights = (
                self.torch_npu.npu_format_cast(self.w1, 29), self.s1.view(torch.int32).to(torch.int64),
                self.torch_npu.npu_format_cast(self.w2, 29), self.s2.view(torch.int32).to(torch.int64),
            )

    def mxfp(self, value, axis=-1):
        return self.torch_npu.npu_dynamic_mx_quant(
            value.contiguous(), axis=axis, dst_type=self.context.torch.float8_e4m3fn, block_size=32
        )

    def int8_weights(self, value):
        torch = self.context.torch
        fp32 = value.float()
        scales = (fp32.abs().amax(1) / 127).clamp_min(1e-12).contiguous()
        payload = (fp32 / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        return payload.contiguous(), scales

    def moe_forward(self, work):
        capacity = max(work.route.tokens)
        if self.mode == "int8":
            capacity = ((capacity + 15) // 16) * 16 * work.topk_ids.shape[1] * self.context.size
        result, _ = self.buffer.fused_deep_moe(
            work.hidden_states.detach().contiguous(), work.topk_ids,
            work.topk_weights.detach().float().contiguous(), *self.native_weights,
            capacity, len(work.route.placement), 0 if self.mode == "mxfp8" else 1,
            fuse_mode=self.fuse_mode, activation="swiglu",
        )
        return result

    def validate_forward(self, work, output):
        from baselines.reference.collectives import CollectiveBackend
        from common.validation import error

        ctx, torch = self.context, self.context.torch

        def decode_mx(value, scales, axis=-1):
            return dequantize(torch, value.cpu(), scales.cpu(), axis).to(ctx.device)

        def activations(value):
            if self.mode == "mxfp8":
                return decode_mx(*self.mxfp(value))
            payload, scales = self.torch_npu.npu_dynamic_quant(value)
            return payload.float() * scales.reshape(-1, 1).float()

        with torch.no_grad():
            x = activations(work.hidden_states)
            if self.mode == "mxfp8":
                w1, w2 = decode_mx(self.w1, self.s1, 1), decode_mx(self.w2, self.s2, 1)
            else:
                w1, w2 = self.w1.float() * self.s1[:, None], self.w2.float() * self.s2[:, None]
            ref = CollectiveBackend(None, "alltoall")
            ref.initialize(ctx)
            layout = ref.dispatch(replace(work, hidden_states=x))
            counts = layout.expert_token_counts
            projected = torch.cat([rows @ weights for rows, weights in
                                   zip(layout.expert_input.split(counts), w1)])
            gate, up = projected.chunk(2, dim=-1)
            within_limit = not len(projected) or bool((gate <= work.limit).all() and (up.abs() <= work.limit).all())
            if not all(ctx.gather(within_limit)):
                raise BenchmarkUnavailable("unsupported", "the native SwiGLU recipe has no matching clip parameter")
            hidden = activations((torch.nn.functional.silu(gate) * up).bfloat16())
            rows = torch.cat([part @ weights for part, weights in zip(hidden.split(counts), w2)])
            expected = ref.combine(work, layout, rows.bfloat16()).output
        return {"scope": f"sglang_{self.mode}_unfused_fp32_reference",
                "relative_l2": error(output, expected, .06)}

    def finalize(self):
        self.buffer = None
        if getattr(self, "group", None) is not None:
            self.context.dist.destroy_process_group(self.group)
