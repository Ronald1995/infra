"""vLLM-Ascend standalone A3 BF16 DispatchFFNCombine inference path."""
from common.backend import BenchmarkUnavailable, EPBackend


class VLLMDispatchFFNBackend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        import torch_npu

        label = context.torch.npu.get_device_name(context.device).casefold()
        supported = "910_93" in label or "atlas a3" in label
        if not all(context.gather(supported)):
            raise BenchmarkUnavailable("unsupported", "the pinned BF16 custom op is compiled for Ascend 910_93 (Atlas A3)")
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
        self.native = getattr(context.torch.ops._C_ascend, "dispatch_ffn_combine", None)
        if self.native is None:
            raise BenchmarkUnavailable("dependency_missing", "vLLM-Ascend DispatchFFNCombine custom op is not loaded")
        self.format_cast = torch_npu.npu_format_cast
        self.group = context.ep_group._get_backend(context.device).get_hccl_comm_name(context.rank)

    def prepare(self, work):
        from baselines.reference.collectives import CollectiveBackend
        from baselines.reference.experts import gmm1

        ctx, torch = self.context, self.context.torch
        # The BF16 entry has no clamp parameter. Establish that this workload
        # does not exercise the shared workload's clipping limit.
        reference = CollectiveBackend(None, "alltoall")
        reference.initialize(ctx)
        with torch.no_grad():
            dispatched = reference.dispatch(work)
            projected = gmm1(work, dispatched.expert_input, dispatched.expert_token_counts, ctx)
            gate, up = projected.chunk(2, dim=-1)
            within_limit = not len(projected) or bool((gate <= work.limit).all() and (up.abs() <= work.limit).all())
        if not all(ctx.gather(within_limit)):
            raise BenchmarkUnavailable("unsupported", "BF16 native SwiGLU has no clamp and this workload clips values")
        first = torch.cat((work.w_gate, work.w_up), dim=1).detach().transpose(1, 2).contiguous()
        second = work.w_down.detach().transpose(1, 2).contiguous()
        self.w1, self.w2 = [self.format_cast(first, 29)], [self.format_cast(second, 29)]
        self.scale1 = [torch.zeros((first.shape[0], first.shape[-1]), dtype=torch.int64, device=ctx.device)]
        self.scale2 = [torch.zeros((second.shape[0], second.shape[-1]), dtype=torch.int64, device=ctx.device)]
        self.receive_capacity = 65536
        if sum(work.route.tokens) * work.topk_ids.shape[1] > self.receive_capacity:
            raise BenchmarkUnavailable("unsupported", "worst-case routes exceed 65536-row native output capacity")

    def moe_forward(self, work):
        ctx, torch = self.context, self.context.torch
        out = torch.empty_like(work.hidden_states)
        counts = torch.empty((1, len(work.local_experts)), dtype=torch.int32, device=ctx.device)
        self.native(
            x=work.hidden_states.detach().contiguous(), weight1=self.w1, weight2=self.w2,
            expert_idx=work.topk_ids.int().contiguous(), scale1=self.scale1, scale2=self.scale2,
            bias1=[], bias2=[], probs=work.topk_weights.detach().float().contiguous(), group=self.group,
            max_output_size=self.receive_capacity, world_size=ctx.size, out=out, expert_token_nums=counts,
        )
        return out


def create(spec):
    return VLLMDispatchFFNBackend(spec)
