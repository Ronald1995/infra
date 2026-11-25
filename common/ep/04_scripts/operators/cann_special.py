"""Direct communication and normalization calls backed by the published op-plugin bindings."""
from dataclasses import dataclass

from common.backend import BenchmarkUnavailable, CombineResult, DispatchResult, EPBackend
from operators.ascendep import MC2V2Backend

OP_PLUGIN_COMMIT = "c83e36b808c4e053c9245f95c7bc45cfaae0000d"


@dataclass
class NativeV1Handle:
    indices: object
    peer_counts: object
    capacity: int


class MC2V1Backend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        import torch_npu

        self.group = None
        self.native = torch_npu
        for name in ("npu_moe_distribute_dispatch", "npu_moe_distribute_combine"):
            if not hasattr(torch_npu, name):
                raise BenchmarkUnavailable("dependency_missing", f"torch_npu does not export {name}")
        self.group = context.dist.new_group(backend="hccl")
        self.communicator = self.group._get_backend(context.device).get_hccl_comm_name(context.rank)

    def route_args(self, work):
        ctx = self.context
        return (self.communicator, ctx.size, ctx.rank, len(work.route.placement))

    def dispatch(self, work, cached=None):
        torch = self.context.torch
        rows, _, indices, counts, peers, _ = self.native.npu_moe_distribute_dispatch(
            work.hidden_states.contiguous(), work.topk_ids.int().contiguous(), *self.route_args(work),
            shared_expert_num=1, shared_expert_rank_num=0,
            global_bs=sum(work.route.tokens), expert_token_nums_type=1,
        )
        counts = tuple(counts.tolist())
        handle = NativeV1Handle(indices, peers, len(rows))
        expert_ids = torch.repeat_interleave(torch.tensor(work.local_experts, device=rows.device),
                                             torch.tensor(counts, device=rows.device))
        return DispatchResult(rows[:sum(counts)], counts, handle,
                              "local_expert_contiguous_opaque_reverse_route", expert_ids)

    def combine(self, work, dispatched, rows):
        difference = dispatched.routing_handle.capacity - len(rows)
        rows = self.context.torch.nn.functional.pad(rows, (0, 0, 0, difference))
        handle = dispatched.routing_handle
        y = self.native.npu_moe_distribute_combine(
            rows, work.topk_ids.int().contiguous(), handle.indices, handle.peer_counts,
            work.topk_weights.float().contiguous(), *self.route_args(work),
            shared_expert_num=1, shared_expert_rank_num=0, global_bs=sum(work.route.tokens),
        )
        return CombineResult(y)

    def moe_forward(self, work):
        from baselines.reference.experts import expert_forward

        dispatched = self.dispatch(work)
        return self.combine(work, dispatched, expert_forward(work, dispatched, self.context)).output

    def finalize(self):
        if getattr(self, "group", None) is not None:
            self.context.dist.destroy_process_group(self.group)


class CombineRMSNormBackend(MC2V2Backend):
    def initialize(self, context):
        super().initialize(context)
        import torch_npu

        self.native = getattr(torch_npu, "npu_moe_distribute_combine_add_rms_norm", None)
        if self.native is None:
            raise BenchmarkUnavailable("dependency_missing", "torch_npu has no combine/add/RMSNorm binding")

    def rmsnorm_combine(self, work, dispatched, rows, residual, gamma):
        handle = dispatched.routing_handle
        rows = self.context.torch.nn.functional.pad(rows, (0, 0, 0, handle.capacity - len(rows)))
        return self.native(
            rows, handle.ids, handle.assist, handle.peer_counts, work.topk_weights.float().contiguous(),
            residual, gamma, *handle.config, shared_expert_num=0, global_bs=handle.batch,
        )
