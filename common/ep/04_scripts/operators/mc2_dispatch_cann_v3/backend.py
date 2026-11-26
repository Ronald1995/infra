"""CANN context-based MC2 V3 operators, reached through the ACLNN V5 symbols."""
from dataclasses import dataclass

from common.backend import BenchmarkUnavailable, CombineResult, DispatchResult, EPBackend


@dataclass
class ContextRoute:
    ids: object
    assist: object
    peers: object
    capacity: int


class MC2ContextBackend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        self.buffer = None
        self.group = None
        from cann_ops_transformer.ops.mc2.common import MoeDistributeBuffer

        self.kind = MoeDistributeBuffer
        name = context.torch.npu.get_device_name(context.device).casefold()
        model = "950" if "950" in name else "a3" if "910_93" in name or "atlas a3" in name else None
        all_models = context.gather(model)
        if not model or len(set(all_models)) != 1:
            raise BenchmarkUnavailable("unsupported", "V3 context path requires homogeneous 950DT or Atlas A3 devices")
        self.is_950 = model == "950"

    def prepare(self, work):
        ctx = self.context
        hidden = work.hidden_states.shape[1]
        self.capacity = max(work.route.tokens)
        self.comm_alg = ctx.config["benchmark"].get("mc2_comm_alg", "")
        ccl_size = self.kind.get_low_latency_ccl_buffer_size(
            ctx.size, self.capacity, hidden, len(work.route.placement), work.topk_ids.shape[1],
            comm_alg=self.comm_alg,
        )
        self.group = ctx.dist.new_group(backend="hccl")
        # MoeDistributeBuffer reads the existing HCCL communicator without
        # requesting lazy initialization itself.
        self.group._get_backend(ctx.device).get_hccl_comm_name(ctx.rank)
        self.buffer = self.kind(self.group, ccl_buffer_size=ccl_size if self.is_950 else 0,
                                comm_alg=self.comm_alg)

    def launch_dispatch(self, work, quant_mode):
        ids = work.topk_ids.int().contiguous()
        labels = None
        torch, rank = self.context.torch, self.context.rank
        if self.is_950:
            first = sum(work.route.tokens[:rank]) * ids.shape[1]
            if first + ids.numel() >= 1 << 24:
                raise BenchmarkUnavailable("unsupported", "V3 route label exceeds exact FP32 integer range")
            labels = torch.arange(first, first + ids.numel(), dtype=torch.float32, device=ids.device).view_as(ids)
        rows, scales, assist, counts, peers, returned_labels = self.buffer.low_latency_dispatch(
            work.hidden_states.detach().contiguous(), ids, len(work.route.placement),
            quant_mode=quant_mode, comm_alg=self.comm_alg, topk_weights=labels, shared_expert_num=0,
            num_max_dispatch_tokens_per_rank=self.capacity,
        )
        counts = tuple(counts.tolist())
        used = sum(counts)
        handle = ContextRoute(ids, assist, peers, len(rows))
        expert_ids = torch.repeat_interleave(
            torch.tensor(work.local_experts, device=ids.device), torch.tensor(counts, device=ids.device),
            output_size=used,
        )
        if labels is None:
            layout, slots = "local_expert_contiguous_opaque_reverse_route", None
        else:
            layout, slots = "local_expert_contiguous", returned_labels[:used].long()
        return DispatchResult(rows[:used], counts, handle, layout, expert_ids, slots), scales

    def dispatch(self, work, cached=None):
        return self.launch_dispatch(work, 0)[0]

    def combine(self, work, dispatched, rows):
        if self.is_950:
            raise BenchmarkUnavailable("unsupported", "the pinned operator V3 docs list context Combine only on Atlas A3")
        handle = dispatched.routing_handle
        padded = self.context.torch.nn.functional.pad(rows, (0, 0, 0, handle.capacity - len(rows)))
        output = self.buffer.low_latency_combine(
            padded.contiguous(), handle.ids, work.topk_weights.detach().float().contiguous(),
            handle.assist, handle.peers, num_experts=len(work.route.placement), comm_alg=self.comm_alg,
            shared_expert_num=0, num_max_dispatch_tokens_per_rank=self.capacity,
        )
        return CombineResult(output)

    def quantized_dispatch(self, work):
        if not self.is_950:
            raise BenchmarkUnavailable("unsupported", "the V3 INT8 route-label check targets 950DT")
        return self.launch_dispatch(work, 2)

    def validate_quantized_dispatch(self, work, result, trace):
        from dataclasses import replace
        from common.validation import CorrectnessError, validate_dispatch

        dispatched, scales = result
        rows = dispatched.expert_input
        if rows.dtype != self.context.torch.int8 or scales[:len(rows)].numel() != len(rows):
            raise CorrectnessError("context per-token quantization did not return INT8 rows and one scale per row")
        reconstructed = rows.float() * scales[:len(rows)].float().reshape(-1, 1)
        checked = replace(dispatched, expert_input=reconstructed)
        return validate_dispatch(work, checked, self.context, trace, threshold=.015)

    def moe_forward(self, work):
        from baselines.reference.experts import expert_forward

        if self.is_950:
            raise BenchmarkUnavailable("unsupported", "the 950DT context route is dispatch-only in the pinned release")
        dispatched = self.dispatch(work)
        return self.combine(work, dispatched, expert_forward(work, dispatched, self.context)).output

    def finalize(self):
        if self.buffer is not None:
            self.buffer._ctx_manager.destroy()
            self.buffer = None
        if self.group is not None:
            self.context.dist.destroy_process_group(self.group)


def create(spec):
    return MC2ContextBackend(spec)
