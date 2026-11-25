"""Ascend DeepEP V2 high-throughput inference, separate from EPBuffer and lueelu."""
from common.backend import BenchmarkUnavailable, EPBackend
from operators.expanded_deepep import ExpandedDeepEPBackend


class AscendV2Backend(ExpandedDeepEPBackend):
    def initialize(self, context):
        EPBackend.initialize(self, context)
        self.layer = None
        try:
            from deep_ep.buffers.elastic import ElasticBuffer
            from deep_ep.buffers._elastic_layout import workspace_layout
        except ImportError as exc:
            raise BenchmarkUnavailable("dependency_missing", "install Ascend DeepEP V2 69090a1") from exc
        self.native = ElasticBuffer
        self.layout = workspace_layout

    def buffer_type(self):
        return self.native, {}

    def prepare(self, work):
        self.layout(self.context.size, len(work.hidden_states), work.topk_ids.shape[1], len(work.route.placement))
        super().prepare(work)

    def moe_forward(self, work):
        from baselines.reference.experts import expert_forward
        from common.backend import DispatchResult

        torch = self.context.torch
        buffer = self.layer.buffer
        with torch.no_grad():
            with torch.profiler.record_function("ep_dispatch"):
                rows, _, weights, handle, _ = buffer.dispatch(
                    work.hidden_states.detach().contiguous(), topk_idx=work.topk_ids,
                    topk_weights=work.topk_weights.detach().float().contiguous(),
                    num_experts=len(work.route.placement), do_expand=True, do_cpu_sync=True, expert_alignment=1,
                )
            prefix = [0, *handle.psum_num_recv_tokens_per_expert.tolist()]
            counts = tuple(stop - start for start, stop in zip(prefix, prefix[1:]))
            output = expert_forward(work, DispatchResult(rows, counts, handle, "local_expert_contiguous"),
                                    self.context)
            with torch.profiler.record_function("ep_combine"):
                return buffer.combine((output * weights[:, None]).to(rows.dtype).contiguous(), handle)[0]


def create(spec):
    return AscendV2Backend(spec)
