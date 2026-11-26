"""Expanded BF16 DeepEP calls with verifiable route and expert metadata."""
from dataclasses import dataclass

from common.backend import BenchmarkUnavailable, CombineResult, DispatchResult
from operators.ascendep import AscendEPBackend


@dataclass
class ExpandedRoute:
    native: object
    counts: tuple[int, ...]
    experts: object
    slots: object


class ExpandedDeepEPBackend(AscendEPBackend):
    def buffer_type(self):
        if self.spec.id == "deepep_direct_lueelu":
            from deep_ep.backends.direct.api import ElasticBuffer
            transport = self.context.config["benchmark"].get("deepep_transport")
            if transport not in {"mte", "udma"}:
                raise BenchmarkUnavailable("environment_unavailable", "set benchmark.deepep_transport to mte or udma")
            return ElasticBuffer, {"transport": transport, "use_1g_huge_pages": False}
        if self.spec.id == "deepep_ascend":
            from deep_ep import EPBuffer
            return EPBuffer, {"allow_hybrid_mode": False}
        from deep_ep import ElasticBuffer
        return ElasticBuffer, {"use_1g_huge_pages": False}

    def prepare(self, work):
        super().prepare(work)
        kind, options = self.buffer_type()
        self.layer.buffer = kind(
            self.context.ep_group, num_max_tokens_per_rank=len(work.hidden_states),
            hidden=work.hidden_states.shape[1], num_topk=work.topk_ids.shape[1],
            explicitly_destroy=True, **options,
        )
        self.layer.release_buffer = self.layer.buffer.destroy
        if self.spec.id in {"deepep_lueelu", "deepep_direct_lueelu"}:
            from ep.deepep_ascend import demo
            self.expanded_forward = demo.expanded_forward

    def moe_forward(self, work):
        if self.spec.id == "deepep_ascend":
            return super().moe_forward(work)
        torch = self.context.torch
        self.layer.gate_up_proj = torch.cat((work.w_gate, work.w_up), dim=1)
        kind, options = self.buffer_type()
        return self.expanded_forward(self.layer, work.hidden_states, work.topk_ids,
                                     work.topk_weights, kind, **options)

    def prepare_communication(self, work):
        from baselines.reference.collectives import gather_rows

        torch, ctx = self.context.torch, self.context
        if sum(work.route.tokens) * work.topk_ids.shape[1] >= 1 << 24:
            raise BenchmarkUnavailable("unsupported", "route-label validation exceeds exact FP32 integer range")
        self.weights = gather_rows(work.topk_weights.detach(), ctx).flatten()
        first = sum(work.route.tokens[:ctx.rank]) * work.topk_ids.shape[1]
        self.labels = torch.arange(first, first + work.topk_ids.numel(), dtype=torch.float32,
                                   device=ctx.device).reshape_as(work.topk_ids)

    def start_dispatch(self, work, cached=None, asynchronous=False):
        buffer, torch = self.layer.buffer, self.context.torch
        x = work.hidden_states.detach().contiguous()
        if cached is None:
            options = dict(topk_idx=work.topk_ids.long().contiguous(), topk_weights=self.labels,
                           num_experts=len(work.route.placement), expert_alignment=1, do_cpu_sync=True)
        else:
            options = dict(handle=cached.native, do_cpu_sync=self.spec.id == "deepep_v2_ascend")
        if asynchronous:
            options["async_with_compute_stream"] = True
        rows, _, labels, handle, event = buffer.dispatch(x, do_expand=True, **options)

        def finish():
            if asynchronous:
                event.current_stream_wait()
            route = cached
            if route is None:
                prefix = [0, *handle.psum_num_recv_tokens_per_expert.tolist()]
                counts = tuple(stop - start for start, stop in zip(prefix, prefix[1:]))
                ids = torch.repeat_interleave(
                    torch.tensor(work.local_experts, device=rows.device),
                    torch.tensor(counts, device=rows.device), output_size=len(rows),
                )
                route = ExpandedRoute(handle, counts, ids, labels.long())
            return DispatchResult(rows, route.counts, route, "local_expert_contiguous", route.experts, route.slots)
        return finish

    def dispatch(self, work, cached=None):
        return self.start_dispatch(work, cached)()

    def issue_dispatch(self, work):
        return self.start_dispatch(work, asynchronous=True)

    def start_combine(self, work, dispatched, rows, asynchronous=False):
        route = dispatched.routing_handle
        payload = (rows * self.weights[route.slots, None]).to(rows.dtype).detach().contiguous()
        options = {"async_with_compute_stream": True} if asynchronous else {}
        output, _, event = self.layer.buffer.combine(payload, route.native, **options)

        def finish():
            if asynchronous:
                event.current_stream_wait()
            return CombineResult(output)
        return finish

    def combine(self, work, dispatched, rows):
        return self.start_combine(work, dispatched, rows)()

    def issue_combine(self, work, dispatched, rows):
        return self.start_combine(work, dispatched, rows, asynchronous=True)
