"""Differentiable EP collectives with explicit split and reverse-route metadata."""
from dataclasses import dataclass, replace
from datetime import timedelta

import torch

from common.backend import BenchmarkUnavailable, CombineResult, DispatchResult, EPBackend


class Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, send, recv, context):
        ctx.send, ctx.recv, ctx.context = send, recv, context
        return exchange(value, send, recv, context)

    @staticmethod
    def backward(ctx, gradient):
        return exchange(gradient, ctx.recv, ctx.send, ctx.context), None, None, None


def exchange(value, send, recv, ctx):
    if ctx.size == 1:
        return value
    output = value.new_empty((sum(recv), *value.shape[1:]))
    ctx.dist.all_to_all_single(output, value.contiguous(), recv, send, group=ctx.ep_group)
    return output


def gather_rows(value, ctx):
    if ctx.size == 1:
        return value.unsqueeze(0)
    buffers = [torch.empty_like(value) for _ in range(ctx.size)]
    ctx.dist.all_gather(buffers, value.contiguous(), group=ctx.ep_group)
    return torch.stack(buffers)


def scatter_sum(value, ctx):
    if ctx.size == 1:
        return value[0]
    output = torch.empty_like(value[0])
    if ctx.device.type == "cpu":
        # Gloo has no list reduce_scatter API.
        reduced = value.contiguous().clone()
        ctx.dist.all_reduce(reduced, group=ctx.ep_group)
        return reduced[ctx.rank]
    ctx.dist.reduce_scatter_tensor(output, value.flatten(0, 1).contiguous(), group=ctx.ep_group)
    return output


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, context):
        ctx.context = context
        return gather_rows(value, context)

    @staticmethod
    def backward(ctx, grad):
        return scatter_sum(grad, ctx.context), None


class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, context):
        ctx.context = context
        return scatter_sum(value, context)

    @staticmethod
    def backward(ctx, grad):
        return gather_rows(grad, ctx.context), None


@dataclass
class AllToAllHandle:
    send: list[int]
    recv: list[int]
    source_order: object
    expert_order: object
    expert_ids: object
    route_slots: object
    counts: tuple[int, ...]


class CollectiveBackend(EPBackend):
    #: A torch.distributed collective returns only after the operation completed on
    #: this rank, so the local buffer holds the peer's data.  ``issue_*`` returns a
    #: deferred wait that calls ``Work.wait()``, which is the same guarantee.
    COMPLETION = ("remote_visible",
                  "the collective returns after the operation completed on this rank")
    #: The algorithm and transport HCCL selected are not exposed by torch.distributed,
    #: so they stay unknown unless a log or profiler record names them.  Guessing one
    #: from the backend name is exactly what the evidence rules forbid.
    ACTUAL_ALGORITHM = None

    def __init__(self, spec, mode):
        super().__init__(spec)
        self.mode = mode
        self.concurrent_group = None

    def actual_algorithm(self):
        """HCCL does not expose its algorithm; ``None`` means "unknown", not "none"."""
        return self.ACTUAL_ALGORITHM

    def prepare_concurrent(self):
        ctx = self.context
        if ctx.device.type != "npu" or ctx.size < 2:
            raise BenchmarkUnavailable("unsupported", "concurrent MoE requires at least two HCCL ranks")
        self.concurrent_group = ctx.dist.new_group(
            ranks=list(range(ctx.size)), backend="hccl",
            timeout=timedelta(seconds=ctx.config["distributed"]["timeout_seconds"]),
        )
        self.concurrent_backend = CollectiveBackend(self.spec, self.mode)
        self.concurrent_backend.initialize(replace(ctx, ep_group=self.concurrent_group))
        self.concurrent_stream = ctx.torch.npu.Stream(device=ctx.device)

    def concurrent_forward(self, work, other):
        ctx = self.context
        npu = ctx.torch.npu
        current = npu.current_stream(ctx.device)
        self.concurrent_stream.wait_stream(current)
        # The two communication lanes own distinct HCCL communicators.
        with npu.stream(self.concurrent_stream):
            second = self.concurrent_backend.moe_forward(other)
        first = self.moe_forward(work)
        current.wait_stream(self.concurrent_stream)
        return first, second

    def finalize(self):
        if self.concurrent_group is not None:
            self.context.dist.destroy_process_group(self.concurrent_group)

    def dispatch(self, workload, cached=None):
        return self.alltoall(workload, cached) if self.mode == "alltoall" else self.allgather(workload)

    def alltoall(self, work, cached):
        ctx = self.context
        k = work.topk_ids.shape[1]
        if cached is None:
            cached = self.pack_routes(work)
        rows = Exchange.apply(work.hidden_states[cached.source_order // k], cached.send, cached.recv, ctx)
        return self.finish_dispatch(rows, cached)

    def pack_routes(self, work):
        ctx = self.context
        ids = work.topk_ids.flatten()
        k = work.topk_ids.shape[1]
        with torch.profiler.record_function("ep_permute"):
            owner = torch.tensor(work.route.placement, device=ids.device)[ids]
            order = owner.argsort(stable=True)
            send_tensor = torch.bincount(owner, minlength=ctx.size)
            recv_tensor = exchange(send_tensor, [1] * ctx.size, [1] * ctx.size, ctx)
            send, recv = send_tensor.tolist(), recv_tensor.tolist()
            global_slots = order + sum(work.route.tokens[:ctx.rank]) * k
            metadata = torch.stack((ids[order], global_slots), dim=-1)
            metadata = exchange(metadata, send, recv, ctx)
            expert_order = metadata[:, 0].argsort(stable=True)
            expert_ids, slots = metadata[expert_order].unbind(1)
            histogram = torch.bincount(expert_ids, minlength=len(work.route.placement)).tolist()
            counts = tuple(histogram[e] for e in work.local_experts)
        return AllToAllHandle(send, recv, order, expert_order, expert_ids, slots, counts)

    @staticmethod
    def finish_dispatch(rows, handle):
        return DispatchResult(rows[handle.expert_order], handle.counts, handle, "local_expert_contiguous",
                              handle.expert_ids, handle.route_slots)

    def issue_dispatch(self, work):
        handle = self.pack_routes(work)
        source = work.hidden_states[handle.source_order // work.topk_ids.shape[1]].contiguous()
        rows, request = self.issue_exchange(source, handle.send, handle.recv)

        def wait():
            if request is not None:
                request.wait()
            return self.finish_dispatch(rows, handle)

        return wait

    def issue_exchange(self, source, send, recv):
        ctx = self.context
        if ctx.size == 1:
            return source, None
        output = source.new_empty((sum(recv), *source.shape[1:]))
        request = ctx.dist.all_to_all_single(output, source, recv, send, group=ctx.ep_group, async_op=True)
        return output, request

    def issue_combine(self, work, dispatched, rows):
        handle = dispatched.routing_handle
        packed, request = self.issue_exchange(rows[handle.expert_order.argsort()], handle.recv, handle.send)

        def wait():
            if request is not None:
                request.wait()
            return self.finish_combine(work, packed[handle.source_order.argsort()])

        return wait

    def allgather(self, work):
        ctx = self.context
        capacity = max(work.route.tokens)
        pad = capacity - len(work.hidden_states)
        functional = torch.nn.functional
        x = Gather.apply(functional.pad(work.hidden_states, (0, 0, 0, pad)), ctx)
        ids = gather_rows(functional.pad(work.topk_ids, (0, 0, 0, pad), value=-1), ctx)
        ranks, tokens, routes = torch.where(ids >= 0)
        experts = ids[ranks, tokens, routes]
        owners = torch.tensor(work.route.placement, device=ids.device)[experts]
        keep = owners == ctx.rank
        experts, ranks, tokens, routes = (v[keep] for v in (experts, ranks, tokens, routes))
        order = experts.argsort(stable=True)
        experts, ranks, tokens, routes = (v[order] for v in (experts, ranks, tokens, routes))
        k = ids.shape[-1]
        offsets = torch.tensor([sum(work.route.tokens[:r]) for r in range(ctx.size)], device=ids.device)
        slots = (offsets[ranks] + tokens) * k + routes
        histogram = torch.bincount(experts, minlength=len(work.route.placement)).tolist()
        counts = tuple(histogram[e] for e in work.local_experts)
        handle = (ranks, tokens, routes, capacity)
        return DispatchResult(x[ranks, tokens], counts, handle, "local_expert_contiguous", experts, slots)

    def combine(self, work, dispatched, rows):
        ctx = self.context
        k = work.topk_ids.shape[1]
        handle = dispatched.routing_handle
        if self.mode == "alltoall":
            ordered = rows[handle.expert_order.argsort()]
            packed = Exchange.apply(ordered, handle.recv, handle.send, ctx)
            returned = packed[handle.source_order.argsort()]
        else:
            ranks, tokens, routes, capacity = handle
            slots = (ranks * capacity + tokens) * k + routes
            output = rows.new_zeros((ctx.size * capacity * k, rows.shape[-1]))
            output = output.index_add(0, slots, rows)
            returned = Scatter.apply(output.reshape(ctx.size, capacity * k, -1), ctx)
            returned = returned[:work.topk_ids.numel()]
        return self.finish_combine(work, returned)

    @staticmethod
    def finish_combine(work, returned):
        k = work.topk_ids.shape[1]
        output = (returned.reshape(len(work.hidden_states), k, returned.shape[-1])
                  * work.topk_weights.unsqueeze(-1)).sum(1)
        return CombineResult(output)

    def moe_forward(self, work):
        from baselines.reference.experts import expert_forward

        with torch.profiler.record_function("ep_dispatch"):
            dispatched = self.dispatch(work)
        rows = expert_forward(work, dispatched, self.context)
        with torch.profiler.record_function("ep_combine"):
            return self.combine(work, dispatched, rows).output
