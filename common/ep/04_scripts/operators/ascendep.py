"""Bridge to pinned ascendep native forward adapters without vendoring upstream code."""
from functools import lru_cache
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from common.backend import BenchmarkUnavailable, CombineResult, DispatchResult, EPBackend

UPSTREAM_COMMIT = "24ff84d3ab8c01e1e2071802465173e624e1f89d"


@lru_cache
def source_checkout(value):
    if not value:
        raise BenchmarkUnavailable("dependency_missing", "set EP_ASCENDEP_ROOT to the pinned ascendep checkout")
    root = Path(value).resolve()
    if not (root / "ep" / "runner.py").is_file():
        raise BenchmarkUnavailable("source_unavailable", f"ascendep source not found at {root}")
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           encoding="ascii", timeout=10).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                         "--untracked-files=no", "--", "ep"],
                                        encoding="utf-8", timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkUnavailable("source_unavailable", f"cannot identify ascendep checkout: {exc}") from exc
    if revision != UPSTREAM_COMMIT or dirty:
        raise BenchmarkUnavailable("source_unavailable", f"expected an unchanged ascendep {UPSTREAM_COMMIT}, found {revision}")
    return root


class AscendEPBackend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        self.layer = None
        root = source_checkout(os.getenv("EP_ASCENDEP_ROOT"))
        sys.path.insert(0, str(root))
        self.demo = import_module(f"ep.{self.spec.upstream or self.spec.id}.demo")
        if not Path(self.demo.__file__).resolve().is_relative_to(root):
            raise BenchmarkUnavailable("source_unavailable", "another ep package shadows the pinned ascendep checkout")

    def prepare(self, work):
        torch = self.context.torch
        self.layer = SimpleNamespace(
            down_proj=work.w_down, hidden_dim=work.hidden_states.shape[1],
            num_global_experts=len(work.route.placement), limit=work.limit,
            act_fn=torch.nn.functional.silu,
            expert_ids_per_ep_rank=torch.arange(len(work.local_experts), device=self.context.device).repeat(
                self.context.size
            ),
        )
        if self.spec.id in {"moonep", "deepep_cann"} and (
            sum(work.route.tokens) * work.topk_ids.shape[1] % len(work.route.placement)
        ):
            raise BenchmarkUnavailable("unsupported", "native adapter requires equal expert receive counts")

    def moe_forward(self, work):
        # Packing is required on each call so the weight gradients retain their graph.
        torch = self.context.torch
        self.layer.gate_up_proj = torch.cat((work.w_gate, work.w_up), dim=1)
        return self.demo.forward(self.layer, work.hidden_states, work.topk_ids, work.topk_weights)

    def finalize(self):
        if self.layer is not None:
            release = getattr(self.layer, "release_buffer", None)
            if release is not None:
                release()


class MC2V2Backend(AscendEPBackend):
    def prepare(self, work):
        super().prepare(work)
        self.group = None

    def exchange(self, work):
        if len(work.hidden_states) > 512:
            raise BenchmarkUnavailable("unsupported", "published 950DT V2 adapter limit is 512 tokens per native invocation")
        if self.group is None:
            self.group = self.context.dist.new_group(backend="hccl")
        return self.demo.Exchange(work.topk_ids, len(work.route.placement), self.group)

    def dispatch(self, work, cached=None):
        exchange = self.exchange(work)
        rows = self.demo.Dispatch.apply(work.hidden_states, exchange)
        return self.result(work, rows, exchange.sizes, exchange.slots, exchange)

    def result(self, work, rows, counts, slots, handle):
        torch = self.context.torch
        experts = torch.repeat_interleave(torch.tensor(work.local_experts, device=rows.device),
                                         torch.tensor(counts, device=rows.device))
        return DispatchResult(rows, tuple(counts), handle, "local_expert_contiguous", experts, slots)

    def quantized_dispatch(self, work):
        import torch_npu

        exchange = self.exchange(work)
        return torch_npu.npu_moe_distribute_dispatch_v2(
            work.hidden_states.contiguous(), exchange.ids, *exchange.config,
            expert_scales=exchange.labels, shared_expert_num=0, global_bs=exchange.batch,
            expert_token_nums_type=1, quant_mode=2,
        )

    def validate_quantized_dispatch(self, work, native, trace):
        from common.validation import CorrectnessError, validate_dispatch

        rows, scales, inverse, counts, _, _, labels = native
        counts = counts.tolist()
        used = sum(counts)
        if rows.dtype != self.context.torch.int8 or scales[:used].numel() != used:
            raise CorrectnessError("quantization mode 2 must return INT8 rows and one scale per route")
        dequantized = rows[:used].float() * scales[:used].reshape(used, 1).float()
        result = self.result(work, dequantized, counts, labels[:used].long(), inverse)
        return validate_dispatch(work, result, self.context, trace, threshold=.015)

    def combine(self, work, dispatched, rows):
        from baselines.reference.collectives import Gather

        exchange = dispatched.routing_handle
        weights = Gather.apply(work.topk_weights, self.context).flatten()[exchange.slots]
        output = self.demo.Combine.apply(rows * weights[:, None], exchange)
        return CombineResult(output)

    def finalize(self):
        try:
            super().finalize()
        finally:
            if getattr(self, "group", None) is not None:
                self.context.dist.destroy_process_group(self.group)


class FusedGMMBackend(AscendEPBackend):
    def route(self, work):
        from baselines.reference.collectives import exchange

        ctx = self.context
        torch = ctx.torch
        ids = work.topk_ids.flatten()
        order = ids.argsort(stable=True)
        send = torch.bincount(ids, minlength=len(work.route.placement))
        n = len(work.local_experts)
        recv = exchange(send, [n] * ctx.size, [n] * ctx.size, ctx)
        counts = recv.view(ctx.size, -1).sum(0)
        return order, send, recv, counts

    def dispatch_gmm(self, work):
        order, send, recv, counts = self.route(work)
        torch = self.context.torch
        w = torch.cat((work.w_gate, work.w_up), dim=1)
        x = work.hidden_states[order // work.topk_ids.shape[1]]
        ctx = self.context
        if self.spec.id == "alltoallv_gmm_cann":
            name = ctx.ep_group._get_backend(x.device).get_hccl_comm_name(ctx.rank)
            return self.demo.DispatchLinear.apply(x, w, name, ctx.size, send.tolist(), recv.tolist(), counts)
        self.ensure_udma()
        route = self.demo.Route(work.topk_ids, len(work.route.placement))
        return self.demo.DispatchLinear.apply(x, w, route)

    def gmm_combine(self, work, activation):
        order, send, recv, counts = self.route(work)
        ctx = self.context
        if self.spec.id == "alltoallv_gmm_cann":
            name = ctx.ep_group._get_backend(activation.device).get_hccl_comm_name(ctx.rank)
            result = self.demo.CombineLinear.apply(
                activation, work.w_down, name, ctx.size, recv.tolist(), send.tolist(), counts
            )
        else:
            self.ensure_udma()
            route = self.demo.Route(work.topk_ids, len(work.route.placement))
            result = self.demo.CombineLinear.apply(activation, work.w_down, route)
        result = result[order.argsort()].reshape(*work.topk_ids.shape, work.w_down.shape[1])
        return (result * work.topk_weights.unsqueeze(-1)).sum(1)

    def ensure_udma(self):
        if self.layer is not None and not hasattr(self.layer, "release_buffer"):
            self.demo.init_udma_group(group=self.context.ep_group)
            self.layer.release_buffer = self.demo.destroy_udma_group
