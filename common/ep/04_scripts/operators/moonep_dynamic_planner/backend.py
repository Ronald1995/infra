"""MoonEP dev_npu planner plus dispatch. The full training cycle is a separate case."""
from common.backend import BenchmarkUnavailable, EPBackend


class MoonEPPlannedDispatchBackend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        from npu_patch import Buffer, ShmemRuntime
        if not hasattr(Buffer, "reduce_projection_grads"):
            raise BenchmarkUnavailable("dependency_missing", "the dev_npu planner branch is required")
        self.kind, self.runtime = Buffer, ShmemRuntime
        self.buffer = None
        self.initialized = False

    def prepare(self, work):
        ctx = self.context
        desired = ctx.config["load_balance"].get("redundant_experts", len(work.local_experts))
        if desired != len(work.local_experts):
            raise BenchmarkUnavailable("unsupported", "native planner requires B == local experts")
        use_udma = ctx.config["benchmark"].get("moon_planner_use_udma", True)
        self.buffer = self.kind(
            len(work.hidden_states), work.hidden_states.shape[1], work.topk_ids.shape[1],
            len(work.route.placement), ctx.size, B=desired, group=ctx.ep_group,
            use_udma=use_udma, planning_use_udma=False, explicitly_destroy=True,
        )
        options = dict(group=ctx.ep_group, enable_udma=use_udma, qp_count=2 if use_udma else 0)
        ip_port = ctx.config["benchmark"].get("shmem_rendezvous")
        if ip_port:
            options["ip_port"] = ip_port
        self.runtime.init_with_buffer(**options)
        self.initialized = True

    def planned_dispatch(self, work):
        torch = self.context.torch
        ids = work.topk_ids.int().contiguous()
        counts = torch.bincount(ids.flatten().long(), minlength=len(work.route.placement)).int()
        return self.buffer.dispatch(
            work.hidden_states.detach().contiguous(), work.topk_weights.detach().float(),
            topk_experts_sk=ids, tokens_per_expert=counts,
        )

    def validate_planned_dispatch(self, work, dispatched):
        from common.validation import CorrectnessError, error

        hidden, weights, offsets, plan = dispatched
        if offsets is None or plan.N != work.topk_ids.numel() or (
            tuple(plan.experts_to_copy.shape) != (self.context.size, len(work.local_experts))
        ):
            raise CorrectnessError("MoonEP planner returned invalid shape or route count")
        weighted = (hidden * weights.unsqueeze(1)).to(hidden.dtype).contiguous()
        output, recovered, _ = self.buffer.combine(plan, weighted, route_weights_nvs=weights)
        expected = work.hidden_states.float() * work.topk_weights.float().sum(-1, keepdim=True)
        return {"scope": "planner_dispatch_and_reverse_route",
                "output_relative_l2": error(output, expected, .04),
                "weights_relative_l2": error(recovered, work.topk_weights.float(), 0),
                "planned_expert_copies": int((plan.experts_to_copy >= 0).sum().item())}

    def finalize(self):
        try:
            if getattr(self, "initialized", False):
                self.runtime.finalize()
        finally:
            if getattr(self, "buffer", None) is not None:
                self.buffer.destroy()


def create(spec):
    return MoonEPPlannedDispatchBackend(spec)
