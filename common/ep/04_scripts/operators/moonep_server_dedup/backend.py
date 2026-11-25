"""Native rank and eight-rank server dedup communication with roundtrip validation."""
from common.backend import BenchmarkUnavailable, EPBackend

SOURCE = "d8495c4745ba6310b40eddbdb9e3e9ef7b2c2b5d"


class MoonEPDedupBackend(EPBackend):
    def initialize(self, ctx):
        super().initialize(ctx)
        from npu_patch import Buffer, ShmemRuntime
        import inspect

        if "dedup_mode" not in inspect.signature(Buffer).parameters:
            raise BenchmarkUnavailable("dependency_missing", "npu_patch lacks dev_server_dedup Buffer")
        self.kind, self.runtime = Buffer, ShmemRuntime
        self.buffer = None
        self.initialized = False

    def prepare(self, work):
        ctx = self.context
        mode = ctx.config["dedup"]["mode"]
        if mode not in {"rank", "server"}:
            raise BenchmarkUnavailable("unsupported", "select native rank or server dedup")
        if mode == "server":
            domains = ctx.config["topology"].get("server_ids")
            if not domains or ctx.size % 8 or len(set(domains)) != ctx.size // 8 or any(
                len(set(domains[i:i + 8])) != 1 for i in range(0, ctx.size, 8)
            ):
                raise BenchmarkUnavailable("unsupported", "server dedup requires an explicit contiguous eight-rank board map")
        self.buffer = self.kind(
            len(work.hidden_states), work.hidden_states.shape[1], work.topk_ids.shape[1],
            len(work.route.placement), ctx.size, B=len(work.local_experts), group=ctx.ep_group,
            use_udma=True, dedup_mode=mode, ranks_per_server=8, explicitly_destroy=True,
        )
        options = dict(group=ctx.ep_group, enable_udma=True, qp_count=2)
        ip_port = ctx.config["benchmark"].get("shmem_rendezvous")
        if ip_port:
            options["ip_port"] = ip_port
        self.runtime.init_with_buffer(**options)
        self.initialized = True

    def deduplicated_dispatch(self, work):
        torch = self.context.torch
        ids = work.topk_ids.int().contiguous()
        counts = torch.bincount(ids.flatten().long(), minlength=len(work.route.placement)).int()
        return self.buffer.dispatch(
            work.hidden_states.detach().contiguous(), work.topk_weights.detach().float(),
            topk_experts_sk=ids, tokens_per_expert=counts,
        )

    def validate_dedup(self, work, dispatched):
        from common.validation import CorrectnessError, error

        hidden, weights, offsets, plan = dispatched
        if offsets is None or plan.N != work.topk_ids.numel() or plan.dedup_mode != (
            self.context.config["dedup"]["mode"]
        ):
            raise CorrectnessError("native dedup planner returned incompatible layout metadata")
        # The plan describes which capacity rows are valid. Combine reads those
        # rows and returns the reverse route scalars with the activation.
        weighted = (hidden * weights.unsqueeze(1)).to(hidden.dtype).contiguous()
        output, recovered, _ = self.buffer.combine(plan, weighted, route_weights_nvs=weights)
        expected = work.hidden_states.float() * work.topk_weights.float().sum(-1, keepdim=True)
        return {"scope": "native_dedup_payload_and_reverse_route", "source_commit": SOURCE,
                "weights_relative_l2": error(recovered, work.topk_weights.float(), 0),
                "output_relative_l2": error(output, expected, .04),
                "native_dup_counts": plan.dup_counts.tolist()}

    def finalize(self):
        try:
            if getattr(self, "initialized", False):
                self.runtime.finalize()
        finally:
            if getattr(self, "buffer", None) is not None:
                self.buffer.destroy()


def create(spec):
    return MoonEPDedupBackend(spec)
