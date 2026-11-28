"""MindSpeed core_r0.8.0 hierarchical AllGather/AllToAllV with the native overlap layer."""
from types import MethodType, SimpleNamespace

from common.backend import BenchmarkUnavailable, EPBackend

MINDSPEED = "3f2fb3701ea26507b739ee4c8ae5f04aea7fa556"
MEGATRON = "baf94af3c667248865f23df73b9fb8e2395e6fd0"


class MindSpeedHierarchicalBackend(EPBackend):
    def initialize(self, ctx):
        super().initialize(ctx)
        self.parallel_state = None
        self.args_module = None
        self.model_parallel_initialized = False
        from operators.hierarchical_alltoallv_mindspeed.native import native_arguments

        self.inner = ctx.config["distributed"].get("hierarchical_inner_size", 0)
        self.check_topology(ctx.gather(ctx.mapping()))
        # The pinned MindSpeed native layer uses torch.cuda APIs internally.
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        from megatron.core import parallel_state
        from megatron.training import global_vars
        from mindspeed.core.transformer.moe import token_dispatcher
        from mindspeed.core.transformer.moe.experts import group_mlp_forward
        from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import MoELayerOverlapAll2All

        self.args_module = global_vars
        self.previous_args = global_vars._GLOBAL_ARGS
        global_vars.set_args(native_arguments())
        self.parallel_state = parallel_state
        self.token_dispatcher_source = token_dispatcher
        self.group_mlp_forward = group_mlp_forward
        self.layer_function = MoELayerOverlapAll2All
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.inner, expert_model_parallel_size=self.outer,
            distributed_timeout_minutes=max(1, ctx.config["distributed"]["timeout_seconds"] // 60),
            order="tp-cp-ep-dp-pp",
        )
        self.model_parallel_initialized = True
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        if not all(ctx.gather(ep_rank * self.inner + tp_rank == ctx.rank)):
            raise BenchmarkUnavailable("unsupported", "Megatron TP-extend-EP ownership does not match benchmark ranks")

    def check_topology(self, mapping):
        ctx = self.context
        if type(self.inner) is not int or self.inner < 2 or ctx.size % self.inner:
            raise BenchmarkUnavailable("unsupported", "set hierarchical_inner_size to a divisor of EP of at least two")
        self.outer = ctx.size // self.inner
        if self.outer < 2:
            raise BenchmarkUnavailable("unsupported", "MindSpeed hierarchy requires at least two physical nodes")
        domains = [m["server"] for m in mapping]
        hosts = [m["host"] for m in mapping]
        if None in domains or len(set(domains)) != self.outer or len(set(hosts)) != self.outer or any(
            len(set(domains[start:start + self.inner])) != 1 or len(set(hosts[start:start + self.inner])) != 1
            for start in range(0, ctx.size, self.inner)
        ):
            raise BenchmarkUnavailable("unsupported", "each inner group must match one physical node and explicit server domain")

    def prepare(self, work):
        from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
        from megatron.core.transformer.transformer_config import TransformerConfig
        from operators.hierarchical_alltoallv_mindspeed.native import configure_dispatcher

        torch, hidden = self.context.torch, work.hidden_states.shape[-1]
        if hidden % self.inner:
            raise BenchmarkUnavailable("unsupported", "the MindSpeed test configuration requires H divisible by the inner group size")
        self.limit = work.limit
        self.local_experts = len(work.local_experts)
        self.config = TransformerConfig(
            num_layers=1, hidden_size=hidden, ffn_hidden_size=work.w_down.shape[-1],
            num_attention_heads=self.inner, tensor_model_parallel_size=self.inner,
            expert_model_parallel_size=self.outer, sequence_parallel=True,
            num_moe_experts=len(work.route.placement), moe_router_topk=work.topk_ids.shape[1],
            moe_token_dispatcher_type="alltoall", moe_grouped_gemm=True,
            gated_linear_unit=True, activation_func=torch.nn.functional.silu, add_bias_linear=False,
            bf16=True, params_dtype=torch.bfloat16,
        )
        dispatcher = MoEAlltoAllTokenDispatcher(self.local_experts, list(work.local_experts), self.config)
        self.dispatcher = configure_dispatcher(dispatcher, self.token_dispatcher_source)

    def native_forward(self, hidden, ids, probabilities, weight1, weight2):
        from baselines.reference.experts import activate

        def router_native(x):
            # The native backward expects the router graph to include a
            # hidden-state derivative, which is zero for this fixed L3 trace.
            return probabilities + x[:, :1] * 0, ids

        expert = SimpleNamespace(weight1=weight1, weight2=weight2, num_local_experts=self.local_experts,
                                 config=self.config, activation_func=lambda x: activate(x, self.limit))
        experts = MethodType(self.group_mlp_forward, expert)
        layer = SimpleNamespace(
            layer_number=1, num_local_experts=self.local_experts, config=self.config,
            token_dispatcher=self.dispatcher, router=router_native, experts=experts,
        )
        output, _ = self.layer_function.apply(hidden, layer)
        return output

    def moe_forward(self, work):
        from operators.hierarchical_alltoallv_mindspeed.native import OverlapLayer

        return OverlapLayer.apply(work.hidden_states, work.topk_ids, work.topk_weights,
                                  work.w_gate, work.w_up, work.w_down, self)

    def finalize(self):
        try:
            if self.model_parallel_initialized:
                self.parallel_state.destroy_model_parallel()
        finally:
            if self.args_module is not None:
                self.args_module.set_args(self.previous_args)


def create(spec):
    return MindSpeedHierarchicalBackend(spec)
