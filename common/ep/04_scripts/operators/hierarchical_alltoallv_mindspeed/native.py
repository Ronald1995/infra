"""Bridge MindSpeed's native overlap autograd to the benchmark's tensor arguments."""
from types import MethodType, SimpleNamespace

import torch


def native_arguments():
    return SimpleNamespace(
        moe_hierarchical_alltoallv=True, moe_alltoall_overlap_comm=True, moe_tp_extend_ep=True,
        moe_zero_memory="disable", moe_zero_memory_num_layers=0, moe_experts_pipeline_degree=0,
        n_shared_experts=None, shared_expert_gate=False, num_layers=1,
        virtual_pipeline_model_parallel_size=None, transformer_pipeline_model_parallel_size=1,
        use_fused_moe_token_permute_and_unpermute=False, moe_router_load_balancing_type="none",
        recompute_activation_function=False, gemm_gradient_accumulation_fusion=False,
    )


class OverlapLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, ids, weights, w_gate, w_up, w_down, backend):
        # The pinned upstream function takes a layer object. Its nested
        # backward accumulates grads on the layer's input/parameter leaves.
        # Keep a separate native graph to return those grads to our explicit
        # benchmark tensor inputs.
        x = hidden.detach().requires_grad_()
        probabilities = weights.detach().requires_grad_()
        first = torch.cat((w_gate, w_up), dim=1).transpose(-1, -2).contiguous().detach().requires_grad_()
        second = w_down.transpose(-1, -2).contiguous().detach().requires_grad_()
        with torch.enable_grad():
            output = backend.native_forward(x, ids, probabilities, first, second)
        ctx.native = (output, x, probabilities, first, second)
        return output.detach()

    @staticmethod
    def backward(ctx, gradient):
        output, x, probabilities, first, second = ctx.native
        torch.autograd.backward(output, gradient)
        if any(value.grad is None for value in (x, probabilities, first, second)):
            raise RuntimeError("MindSpeed native backward did not produce all required gradients")
        gate, up = first.grad.transpose(-1, -2).chunk(2, dim=1)
        return x.grad, None, probabilities.grad, gate, up, second.grad.transpose(-1, -2), None


def configure_dispatcher(dispatcher, source):
    dispatcher.preprocess = MethodType(source.preprocess_tp_extend_ep, dispatcher)
    dispatcher.token_permutation = MethodType(source.alltoall_token_permutation_new, dispatcher)
    dispatcher.token_unpermutation = MethodType(source.alltoall_token_unpermutation_new, dispatcher)
    return dispatcher
