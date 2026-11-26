from torch_npu.contrib import transfer_to_npu
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from ep.megatron import dispatch
from ep.runner import run


def forward(layer, x, ids, weights):
    return dispatch(layer, x, ids, weights, MoEAlltoAllTokenDispatcher)


if __name__ == "__main__":
    run(forward, "alltoall_megatron")
