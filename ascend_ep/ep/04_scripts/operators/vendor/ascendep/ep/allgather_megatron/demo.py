from torch_npu.contrib import transfer_to_npu
from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher
from ep.megatron import dispatch
from ep.runner import run


def forward(layer, x, ids, weights):
    return dispatch(layer, x, ids, weights, MoEAllGatherTokenDispatcher)


if __name__ == "__main__":
    run(forward, "allgather_megatron")
