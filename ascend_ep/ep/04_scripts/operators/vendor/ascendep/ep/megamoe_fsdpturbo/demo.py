"""FSDPTurbo native BF16 training dispatcher with its SHMEM window pool."""
import torch
import torch.distributed as dist
from fsdp_turbo.distributed.expert_parallel import dispatcher_mega_moe as native
from ep.runner import run


def release_pool():
    # Each validation shape gets a new heap and a pool sized by the native dispatcher.
    torch.npu.synchronize()
    dist.barrier()
    instance = native._MEGA_MOE_STATE["shmem"]
    pool = native._WINDOW_POOL_STATE["pools"].pop(id(instance))
    for op in [*pool["ops"], *pool["modules"].values()]:
        op.close()
    instance.destroy()
    native._MEGA_MOE_STATE["shmem"] = None


def forward(layer, x, ids, weights):
    if not hasattr(layer, "native_forward"):
        layer.native_forward = native.get_mega_moe_forward_fn(dist.group.WORLD, fixed_router=False)
        layer.release_buffer = release_pool
    return layer.native_forward(layer, x, ids.int(), weights)


if __name__ == "__main__":
    with torch.autograd.set_multithreading_enabled(False):
        run(forward, "megamoe_fsdpturbo")
