"""Framework allocator measurements and explicit gaps for device counters."""


def reset(ctx):
    if ctx.device.type == "npu":
        ctx.torch.npu.reset_peak_memory_stats(ctx.device)


def snapshot(ctx):
    npu = ctx.torch.npu if ctx.device.type == "npu" else None
    return {
        "unit": "bytes", "scope": "rank", "source": "torch_npu_allocator" if npu else "unavailable",
        "allocated": npu.memory_allocated(ctx.device) if npu else None,
        "reserved": npu.memory_reserved(ctx.device) if npu else None,
        "peak_allocated": npu.max_memory_allocated(ctx.device) if npu else None,
        "peak_reserved": npu.max_memory_reserved(ctx.device) if npu else None,
        "device_hbm": None, "peak_hbm": None, "communication_buffer": None,
        "shmem_heap": None, "workspace": None, "activation": None, "redundant_weight": None,
        "aic": None, "aiv": None,
        "unavailable_reason": "CPU run" if not npu else "allocator excludes native buffers and hardware counters",
    }
