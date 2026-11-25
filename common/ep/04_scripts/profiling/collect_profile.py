"""Collect a separate rank-local trace with the installed PyTorch profiler."""


def collect(ctx, prepared, directory):
    options = ctx.config["profiling"]
    directory.mkdir(parents=True, exist_ok=True)
    if ctx.device.type == "npu":
        from torch_npu import profiler
        settings = {
            "activities": [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            "experimental_config": profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1),
        }
    else:
        from torch import profiler
        settings = {"activities": [profiler.ProfilerActivity.CPU]}
    warmup, active = options["warmup"], options["active"]
    with profiler.profile(
        **settings, profile_memory=True, record_shapes=True,
        schedule=profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
    ) as profile:
        for i in range(warmup + active):
            inputs = prepared.setup(i)
            ctx.barrier()
            ctx.synchronize()
            prepared.wait(prepared.call(inputs))
            ctx.synchronize()
            profile.step()
