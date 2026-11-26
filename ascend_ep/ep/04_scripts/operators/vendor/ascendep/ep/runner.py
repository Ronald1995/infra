"""Entry point shared by individual EP demos (torchrun, one rank per NPU)."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from statistics import median, quantiles

import torch
import torch.distributed as dist
from torch.nn import functional as F


def sample(rank, tokens, hidden, ffn, world=8):
    gen = torch.Generator().manual_seed(1729 + rank)
    ids = torch.arange(tokens * 6).reshape(tokens, 6) % 24
    ids = ids[torch.randperm(tokens, generator=gen)]
    ids = ids.gather(1, torch.rand(tokens, 6, generator=gen).argsort(1))
    weights = torch.rand(tokens, 6, generator=gen).softmax(1).mul(2.5)
    x = torch.randn(tokens, hidden, generator=gen)
    dy = torch.randn(tokens, hidden, generator=gen) / hidden**0.5
    gu = torch.randn(24 // world, ffn * 2, hidden, generator=gen) / hidden**0.5
    down = torch.randn(24 // world, hidden, ffn, generator=gen) / ffn**0.5
    return ids, *[t.bfloat16() for t in (weights, x, dy, gu, down)]


def oracle(rank, world=8, dims=(32, 32, 64)):
    # Independent FP32 CPU oracle over the entire small global batch.
    batch = [sample(r, *dims, world=world) for r in range(world)]
    ids, weights, x, dy, gu, down = [torch.cat(items) for items in zip(*batch)]
    weights, x, gu, down = [t.float().requires_grad_() for t in (weights, x, gu, down)]
    out = torch.zeros_like(x)
    for e in range(24):
        rows, cols = torch.where(ids == e)
        gate, up = F.linear(x[rows], gu[e]).chunk(2, -1)
        v = F.linear(F.silu(gate.clamp(max=10)) * up.clamp(-10, 10), down[e])
        out = out.index_add(0, rows, v * weights[rows, cols, None])
    out.backward(dy.float())
    return [t.chunk(world)[rank].detach() for t in (out, x.grad, weights.grad, gu.grad, down.grad)]


def run(forward, name, small_dims=(64, 512, 256)):
    import torch_npu
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--samples", default=50, type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    assert world in (2, 8) and torch.npu.device_count() == world
    validation = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for small in (True, False):
        dims = small_dims if small else (4096, 7168, 3072)
        cpu = sample(rank, *dims, world=world)
        h = hashlib.sha256()
        for t in cpu[:4]:
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        ids, weights, x, dy, gu, down = [t.npu() for t in cpu]
        del cpu
        weights.requires_grad_()
        x.requires_grad_()
        layer = torch.nn.Module()
        layer.gate_up_proj = torch.nn.Parameter(gu)
        layer.down_proj = torch.nn.Parameter(down)
        layer.hidden_dim = dims[1]
        layer.num_global_experts = 24
        layer.expert_ids_per_ep_rank = torch.arange(24 // world, device=x.device).repeat(world)
        layer.act_fn = F.silu
        layer.limit = 10.
        def execute():
            return forward(layer, x, ids, weights)
        def clear():
            layer.zero_grad(set_to_none=True)
            x.grad = weights.grad = None
        out = execute()
        out.backward(dy)
        actual = [out, x.grad, weights.grad, layer.gate_up_proj.grad, layer.down_proj.grad]
        if small:
            ref = oracle(rank, world, dims)
        else:
            from ep.alltoall_hccl.demo import forward as baseline
            actual = [v.detach().cpu() for v in actual]
            clear()
            expected = baseline(layer, x, ids, weights)
            expected.backward(dy)
            ref = [expected, x.grad, weights.grad, layer.gate_up_proj.grad, layer.down_proj.grad]
        errors = []
        for a, b in zip(actual, ref):
            # Validate in bounded CPU chunks, outside the measured region.
            av = a.detach().cpu().reshape(-1)
            bv = b.detach().cpu().reshape(-1)
            difference_sq = reference_sq = 0.0
            for ac, bc in zip(av.split(1 << 20), bv.split(1 << 20)):
                target = bc.float()
                difference_sq += (ac.float() - target).square().sum().item()
                reference_sq += target.square().sum().item()
            errors.append(difference_sq**0.5 / max(reference_sq**0.5, 1e-10))
            if not errors[-1] < 0.035:
                raise ValueError(f"{name}: relative L2 error {errors} exceeds 0.035")
        if small:
            with torch.no_grad():
                for p, expected_grad in zip(layer.parameters(), ref[-2:]):
                    expected_update = p.cpu().float() - 0.01 * expected_grad
                    p.add_(p.grad, alpha=-0.01)
                    torch.testing.assert_close(p.cpu().float(), expected_update, atol=.004, rtol=.01)
        validation["cpu_fp32_small" if small else "hccl_target_shape"] = errors
        del actual, ref, out, a, b, av, bv, ac, bc, target
        clear()
        if small and hasattr(layer, "release_buffer"):
            layer.release_buffer()
    del expected, gu, down
    torch.npu.empty_cache()
    torch.npu.synchronize()
    raw = []
    for i in range(args.warmup + args.samples):
        clear()
        if i == args.warmup:
            torch.npu.reset_peak_memory_stats()
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        out = execute()
        torch.npu.synchronize()
        mid = time.perf_counter()
        out.backward(dy)
        torch.npu.synchronize()
        end = time.perf_counter()
        clear()
        dist.barrier()
        torch.npu.synchronize()
        s = time.perf_counter()
        execute().backward(dy)
        torch.npu.synchronize()
        if i >= args.warmup:
            raw.append([(mid-start)*1000, (end-mid)*1000, (time.perf_counter()-s)*1000])
    record = dict(rank=rank, trace_sha256=h.hexdigest(), raw_ms=raw, relative_l2=errors, validation=validation,
                  allocated=torch.npu.max_memory_allocated(), reserved=torch.npu.max_memory_reserved())
    records = [None]*world
    dist.all_gather_object(records, record)
    if rank == 0:
        metrics = {}
        for j, key in enumerate(("fwd_ms","bwd_ms","fwd_bwd_ms")):
            maxima = [max(r["raw_ms"][i][j] for r in records) for i in range(args.samples)]
            metrics[key] = dict(p50=median(maxima), p95=quantiles(maxima,n=100,method="inclusive")[94])
        result = dict(local_experts=24 // world, backend=name, status="pass", shape=dims, experts=24, top_k=6, ep=world, visible_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                      physical_device_nodes=sorted(p.name for p in Path("/dev").glob("davinci[0-9]*")),
                      dtype="bfloat16", planner="off", torch=str(torch.__version__), torch_npu=torch_npu.__version__,
                      warmup=args.warmup, samples=args.samples, metrics=metrics, ranks=records,
                      source_tokens_per_second=(world*dims[0]*1000)/metrics["fwd_bwd_ms"]["p50"])
        (args.output/f"{name}.json").write_text(json.dumps(result,indent=2))
    if args.profile:
        from torch_npu import profiler
        directory = args.output / "profiling" / name / f"rank{rank}"
        directory.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            schedule=profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            experimental_config=profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1),
            on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        ) as prof:
            for _ in range(4):
                clear()
                dist.barrier()
                torch.npu.synchronize()
                with torch.profiler.record_function("ep_forward"):
                    out = execute()
                with torch.profiler.record_function("ep_backward"):
                    out.backward(dy)
                torch.npu.synchronize()
                prof.step()
    if hasattr(layer, "release_buffer"):
        layer.release_buffer()
    dist.destroy_process_group()
