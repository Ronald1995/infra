"""Legacy ``HW-012``: AllToAll on a second stream while GMM1 runs on the default one."""
import torch

from baselines.reference.collectives import CollectiveBackend
from baselines.reference.experts import gmm1
from baselines.l0.contract import Prepared
from baselines.l0.verification.vocabulary import requested
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError


def prepare(ctx, work, size):
    if ctx.size < 2 or ctx.device.type == "cpu":
        raise BenchmarkUnavailable("unsupported",
                                   "multi-stream NPU baseline requires at least two devices")
    reference = CollectiveBackend(None, "alltoall")
    reference.initialize(ctx)
    dispatched = reference.dispatch(work)
    rows = dispatched.expert_input.detach().requires_grad_()
    counts = dispatched.expert_token_counts
    value = torch.full((size,), ctx.rank % 127, dtype=torch.uint8, device=ctx.device)
    destination = torch.empty_like(value)
    stream = torch.npu.Stream(device=ctx.device)

    def run():
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            handle = ctx.dist.all_to_all_single(destination, value, group=ctx.ep_group,
                                                async_op=True)
        result = gmm1(work, rows, counts, ctx)
        handle.wait()
        torch.npu.current_stream().wait_stream(stream)
        return result

    def validate():
        expected = gmm1(work, rows, counts, ctx)
        got = run()
        stripe = size // ctx.size
        peers = (torch.arange(ctx.size, device=ctx.device) % 127).to(torch.uint8)
        if stripe and not torch.equal(destination, peers.repeat_interleave(stripe)):
            raise CorrectnessError("concurrent AllToAll payload did not match")
        return {"relative_l2": [float((got - expected).norm() /
                                     (expected.norm() + 1e-12))]}

    return Prepared(run=run, validate=validate,
                    request=requested(backend="torch_hccl", operation="concurrent",
                                      transport_protocol="auto"),
                    payload=size * (ctx.size - 1))
