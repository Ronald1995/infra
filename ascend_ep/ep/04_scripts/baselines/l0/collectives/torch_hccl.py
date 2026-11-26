"""PyTorch distributed HCCL collective baselines.

These are the original L0 collective cases (``HW-001`` .. ``HW-006``).  They
measure what a framework actually gets from HCCL, with no visibility into the
algorithm or transport HCCL selected, so every result carries
``transport_protocol: auto`` and stays ``unverified`` unless a log or profiler
source names the path.
"""
import torch

from baselines.l0.contract import Prepared
from baselines.l0.verification.vocabulary import requested
from baselines.reference.collectives import scatter_sum
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError


def prepare(ctx, case, size, variant):
    operation = case.operation
    if operation in ("p2p_unidirectional", "p2p_bidirectional", "multi_peer"):
        return _p2p(ctx, operation, size)
    if ctx.size < 2:
        raise BenchmarkUnavailable("unsupported", "communication baseline needs at least two ranks")
    run, result, expected, payload = _collective(ctx, operation, variant, size)
    request = requested(backend="torch_hccl",
                        operation=_operation_name(operation, variant),
                        transport_protocol="auto")

    def validate():
        run()
        got = result if isinstance(result, list) else [result]
        want = expected if isinstance(expected, list) else [expected]
        if any(not torch.equal(a, b) for a, b in zip(got, want)):
            raise CorrectnessError(
                f"{case.id} output bytes disagree with the expected collective payload")
        return {"checked": "byte_exact"}

    return Prepared(run=run, validate=validate, request=request, payload=payload)


def _operation_name(operation, variant):
    if operation == "gather_reduce":
        return variant if variant in ("allgather", "reducescatter") else "allgather"
    return operation


def _collective(ctx, operation, variant, size):
    n, rank, dist = ctx.size, ctx.rank, ctx.dist
    value = torch.full((size,), rank % 127 + 1, dtype=torch.uint8, device=ctx.device)
    if operation == "alltoall":
        if size % n:
            raise BenchmarkUnavailable("unsupported", "AllToAll message must divide EP size")
        result = torch.empty_like(value)
        run = lambda: dist.all_to_all_single(result, value, group=ctx.ep_group)
        expected = torch.arange(1, n + 1, device=ctx.device).sub(1).remainder(127).add(1)
        expected = expected.to(torch.uint8).repeat_interleave(size // n)
        return run, result, expected, (size - size // n) * n
    if operation == "alltoallv":
        # Rotate a deliberately skewed split while preserving the total bytes per rank.
        shares = [size * (index + 1) // (n * (n + 1) // 2) for index in range(n)]
        shares[-1] += size - sum(shares)
        send = [shares[(peer - rank) % n] for peer in range(n)]
        recv = [shares[(rank - peer) % n] for peer in range(n)]
        result = torch.empty_like(value)
        run = lambda: dist.all_to_all_single(result, value, recv, send, group=ctx.ep_group)
        expected = torch.cat([value.new_full((count,), peer % 127 + 1)
                              for peer, count in enumerate(recv)])
        return run, result, expected, (size - shares[0]) * n
    if variant == "allgather":
        buffers = [torch.empty_like(value) for _ in range(n)]
        run = lambda: dist.all_gather(buffers, value, group=ctx.ep_group)
        expected = [value.new_full(value.shape, peer % 127 + 1) for peer in range(n)]
        return run, buffers, expected, size * n * (n - 1)
    inputs = value.repeat(n).view(n, -1)
    result = torch.empty_like(value)
    if ctx.device.type == "npu":
        run = lambda: dist.reduce_scatter_tensor(result, inputs.flatten(), group=ctx.ep_group)
    else:
        run = lambda: result.copy_(scatter_sum(inputs, ctx))
    expected = value.new_full(value.shape, sum(peer % 127 + 1 for peer in range(n)) % 256)
    return run, result, expected, size * n * (n - 1)


def _p2p(ctx, operation, size):
    n, rank, dist = ctx.size, ctx.rank, ctx.dist
    if operation == "multi_peer":
        sends = receives = [peer for peer in range(n) if peer != rank]
    elif operation == "p2p_bidirectional":
        sends = receives = [1 - rank] if rank < 2 else []
    else:
        sends, receives = ([1] if rank == 0 else []), ([0] if rank == 1 else [])
    value = torch.full((size,), rank % 127 + 1, dtype=torch.uint8, device=ctx.device)
    buffers = {peer: torch.empty_like(value) for peer in receives}

    def run():
        requests = [dist.P2POp(dist.irecv, buffers[peer], peer, ctx.ep_group) for peer in receives]
        requests.extend(dist.P2POp(dist.isend, value, peer, ctx.ep_group) for peer in sends)
        if requests:
            for request in dist.batch_isend_irecv(requests):
                request.wait()

    def validate():
        run()
        for peer, buffer in buffers.items():
            if not torch.equal(buffer, value.new_full(value.shape, peer % 127 + 1)):
                raise CorrectnessError(f"P2P received corrupt payload from rank {peer}")
        return {"checked": "byte_exact", "receivers": receives}

    transfers = {"multi_peer": n * (n - 1), "p2p_bidirectional": 2, "p2p_unidirectional": 1}
    return Prepared(run=run, validate=validate,
                    request=requested(backend="torch_hccl", operation="p2p",
                                      transport_protocol="auto"),
                    payload=int(value.numel() * transfers[operation]))
