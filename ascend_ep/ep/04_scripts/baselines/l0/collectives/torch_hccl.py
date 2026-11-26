"""PyTorch distributed HCCL collective baselines.

These are the original L0 collective cases (``HW-001`` .. ``HW-006``).  They
measure what a framework actually gets from HCCL, with no visibility into the
algorithm or transport HCCL selected, so every result carries
``transport_protocol: auto`` and stays ``unverified`` unless a log or profiler
source names the path.

Payload
-------
Every collective now uses the shared float32 integer payload from
:mod:`baselines.l0.payload`:

* ``uint8`` was used before, and HCCL ReduceScatter rejects ``at::kByte``, so that
  case had to carry a private workaround;
* float32 values are exact small integers and the reduced reference stays inside
  the float32 exact-integer range at every supported EP size, so the check is
  element-wise equality rather than a tolerance;
* the element count is ``message_bytes / 4``, so the transferred byte count (and
  therefore the bandwidth denominator) is unchanged from the ``uint8`` payload.

Verification compares every element against a reference computed from the same
generator, including which peer a byte came from.  "The call returned" is not a
correctness check.
"""
import torch

from baselines.l0 import payload as payloads
from baselines.l0.contract import Prepared
from baselines.l0.verification.vocabulary import requested
from baselines.reference.collectives import scatter_sum
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError

#: Bytes per payload element.
ELEMENT_BYTES = 4


def prepare(ctx, case, size, variant):
    operation = case.operation
    if operation in ("p2p_unidirectional", "p2p_bidirectional", "multi_peer"):
        return _p2p(ctx, operation, size)
    if ctx.size < 2:
        raise BenchmarkUnavailable("unsupported", "communication baseline needs at least two ranks")
    payloads.assert_exact(ctx.size)
    run, result, expected, payload = _collective(ctx, operation, variant, size)
    request = requested(backend="torch_hccl",
                        operation=_operation_name(operation, variant),
                        transport_protocol="auto")

    def validate():
        run()
        got = result if isinstance(result, list) else [result]
        want = expected if isinstance(expected, list) else [expected]
        if len(got) != len(want):
            raise CorrectnessError(
                f"{case.id} produced {len(got)} output buffer(s), expected {len(want)}")
        mismatched = [index for index, (actual, reference) in enumerate(zip(got, want))
                      if not torch.equal(actual, reference)]
        if mismatched:
            raise CorrectnessError(
                f"{case.id} output {mismatched} disagrees with the expected float32 "
                f"collective payload (element-wise, not a tolerance)")
        return {"checked": "element_exact", "dtype": "float32",
                "elements": int(sum(item.numel() for item in want))}

    return Prepared(run=run, validate=validate, request=request, payload=payload,
                    completion="remote_visible",
                    completion_detail=("torch.distributed collective returns after the HCCL "
                                       "operation completed on this rank"))


def _operation_name(operation, variant):
    if operation == "gather_reduce":
        return variant if variant in ("allgather", "reducescatter") else "allgather"
    return operation


def _elements(size):
    """Payload element count for a byte-sized sweep point."""
    return max(1, int(size) // ELEMENT_BYTES)


def _collective(ctx, operation, variant, size):
    n, rank, dist = ctx.size, ctx.rank, ctx.dist
    numel = _elements(size)
    value = payloads.pattern(numel, rank, ctx.device)
    if operation == "alltoall":
        if numel % n:
            raise BenchmarkUnavailable("unsupported", "AllToAll payload must divide EP size")
        result = torch.empty_like(value)
        run = lambda: dist.all_to_all_single(result, value, group=ctx.ep_group)
        expected = payloads.alltoall_reference(numel, n, ctx.device)
        remote = (numel - numel // n) * n * ELEMENT_BYTES
        return run, result, expected, remote
    if operation == "alltoallv":
        # A deliberately skewed, uniform split: every rank sends the same per-peer
        # counts, so the received slice of each peer's payload is predictable.
        shares = [numel * (index + 1) // (n * (n + 1) // 2) for index in range(n)]
        shares[-1] += numel - sum(shares)
        result = torch.empty_like(value)
        run = lambda: dist.all_to_all_single(result, value, shares, shares, group=ctx.ep_group)
        expected = torch.cat([
            payloads.pattern(numel, peer, ctx.device)[sum(shares[:rank]):sum(shares[:rank]) + shares[rank]]
            for peer in range(n)
        ])
        remote = (numel - shares[rank]) * n * ELEMENT_BYTES
        return run, result, expected, remote
    if operation == "allreduce":
        result = torch.empty_like(value)
        run = lambda: dist.all_reduce(result.copy_(value), group=ctx.ep_group)
        expected = payloads.allreduce_reference(numel, n, ctx.device)
        return run, result, expected, numel * (n - 1) * n * ELEMENT_BYTES
    if variant == "allgather":
        buffers = [torch.empty_like(value) for _ in range(n)]
        run = lambda: dist.all_gather(buffers, value, group=ctx.ep_group)
        expected = [payloads.pattern(numel, peer, ctx.device) for peer in range(n)]
        return run, buffers, expected, numel * n * (n - 1) * ELEMENT_BYTES
    # ReduceScatter: float32 payload, exact elementwise reference.  Each rank
    # contributes its own pattern to every destination chunk, so the reduced value
    # is the elementwise sum over ranks -- exactly what reduce_reference returns.
    inputs = value.repeat(n)
    result = torch.empty_like(value)
    if ctx.device.type == "npu":
        run = lambda: dist.reduce_scatter_tensor(result, inputs, group=ctx.ep_group)
    else:
        # Gloo has no reduce_scatter_tensor; the reference implementation sums the
        # same chunks so the CPU path validates the same arithmetic.
        run = lambda: result.copy_(scatter_sum(inputs.view(n, -1), ctx))
    expected = payloads.reduce_reference(numel, n, ctx.device)
    return run, result, expected, numel * (n - 1) * n * ELEMENT_BYTES


def _p2p(ctx, operation, size):
    n, rank, dist = ctx.size, ctx.rank, ctx.dist
    payloads.assert_exact(n)
    numel = _elements(size)
    value = payloads.pattern(numel, rank, ctx.device)
    if operation == "multi_peer":
        sends = receives = [peer for peer in range(n) if peer != rank]
    elif operation == "p2p_bidirectional":
        sends = receives = [1 - rank] if rank < 2 else []
    else:
        sends, receives = ([1] if rank == 0 else []), ([0] if rank == 1 else [])
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
            expected = payloads.pattern(numel, peer, ctx.device)
            if not torch.equal(buffer, expected):
                raise CorrectnessError(
                    f"P2P received a corrupt payload from rank {peer} (element-wise check)")
        return {"checked": "element_exact", "dtype": "float32", "receivers": receives}

    transfers = {"multi_peer": n * (n - 1), "p2p_bidirectional": 2, "p2p_unidirectional": 1}
    return Prepared(run=run, validate=validate,
                    request=requested(backend="torch_hccl", operation="p2p",
                                      transport_protocol="auto"),
                    payload=int(value.numel() * ELEMENT_BYTES * transfers[operation]),
                    completion="remote_visible",
                    completion_detail="batch_isend_irecv work objects waited by the caller")
