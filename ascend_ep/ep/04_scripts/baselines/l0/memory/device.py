"""HBM device-copy baselines (``MEM-001``, ``MEM-004``, legacy ``HW-009``).

``MEM-001`` copies contiguous device memory; ``MEM-004`` copies the same volume
with a stride of two, which changes the access pattern while keeping the byte
count identical, so the two bandwidth curves are directly comparable.
"""
import torch

from baselines.l0.contract import Prepared
from baselines.l0.verification.vocabulary import requested
from common.validation import CorrectnessError


def prepare(ctx, case, size, params):
    rank = ctx.rank
    value = torch.full((size,), rank % 127 + 1, dtype=torch.uint8, device=ctx.device)
    strided = case.id == "MEM-004"
    result = torch.empty_like(value)
    request = requested(backend="torch_memory", operation="copy",
                        transport_protocol="auto")

    if strided:
        def run():
            result[::2].copy_(value[::2])
    else:
        run = lambda: result.copy_(value)

    def validate():
        run()
        reference = value.clone()
        if strided:
            expected = torch.zeros_like(reference)
            expected[::2] = reference[::2]
        else:
            expected = reference
        if not torch.equal(result, expected):
            raise CorrectnessError(f"{case.id} device copy produced different bytes")
        return {"checked": "byte_exact", "pattern": "strided" if strided else "contiguous"}

    # One read plus one write per element: the logical traffic, not the physical
    # HBM counter, which the report keeps empty when no counter is available.
    payload = {"per_rank_bytes": size * 2, "global_bytes": size * 2 * ctx.size,
               "unit": "bytes", "scope": "per_rank_and_global", "direction": "bidirectional",
               "definition": "logical read + write bytes of one copy"}
    # A torch copy is not evidence about which engine executed it; the record
    # stays unverified unless profiling names the kernel.
    return Prepared(run=run, validate=validate, request=request, payload=payload)
