"""Exactly verifiable payloads for collective correctness checks.

Background
----------
The framework collectives used to validate an ``uint8`` byte pattern.  HCCL
ReduceScatter rejects ``at::kByte`` as an input dtype, so that case had to grow a
private ``float32`` workaround, and the AllReduce path never got one.  The result
was three different payload conventions in one file, one of which could not run.

A second problem is arithmetic: a reduction over 128 ranks multiplies every value
by 128, and ``float32`` only represents integers exactly up to ``2**24``.  A naive
``arange(numel)`` payload at EP128 overflows that bound and the "reference" result
stops being exact, which turns a correctness check into a tolerance check nobody
chose.

Design
------
One payload generator for every collective:

``pattern(numel, rank)``
    ``(arange(numel) % PERIOD) + rank * STRIDE`` with ``PERIOD == STRIDE == 64``.

Because the stride equals the period:

* every value is a small non-negative integer, exactly representable in float32;
* two ranks can never produce the same value, so a received element identifies the
  rank that sent it (needed by AllToAll/AllGather verification);
* the largest possible reduction is ``size * ((size - 1) * STRIDE + PERIOD - 1)``,
  which is ~1.05e6 at EP128 -- two orders of magnitude below ``2**24``, so the
  reduced reference stays exact;
* the reference is computed with the same integer arithmetic, so the check is
  element-wise equality rather than an error threshold.

The generator is shared by every collective and by the tests, so a payload cannot
silently change meaning between the measurement and its reference.
"""
import torch

#: Values repeat with this period, and each rank is offset by exactly one period.
PAYLOAD_PERIOD = 64
PAYLOAD_STRIDE = 64

#: Largest exactly representable integer in float32.
FLOAT32_EXACT_INTEGER = 1 << 24

#: Copy-only baselines (native transport, local memory movement) still compare raw
#: bytes: no reduction happens there, so ``uint8`` is the right type and there is no
#: HCCL dtype restriction to work around.  Collectives must not use it.
COPY_PATTERN_MODULUS = 251


def pattern(numel, rank, device, dtype=torch.float32):
    """The deterministic payload for one rank."""
    if numel < 0:
        raise ValueError("numel must be non-negative")
    index = torch.arange(int(numel), dtype=torch.float32, device=device)
    return ((index % PAYLOAD_PERIOD) + float(rank * PAYLOAD_STRIDE)).to(dtype)


def rank_of(value):
    """The rank a pattern value came from, or ``None`` when it is not a pattern.

    Every element of one rank's pattern falls inside ``[rank*STRIDE,
    rank*STRIDE + PERIOD)``, so the whole tensor must agree on the rank.
    """
    if value.numel() == 0 or not value.is_floating_point():
        return None
    low = int(value.min().item())
    high = int(value.max().item())
    rank = low // PAYLOAD_STRIDE
    if not (rank * PAYLOAD_STRIDE <= low <= high < (rank + 1) * PAYLOAD_STRIDE):
        return None
    return rank


def reduce_reference(numel, size, device, dtype=torch.float32):
    """Elementwise sum of every rank's pattern, computed exactly."""
    index = torch.arange(int(numel), dtype=torch.float32, device=device)
    base = (index % PAYLOAD_PERIOD) * size
    offset = PAYLOAD_STRIDE * size * (size - 1) / 2
    result = base + offset
    if float(result.max()) >= FLOAT32_EXACT_INTEGER:
        raise ValueError(
            f"reduce payload for numel={numel} and size={size} exceeds the float32 "
            f"exact-integer range; reduce the payload size or the rank count")
    return result.to(dtype)


def gather_reference(numel, size, device, dtype=torch.float32):
    """The expected AllGather output: one rank pattern per peer, in rank order."""
    return torch.cat([pattern(numel, rank, device, dtype) for rank in range(size)])


def alltoall_reference(numel, size, device, dtype=torch.float32):
    """The expected AllToAll output for one rank: peer ``p``'s slice of ``p``'s data.

    Every rank sends the same shape, so the result is a stack of the peers' slices.
    """
    chunk = numel // size
    return torch.cat([pattern(chunk, peer, device, dtype) for peer in range(size)])


def allreduce_reference(numel, size, device, dtype=torch.float32):
    """The expected AllReduce output: the exact elementwise sum."""
    return reduce_reference(numel, size, device, dtype)


def max_reduced_value(size):
    """The largest value a reduction over ``size`` ranks can produce."""
    return size * ((size - 1) * PAYLOAD_STRIDE + PAYLOAD_PERIOD - 1)


def assert_exact(size):
    """Fail early when an EP group would overflow the exact-integer budget."""
    if max_reduced_value(size) >= FLOAT32_EXACT_INTEGER:
        raise ValueError(
            f"an EP{size} reduction cannot be verified exactly in float32 "
            f"(max {max_reduced_value(size)} >= {FLOAT32_EXACT_INTEGER})")
    return max_reduced_value(size)
