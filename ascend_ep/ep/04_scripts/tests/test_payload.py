"""L0 communication payloads must be exactly verifiable.

The collective payload used to be ``uint8``; HCCL ReduceScatter rejects that dtype,
and a reduce over 128 ranks overflows the float32 exact-integer range well before the
largest swept message size.  These tests pin the arithmetic that makes the reference
exact, so a future payload change cannot quietly turn a correctness check into a
tolerance check.
"""
import pytest
import torch

from baselines.l0 import payload as payloads


def test_pattern_is_a_small_exact_integer():
    value = payloads.pattern(1024, 3, "cpu")
    assert value.dtype == torch.float32
    assert bool((value == value.round()).all())
    assert float(value.min()) == 64.0 * 3
    assert float(value.max()) <= 64.0 * 3 + payloads.PAYLOAD_PERIOD - 1


def test_pattern_identifies_its_rank():
    for rank in (0, 1, 7, 127):
        value = payloads.pattern(16, rank, "cpu")
        assert payloads.rank_of(value) == rank


def test_ranks_never_collide():
    first = set(payloads.pattern(64, 0, "cpu").tolist())
    second = set(payloads.pattern(64, 1, "cpu").tolist())
    assert not first & second


@pytest.mark.parametrize("size", [2, 8, 64, 128])
def test_reduce_reference_is_exact_at_every_supported_ep_size(size):
    payloads.assert_exact(size)
    reference = payloads.reduce_reference(4096, size, "cpu")
    assert float(reference.max()) < payloads.FLOAT32_EXACT_INTEGER
    assert bool((reference == reference.round()).all())
    # The reference is the elementwise sum of the participating ranks.
    expected = sum(payloads.pattern(4096, rank, "cpu") for rank in range(size))
    assert torch.equal(reference, expected)


def test_reduce_reference_is_the_same_arithmetic_as_a_real_reduction():
    size = 8
    reduced = 0
    for rank in range(size):
        reduced = reduced + payloads.pattern(64, rank, "cpu")
    assert torch.equal(reduced, payloads.reduce_reference(64, size, "cpu"))


def test_reduce_budget_overflow_is_refused_not_silently_wrong():
    # The modular pattern keeps a single rank's values small, so the budget is
    # exceeded by the rank *count*, not by the element count.
    assert payloads.max_reduced_value(128) < payloads.FLOAT32_EXACT_INTEGER
    with pytest.raises(ValueError, match="float32"):
        payloads.reduce_reference(4096, 1024, "cpu")
    original = payloads.PAYLOAD_STRIDE
    try:
        payloads.PAYLOAD_STRIDE = 1 << 22
        with pytest.raises(ValueError, match="exact"):
            payloads.assert_exact(128)
    finally:
        payloads.PAYLOAD_STRIDE = original


def test_gather_and_alltoall_references_are_concatenations_of_patterns():
    gathered = payloads.gather_reference(8, 4, "cpu")
    assert gathered.numel() == 32
    assert torch.equal(gathered, torch.cat([payloads.pattern(8, rank, "cpu")
                                            for rank in range(4)]))
    alltoall = payloads.alltoall_reference(32, 4, "cpu")
    assert alltoall.numel() == 32
    assert torch.equal(alltoall, torch.cat([payloads.pattern(8, rank, "cpu")
                                            for rank in range(4)]))


def test_allreduce_reference_matches_reduce_reference():
    assert torch.equal(payloads.allreduce_reference(256, 8, "cpu"),
                       payloads.reduce_reference(256, 8, "cpu"))


def test_payload_byte_budget_matches_the_uint8_baseline():
    """The dtype change must not change the bandwidth denominator."""
    from baselines.l0.collectives import torch_hccl

    assert torch_hccl.ELEMENT_BYTES == 4
    for size in (4096, 65536, 1 << 20):
        elements = torch_hccl._elements(size)
        assert elements * torch_hccl.ELEMENT_BYTES == size
