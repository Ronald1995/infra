"""Native benchmark adapter for alltoall_megatron."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
