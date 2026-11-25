"""Native benchmark adapter for allgather_megatron."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
