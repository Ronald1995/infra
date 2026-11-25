"""Native benchmark adapter for alltoall_fsdpturbo."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
