"""Native benchmark adapter for allgather_fsdpturbo."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
