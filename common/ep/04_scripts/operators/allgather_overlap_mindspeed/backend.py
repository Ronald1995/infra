"""Native benchmark adapter for allgather_overlap_mindspeed."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
