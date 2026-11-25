"""Native benchmark adapter for chunkmoe_fsdpturbo."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
