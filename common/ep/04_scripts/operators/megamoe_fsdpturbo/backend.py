"""Native benchmark adapter for megamoe_fsdpturbo."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
