"""Native benchmark adapter for mc2_fsdpturbo."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
