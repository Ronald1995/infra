"""Native benchmark adapter for deepep_sglang."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
