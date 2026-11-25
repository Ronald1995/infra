"""Native benchmark adapter for deepep_cann."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
