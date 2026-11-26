"""Native benchmark adapter for moonep."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
