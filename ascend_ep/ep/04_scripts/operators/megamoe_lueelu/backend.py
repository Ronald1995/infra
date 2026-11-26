"""Native benchmark adapter for megamoe_lueelu."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
