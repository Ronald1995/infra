"""Native benchmark adapter for megamoe_td."""
from operators.ascendep import AscendEPBackend


def create(spec):
    return AscendEPBackend(spec)
