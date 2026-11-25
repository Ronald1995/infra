"""Native benchmark adapter for ep_gmm_fused_lueelu."""
from operators.ascendep import FusedGMMBackend


def create(spec):
    return FusedGMMBackend(spec)
