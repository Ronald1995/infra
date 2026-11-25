"""Native benchmark adapter for alltoallv_gmm_cann."""
from operators.ascendep import FusedGMMBackend


def create(spec):
    return FusedGMMBackend(spec)
