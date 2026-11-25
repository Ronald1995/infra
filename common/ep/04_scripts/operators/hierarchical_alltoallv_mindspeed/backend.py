"""Native benchmark adapter for hierarchical_alltoallv_mindspeed."""
from common.backend import UnavailableBackend


def create(spec):
    return UnavailableBackend(spec)
