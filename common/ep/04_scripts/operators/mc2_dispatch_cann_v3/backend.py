"""Native benchmark adapter for mc2_dispatch_cann_v3."""
from common.backend import UnavailableBackend


def create(spec):
    return UnavailableBackend(spec)
