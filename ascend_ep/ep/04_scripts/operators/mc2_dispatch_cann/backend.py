"""Native benchmark adapter for mc2_dispatch_cann."""
from operators.ascendep import MC2V2Backend


def create(spec):
    return MC2V2Backend(spec)
