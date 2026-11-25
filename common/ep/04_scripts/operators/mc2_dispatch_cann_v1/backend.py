"""CANN V1 dispatch and combine with an opaque reverse route."""
from operators.cann_special import MC2V1Backend


def create(spec):
    return MC2V1Backend(spec)
