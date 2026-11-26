"""Expanded native communication and expert backend for deepep_ascend."""
from operators.expanded_deepep import ExpandedDeepEPBackend


def create(spec):
    return ExpandedDeepEPBackend(spec)
