"""Native benchmark adapter for allgather_hccl."""
from baselines.reference.collectives import CollectiveBackend


def create(spec):
    return CollectiveBackend(spec, "allgather")
