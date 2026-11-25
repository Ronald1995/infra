"""Native benchmark adapter for alltoall_hccl."""
from baselines.reference.collectives import CollectiveBackend


def create(spec):
    return CollectiveBackend(spec, "alltoall")
