"""CANN fused combine, residual add and RMSNorm."""
from operators.cann_special import CombineRMSNormBackend


def create(spec):
    return CombineRMSNormBackend(spec)
