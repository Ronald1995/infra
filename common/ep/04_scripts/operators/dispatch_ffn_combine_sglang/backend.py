"""SGLang FuseEP DispatchFFNCombine W8A8."""
from operators.sglang_fused import SGLangFusedBackend


def create(spec):
    return SGLangFusedBackend(spec, "int8")
