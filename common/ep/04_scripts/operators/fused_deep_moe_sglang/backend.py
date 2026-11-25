"""SGLang FuseEP FusedDeepMoe MXFP8."""
from operators.sglang_fused import SGLangFusedBackend


def create(spec):
    return SGLangFusedBackend(spec, "mxfp8")
