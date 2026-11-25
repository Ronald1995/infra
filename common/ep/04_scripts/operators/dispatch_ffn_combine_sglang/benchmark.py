"""Run only dispatch_ffn_combine_sglang through the shared controller."""
from run_benchmark import single_operator


if __name__ == "__main__":
    raise SystemExit(single_operator("dispatch_ffn_combine_sglang"))
