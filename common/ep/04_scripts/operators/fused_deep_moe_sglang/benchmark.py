"""Run only fused_deep_moe_sglang through the shared controller."""
from run_benchmark import single_operator


if __name__ == "__main__":
    raise SystemExit(single_operator("fused_deep_moe_sglang"))
