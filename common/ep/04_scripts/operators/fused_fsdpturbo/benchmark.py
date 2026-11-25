"""Run only fused_fsdpturbo through the shared controller."""
from run_benchmark import single_operator


if __name__ == "__main__":
    raise SystemExit(single_operator("fused_fsdpturbo"))
