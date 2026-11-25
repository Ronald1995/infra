"""Run only ep_gmm_fused_lueelu through the shared controller."""
from run_benchmark import single_operator


if __name__ == "__main__":
    raise SystemExit(single_operator("ep_gmm_fused_lueelu"))
