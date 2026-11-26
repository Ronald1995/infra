"""Run only moonep_server_dedup through the shared controller."""
from run_benchmark import single_operator


if __name__ == "__main__":
    raise SystemExit(single_operator("moonep_server_dedup"))
