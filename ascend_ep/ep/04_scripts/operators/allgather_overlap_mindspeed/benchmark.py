"""Run only allgather_overlap_mindspeed through the shared controller."""
from run_benchmark import single_operator


if __name__ == "__main__":
    raise SystemExit(single_operator("allgather_overlap_mindspeed"))
