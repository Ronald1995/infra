"""One worker per device with bounded process-group lifetime."""
from dataclasses import dataclass
from datetime import timedelta
import os
import socket
from contextlib import contextmanager

from common.backend import BenchmarkUnavailable


@dataclass
class ExecutionContext:
    config: dict
    torch: object
    dist: object
    device: object
    rank: int
    size: int
    local_rank: int
    ep_group: object
    tp_group: object
    metadata_group: object

    def synchronize(self):
        if self.device.type != "cpu":
            self.torch.npu.synchronize(self.device)

    def barrier(self):
        if self.size > 1:
            self.dist.barrier(group=self.ep_group)

    def gather(self, value):
        if self.size == 1:
            return [value]
        objects = [None] * self.size
        self.dist.all_gather_object(objects, value, group=self.metadata_group)
        return objects

    def mapping(self):
        visibility = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        return {
            "rank": self.rank, "local_rank": self.local_rank,
            "node_rank": int(os.getenv("GROUP_RANK", "0")), "host": socket.gethostname(),
            "device": str(self.device),
            "physical_device": visibility[self.local_rank] if self.device.type == "npu"
            and self.local_rank < len(visibility) and visibility[self.local_rank] else None,
            "server": self.config["topology"].get("server_ids", [None] * self.size)[self.rank],
            "supernode": self.config["topology"].get("supernode_ids", [None] * self.size)[self.rank],
        }


@contextmanager
def open_context(config):
    try:
        import torch
        import torch.distributed as dist
    except (ImportError, OSError) as exc:
        raise BenchmarkUnavailable("dependency_missing", f"PyTorch failed to import: {exc}") from exc

    device_type = config["device"]
    d = config["distributed"]
    size, tp = d["ep_size"], d["tp_size"]
    rank, local_rank = (int(os.getenv(key, "0")) for key in ("RANK", "LOCAL_RANK"))
    if int(os.getenv("WORLD_SIZE", "1")) != size * tp:
        raise BenchmarkUnavailable("environment_unavailable", "WORLD_SIZE must equal EP x TP")
    if tp != 1:
        raise BenchmarkUnavailable("not_implemented", "TP-sharded expert and router semantics are not adapted")
    if device_type == "npu":
        try:
            import torch_npu  # noqa: F401
        except (ImportError, OSError) as exc:
            raise BenchmarkUnavailable("dependency_missing", f"torch_npu failed to import: {exc}") from exc
        if not torch.npu.is_available() or local_rank >= torch.npu.device_count():
            raise BenchmarkUnavailable("environment_unavailable", f"NPU local rank {local_rank} unavailable")
        torch.npu.set_device(local_rank)
    else:
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))
    device = torch.device(f"npu:{local_rank}" if device_type == "npu" else "cpu")
    try:
        # Native demos need WORLD even with one NPU.
        if size > 1 or device_type == "npu":
            store = os.getenv("EP_GLOO_FILE_STORE") if device_type == "cpu" else None
            rendezvous = {"init_method": store, "rank": rank, "world_size": size} if store else {}
            dist.init_process_group("hccl" if device_type == "npu" else "gloo",
                                    timeout=timedelta(seconds=d["timeout_seconds"]), **rendezvous)
        group = dist.group.WORLD if dist.is_initialized() else None
        yield ExecutionContext(config, torch, dist, device, rank, size, local_rank,
                               group, None, group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
