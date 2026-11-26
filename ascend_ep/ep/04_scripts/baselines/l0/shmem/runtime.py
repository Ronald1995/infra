"""SHMEM runtime plumbing shared by the MTE and UDMA baselines.

The SHMEM Python package exposes host-stream ``put``/``get``/``quiet`` and a
symmetric heap; device kernels are reached through a small shared library built
from this repository.  Nothing here assumes which engine is active: the caller
passes the engine it initialised, and the observed evidence records that same
engine plus the API symbol that was actually invoked.
"""
import ctypes
from contextlib import ExitStack
import random

from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError

SOURCE_COMMIT = "f0bcfc0fe51bce1d866ba6935d2adc6e5e5800e5"
PATTERN_MAX = 254
SAMPLE_BYTES = 4096


def pattern(rank):
    return 1 + rank % PATTERN_MAX


class Runtime:
    """Initialised SHMEM instance with the symmetric buffers a case needs."""

    def __init__(self, ctx, engine, local_mem_bytes, buffer_sizes, qp_num=0):
        self.ctx = ctx
        self.engine = engine
        self.stack = ExitStack()
        self.buffers = []
        try:
            import acl
            import shmem
            from shmem import core
        except (ImportError, OSError) as exc:
            raise BenchmarkUnavailable("dependency_missing", f"SHMEM is not importable: {exc}") from exc
        self.acl = acl
        self.shmem = shmem
        self.core = core
        rendezvous = ctx.config["benchmark"].get("shmem_rendezvous")
        if not rendezvous:
            raise BenchmarkUnavailable(
                "environment_unavailable",
                "set benchmark.shmem_rendezvous to the SHMEM TCP endpoint")
        try:
            if qp_num:
                configure = getattr(shmem, "aclshmemx_set_qp_num", None)
                if configure is None:
                    raise BenchmarkUnavailable(
                        "unsupported",
                        "this SHMEM build exposes no aclshmemx_set_qp_num; "
                        "multi-QP submission is unavailable")
                result = configure(shmem.OpEngineType.UDMA, int(qp_num))
                if result != 0:
                    raise BenchmarkUnavailable(
                        "unsupported", f"aclshmemx_set_qp_num(UDMA, {qp_num}) returned {result}")
            attributes = shmem.InitAttr()
            attributes.my_rank = ctx.rank
            attributes.n_ranks = ctx.size
            attributes.ip_port = rendezvous
            attributes.local_mem_size = max(int(local_mem_bytes), 1 << 20)
            attributes.option_attr.data_op_engine_type = {
                "mte": shmem.OpEngineType.MTE,
                "udma": shmem.OpEngineType.UDMA,
                "roce": shmem.OpEngineType.ROCE,
            }[engine]
            self._checked(shmem.aclshmem_init(attributes), "aclshmem_init")
            self.stack.callback(shmem.aclshmem_finalize)
            for size in buffer_sizes:
                buffer = core.buffer(int(size))
                self.stack.callback(core.free, buffer)
                self.buffers.append(buffer)
            stream, status = acl.rt.create_stream()
            self._checked(status, "create_stream")
            self.stack.callback(acl.rt.destroy_stream, stream)
            self.stream = stream
        except BaseException:
            self.stack.close()
            raise

    def _checked(self, result, operation):
        if result != 0:
            raise RuntimeError(f"{operation} returned {result}")

    def fill(self, buffer, value, size=None):
        size = buffer.length if size is None else size
        self._checked(self.acl.rt.memset(buffer.addr, size, int(value), size), "memset")

    def read(self, buffer, offset, count):
        host = (ctypes.c_uint8 * count)()
        self._checked(self.acl.rt.memcpy(ctypes.addressof(host), count,
                                         buffer.addr + offset, count, 2), "device_to_host")
        return bytes(host)

    def validate_slots(self, buffer, slot_bytes, expectations):
        """Sample head, tail and interior offsets of every peer slot.

        Full-payload verification would copy the whole message back to the host;
        sampling keeps validation affordable at 1 GiB while still catching a
        payload that landed in the wrong slot or was never written.
        """
        offsets = sorted({0, max(0, slot_bytes - SAMPLE_BYTES),
                          slot_bytes // 3, (2 * slot_bytes) // 3})
        offsets = [offset for offset in offsets if offset < slot_bytes]
        generator = random.Random(0)
        for _ in range(2):
            candidate = generator.randrange(max(1, slot_bytes - SAMPLE_BYTES))
            if candidate not in offsets:
                offsets.append(candidate)
        checked = 0
        for peer, expected in expectations.items():
            base = peer * slot_bytes
            for offset in offsets:
                count = min(64, slot_bytes - offset)
                if count <= 0:
                    continue
                block = self.read(buffer, base + offset, count)
                if block != bytes([expected]) * count:
                    raise CorrectnessError(
                        f"UDMA/MTE payload mismatch at peer slot {peer} offset {offset}: "
                        f"expected byte {expected}, got {block[:8].hex()}")
                checked += count
        return {"checked": "sampled", "bytes_checked": checked,
                "slots": len(expectations), "slot_bytes": slot_bytes}

    def close(self):
        self.stack.close()


def native_library(ctx, config_key, environment, hint):
    """Load a case's native kernel library or raise a precise dependency error."""
    path = ctx.config["benchmark"].get(config_key)
    if not path:
        raise BenchmarkUnavailable(
            "dependency_missing", f"{config_key} is not configured; {hint}")
    try:
        return ctypes.CDLL(str(path))
    except OSError as exc:
        raise BenchmarkUnavailable("dependency_missing",
                                   f"cannot load {config_key}={path}: {exc}") from exc
