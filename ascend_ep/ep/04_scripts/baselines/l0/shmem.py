"""Native SHMEM host-stream MTE and device-side UDMA put baselines."""
import ctypes
from contextlib import ExitStack

from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError

SOURCE_COMMIT = "f0bcfc0fe51bce1d866ba6935d2adc6e5e5800e5"


def prepare(ctx, size, engine):
    if ctx.device.type == "cpu" or ctx.size < 2:
        raise BenchmarkUnavailable("unsupported", "SHMEM needs at least two Ascend devices")
    if engine == "urma":
        raise BenchmarkUnavailable("unsupported", "pinned CANN SHMEM exposes UDMA, with no distinct URMA engine")
    library = ctx.config["benchmark"].get("shmem_udma_library")
    if engine == "udma" and not library:
        raise BenchmarkUnavailable("dependency_missing", "build native/build_udma.sh and set benchmark.shmem_udma_library")
    rendezvous = ctx.config["benchmark"].get("shmem_rendezvous")
    if not rendezvous:
        raise BenchmarkUnavailable("environment_unavailable", "set benchmark.shmem_rendezvous to the SHMEM TCP endpoint")
    import acl
    import shmem
    from shmem import core

    def checked(result, operation):
        if result != 0:
            raise RuntimeError(f"{operation} returned {result}")

    stack = ExitStack()
    try:
        attributes = shmem.InitAttr()
        attributes.my_rank = ctx.rank
        attributes.n_ranks = ctx.size
        attributes.ip_port = rendezvous
        attributes.local_mem_size = max(1 << 26, size * 4)
        attributes.option_attr.data_op_engine_type = (
            shmem.OpEngineType.UDMA if engine == "udma" else shmem.OpEngineType.MTE
        )
        checked(shmem.aclshmem_init(attributes), "aclshmem_init")
        stack.callback(shmem.aclshmem_finalize)
        src, dst = core.buffer(size), core.buffer(size)
        stack.callback(core.free, src)
        stack.callback(core.free, dst)
        payload = 1 + ctx.rank % 254
        checked(acl.rt.memset(src.addr, size, payload, size), "source memset")
        checked(acl.rt.memset(dst.addr, size, 0, size), "destination memset")
        stream, status = acl.rt.create_stream()
        checked(status, "create_stream")
        stack.callback(acl.rt.destroy_stream, stream)
        if engine == "udma":
            try:
                native = ctypes.CDLL(str(library))
            except OSError as exc:
                raise BenchmarkUnavailable("dependency_missing", f"cannot load UDMA kernel: {exc}") from exc
            put = native.ep_benchmark_udma_put
            put.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                            ctypes.c_int, ctypes.c_void_p]
            put.restype = None
    except BaseException:
        stack.close()
        raise

    def run():
        peer = (ctx.rank + 1) % ctx.size
        if engine == "udma":
            put(dst.addr, src.addr, size, peer, stream)
        else:
            core.put(dst, src, peer, stream)
        checked(acl.rt.synchronize_stream(stream), "synchronize_stream")

    def validate():
        run()
        ctx.barrier()
        expected = 1 + ((ctx.rank - 1) % ctx.size) % 254
        for offset in {0, size - 1}:
            byte = ctypes.c_uint8()
            checked(acl.rt.memcpy(ctypes.addressof(byte), 1, dst.addr + offset, 1, 2), "device_to_host")
            if byte.value != expected:
                raise CorrectnessError("SHMEM received byte differs from peer payload")
        return {"checked": "first_and_last_byte", "source_commit": SOURCE_COMMIT}

    return run, validate, stack.close
