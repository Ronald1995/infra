"""ctypes binding for the native L0 transport bridge.

The bridge is optional.  Every failure mode is translated into
:class:`BridgeUnavailable` with one of the benchmark status codes so the case
reports ``dependency_missing`` or ``environment_unavailable`` instead of a
performance number it cannot support.
"""
import ctypes
import json
from pathlib import Path

DEFAULT_NAMES = ("libl0_transport_bridge.so", "l0_transport_bridge.dll",
                 "libl0_transport_bridge.dylib")
BUFFER_BYTES = 1 << 20


class BridgeUnavailable(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


def locate(config=None):
    """Return the configured bridge library path, or ``None``."""
    benchmark = ((config or {}).get("benchmark") or {})
    transport = benchmark.get("transport") or {}
    candidates = [transport.get("library"),
                  (benchmark.get("native") or {}).get("transport_library")]
    import os
    candidates.append(os.getenv("EP_L0_TRANSPORT_LIBRARY"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


_LIBRARIES = {}


def _library(config=None):
    path = locate(config)
    if path is None:
        raise BridgeUnavailable(
            "dependency_missing",
            "native transport bridge is not built; run "
            "baselines/l0/transport/native/build.sh and set "
            "benchmark.transport.library or EP_L0_TRANSPORT_LIBRARY")
    key = (str(path), path.stat().st_mtime_ns)
    if key in _LIBRARIES:
        return _LIBRARIES[key]
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise BridgeUnavailable("dependency_missing", f"cannot load bridge: {exc}") from exc
    library.l0_bridge_describe.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    library.l0_bridge_describe.restype = ctypes.c_int
    library.l0_bridge_last_error.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    library.l0_bridge_last_error.restype = ctypes.c_int
    library.l0_bridge_probe_endpoint.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_char_p, ctypes.c_uint32]
    library.l0_bridge_probe_endpoint.restype = ctypes.c_int
    library.l0_bridge_root_info.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    library.l0_bridge_root_info.restype = ctypes.c_int
    library.l0_session_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_open.restype = ctypes.c_int64
    library.l0_session_comm_init.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_uint32,
                                             ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_comm_init.restype = ctypes.c_int
    library.l0_session_rank_graph.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_rank_graph.restype = ctypes.c_int
    library.l0_session_register.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_uint64,
                                            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_register.restype = ctypes.c_int
    library.l0_session_channels.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p,
                                            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                            ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_channels.restype = ctypes.c_int
    library.l0_session_transfer.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_char_p,
                                            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                                            ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_transfer.restype = ctypes.c_int
    library.l0_session_thread_alloc.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_uint32,
                                                ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_thread_alloc.restype = ctypes.c_int
    library.l0_session_quiet.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_uint32]
    library.l0_session_quiet.restype = ctypes.c_int
    library.l0_session_close.argtypes = [ctypes.c_int64]
    library.l0_session_close.restype = ctypes.c_int
    _LIBRARIES[key] = library
    return library


def _call(config, function, *args):
    library = _library(config)
    buffer = ctypes.create_string_buffer(BUFFER_BYTES)
    result = function(library, buffer, *args)
    if result < 0:
        error = ctypes.create_string_buffer(4096)
        library.l0_bridge_last_error(error, 4096)
        raise BridgeUnavailable("environment_unavailable",
                                error.value.decode("utf-8", "replace") or "bridge call failed")
    return buffer.value.decode("utf-8", "replace")


def describe(config=None):
    """Return the bridge self-description (compiled channels and features)."""
    library = _library(config)
    buffer = ctypes.create_string_buffer(BUFFER_BYTES)
    if library.l0_bridge_describe(buffer, BUFFER_BYTES) < 0:
        raise BridgeUnavailable("environment_unavailable", "bridge describe failed")
    descriptor = json.loads(buffer.value.decode("utf-8", "replace"))
    if not descriptor.get("hccl") or not descriptor.get("hcomm"):
        raise BridgeUnavailable(
            "dependency_missing",
            "bridge was built without HCCL/HCOMM headers; rebuild with ASCEND_HOME_PATH set")
    return descriptor


def probe_endpoint(config, protocol, loc_type, device_id):
    """Ask the runtime to create one endpoint for ``protocol``/``loc_type``."""
    library = _library(config)
    buffer = ctypes.create_string_buffer(BUFFER_BYTES)
    result = library.l0_bridge_probe_endpoint(
        protocol.encode(), 1 if loc_type == "HOST" else 0, device_id, buffer, BUFFER_BYTES)
    if result < 0:
        raise BridgeUnavailable("environment_unavailable", "endpoint probe call failed")
    return json.loads(buffer.value.decode("utf-8", "replace"))


def root_info(config):
    """Return this rank's opaque HCCL root-information blob."""
    library = _library(config)
    buffer = ctypes.create_string_buffer(65536)
    size = library.l0_bridge_root_info(buffer, 65536)
    if size < 0:
        error = ctypes.create_string_buffer(4096)
        library.l0_bridge_last_error(error, 4096)
        raise BridgeUnavailable("environment_unavailable",
                                error.value.decode("utf-8", "replace"))
    return buffer.raw[:size]


class Session:
    """One rank's bridge session: comm, registered memory, channels, transfers."""

    def __init__(self, config, world, rank, device):
        self.config = config
        self.library = _library(config)
        self.buffer = ctypes.create_string_buffer(BUFFER_BYTES)
        handle = self.library.l0_session_open(world, rank, device, self.buffer, BUFFER_BYTES)
        if handle <= 0:
            raise BridgeUnavailable("environment_unavailable",
                                    "l0_session_open failed: " + self._text())
        self.handle = handle
        self.closed = False
        self.peers = []

    def _text(self):
        error = ctypes.create_string_buffer(4096)
        self.library.l0_bridge_last_error(error, 4096)
        return error.value.decode("utf-8", "replace")

    def _json(self, function, *args):
        buffer = ctypes.create_string_buffer(BUFFER_BYTES)
        if function(self.handle, *args, buffer, BUFFER_BYTES) < 0:
            raise BridgeUnavailable("environment_unavailable", self._text())
        return json.loads(buffer.value.decode("utf-8", "replace"))

    def comm_init(self, root_info_bytes):
        return self._json(self.library.l0_session_comm_init, root_info_bytes,
                          len(root_info_bytes))

    def rank_graph(self):
        return self._json(self.library.l0_session_rank_graph)

    def register(self, address, bytes_, tag):
        return self._json(self.library.l0_session_register, ctypes.c_void_p(int(address)),
                          int(bytes_), tag.encode())

    def channels(self, engine, protocol, peers):
        array = (ctypes.c_int * len(peers))(*peers)
        result = self._json(self.library.l0_session_channels, engine.encode(),
                            protocol.encode(), ctypes.cast(array, ctypes.POINTER(ctypes.c_int)),
                            len(peers))
        if result.get("ok"):
            self.peers = result.get("peers", [])
        return result

    def transfer(self, peer_index, mode, local_offset, remote_offset, bytes_):
        return self._json(self.library.l0_session_transfer, int(peer_index), mode.encode(),
                          int(local_offset), int(remote_offset), int(bytes_))

    def thread_alloc(self, engine, notify=4):
        return self._json(self.library.l0_session_thread_alloc, engine.encode(), int(notify))

    def quiet(self):
        return self._json(self.library.l0_session_quiet)

    def close(self):
        if not self.closed:
            self.library.l0_session_close(self.handle)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def available(config=None):
    try:
        describe(config)
        return True, None
    except BridgeUnavailable as exc:
        return False, str(exc)
