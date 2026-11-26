"""Case-specific parameter variants.

Parameter Variant is its own layer so ``scheduler.py`` never grows a table of
magic numbers.  Two rules keep the matrix honest and small:

1. A variant is *case specific*.  Message size, peer count and shape mode are
   selected per case from a shared candidate pool instead of taking a Cartesian
   product over every dimension of every case.
2. A variant that is structurally impossible (an AllToAll message that does not
   divide the EP group, a peer count beyond the available peers) is returned with
   an explicit status and reason; it is never dropped.
"""
from common.cases import L0_PARAMETERS, catalog, l0_family

#: Named variants for cases whose operation has several distinct kernels/stages.
CASE_VARIANTS = {
    # L0
    "HW-006": ("allgather", "reduce_scatter"),
    "HW-007": ("mte",),
    "HW-008": ("urma", "udma"),
    "HW-010": ("gmm1", "gmm2", "gmm1_dgrad", "gmm1_wgrad", "gmm2_dgrad", "gmm2_wgrad"),
    "HW-011": ("forward", "backward"),
    # L2/L4
    "LAYER-003": ("fresh_plan", "cached_plan"),
    "FUSE-005": ("chunks_2", "chunks_4", "chunks_8"),
    "FUSE-006": ("tile_128_256_256", "tile_256_128_256", "tile_256_256_256"),
    "FUSE-007": ("copy_cores_1", "copy_cores_2", "copy_cores_4", "copy_cores_8"),
}

#: Backwards-compatible name used by older call sites.
L0_VARIANTS = {key: value for key, value in CASE_VARIANTS.items() if key.startswith(("HW-", "LINK-"))}

#: Message-size *window* per L0 family.  The candidate pool comes from
#: ``benchmark.message_bytes``; a case keeps the candidates inside its window, so
#: latency cases are not swept over 1 GiB and throughput cases are not swept over
#: 4 KiB.  The union of all windows still covers the 4 KiB .. 1 GiB range.
SIZE_WINDOWS = {
    "latency": (0, 1 << 20),
    "throughput": (1 << 20, 1 << 30),
    "memory": (0, 1 << 30),
    "compute": (0, 0),
    "overlap": (0, 1 << 24),
}

LINK_THROUGHPUT = frozenset({"LINK-009", "LINK-010", "LINK-011"})
LINK_LATENCY = frozenset({"LINK-001", "LINK-002", "LINK-003", "LINK-004",
                          "LINK-005", "LINK-006", "LINK-007", "LINK-008"})
LINK_SHMEM = frozenset({"LINK-012", "LINK-013"})

#: Framework collective cases that split one message across the EP group.
ALLTOALL_CASES = frozenset({"HW-004", "EXEC-001"})

COMPUTE_SHAPE_MODES = ("fixed", "workload")


def variants_for(case):
    """Named variants for ``case``; ``("default",)`` when it has none."""
    return CASE_VARIANTS.get(case.id, ("default",))


def _window(family, case_id):
    if family == "transport":
        if case_id in LINK_THROUGHPUT:
            return SIZE_WINDOWS["throughput"]
        if case_id in LINK_SHMEM:
            return SIZE_WINDOWS["memory"]
        return SIZE_WINDOWS["latency"]
    if family == "execution":
        return SIZE_WINDOWS["throughput"]
    if family == "memory":
        return SIZE_WINDOWS["memory"]
    if family == "overlap":
        return SIZE_WINDOWS["overlap"]
    if family == "compute":
        return SIZE_WINDOWS["compute"]
    return SIZE_WINDOWS["latency"]


def message_sizes(case, settings):
    """Select the message-size candidates that apply to ``case``."""
    family = l0_family(case)
    candidates = [int(size) for size in settings.get("message_bytes") or [4096]]
    if family == "compute":
        return [None]
    lower, upper = _window(family, case.id)
    selected = [size for size in candidates if lower <= size <= upper]
    # Never return an empty sweep: fall back to the closest candidate and let the
    # capability/parameter checks decide whether it is runnable.
    return selected or [candidates[0]]


def peer_counts(case, settings, ep_size):
    """Return ``(count, reason)`` pairs for the transport peer sweep.

    ``0`` means "every other rank".  A count that exceeds the peers available in
    this EP group stays in the matrix as ``not_applicable`` instead of silently
    disappearing, which is what makes EP8 and EP128 comparable.
    """
    if l0_family(case) != "transport" or case.id in ("HW-007", "HW-008", "LINK-012", "LINK-013"):
        return [(0, None)]
    available = ep_size - 1
    result = []
    for count in settings.get("peer_counts") or [0]:
        count = int(count)
        if count and count > available:
            result.append((count, f"peer_count={count} exceeds the {available} peers of EP{ep_size}"))
        else:
            result.append((count, None))
    return result


def select_peers_rotation():
    """Peer selection policy recorded on every transport task.

    The native driver resolves the policy per rank, so a fixed peer count picks
    the same *kind* of neighbours at EP8 and EP128 instead of the first N ranks.
    """
    return "ring"


def _transport_sets(case, settings, size, variant, ep_size):
    static = dict(L0_PARAMETERS.get(case.id, {}))
    outstanding = [int(value) for value in settings.get("outstanding") or [1]]
    transport = settings.get("transport") or {}
    policies = transport.get("peer_policies") or [transport.get("peer_policy") or select_peers_rotation()]
    if case.id == "HW-008":
        return [{"variant": engine, "message_bytes": size} for engine in ("mte", "udma", "urma")]
    base = {"variant": variant, "message_bytes": size}
    if case.id in ("LINK-012", "LINK-013"):
        # The SHMEM engines exchange with the whole group; peer policy does not
        # change what they submit, so it is not multiplied into more tasks.
        variants = ("default",) if case.id == "LINK-012" else ("device", "host")
        return [{**base, **static, "outstanding": count, "peer_count": peers,
                 "peer_policy": "group", "variant": item,
                 "cores": (settings.get("cores") or [1])[-1],
                 "quiet_mode": (settings.get("quiet_modes") or [0])[-1],
                 "mode": settings.get("transport_mode", "write")}
                for item in variants for count in outstanding
                for peers, _ in peer_counts(case, settings, ep_size)]
    return [{**base, **static, "outstanding": count, "peer_count": peers,
             "peer_policy": policy, "mode": settings.get("transport_mode", "write")}
            for policy in policies
            for count in outstanding
            for peers, _ in peer_counts(case, settings, ep_size)]


def _execution_sets(case, settings, size, variant):
    static = dict(L0_PARAMETERS.get(case.id, {}))
    modes = (settings.get("execution") or {}).get("modes") or ["default"]
    return [{**{"variant": variant, "message_bytes": size}, **static, "accelerator": mode,
             "operation": static.get("operation", case.operation)} for mode in modes]


def _memory_sets(size, variant):
    return [{"variant": "default", "message_bytes": size}]


def _overlap_sets(case, size, variant):
    static = dict(L0_PARAMETERS.get(case.id, {}))
    return [{"variant": variant, "message_bytes": size,
             "transport": static.get("mechanism", "auto"),
             "compute": static.get("compute", "gmm")}]


def _compute_sets(case, settings, variant):
    fixed = dict((settings.get("compute") or {}).get("fixed_shape") or {})
    return [{"variant": variant, "message_bytes": None, "shape_mode": mode,
             "shape": fixed if mode == "fixed" else None}
            for mode in COMPUTE_SHAPE_MODES]


def _collective_sets(case, size, variant):
    return [{"variant": variant, "message_bytes": size}]


def l0_parameter_sets(case, settings, size, variant, ep_size, matrix=None):
    """Expand one L0 case into parameter dictionaries.

    The result is a list so a single (case, size, variant) point can still fan out
    over accelerator modes or peer counts.  ``matrix`` is accepted for signature
    compatibility with the capability probe; parameter expansion itself stays
    hardware independent so a manifest can be generated on a CPU host.
    """
    family = l0_family(case)
    if family == "transport":
        return _transport_sets(case, settings, size, variant, ep_size)
    if family == "execution":
        return _execution_sets(case, settings, size, variant)
    if family == "overlap":
        return _overlap_sets(case, size, variant)
    if family == "memory":
        return _memory_sets(size, variant)
    if family == "compute":
        return _compute_sets(case, settings, variant)
    if family == "collective":
        return _collective_sets(case, size, variant)
    return [{"variant": variant, "message_bytes": size}]


def parameter_problems(case, params, ep_size):
    """Structural constraints that make one parameter set unrunnable.

    Returns a reason string or ``None``.  These are checked before a task is
    handed to a worker so the manifest says *why* a point is unsupported.
    """
    size = params.get("message_bytes")
    if case.id in ALLTOALL_CASES and isinstance(size, int) and size % ep_size:
        return f"AllToAll message_bytes={size} must divide EP{ep_size}"
    peer_count = params.get("peer_count")
    if isinstance(peer_count, int) and peer_count and peer_count > ep_size - 1:
        return f"peer_count={peer_count} exceeds the {ep_size - 1} peers of EP{ep_size}"
    return None


def task_parameters(case, backend_id, settings, ep_size):
    """All parameter dictionaries for one (case, operator) pair.

    L0 cases ignore the operator dimension, so ``backend_id`` is only used to
    decide whether operator variants apply at all.
    """
    entry = catalog()[case.id]
    if entry.level != "L0":
        return [{"variant": variant, "message_bytes": None} for variant in variants_for(entry)]
    result = []
    for size in message_sizes(entry, settings):
        for variant in variants_for(entry):
            result.extend(l0_parameter_sets(entry, settings, size, variant, ep_size))
    return result
