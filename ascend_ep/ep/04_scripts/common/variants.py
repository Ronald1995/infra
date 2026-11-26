"""Case-specific parameter variants.

Parameter Variant is its own layer so ``scheduler.py`` never grows a table of
magic numbers.  Three rules keep the matrix honest and small:

1. A variant is *case specific*.  Message size, peer count and shape mode are
   selected per case from a shared candidate pool instead of taking a Cartesian
   product over every dimension of every case.
2. A variant that is structurally impossible (an AllToAll message that does not
   divide the EP group, a peer count beyond the available peers, a cross-server
   policy on a single-server EP group) is reported with an explicit status and
   reason; it is never dropped and never silently replaced by another variant.
3. A mechanism that the configuration disables
   (``benchmark.transport.enabled``) never produces an executable variant: the
   case stays in the manifest as ``not_applicable`` instead of disappearing.
"""
from collections import Counter

from common.cases import L0_PARAMETERS, catalog, l0_family
from common.config import PEER_POLICIES

#: Marker carried by a parameter set that must not be executed because its
#: transport mechanism is disabled by the configuration.
TRANSPORT_DISABLED = "transport_disabled"

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


def transport_mechanism(case):
    """The transport mechanism a ``LINK-*`` case drives, or ``None`` off-transport."""
    if l0_family(case) != "transport":
        return None
    return L0_PARAMETERS.get(case.alias_of or case.id, {}).get("mechanism")


def transport_availability(case, settings):
    """Return ``(allowed, mechanism, reason)`` for one transport case.

    ``benchmark.transport.enabled`` is the single source of truth: a mechanism
    that is not listed may not produce an executable task.  ``None`` keeps the
    legacy meaning ("no subset chosen -> every mechanism allowed") and an
    explicit empty list disables all transport mechanisms.
    """
    mechanism = transport_mechanism(case)
    if mechanism is None:
        return True, None, None
    enabled = (settings.get("transport") or {}).get("enabled")
    if enabled is None or mechanism in enabled:
        return True, mechanism, None
    return False, mechanism, (
        f"transport mechanism {mechanism!r} is not enabled by benchmark.transport.enabled")


def transport_gate(case, settings):
    """``(status, reason)`` when ``case`` may not execute at this configuration."""
    allowed, _mechanism, reason = transport_availability(case, settings)
    return None if allowed else ("not_applicable", reason)


def peer_candidates(size, policy="ring", server_ids=None):
    """Smallest number of peers every rank can reach under ``policy``.

    The planner uses the most constrained rank, so a peer sweep is only runnable
    when *every* rank can satisfy it; the executor applies the same rule per rank.
    """
    if policy not in PEER_POLICIES:
        return 0
    if policy == "ring":
        return max(size - 1, 0)
    if not server_ids or len(server_ids) != size:
        return 0
    counts = Counter(server_ids)
    if policy == "same_server":
        return max(min(counts.values()) - 1, 0)
    return max(size - max(counts.values()), 0)


def topology_domains(config):
    """The explicit rank-to-server map, or ``None`` when it is absent/incomplete."""
    topology = (config or {}).get("topology") or {}
    server_ids = topology.get("server_ids")
    if not isinstance(server_ids, list):
        return None
    size = ((config or {}).get("distributed") or {}).get("ep_size")
    return server_ids if size is None or len(server_ids) == size else None


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
    allowed, mechanism, reason = transport_availability(case, settings)
    if not allowed:
        # Defensive: ``task_parameters`` already collapses a disabled mechanism to
        # one explicit parameter set, so no caller can turn it into variants.
        return [{TRANSPORT_DISABLED: True, "variant": variant, "message_bytes": size,
                 "mechanism": mechanism, "reason": reason}]
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


def structural_problem(case, params, ep_size):
    """``(status, reason)`` for an arithmetic constraint of the parameter set.

    A message that does not divide the group or a peer count larger than the EP
    group is impossible regardless of hardware, so it is reported as
    ``unsupported`` before any capability is consulted.
    """
    if isinstance(params.get("message_bytes"), int) and case.id in ALLTOALL_CASES:
        size = params["message_bytes"]
        if size % ep_size:
            return "unsupported", f"AllToAll message_bytes={size} must divide EP{ep_size}"
    peer_count = params.get("peer_count")
    if isinstance(peer_count, int) and peer_count and peer_count > ep_size - 1:
        return "unsupported", f"peer_count={peer_count} exceeds the {ep_size - 1} peers of EP{ep_size}"
    return None


def policy_problem(case, params, ep_size, config=None):
    """``(status, reason)`` when the topology cannot satisfy the peer policy.

    ``same_server`` / ``cross_server`` need an explicit rank-to-server map; a
    policy with no candidate peer (or fewer candidates than requested) is
    ``not_applicable`` with the reason, never a quiet fallback to plain peers.
    """
    policy = params.get("peer_policy")
    if policy not in ("same_server", "cross_server"):
        return None
    peer_count = params.get("peer_count")
    domains = topology_domains(config)
    if domains is None:
        return "not_applicable", (
            f"{policy} peer selection requires an explicit rank-to-server topology for "
            f"EP{ep_size}; the plan carries no complete map")
    available = peer_candidates(ep_size, policy, domains)
    if not available:
        label = "cross-server" if policy == "cross_server" else "same-server"
        return "not_applicable", f"no {label} peer available at EP{ep_size}"
    if isinstance(peer_count, int) and peer_count and peer_count > available:
        return "not_applicable", (
            f"peer_count={peer_count} exceeds the {available} available {policy} peers "
            f"at EP{ep_size}")
    return None


def parameter_availability(case, params, ep_size, config=None):
    """Return ``(status, reason)`` for an unrunnable parameter set, else ``None``."""
    return structural_problem(case, params, ep_size) or policy_problem(
        case, params, ep_size, config)


def parameter_problems(case, params, ep_size, config=None):
    """Reason string for one unrunnable parameter set, or ``None``."""
    problem = parameter_availability(case, params, ep_size, config)
    return problem[1] if problem else None


def task_parameters(case, backend_id, settings, ep_size):
    """All parameter dictionaries for one (case, operator) pair.

    L0 cases ignore the operator dimension, so ``backend_id`` is only used to
    decide whether operator variants apply at all.  A disabled transport
    mechanism collapses to a single explicit parameter set: the case must stay in
    the manifest, but it may not fan out into executable variants.
    """
    entry = catalog()[case.id]
    if entry.level != "L0":
        return [{"variant": variant, "message_bytes": None} for variant in variants_for(entry)]
    if l0_family(entry) == "transport":
        allowed, mechanism, reason = transport_availability(entry, settings)
        if not allowed:
            return [{TRANSPORT_DISABLED: True, "variant": "default",
                     "message_bytes": None, "mechanism": mechanism, "reason": reason}]
    result = []
    for size in message_sizes(entry, settings):
        for variant in variants_for(entry):
            result.extend(l0_parameter_sets(entry, settings, size, variant, ep_size))
    return result
