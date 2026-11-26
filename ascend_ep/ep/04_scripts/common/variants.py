"""Case-specific parameter variants.

Parameter Variant is its own layer so ``scheduler.py`` never grows a table of
magic numbers.  The case *definition* (which variants exist, which dimensions it
sweeps, which message-size window applies) lives on the ``CaseSpec``; this module
only turns that declaration into parameter dictionaries.  Three rules keep the
matrix honest:

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

Expansion never consults a capability verdict.  A task's *parameters* are a
property of the plan, not of the hardware, which is what makes EP8, EP64 and
EP128 produce the same parameter matrix and therefore comparable task counts.
"""
from collections import Counter

from common.case_spec import SIZE_WINDOWS as _SPEC_WINDOWS
from common.case_spec import L0_PARAMETERS, spec as case_spec
from common.config import PEER_POLICIES

#: Marker carried by a parameter set that must not be executed because its
#: transport mechanism is disabled by the configuration.
TRANSPORT_DISABLED = "transport_disabled"

#: Backwards-compatible name for the windows, now declared per case.
SIZE_WINDOWS = {
    "latency": (0, 1 << 20),
    "throughput": (1 << 20, 1 << 30),
    "memory": (0, 1 << 30),
    "compute": (0, 0),
    "overlap": (0, 1 << 24),
}

COMPUTE_SHAPE_MODES = ("fixed", "workload")

#: Which transport mechanisms actually expose the ``cores`` and ``quiet_mode``
#: controls, and to which variants.  This is a *runtime* fact, not a preference:
#:
#: * ``shmem_udma``: ``baselines/l0/shmem/udma.py`` passes ``cores`` as the QP
#:   count and ``quiet_mode`` into ``ep_l0_udma_submit``, and records both in the
#:   validation evidence.  Its ``host`` variant has no QP/quiet control at all
#:   (``aclshmemx_putmem_on_stream``), so those values are fixed there.
#: * every other mechanism: the native transport bridge has no core-count or
#:   async-submit control in its ABI (``l0_session_open``/``channels``/
#:   ``transfer``/``thread_alloc``), and ``l0_session_quiet`` is a drain fence,
#:   not a submit mode.  Declaring a sweep there would create tasks whose
#:   parameter the runtime never reads, so the dimension is ``not_applicable``
#:   with that reason instead of silently running the last configured value.
MECHANISM_DIMENSIONS = {
    "shmem_udma": {"cores": "sweep", "quiet_mode": "sweep",
                   "variants": {"device": "sweep", "host": "fixed"}},
    "shmem_mte": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
    "ub_memory": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
    "urma": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
    "ub_ctp": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
    "ub_rtp": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
    "hccs": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
    "roce": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
    "uboe": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
}

#: Human labels for the reason strings.
DIMENSION_LABELS = {"cores": "core-count", "quiet_mode": "async-submit (quiet) mode"}

#: Where each swept dimension is consumed at runtime.  The plan audit checks that
#: the file exists and mentions the parameter, so a "swept" dimension always has a
#: consumer instead of only a config entry.
DIMENSION_CONSUMERS = {
    "message_bytes": ("baselines/l0/transport/dataplane.py", "message"),
    "outstanding": ("baselines/l0/transport/dataplane.py", "outstanding"),
    "peer_count": ("baselines/l0/transport/channel.py", "peer"),
    "cores": ("baselines/l0/shmem/udma.py", "cores"),
    "quiet_mode": ("baselines/l0/shmem/udma.py", "quiet_mode"),
}


def case_dimensions(case, settings, ep_size=None, variant=None):
    """Per-dimension applicability for one transport case.

    Returns ``{dimension: {"type": "sweep"|"fixed"|"not_applicable", "values":
    [...], "value": x, "reason": "..."}}``.  Only ``sweep`` dimensions multiply
    the task matrix; ``not_applicable`` is a declared decision with a reason, so a
    plan can report *why* a dimension has no coverage instead of pretending it ran.
    """
    from common.case_spec import l0_family

    dimensions = {}
    if l0_family(case) != "transport":
        return dimensions
    mechanism = transport_mechanism(case)
    dimensions["message_bytes"] = {
        "type": "sweep", "values": message_sizes(case, settings),
        "consumer": _consumer("message_bytes")}
    if case.sweeps("outstanding"):
        dimensions["outstanding"] = {
            "type": "sweep", "values": [int(value) for value in
                                        settings.get("outstanding") or [1]],
            "consumer": _consumer("outstanding")}
    else:
        dimensions["outstanding"] = {"type": "fixed", "value": 1,
                                     "reason": "this case measures one in-flight transfer"}
    if case.sweeps("peer_count"):
        counts, reasons = [], []
        for count, reason in peer_counts(case, settings, ep_size or 1):
            if reason:
                reasons.append(reason)
            else:
                counts.append(count)
        dimensions["peer_count"] = {
            "type": "sweep", "values": counts or [0],
            "consumer": _consumer("peer_count"),
            "invalid": reasons}
    else:
        dimensions["peer_count"] = {"type": "fixed", "value": 0,
                                    "reason": "this case uses the whole EP group as peers"}
    support = MECHANISM_DIMENSIONS.get(mechanism) or {}
    variant_support = support.get("variants") or {}
    for name, configured in (("cores", settings.get("cores") or [1]),
                             ("quiet_mode", settings.get("quiet_modes") or [0])):
        kind = support.get(name, "not_applicable")
        if kind == "sweep" and variant is not None:
            kind = variant_support.get(variant, kind)
        if kind == "sweep":
            dimensions[name] = {"type": "sweep",
                                "values": [int(value) for value in configured],
                                "consumer": _consumer(name)}
        elif kind == "fixed":
            dimensions[name] = {
                "type": "fixed", "value": 1 if name == "cores" else 0,
                "reason": (f"{mechanism} {variant or ''} variant has no "
                           f"{DIMENSION_LABELS[name]} control; the value is fixed by the API")
                           .replace("  ", " ")}
        else:
            dimensions[name] = {
                "type": "not_applicable",
                "reason": (f"{mechanism} does not expose any {DIMENSION_LABELS[name]} control; "
                           f"the dimension cannot be swept without changing the runtime")}
    return dimensions


def _consumer(name):
    path, token = DIMENSION_CONSUMERS[name]
    return {"path": path, "token": token}


def dimension_values(dimensions, name):
    """The values a dimension contributes, or the single fixed value."""
    record = dimensions.get(name) or {}
    if record.get("type") == "sweep":
        return list(record.get("values") or [])
    if record.get("type") == "fixed":
        return [record.get("value")]
    return []


def variants_for(case):
    """Named variants for ``case``; ``("default",)`` when it has none."""
    return case_spec(case.id).variants


def transport_mechanism(case):
    """The transport mechanism a ``LINK-*`` case drives, or ``None`` off-transport."""
    from common.case_spec import l0_family

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


def _window(case):
    """``(lower, upper)`` byte window for ``case``."""
    from common.case_spec import l0_family

    window = case.size_window
    if window is None:
        window = {"transport": "latency", "execution": "throughput", "memory": "memory",
                  "overlap": "overlap", "compute": "compute"}.get(l0_family(case), "latency")
    return SIZE_WINDOWS[window]


def message_sizes(case, settings):
    """Select the message-size candidates that apply to ``case``."""
    from common.case_spec import l0_family

    if l0_family(case) == "compute":
        return [None]
    candidates = [int(size) for size in settings.get("message_bytes") or [4096]]
    lower, upper = _window(case)
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
    from common.case_spec import l0_family

    if l0_family(case) != "transport" or not case.sweeps("peer_count"):
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
    static = case.requested()
    transport = settings.get("transport") or {}
    base = {"variant": variant, "message_bytes": size}
    mode = settings.get("transport_mode", "write")
    dimensions = case_dimensions(case, settings, ep_size, variant)
    outstanding = dimension_values(dimensions, "outstanding") or [1]
    cores = dimension_values(dimensions, "cores")
    quiet_modes = dimension_values(dimensions, "quiet_mode")
    if static.get("backend") == "shmem":
        # The SHMEM engines exchange with the whole group; peer selection does not
        # change what they submit, so it is not multiplied into more tasks.  The
        # core count and the quiet mode *do* change the submission (QP count and
        # the async-submit flag), so every applicable combination becomes its own
        # task; a variant without those controls keeps exactly one fixed value.
        sets = []
        for count in outstanding:
            for core in cores or [1]:
                for quiet in quiet_modes or [0]:
                    sets.append({**base, **static, "outstanding": count, "peer_count": 0,
                                 "peer_policy": "group", "cores": core,
                                 "quiet_mode": quiet, "mode": mode})
        return sets
    policies = transport.get("peer_policies") or [
        transport.get("peer_policy") or select_peers_rotation()]
    # A peer count that EP cannot satisfy stays in the matrix as an explicit
    # record (the structural check marks it unsupported); filtering it out here
    # would make the test scope shrink with the EP size.
    peers = [value for value, _problem in peer_counts(case, settings, ep_size)]
    return [{**base, **static, "outstanding": count, "peer_count": peer,
             "peer_policy": policy, "mode": mode}
            for policy in policies
            for count in outstanding
            for peer in peers]


def _execution_sets(case, settings, size, variant):
    static = case.requested()
    modes = (settings.get("execution") or {}).get("modes") or ["default"]
    return [{**{"variant": variant, "message_bytes": size}, **static, "accelerator": mode,
             "operation": static.get("operation", case.operation)} for mode in modes]


def _memory_sets(size, variant):
    return [{"variant": "default", "message_bytes": size}]


def _overlap_sets(case, size, variant):
    static = case.requested()
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
    from common.case_spec import l0_family

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
    if isinstance(params.get("message_bytes"), int) and case.operation == "alltoall":
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


def expand_variants(l0, settings, ep_size):
    """All parameter dictionaries for one L0 case (capability independent)."""
    result = []
    for size in message_sizes(l0, settings):
        for variant in variants_for(l0):
            result.extend(l0_parameter_sets(l0, settings, size, variant, ep_size))
    return result


def operator_variants(case, operator_id=None):
    """All parameter dictionaries for one L1-L4 case.

    A fixed-owner case is measured on exactly one adapter, so *that* adapter's
    parameter set names the owner instead of an adapter sweep; every other adapter
    keeps the case's own variants.
    """
    from common.case_spec import FIXED_OWNER

    if case.owner is not None and case.owner == operator_id:
        return [{"variant": FIXED_OWNER, "message_bytes": None}]
    return [{"variant": variant, "message_bytes": None} for variant in variants_for(case)]


def task_parameters(case, backend_id, settings, ep_size):
    """All parameter dictionaries for one (case, operator) pair.

    L0 cases ignore the operator dimension, so ``backend_id`` is only used to
    decide whether operator variants apply at all.  A disabled transport
    mechanism collapses to a single explicit parameter set: the case must stay in
    the manifest, but it may not fan out into executable variants.
    """
    from common.case_spec import l0_family

    l0 = case_spec(case.id)
    if l0.level != "L0":
        return operator_variants(l0, backend_id)
    if l0_family(l0) == "transport":
        allowed, mechanism, reason = transport_availability(l0, settings)
        if not allowed:
            # One explicit record: the case stays in the manifest, the mechanism is
            # named, and the message size is the case's first swept point so the
            # record keeps the same shape as a runnable-but-unsupported point.
            return [{TRANSPORT_DISABLED: True, "variant": "default",
                     "message_bytes": message_sizes(l0, settings)[0],
                     "mechanism": mechanism, "reason": reason}]
    return expand_variants(l0, settings, ep_size)


def size_window_names():
    """The declared window names, for documentation and schema checks."""
    return tuple(_SPEC_WINDOWS)

def case_dimension_records(cases, settings, ep_size=None):
    """Per-variant dimension declarations for the transport cases of one point.

    Written into the plan metadata, so a reader can see that ``cores=4`` became a
    task for LINK-013 and that it is ``not_applicable`` for LINK-001 *before*
    looking at a single result.
    """
    from common.case_spec import l0_family

    records = {}
    for case in cases:
        if l0_family(case) != "transport":
            continue
        variants = case.variants or ("default",)
        records[case.id] = {
            variant: case_dimensions(case, settings, ep_size, variant)
            for variant in variants}
    return records
