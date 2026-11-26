"""Native channel benchmark driver shared by every ``LINK-*`` case.

One case is one (engine, protocol, endpoint location) triple.  The driver:

1. opens a native bridge session and initialises an HCCL communicator using the
   root information exchanged through torch.distributed;
2. queries the rank graph before creating anything, so the link protocol the
   runtime reports is recorded as evidence, not assumed;
3. registers two local symmetric buffers, acquires one channel per peer for the
   requested engine and protocol, and reads back the remote memory descriptors;
4. measures submission and completion separately, sweeping ``outstanding``
   requests and peers;
5. validates the received bytes against the per-rank pattern.

A case that cannot reach any of these steps raises
:class:`~common.backend.BenchmarkUnavailable` with the precise status; it never
falls back to another engine or protocol.
"""
import base64

from baselines.l0.contract import Prepared
from baselines.l0.transport import bridge, dataplane, peers
from baselines.l0.transport.common import profiles
from baselines.l0.verification import evidence as evidence_collector
from baselines.l0.verification.path_verifier import Evidence
from baselines.l0.verification.vocabulary import requested
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError

CHUNK_BYTES = 256 * 1024 * 1024
PATTERN_MODULUS = 251


def payload(ctx, size, peers):
    return {"per_rank_bytes": size * len(peers), "global_bytes": size * len(peers) * ctx.size,
            "unit": "bytes", "scope": "per_rank_and_global", "direction": "one_way",
            "definition": ("per_rank_bytes counts one payload copy per destination peer; "
                           "global_bytes sums every rank's fan-out")}


def _data_plane(profile, ctx):
    """Resolve the data plane that must serve this case's execution resource.

    An ``AIV``/``CCU`` case is only runnable with the compiled ASC-COMM kernel.  If
    it is missing the case is blocked; if it is present but the benchmark buffer
    registration has never been executed on hardware, the case is
    ``not_implemented``.  Neither may fall back to the host NBI path, because a
    host-submitted transfer is not an AIV or CCU data plane.
    """
    plane = dataplane.for_resource(profile.execution_resource, ctx.config)
    if plane.kind == "host":
        return plane
    blocker = plane.blocker()
    if blocker is not None:
        raise BenchmarkUnavailable(*blocker)
    plane.initialize(ctx)
    raise BenchmarkUnavailable(
        "not_implemented",
        f"the {plane.engine} ASC-COMM kernel is built but this benchmark's buffer "
        f"registration for the device ABI has never run on hardware; see "
        f"kernels/asc_comm/README.md for the verification plan")


def prepare(ctx, case, size, params):
    profile = profiles.profile(case.id)
    if ctx.device.type == "cpu":
        raise BenchmarkUnavailable("unsupported", "native transport baselines require Ascend devices")
    if ctx.size < 2:
        raise BenchmarkUnavailable("unsupported", "native transport baseline needs at least two ranks")

    plane = _data_plane(profile, ctx)

    descriptor = bridge.describe(ctx.config)  # raises dependency_missing when not built
    compiled = {item["channel"] for item in descriptor.get("channels", ()) if item.get("compiled")}
    if profile.channel_hint and compiled and profile.channel_hint not in compiled:
        raise BenchmarkUnavailable(
            "unsupported",
            f"bridge was not compiled with {profile.channel_hint}")

    session = bridge.Session(ctx.config, ctx.size, ctx.rank, ctx.local_rank)
    stack = [session.close]
    try:
        blob = bridge.root_info(ctx.config)
        gathered = ctx.gather(base64.b64encode(blob).decode("ascii"))
        # Every rank initialises from the root rank's blob, not from its own.
        root_info = base64.b64decode(gathered[0])
        initialized = session.comm_init(root_info)
        if not initialized.get("ok"):
            raise BenchmarkUnavailable("environment_unavailable",
                                       f"HCCL communicator initialisation failed: {initialized}")
        graph = session.rank_graph()
        links = _link_protocols(graph, ctx.rank)
        peer_ranks = _peers(ctx, params)

        torch = ctx.torch
        pattern = torch.full((size,), ctx.rank % PATTERN_MODULUS + 1, dtype=torch.uint8,
                             device=ctx.device)
        stride = max(size, 1)
        # Destination matrix has one slot per (source rank): validating a slot
        # proves which peer's payload landed where.
        destination = torch.zeros((ctx.size * stride,), dtype=torch.uint8, device=ctx.device)
        registered = session.register(pattern.data_ptr(), pattern.numel(), "src")
        if not registered.get("ok"):
            raise BenchmarkUnavailable("environment_unavailable",
                                       f"memory registration failed: {registered}")
        target = session.register(destination.data_ptr(), destination.numel(), "dst")
        if not target.get("ok"):
            raise BenchmarkUnavailable("environment_unavailable",
                                       f"memory registration failed: {target}")
        acquired = session.channels(profile.engine, profile.protocol, peer_ranks)
        if not acquired.get("ok"):
            raise BenchmarkUnavailable(
                "unsupported",
                f"HcclChannelAcquire failed for {profile.engine}/{profile.protocol}: {acquired}")
        session.thread_alloc(profile.engine)
        observed_protocols = {entry["rank"]: entry.get("link_protocol_from_graph")
                              for entry in acquired.get("peers", ())}

        outstanding = max(1, int(params.get("outstanding") or 1))
        chunks = max(1, min(outstanding, max(1, size // CHUNK_BYTES)))
        per_chunk = max(1, size // chunks)
        peers = [(index, entry["rank"]) for index, entry in enumerate(acquired["peers"])]
        # Each peer exposes its registered buffers by tag; address the peer's
        # destination buffer, not whichever buffer happened to be listed first.
        remote_index = {}
        for entry in acquired["peers"]:
            memories = entry.get("remote_mems") or []
            remote_index[entry["rank"]] = next(
                (position for position, memory in enumerate(memories)
                 if memory.get("tag") == "dst"),
                1 if len(memories) > 1 else 0)
    except BaseException:
        for closing in reversed(stack):
            closing()
        raise

    evidence = _evidence(profile, links, observed_protocols, acquired)
    evidence.append(Evidence(
        source="native_descriptor",
        claim={"execution_resource": profile.execution_resource},
        detail=(f"data plane {plane.kind} serves {profile.execution_resource}; "
                f"completion is {plane.completion.name} ({plane.completion.level})"),
    ))
    # The completion contract is recorded with the result, so a fenced host drain is
    # never presented as a completed device transfer.
    host_completion = plane.completion

    # The verification ladder is derived from evidence only.  `configured` is where a
    # case always starts; an acquired channel and a rank-graph protocol are what can
    # move it, and a profiler record is required before `kernel_verified`.
    protocols = {value for value in observed_protocols.values() if value}
    collector = evidence_collector.EvidenceCollector(
        requested=request_for(case, params),
        endpoint_created=bool(initialized.get("ok")),
        channel_kind=next(iter([entry.get("channel") for entry in acquired.get("peers", ())
                                if entry.get("channel")]), None),
        engine=profile.execution_resource,
        protocol=next(iter(protocols)) if len(protocols) == 1 else None,
    )
    levels = collector.levels()

    def issue():
        for peer_index, peer_rank in peers:
            remote_offset = peer_rank * stride
            for chunk in range(chunks):
                offset = chunk * per_chunk
                length = min(per_chunk, size - offset)
                if length <= 0:
                    break
                answer = session.transfer(peer_index, params.get("mode", "write"),
                                          0, remote_index[peer_rank],
                                          offset, remote_offset + offset, length)
                if not answer.get("ok"):
                    raise RuntimeError(f"transfer to rank {peer_rank} failed: {answer}")

    def run():
        issue()
        return None

    def wait(_):
        quiet = session.quiet()
        if not quiet.get("ok"):
            raise RuntimeError(f"quiet failed: {quiet}")

    def validate():
        torch.npu.synchronize(ctx.device)
        ctx.barrier()
        # Each peer wrote its own pattern into destination[peer * stride].
        received = destination.view(ctx.size, stride)[:, :size].cpu()
        for peer_rank in range(ctx.size):
            if peer_rank == ctx.rank:
                continue
            expected = peer_rank % PATTERN_MODULUS + 1
            slot = received[peer_rank]
            if not bool((slot == expected).all()):
                raise CorrectnessError(
                    f"rank {ctx.rank} received corrupt bytes from rank {peer_rank}")
        return {"checked": "bytewise", "peers": len(peers),
                "submission": _submission(acquired)}

    def cleanup():
        for closing in reversed(stack):
            closing()

    return Prepared(run=run, wait=wait, validate=validate, cleanup=cleanup,
                    request=request_for(case, params), evidence=evidence,
                    payload=payload(ctx, size, peers),
                    completion=host_completion.level,
                    completion_detail=host_completion.detail,
                    levels=levels)


def _peers(ctx, params):
    """Resolve the destination peers for this rank.

    The policy is resolved by :mod:`baselines.l0.transport.peers`: ``ring`` uses a
    rank rotation, ``same_server``/``cross_server`` restrict the candidate set to
    the configured rank-to-server map, and a policy the topology cannot satisfy
    raises ``BenchmarkUnavailable`` instead of falling back to plain peers.
    """
    return peers.select(
        ctx.size, ctx.rank,
        policy=params.get("peer_policy") or "ring",
        peer_count=params.get("peer_count") or 0,
        domains=peers.server_ids(ctx.config, ctx.size),
        rank_offset=params.get("rank_offset"),
    )


def _link_protocols(graph, rank):
    result = {}
    for layer in graph.get("layers", ()):
        for link in layer.get("links", ()):
            if link.get("peer") == rank:
                continue
            result.setdefault(link["peer"], link.get("link_protocol"))
    return result


def _submission(acquired):
    peers = acquired.get("peers", ())
    return "descriptor_accepted" if peers else "none"


def _evidence(profile, links, observed_protocols, acquired):
    """Build strong evidence items from the runtime answers.

    The channel descriptor proves which engine and protocol were requested and
    accepted.  The rank graph proves which protocol the runtime advertises for
    the link to each peer.  The URMA cases need both, because URMA names a
    channel family rather than a ``CommProtocol`` value; the protocol cases need
    the rank-graph protocol to match.
    """
    evidence = []
    graph_protocols = {value for value in observed_protocols.values() if value}
    for peer, protocol in sorted(observed_protocols.items()):
        if not protocol or protocol == "RESERVED":
            continue
        evidence.append(Evidence(
            source="rank_graph",
            claim={"transport_protocol": _mechanism_for_protocol(protocol)},
            detail=f"rank graph reports {protocol} towards rank {peer}",
        ))
    channel_names = [entry.get("channel") for entry in acquired.get("peers", ()) if entry.get("channel")]
    if channel_names:
        claim = {"execution_resource": profile.execution_resource}
        if profile.mechanism == "urma":
            # Only claim URMA when the runtime also advertises the UB_CTP
            # protocol the URMA-family channels are created with.
            if profile.protocol in graph_protocols:
                claim["transport_protocol"] = "urma"
        else:
            claim["transport_protocol"] = profile.mechanism
        evidence.append(Evidence(
            source="native_descriptor", claim=claim,
            detail=(f"HcclChannelAcquire accepted engine={profile.engine} "
                    f"protocol={profile.protocol} for {len(channel_names)} peer channel(s)"),
        ))
    return evidence


_PROTOCOL_MECHANISM = {
    "UB_MEM": "ub_memory", "UB_CTP": "ub_ctp", "UBC_TP": "ub_ctp", "UB_RTP": "ub_rtp",
    "ROCE": "roce", "HCCS": "hccs", "HCCS_ONLY": "hccs", "UBOE": "uboe",
}


def _mechanism_for_protocol(protocol):
    return _PROTOCOL_MECHANISM.get(protocol)


def request_for(case, params):
    profile = profiles.profile(case.id)
    return requested(backend="hcomm", operation=case.operation,
                     execution_resource=profile.execution_resource,
                     transport_protocol=profile.mechanism,
                     accelerator=params.get("accelerator"),
                     channel=params.get("channel") or profile.channel_hint,
                     topology_mode=params.get("topology_mode"))
