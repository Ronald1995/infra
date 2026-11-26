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
from itertools import islice

from baselines.l0.contract import Prepared
from baselines.l0.transport import bridge
from baselines.l0.transport.common import profiles
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


def prepare(ctx, case, size, params):
    profile = profiles.profile(case.id)
    if ctx.device.type == "cpu":
        raise BenchmarkUnavailable("unsupported", "native transport baselines require Ascend devices")
    if ctx.size < 2:
        raise BenchmarkUnavailable("unsupported", "native transport baseline needs at least two ranks")

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
        root_info = base64.b64decode(gathered[ctx.rank])
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
    except BaseException:
        for closing in reversed(stack):
            closing()
        raise

    evidence = _evidence(profile, links, observed_protocols, acquired)

    def issue():
        for peer_index, peer_rank in peers:
            remote_offset = peer_rank * stride
            for chunk in range(chunks):
                offset = chunk * per_chunk
                length = min(per_chunk, size - offset)
                if length <= 0:
                    break
                answer = session.transfer(peer_index, params.get("mode", "write"),
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
                    payload=payload(ctx, size, peers))


def _peers(ctx, params):
    limit = int(params.get("peer_count") or 0)
    ranks = [rank for rank in range(ctx.size) if rank != ctx.rank]
    if limit > 0:
        ranks = list(islice(ranks, limit))
    if not ranks:
        raise BenchmarkUnavailable("unsupported", "no destination peer is available")
    return ranks


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
