"""Deterministic rank-local routes and communication statistics."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from common.storage import fingerprint, load_json


@dataclass
class Routing:
    ids: np.ndarray
    weights: np.ndarray
    placement: tuple[int, ...]
    tokens: tuple[int, ...]
    seed: int


def geometry(config, step=0):
    work, model = config["workload"], config["model"]
    size = config["distributed"]["ep_size"]
    owner = tuple(work.get("expert_placement") or np.repeat(
        np.arange(size), model["num_experts"] // size
    ).tolist())
    source = work.get("token_steps", [work["tokens_per_rank"]])
    tokens = source[step % len(source)]
    tokens = (tokens,) * size if isinstance(tokens, int) else tuple(tokens)
    if len(tokens) != size or any(type(n) is not int or n < 0 for n in tokens) or not sum(tokens):
        raise ValueError("every token step must contain valid counts for all ranks")
    return owner, tokens


def generate(config, rank, step=0):
    work, model = config["workload"], config["model"]
    placement, tokens = geometry(config, step)
    ep, k, experts = len(tokens), model["top_k"], model["num_experts"]
    count = tokens[rank]
    rng = np.random.default_rng([work["seed"], rank, step])
    strategy = work["routing"]
    weights = None
    if strategy == "real_trace_replay":
        record = load_json(work["trace_path"])["steps"][step]["ranks"][rank]
        ids = np.asarray(record["topk_ids"], dtype=np.int64).reshape(count, k)
        weights = np.asarray(record["topk_weights"], dtype=np.float32).reshape(count, k)
    elif strategy == "balanced":
        offset = sum(tokens[:rank]) * k
        ids = (np.arange(offset, offset + count * k).reshape(count, k) % experts)
        ids = rng.permuted(ids, axis=1)
    elif strategy == "uniform_random":
        ids = np.empty((count, k), dtype=np.int64)
        for row in ids:
            row[:] = rng.choice(experts, k, replace=False)
    else:
        owners = np.asarray(placement)
        servers = config["topology"].get("server_ids")
        target = work.get("hot_rank", step % ep if strategy == "dynamic_trace" else 0)
        if not 0 <= target < ep:
            raise ValueError("hot_rank outside EP group")
        if strategy in {"server_hotspot", "cross_server_heavy"} and servers is None:
            raise ValueError(f"{strategy} requires explicit topology.server_ids")
        if strategy == "expert_hotspot":
            hot_expert = work.get("hot_expert", 0)
            if not 0 <= hot_expert < experts:
                raise ValueError("hot_expert outside expert placement")
            mask = np.arange(experts) == hot_expert
        elif strategy == "server_hotspot":
            mask = np.asarray(servers)[owners] == servers[target]
        elif strategy == "cross_server_heavy":
            mask = np.asarray(servers)[owners] != servers[rank]
        else:
            if strategy == "high_locality":
                target = rank
            elif strategy == "dedup_controlled":
                target = (rank + 1) % ep
            mask = owners == target

        preferred, other = np.flatnonzero(mask), np.flatnonzero(~mask)
        fraction = work.get("preferred_route_ratio", 0.8)
        if not 0 <= fraction <= 1:
            raise ValueError("preferred_route_ratio must be in [0, 1]")
        requested = round(count * k * fraction)
        minimum, maximum = max(0, k - len(other)), min(k, len(preferred))
        achieved = min(max(requested, count * minimum), count * maximum)
        base, extra = divmod(achieved, count) if count else (0, 0)
        choices = rng.permutation(count) < extra
        ids = np.empty((count, k), dtype=np.int64)
        for i, row in enumerate(ids):
            first = base + int(choices[i])
            row[:] = np.concatenate((rng.choice(preferred, first, replace=False),
                                     rng.choice(other, k - first, replace=False)))
            rng.shuffle(row)
    if weights is None:
        weights = rng.uniform(0.1, 1.0, size=(count, k)).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
    validate_routes(ids, weights, count, k, experts)
    return Routing(ids, weights, placement, tokens, work["seed"])


def validate_routes(ids, weights, tokens, k, experts):
    if ids.shape != (tokens, k) or weights.shape != ids.shape:
        raise ValueError("routing tensors do not match T x K")
    if np.any((ids < 0) | (ids >= experts)) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("Top-K routes must be unique and in expert range")
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(axis=1) <= 0):
        raise ValueError("routing weights must be finite, nonnegative and nonempty")


def summarize_local(route, rank, servers):
    ids = route.ids
    owners = np.asarray(route.placement)[ids]
    counts = np.bincount(ids.ravel(), minlength=len(route.placement))
    rank_counts = np.bincount(owners.ravel(), minlength=len(route.tokens))
    distinct = np.sort(owners, axis=1)
    unique = np.ones_like(distinct, dtype=bool)
    unique[:, 1:] = distinct[:, 1:] != distinct[:, :-1]
    dedup = np.bincount(distinct[unique], minlength=len(route.tokens))
    server_dedup = cross_routes = None
    if servers is not None:
        domains = np.asarray([str(server) for server in servers])
        destination = domains[owners]
        cross_routes = int(np.count_nonzero(destination != domains[rank]))
        distinct_servers = np.sort(destination, axis=1)
        unique_servers = np.ones_like(distinct_servers, dtype=bool)
        unique_servers[:, 1:] = distinct_servers[:, 1:] != distinct_servers[:, :-1]
        server_dedup = int(np.count_nonzero(unique_servers & (distinct_servers != domains[rank])))
    digest = sha256()
    digest.update(ids.astype("<i8", copy=False).tobytes())
    digest.update(route.weights.astype("<f4", copy=False).tobytes())
    return {
        "rank": rank, "digest": digest.hexdigest(), "expert_counts": counts.tolist(),
        "rank_counts": rank_counts.tolist(), "unique_destination_ranks": dedup.tolist(),
        "unique_destination_servers": server_dedup, "cross_server_routes": cross_routes,
    }


def max_mean(values):
    return max(values) * len(values) / sum(values) if sum(values) else None


def summarize(route, records, config):
    records = sorted(records, key=lambda item: item["rank"])
    matrix = np.asarray([r["rank_counts"] for r in records])
    expert_counts = np.sum([r["expert_counts"] for r in records], axis=0).tolist()
    rank_counts = matrix.sum(axis=0).tolist()
    total = sum(route.tokens) * route.ids.shape[1]
    if matrix.sum() != total or sum(expert_counts) != total:
        raise ValueError("aggregated routing counts do not match source tokens x top-k")
    remote = matrix.copy()
    np.fill_diagonal(remote, 0)
    cross = [r["cross_server_routes"] for r in records]
    summary = {
        "trace_hash": fingerprint([route.seed, route.placement, route.tokens, [r["digest"] for r in records]]),
        "seed": route.seed, "expert_counts": expert_counts, "rank_counts": rank_counts,
        "source_destination_matrix": matrix.tolist(),
        "local_route_ratio": float(matrix.trace() / total),
        "cross_server_ratio": sum(cross) / total if None not in cross else None,
        "unique_destination_ranks": [r["unique_destination_ranks"] for r in records],
        "unique_destination_servers": [r["unique_destination_servers"] for r in records],
        "expert_max_mean": max_mean(expert_counts), "rank_max_mean": max_mean(rank_counts),
        "rank_tx_max_mean": max_mean(remote.sum(axis=1).tolist()),
        "rank_rx_max_mean": max_mean(remote.sum(axis=0).tolist()),
        "targets": {},
    }
    for name in ("expert_max_mean", "rank_max_mean", "cross_server_ratio"):
        expected = config["workload"].get(f"target_{name}")
        if expected is not None:
            actual = summary[name]
            met = actual is not None and abs(expected - actual) < 1e-6
            summary["targets"][name] = {
                "target": expected, "actual": actual, "met": met,
                "reason": None if met else "target not reached by the configured strategy, topology and Top-K constraints",
            }
    return summary
