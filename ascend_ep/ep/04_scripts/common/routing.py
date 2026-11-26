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
    elif strategy == "target_distribution":
        ids = target_routes(config, rank, step)
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
    control = config["workload"].get("routing_control") or {}
    if config["workload"].get("routing") == "target_distribution" and control:
        metric = control["target_metric"]
        target = float(control["target_value"])
        tolerance = float((control.get("tolerance") or {}).get("relative", 0.02))
        actual = summary.get(metric)
        plan = target_plan(config)
        met = actual is not None and abs(actual - target) <= tolerance * abs(target)
        summary["target_control"] = {
            "metric": metric, "target": target, "actual": actual,
            "tolerance_relative": tolerance,
            "feasible_upper_bound": plan.get("bound"),
            "status": plan.get("status"),
            "met": met,
            "reason": (plan.get("reason") if plan.get("status") != "generated"
                       else None if met else
                       f"generated {actual:.4f} against a target of {target:.4f} "
                       f"(tolerance {tolerance:.0%})"),
            "on_target_miss": control.get("on_target_miss", "failed"),
        }
    return summary


# ---------------------------------------------------------------------------
# Target-distribution routing
# ---------------------------------------------------------------------------
#: Metrics this module can *generate* to.  Everything else is a measurement.
TARGET_METRICS = ("rank_max_mean", "expert_max_mean", "local_route_ratio")
#: Metrics that stay statistical: they are reported, never generated.  Saying so
#: is the difference between "we did not test it" and "we tested something else".
STATISTICAL_METRICS = ("server_max_mean", "rank_tx_max_mean", "rank_rx_max_mean",
                       "cross_server_ratio")


def owners_by_rank(placement, ep):
    """``rank -> [expert ids it owns]`` from the placement tuple."""
    owners = [[] for _ in range(ep)]
    for expert, rank in enumerate(placement):
        owners[rank].append(expert)
    return owners


def feasible_bounds(config):
    """The largest value each target metric can reach, and why.

    The Top-K uniqueness rule is the binding constraint:

    * one Expert can take at most one route per token, so
      ``expert_max_mean <= E / K``;
    * one Rank can take at most ``min(K, experts_on_rank)`` routes per token, so
      ``rank_max_mean <= P * min(K, experts_on_rank) / K``.

    These are computed from the effective placement, not assumed.
    """
    ep = config["distributed"]["ep_size"]
    k = config["model"]["top_k"]
    experts = config["model"]["num_experts"]
    placement = geometry(config)[0]
    per_rank = owners_by_rank(placement, ep)
    smallest = min(len(items) for items in per_rank) if per_rank else 0
    return {
        "expert_max_mean": experts / k if k else None,
        "rank_max_mean": ep * min(k, smallest) / k if k else None,
        "local_route_ratio": min(k, smallest) / k if k else None,
    }


def _allocate(total, weights, caps):
    """Integer allocation of ``total`` units by ``weights``, never above ``caps``.

    Proportional floor first, then the remainder one unit at a time to the largest
    fractional part that still has capacity.  Returns a list of ints summing to
    ``total`` (``sum(caps) >= total`` is the caller's feasibility guarantee).
    """
    n = len(weights)
    weights = [max(0.0, float(w)) for w in weights]
    caps = [int(c) for c in caps]
    if total > sum(caps):
        raise ValueError("target allocation exceeds the per-token capacity")
    if not sum(weights):
        weights = [1.0] * n
    shares = [total * w / sum(weights) for w in weights]
    out = [min(int(share), caps[i]) for i, share in enumerate(shares)]
    remaining = total - sum(out)
    while remaining > 0:
        best, best_gap = None, None
        for i in range(n):
            if out[i] >= caps[i]:
                continue
            gap = shares[i] - out[i]
            if best is None or gap > best_gap + 1e-12 or (abs(gap - best_gap) <= 1e-12
                                                          and i < best):
                best, best_gap = i, gap
        if best is None:  # pragma: no cover - guarded by the capacity check
            raise ValueError("target allocation has no capacity left")
        out[best] += 1
        remaining -= 1
    return out


def _even_split(total, indices, caps):
    """Split ``total`` over ``indices`` evenly, respecting ``caps``."""
    weights = [1.0] * len(indices)
    shares = _allocate(total, weights, [caps[i] for i in indices]) if total else [0] * len(indices)
    result = {index: share for index, share in zip(indices, shares)}
    return result


def target_plan(config, step=0):
    """The deterministic quota plan that every rank computes identically.

    ``rows[r][d]`` is how many of source rank ``r``'s routes go to destination
    rank ``d``.  The plan never depends on what another rank *chooses*: each rank
    derives the whole table from the seed, the token counts, the placement and the
    target, then keeps its own row.  That is what makes the same trace usable by
    every operator without a collective.
    """
    work, model = config["workload"], config["model"]
    ep, k = config["distributed"]["ep_size"], model["top_k"]
    experts = model["num_experts"]
    placement, tokens = geometry(config, step)
    control = work.get("routing_control") or {}
    metric = control.get("target_metric", "rank_max_mean")
    if metric not in TARGET_METRICS:
        raise ValueError(f"routing_control.target_metric must be one of {TARGET_METRICS}; "
                         f"{metric!r} is statistical only ({STATISTICAL_METRICS})")
    total = sum(tokens) * k
    mean = total / ep
    bounds = feasible_bounds(config)
    hotspot = control.get("hotspot") or {}
    hot_ranks = list(hotspot.get("ranks") or [0])
    if any(not 0 <= rank < ep for rank in hot_ranks):
        raise ValueError("routing_control.hotspot.ranks outside the EP group")
    tolerance = float((control.get("tolerance") or {}).get("relative", 0.02))
    requested = float(control.get("target_value", 1.0))
    per_rank = owners_by_rank(placement, ep)
    caps_per_token = [min(k, len(items)) for items in per_rank]

    status, reason = "generated", None
    uniform_expert = metric == "expert_max_mean" and requested <= 1.0 + 1e-9
    if metric in ("rank_max_mean", "expert_max_mean", "local_route_ratio"):
        bound = bounds[metric]
        if requested > bound + 1e-9:
            status = "not_applicable"
            reason = ("target is unreachable: the Top-K uniqueness rule caps "
                      f"{metric} at {bound:.3f} (requested {requested:.3f})")
            return {"metric": metric, "requested": requested, "bound": bound,
                    "status": status, "reason": reason, "rows": [], "tolerance": tolerance}

    weights = [1.0] * ep
    expert_target = None
    if uniform_expert:
        weights = [1.0] * ep
        expert_target = None
    elif metric == "rank_max_mean":
        hot_total = requested * mean
        hot_total = min(max(hot_total, mean), bounds["rank_max_mean"] * mean)
        weights = [0.0] * ep
        if len(hot_ranks) == 1 and ep > 1:
            weights[hot_ranks[0]] = hot_total
            others = [d for d in range(ep) if d != hot_ranks[0]]
            for d in others:
                weights[d] = (total - hot_total) / len(others)
        else:
            weights = [1.0] * ep
            for d in hot_ranks:
                weights[d] = hot_total / max(1, len(hot_ranks))
            rest = [d for d in range(ep) if d not in hot_ranks]
            if rest:
                remaining = total - hot_total / max(1, len(hot_ranks)) * len(hot_ranks)
                for d in rest:
                    weights[d] = remaining / len(rest)
    elif metric == "expert_max_mean":
        # The hot expert can take one route per token, so its quota is bounded by
        # the number of source tokens; the owner rank's quota must cover it.
        hot_expert = int(hotspot.get("expert", 0))
        if not 0 <= hot_expert < experts:
            raise ValueError("routing_control.hotspot.expert outside the placement")
        hot_owner = placement[hot_expert]
        hot_total = min(requested * total / experts, sum(tokens))
        expert_target = {"expert": hot_expert, "owner": hot_owner, "routes": hot_total}
        weights = [1.0] * ep
        weights[hot_owner] = max(hot_total, mean)
        rest = [d for d in range(ep) if d != hot_owner]
        remaining = max(0.0, total - weights[hot_owner])
        for d in rest:
            weights[d] = remaining / len(rest) if rest else 0.0

    rows = []
    local_ratio = None
    if metric == "local_route_ratio":
        local_ratio = min(max(requested, 0.0), 1.0)
    for rank, count in enumerate(tokens):
        routes = count * k
        caps = [caps_per_token[d] * count for d in range(ep)]
        if expert_target is not None:
            # The owner rank already has to absorb the hot expert's routes.
            caps[expert_target["owner"]] = max(caps[expert_target["owner"]], routes)
        if local_ratio is not None:
            local = round(local_ratio * routes)
            row = _even_split(routes - local, [d for d in range(ep) if d != rank], caps)
            row[rank] = local
            quotas = [row[d] for d in range(ep)]
        else:
            quotas = _allocate(routes, weights, caps)
        if expert_target is not None:
            owner = expert_target["owner"]
            quotas[owner] = max(quotas[owner], min(
                int(round(expert_target["routes"] * count / max(1, sum(tokens)))), routes))
        rows.append(quotas)
    return {"metric": metric, "requested": requested, "bound": bounds[metric],
            "status": status, "reason": reason, "rows": rows, "tolerance": tolerance,
            "expert_target": expert_target, "local_ratio": local_ratio,
            "mean": mean}


def target_routes(config, rank, step=0):
    """Rank-local Top-K ids generated to the target quota plan.

    The row quota is exact: every destination receives exactly the number of routes
    the plan allocated to it, each token takes K *distinct* experts, and every route
    is counted once.  When the plan targets an expert, one slot of each hot token is
    reserved for that expert first -- otherwise a token that sends several routes to
    the owner rank would dilute the expert quota and silently miss the target.
    """
    work, model = config["workload"], config["model"]
    placement, tokens = geometry(config, step)
    ep, k, experts = len(tokens), model["top_k"], model["num_experts"]
    plan = target_plan(config, step)
    if plan["status"] != "generated":
        from common.backend import BenchmarkUnavailable

        raise BenchmarkUnavailable(plan["status"], plan["reason"])
    quota = list(plan["rows"][rank])
    count = tokens[rank]
    owners = owners_by_rank(placement, ep)
    caps = [min(k, len(items)) for items in owners]
    rng = np.random.default_rng([work["seed"], rank, step, 17])
    remaining = list(quota)
    ids = np.empty((count, k), dtype=np.int64)
    used_per_expert = np.zeros(experts, dtype=np.int64)
    offset = sum(tokens[:rank])
    expert_target = plan.get("expert_target")
    hot_quota = int(round(expert_target["routes"])) if expert_target else 0
    for token in range(count):
        reserved = None
        if expert_target is not None:
            owner, hot = expert_target["owner"], expert_target["expert"]
            global_token = offset + token
            if (global_token < hot_quota and remaining[owner] > 0
                    and used_per_expert[hot] < hot_quota):
                reserved = hot
                used_per_expert[hot] += 1
                remaining[owner] -= 1
        chosen = []
        # The reserved hot slot counts against the owner's per-token capacity.
        reserved_owner = expert_target["owner"] if reserved is not None else None
        def _slots(destination):
            return chosen.count(destination) + (1 if destination == reserved_owner else 0)
        # Mandatory picks: a destination whose remaining quota cannot fit into the
        # tokens left (at its per-token cap) must be served now.
        tokens_left = count - token - 1
        # Future hot tokens will consume one owner slot each; do not spend them now.
        if expert_target is not None:
            hot_left = max(0, min(hot_quota - (offset + token + 1), tokens_left + 1))
        else:
            hot_left = 0

        def _effective(destination):
            return remaining[destination] - (hot_left if destination == reserved_owner else 0)

        for destination in range(ep):
            need = _effective(destination) - tokens_left * caps[destination]
            take = min(max(0, need), caps[destination] - _slots(destination))
            for _ in range(take):
                chosen.append(destination)
                remaining[destination] -= 1
        slots = k - (1 if reserved is not None else 0)
        while len(chosen) < slots:
            best, best_value = None, None
            for destination in range(ep):
                if _effective(destination) <= 0:
                    continue
                if _slots(destination) >= caps[destination]:
                    continue
                value = _effective(destination) / max(1, caps[destination])
                if best is None or value > best_value + 1e-12:
                    best, best_value = destination, value
            if best is None:
                options = [d for d in range(ep) if _slots(d) < caps[d]]
                if not options:
                    raise ValueError("the Top-K uniqueness rule leaves no destination")
                best = min(options, key=lambda d: (remaining[d], d))
            chosen.append(best)
            remaining[best] -= 1
        token_experts = [reserved] if reserved is not None else []
        for destination in chosen:
            candidates = [expert for expert in owners[destination]
                          if expert not in token_experts]
            if not candidates:
                raise ValueError("a destination rank has no unused expert for this token")
            pick = min(candidates, key=lambda expert: (used_per_expert[expert], expert))
            token_experts.append(pick)
            used_per_expert[pick] += 1
        ids[token] = rng.permuted(np.asarray(token_experts, dtype=np.int64))
    if any(remaining):
        raise ValueError(f"row quotas were not satisfied: {remaining}")
    return ids
