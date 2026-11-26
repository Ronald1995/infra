"""Derive metrics with explicit scale, scope and evidence."""
import math
from statistics import mean, pstdev

from common.config import DTYPES


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stage_latency(rank_records):
    groups = list(zip(*(rank["samples"] for rank in rank_records)))
    if not groups or "stage_ns" not in groups[0][0]:
        return {}
    return {
        name: percentile([max(sample["stage_ns"][name] for sample in group) for group in groups], .5) / 1e6
        for name in groups[0][0]["stage_ns"]
    }


def latency(rank_records):
    ranks = [record["samples"] for record in rank_records]
    sizes = {len(record) for record in ranks}
    if len(sizes) != 1 or not sizes or sizes == {0}:
        raise ValueError("latency requires an equal, nonzero sample count from all ranks")
    groups = list(zip(*ranks))
    maxima = [max(v["latency_ns"] for v in group) for group in groups]
    result = {
        "unit": "ms", "scope": "per_iteration_rank_max",
        "source": "measured", "rank_min": [], "rank_mean": [], "rank_max": [],
    }
    for group in groups:
        values = [v["latency_ns"] / 1e6 for v in group]
        for key, value in zip(("rank_min", "rank_mean", "rank_max"), (min(values), mean(values), max(values))):
            result[key].append(value)
    for p in (50, 95, 99):
        result[f"p{p}"] = percentile(maxima, p / 100) / 1e6
    # Rank Max is the headline number; the rank-average percentile is reported next
    # to it so an imbalance is visible instead of being averaged away.
    result["rank_avg_p50_ms"] = percentile(
        [mean([v["latency_ns"] for v in group]) for group in groups], .5) / 1e6
    result["rank_min_p50_ms"] = percentile(
        [min(v["latency_ns"] for v in group) for group in groups], .5) / 1e6
    result["imbalance_ratio"] = (result["rank_max"][0] / mean(result["rank_max"])
                                 if mean(result["rank_max"]) else None)
    result["stddev"] = pstdev(maxima) / 1e6
    result["cv"] = result["stddev"] / mean(result["rank_max"]) if mean(maxima) else None
    result["p95_p50_ratio"] = result["p95"] / result["p50"] if result["p50"] else None
    result["host_issue_p50_ms"] = percentile([max(x["host_issue_ns"] for x in g) for g in groups], .5) / 1e6
    result["host_wait_p50_ms"] = percentile([max(x["host_wait_ns"] for x in g) for g in groups], .5) / 1e6
    result["output_ready_p50_ms"] = percentile([max(x["output_ready_ns"] for x in g) for g in groups], .5) / 1e6
    return result


def traffic(trace, config, tokens, traffic_model, directions=1):
    model = config["model"]
    width = model["hidden_size"] * DTYPES[model["dtype"]]
    matrix = trace["source_destination_matrix"]
    n = len(matrix)
    expanded = sum(sum(row) - row[i] for i, row in enumerate(matrix))
    rank_dedup = sum(sum(row) - row[i] for i, row in enumerate(trace["unique_destination_ranks"]))
    server_counts = trace["unique_destination_servers"]
    mode = config["dedup"]["mode"]
    if None in server_counts:
        cross = None
    elif mode == "server":
        cross = sum(server_counts)
    elif mode == "rank":
        domains = config["topology"]["server_ids"]
        cross = sum(value for i, row in enumerate(trace["unique_destination_ranks"])
                    for j, value in enumerate(row) if domains[i] != domains[j])
    else:
        cross = round(trace["cross_server_ratio"] * sum(map(sum, matrix)))
    data_model = traffic_model
    payload = None
    padding = None
    metadata = None
    if traffic_model == "collective_alltoallv":
        payload = expanded
        # Two int64 route fields per remote row plus a rank-split scalar per remote peer.
        # Dispatch and combine have distinct metadata requirements.
    elif traffic_model == "collective_allgather":
        # Logical collective peer copies. Collective schedules and wire traffic are unobserved.
        data_model = "collective_allgather_logical_peer_copies"
        payload = max(tokens) * n * (n - 1)
        padding = (max(tokens) * n - sum(tokens)) * (n - 1) * width * directions
    elif traffic_model == "rank_dedup":
        payload = rank_dedup

    return {
        "unit": "bytes", "scope": "global_ep_group", "source": "routing_trace_model",
        "traffic_model": data_model, "logical_bytes": sum(tokens) * model["top_k"] * width * directions,
        "remote_expanded_bytes": expanded * width * directions,
        "rank_dedup_payload_bytes": rank_dedup * width,
        "server_dedup_cross_server_payload_bytes": sum(server_counts) * width if None not in server_counts else None,
        "cross_server_payload_bytes": None if cross is None else cross * width,
        "payload_bytes": None if payload is None else payload * width * directions,
        "metadata_bytes": metadata, "padding_bytes": padding,
        "dedup_opportunity_ratio": 1 - rank_dedup / expanded if expanded else None,
    }


def bandwidth(values, milliseconds):
    out = dict(values, bandwidth_unit="GB/s")
    for target, source in (("logical_bandwidth_GBps", "logical_bytes"),
                           ("payload_bandwidth_GBps", "payload_bytes")):
        value = values.get(source)
        out[target] = value / (milliseconds * 1e6) if value is not None and milliseconds else None
    return dict(out, physical_bandwidth_GBps=None, hardware_bandwidth_utilization=None,
                baseline_bandwidth_efficiency=None,
                hardware_unavailable_reason="hardware link counters unavailable")


def throughput(tokens, ep, milliseconds, layers=1):
    rate = tokens * layers * 1000 / milliseconds if milliseconds else None
    return {"source": "routing_trace_and_measured_rank_max", "unit": "tokens/s", "scope": "EP group",
            "global_source_tokens_per_second": rate,
            "tokens_per_second_per_rank": rate / ep if rate is not None else None}


def compute(config, routes, op, milliseconds):
    m = config["model"]
    single = 2 * routes * m["hidden_size"] * m["intermediate_size"]
    forward, backward = 3 * single, 6 * single
    if op in {"backward", "fused_backward"}:
        effective = backward
    elif op in {"forward_backward", "recompute", "router_training", "router_large_training",
                "planner", "redundant_experts", "planner_training_steps", "rank_dedup_layer",
                "server_dedup_layer"}:
        effective = forward + backward
    elif op in {"serial_dispatch_gmm", "dispatch_gmm", "quantized_fusion", "tile_schedule", "comm_resources"}:
        effective = 2 * single
    elif op in {"serial_gmm_combine", "gmm_combine"}:
        effective = single
    elif op in {"multilayer", "concurrent"}:
        effective = 2 * forward
    else:
        effective = forward
    return {"source": "effective_expert_matrix_math", "scope": "EP group",
            "forward_flops": forward, "backward_flops": backward, "effective_flops": effective,
            "effective_tflops": effective / (milliseconds * 1e9) if milliseconds else None,
            "executed_flops": None, "executed_tflops": None, "padding_ratio": None,
            "gmm_baseline_efficiency": None, "recompute_overhead": None}
