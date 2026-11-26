"""Long-tail analysis: what a load imbalance or a slow rank costs, measured.

The analysis pairs each "stressed" row (generated routing target, injected straggler,
or both) with a *baseline* row measured under the same effective configuration except
for the stress itself, and reports:

* ``slowdown_ratio``      stressed latency / baseline latency (1.0 = no cost);
* ``throughput_retention`` baseline / stressed (1.0 = no loss, i.e. 1/slowdown);
* ``amplification``       extra group latency / injected delay.  A value below 1 means
  overlap hid part of the injection; above 1 means the slow rank dragged the group;
* ``calibration_error``   measured vs requested injection delay, as a fraction.

Pairing uses a *condition key* that deliberately excludes the stress block: the same
case, operator, EP size, tokens, routing target and model must be present in both
rows, otherwise the point is reported as ``not_applicable`` with the reason. No
baseline means no ratio -- a made-up denominator is worse than a missing number.
"""
import json

#: Keys that carry a median latency in a result record, in order of preference.
LATENCY_KEYS = ("p50_ms", "p50", "median_ms", "latency_ms")


def latency_ms(row):
    """The median latency of one result row in ms, or ``None``."""
    for container in (row.get("latency"), row.get("metrics")):
        if isinstance(container, dict):
            for key in LATENCY_KEYS:
                value = container.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        elif isinstance(container, (int, float)):
            return float(container)
    return None


def _block(row):
    return ((row.get("config") or {}).get("benchmark") or {}).get("straggler") or {}


def _enabled(row):
    return bool(_block(row).get("enabled"))


def _routing_control(row):
    workload = (row.get("config") or {}).get("workload") or {}
    return workload.get("routing_control") or {}


def condition_key(row):
    """Everything that must match between a baseline and a stressed measurement."""
    config = row.get("config") or {}
    workload = config.get("workload") or {}
    model = config.get("model") or {}
    return json.dumps({
        "case": row.get("case_id"), "level": row.get("benchmark_level") or row.get("level"),
        "operator": row.get("backend_id") or row.get("operator"), "ep": row.get("ep_size"),
        "tokens": workload.get("tokens_per_rank"), "routing": workload.get("routing"),
        "routing_control": _routing_control(row), "model": model,
        "model_num_experts": model.get("num_experts"), "top_k": model.get("top_k"),
        "dtype": model.get("dtype"),
        "dedup": (config.get("dedup") or {}).get("mode"),
        "load_balance": (config.get("load_balance") or {}).get("enabled"),
        "case_parameters": row.get("case_parameters") or row.get("parameters"),
    }, sort_keys=True, default=str)


def balanced_key(row):
    """The condition without the routing fields: a routing target's baseline."""
    config = row.get("config") or {}
    workload = config.get("workload") or {}
    model = config.get("model") or {}
    return json.dumps({
        "case": row.get("case_id"), "level": row.get("benchmark_level") or row.get("level"),
        "operator": row.get("backend_id") or row.get("operator"), "ep": row.get("ep_size"),
        "tokens": workload.get("tokens_per_rank"), "model": model,
        "case_parameters": row.get("case_parameters") or row.get("parameters"),
    }, sort_keys=True, default=str)


def _baseline(row, stress, by_full, by_balanced):
    if stress["kind"] == "straggler":
        candidates = by_full.get(condition_key(row), [])
        return next((item for item in candidates
                     if not _enabled(item) and latency_ms(item) is not None), None)
    candidates = by_balanced.get(balanced_key(row), [])
    workload = (row.get("config") or {}).get("workload") or {}
    return next((item for item in candidates
                 if (item.get("config") or {}).get("workload", {}).get("routing") == "balanced"
                 and not _routing_control(item) and not _enabled(item)
                 and latency_ms(item) is not None and
                 (row.get("config") or {}).get("workload", {}).get("routing") != "balanced"),
                None)


def _stress(row):
    block = _block(row)
    if _enabled(row):
        injection = block.get("injection") or {}
        return {"kind": "straggler", "method": injection.get("method"),
                "stage": injection.get("stage"),
                "requested_delay_us": injection.get("delay_us")}
    control = _routing_control(row)
    if control:
        return {"kind": "routing", "metric": control.get("target_metric"),
                "target": control.get("target_value")}
    return None


def _calibration(row):
    record = row.get("straggler") or {}
    ranks = record.get("ranks") or []
    requested = [item.get("requested_delay_us") for item in ranks
                 if item.get("requested_delay_us")]
    actual = sum(item.get("actual_delay_us_sum") or 0.0 for item in ranks)
    injections = sum(item.get("injections") or 0 for item in ranks)
    if not requested or not injections:
        return None
    requested_total = sum(requested)
    return {"requested_us": requested_total,
            "actual_us": actual / max(1, injections),
            "error": abs(actual / max(1, injections) - requested_total)
                     / requested_total if requested_total else None,
            "failures": record.get("calibration_failures") or 0}


def analyze(rows):
    """Pair stressed rows with baselines and compute the long-tail metrics."""
    by_full, by_balanced = {}, {}
    for row in rows:
        by_full.setdefault(condition_key(row), []).append(row)
        by_balanced.setdefault(balanced_key(row), []).append(row)
    pairs, missing = [], []
    for _key, group in sorted(by_full.items()):
        for row in group:
            stress = _stress(row)
            if stress is None:
                continue
            baseline = _baseline(row, stress, by_full, by_balanced)
            base_latency = latency_ms(baseline) if baseline is not None else None
            stressed = latency_ms(row)
            if baseline is None or base_latency is None or stressed is None:
                missing.append({
                    "task_id": row.get("task_id"), "case_id": row.get("case_id"),
                    "operator": row.get("backend_id") or row.get("operator"),
                    "ep_size": row.get("ep_size"), "stress": stress,
                    "status": "not_applicable",
                    "reason": ("no baseline measurement under the same configuration"
                               if baseline is None else "missing latency on one side"),
                })
                continue
            extra_ms = stressed - base_latency
            injection = stress.get("requested_delay_us") if stress["kind"] == "straggler" else None
            pairs.append({
                "task_id": row.get("task_id"), "case_id": row.get("case_id"),
                "level": row.get("benchmark_level") or row.get("level"),
                "operator": row.get("backend_id") or row.get("operator"),
                "ep_size": row.get("ep_size"), "stress": stress,
                "baseline_task": baseline.get("task_id"),
                "baseline_ms": base_latency, "stressed_ms": stressed,
                "slowdown_ratio": stressed / base_latency if base_latency else None,
                "throughput_retention": base_latency / stressed if stressed else None,
                "amplification": (extra_ms / (injection / 1000.0)
                                  if injection else None),
                "calibration": _calibration(row),
            })
    ratios = [item["slowdown_ratio"] for item in pairs if item["slowdown_ratio"]]
    return {
        "schema_version": 1,
        "pairs": pairs,
        "not_applicable": missing,
        "summary": {
            "pairs": len(pairs), "missing_baseline": len(missing),
            "slowdown_p50": _percentile(ratios, .5),
            "slowdown_p95": _percentile(ratios, .95),
            "slowdown_max": max(ratios) if ratios else None,
        },
    }


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]
