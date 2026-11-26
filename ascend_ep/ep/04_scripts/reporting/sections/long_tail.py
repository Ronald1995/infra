"""Section: long-tail cost of a routing target or an injected slow rank."""
from reporting.empty import unavailable
from reporting.formatting import table

TITLE = "Routing Imbalance, Stragglers and Combined Long Tail"


def render(ctx):
    return [f"## 9. {TITLE}", "", *body(ctx)]


def body(ctx):
    result = (ctx.analysis or {}).get("long_tail") or {}
    pairs = result.get("pairs") or []
    missing = result.get("not_applicable") or []
    if not pairs and not missing:
        return unavailable(
            f"{TITLE}: unavailable",
            findings="This run produced no routing-target or straggler point, so there is "
                     "no long-tail measurement to analyse.",
            causes=["the plan did not include the `routing_imbalance`, `straggler` or "
                    "`long_tail` suite, or those points did not run"],
            required=["plan at least one of those suites and let its points execute"],
            conclusion="no slowdown, retention or amplification number can be claimed for "
                       "this run.")
    lines = []
    if pairs:
        rows = [[item.get("case_id"), item.get("operator"), item.get("ep_size"),
                 (item.get("stress") or {}).get("kind"),
                 _number(item.get("baseline_ms")), _number(item.get("stressed_ms")),
                 _ratio(item.get("slowdown_ratio")),
                 _ratio(item.get("throughput_retention")),
                 _ratio(item.get("amplification")),
                 _calibration(item.get("calibration"))]
                for item in pairs]
        lines += ["### Measured cost per stressed point", "",
                  table(["Case", "Operator", "EP", "Stress", "Baseline ms", "Stressed ms",
                         "Slowdown", "Retention", "Amplification", "Calibration"], rows),
                  "",
                  "Amplification is extra group latency divided by the injected delay: below 1 "
                  "means overlap hid part of it, above 1 means the slow rank dragged the group.",
                  ""]
        summary = result.get("summary") or {}
        lines += ["### Distribution", "",
                  f"- pairs: {summary.get('pairs')}; without a baseline: "
                  f"{summary.get('missing_baseline')}",
                  f"- slowdown p50 / p95 / max: {_ratio(summary.get('slowdown_p50'))} / "
                  f"{_ratio(summary.get('slowdown_p95'))} / "
                  f"{_ratio(summary.get('slowdown_max'))}", ""]
    if missing:
        lines += ["### Points that cannot be compared", "",
                  table(["Case", "Operator", "EP", "Stress", "Reason"],
                        [[item.get("case_id"), item.get("operator"), item.get("ep_size"),
                          (item.get("stress") or {}).get("kind"), item.get("reason")]
                         for item in missing[:20]]),
                  "", f"{len(missing)} point(s) reported `not_applicable`; no ratio was "
                  "invented for them.", ""]
    return lines


def _number(value):
    return "n/a" if value is None else f"{value:.3f}"


def _ratio(value):
    return "n/a" if value is None else f"{value:.3f}"


def _calibration(record):
    if not record:
        return "n/a"
    error = record.get("error")
    return "n/a" if error is None else f"{error:.1%}"
