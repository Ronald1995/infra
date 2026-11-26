"""Coverage matrices and cross-EP-size performance comparison.

Four dimensions are reported: ``EP Size x Case x Operator x Parameter Variant``.
Three rules keep the report honest:

* ``unsupported`` / ``dependency_missing`` / ``not_applicable`` never count as
  ``passed``;
* a case that produced no task at all is shown as ``missing`` instead of being
  quietly omitted;
* two measurements are only put in the same performance group when their
  parameter variant, model shape, per-rank workload and timing boundary match,
  and L0 transport rows additionally require a ``verified`` execution path.
"""
from collections import defaultdict
import csv

from common.storage import fingerprint, write_json
from schemas import SEVERITY

NOT_RUN_STATUSES = {"not_run", "not_applicable", "unsupported", "dependency_missing",
                    "environment_unavailable", "source_unavailable", "not_implemented"}
FAILED_STATUSES = {"runtime_failed", "correctness_failed", "timeout", "oom"}


def _severity(status):
    return SEVERITY.index(status) if status in SEVERITY else -1


def worst(statuses):
    return max(statuses, key=_severity) if statuses else "not_run"


def aggregate(statuses):
    """Case-level status: a failure dominates, then a real pass, then the gap.

    A case is covered once *any* operator measured it successfully, so one pass
    plus several unsupported operators is ``passed`` at case level; the operator
    table keeps the unsupported count and the case/operator table keeps every
    cell.  A failed attempt always dominates, so a regression cannot hide behind
    another operator's pass.
    """
    if not statuses:
        return "not_run"
    failed = [status for status in statuses if status in FAILED_STATUSES]
    if failed:
        return worst(failed)
    if "passed" in statuses:
        return "passed"
    return worst(statuses)


def _ep_size(row):
    return row["config"]["distributed"]["ep_size"]


def _matrix(manifest):
    return manifest.get("matrix") or {}


def _ep_sizes(rows, manifest):
    declared = (_matrix(manifest).get("selected_ep_sizes")
                or _matrix(manifest).get("ep_sizes") or [])
    observed = {_ep_size(row) for row in rows}
    return sorted(set(declared) | observed)


def case_matrix(rows, manifest):
    """One row per catalog case with the worst status at each EP size.

    A case the plan expected but that produced no record is ``missing``; a case
    outside the plan scope is ``out_of_scope``.  Neither is silently omitted.
    """
    catalog = _matrix(manifest).get("catalog") or {}
    sizes = _ep_sizes(rows, manifest)
    expected = set(_matrix(manifest).get("expected_cases") or ())
    if not expected:
        expected = {row["case_id"] for row in rows}
    statuses = defaultdict(list)
    reasons = defaultdict(list)
    for row in rows:
        key = (row["case_id"], _ep_size(row))
        statuses[key].append(row["status"])
        if row.get("error_reason"):
            reasons[key].append(row["error_reason"])
    if not catalog:
        catalog = {row["case_id"]: {"level": row["benchmark_level"], "category": None,
                                    "alias_of": None} for row in rows}
    rows_out = []
    for case_id in sorted(catalog):
        entry = catalog[case_id]
        record = {"case_id": case_id, "level": entry.get("level"),
                  "category": entry.get("category"), "alias_of": entry.get("alias_of"),
                  "verification": {}}
        for size in sizes:
            if entry.get("alias_of"):
                record[str(size)] = f"alias->{entry['alias_of']}"
                record["verification"][str(size)] = "alias"
                continue
            if case_id not in expected:
                record[str(size)] = "out_of_scope"
                record["verification"][str(size)] = "out_of_scope"
                continue
            key = (case_id, size)
            if key not in statuses:
                record[str(size)] = "missing"
                record["verification"][str(size)] = "missing"
                continue
            status = aggregate(statuses[key])
            record[str(size)] = status
            record["verification"][str(size)] = status
        record["reasons"] = {str(size): (reasons.get((case_id, size)) or [None])[0]
                             for size in sizes}
        rows_out.append(record)
    return rows_out


def operator_matrix(rows, manifest):
    """Per EP size and operator: supported / executed / passed / failed counts."""
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        entry = counts[(_ep_size(row), row["backend_id"])]
        entry[row["status"]] += 1
        entry["cases"] = entry.get("cases", 0) + 1
        if row["status"] == "passed":
            entry["passed_cases_set"] = entry.get("passed_cases_set", set())
            entry["passed_cases_set"].add(row["case_id"])
        entry["case_ids"] = entry.get("case_ids", set())
        entry["case_ids"].add(row["case_id"])
    output = []
    for (size, operator), entry in sorted(counts.items()):
        total = entry["cases"]
        passed = entry.get("passed", 0)
        output.append({
            "ep_size": size, "operator": operator, "tasks": total,
            "cases": len(entry.get("case_ids", ())),
            "passed": passed,
            "passed_cases": len(entry.get("passed_cases_set", ())),
            "failed": sum(entry.get(status, 0) for status in FAILED_STATUSES),
            "unsupported": entry.get("unsupported", 0),
            "dependency_missing": entry.get("dependency_missing", 0),
            "environment_unavailable": entry.get("environment_unavailable", 0),
            "not_applicable": entry.get("not_applicable", 0),
            "not_run": entry.get("not_run", 0) + entry.get("not_implemented", 0),
            "statuses": {status: count for status, count in sorted(entry.items())
                         if status not in {"cases", "case_ids", "passed_cases_set"}},
        })
    return output


def _comparable_row(row):
    return (row["status"] == "passed" and row.get("correctness") and row["correctness"]["passed"]
            and row.get("latency") and row["latency"].get("p50"))


def _parameters_key(row):
    return dict(row.get("case_parameters") or {})


def _workload_key(row):
    cfg = row["config"]
    tokens = cfg["workload"]["tokens_per_rank"]
    tokens = tuple(tokens) if isinstance(tokens, list) else (tokens,)
    # Per-rank shape, not the global total: EP8 and EP128 are comparable only
    # when every rank does the same work.  The global expert count is *not* part
    # of the key: it changes with the EP size by design, so it is reported with
    # each point instead of excluding the comparison.
    model = {key: value for key, value in cfg["model"].items() if key != "num_experts"}
    return {
        "model": model,
        "tokens_per_rank": tokens,
        "routing": cfg["workload"]["routing"],
        "expert_placement": cfg["workload"].get("expert_placement"),
        "seed": cfg["workload"]["seed"],
        "dedup": cfg["dedup"],
        "load_balance": cfg["load_balance"],
    }


def comparison_key(row):
    cfg = row["config"]
    latency = row.get("latency") or {}
    bandwidth = row.get("bandwidth") or {}
    return fingerprint([
        row["case_id"], row["backend_id"], row["benchmark_level"], row["operation"],
        _parameters_key(row), _workload_key(row),
        cfg["benchmark"]["timing"], latency.get("unit"), latency.get("scope"),
        bandwidth.get("unit"), bandwidth.get("scope"),
    ])


def ep_size_comparison(rows, manifest):
    """Group compatible measurements by (case, operator, variant) across EP sizes."""
    groups = defaultdict(list)
    excluded = []
    for row in rows:
        if not _comparable_row(row):
            continue
        groups[comparison_key(row)].append(row)
    output = []
    for key, group in groups.items():
        sizes = sorted({_ep_size(row) for row in group})
        if len(sizes) < 2:
            continue
        verified = []
        unverified = []
        for row in sorted(group, key=_ep_size):
            latency = row.get("latency") or {}
            bandwidth = row.get("bandwidth") or {}
            rate = row.get("throughput") or {}
            status = ((row.get("verification") or {}).get("status")
                      or (row.get("observed") or {}).get("verification_status"))
            point = {
                "ep_size": _ep_size(row), "task_id": row["task_id"], "status": row["status"],
                "unit": latency.get("unit"), "scope": latency.get("scope"),
                "p50": latency.get("p50"), "p95": latency.get("p95"),
                "bandwidth_GBps": bandwidth.get("payload_bandwidth_GBps"),
                "bandwidth_bytes": bandwidth.get("payload_bytes"),
                "tokens_per_second": rate.get("global_source_tokens_per_second"),
                "num_experts": row["config"]["model"]["num_experts"],
                "verification": status,
            }
            if row["benchmark_level"] == "L0" and status != "verified":
                unverified.append(point)
            else:
                verified.append(point)
        first = group[0]
        entry = {
            "case_id": first["case_id"], "operator": first["backend_id"],
            "level": first["benchmark_level"], "parameter": _parameters_key(first),
            "variant": (first.get("case_parameters") or {}).get("variant"),
            "timing": first["config"]["benchmark"]["timing"],
            "points": verified,
            "unverified": unverified,
            "comparable": bool(verified),
            "reason": None if verified else "no verified execution path (L0)",
        }
        if len(verified) and len({point["ep_size"] for point in verified}) < 2:
            entry["comparable"] = False
            entry["reason"] = "fewer than two EP sizes with verified measurements"
        output.append(entry)
    output.sort(key=lambda item: (item["level"], item["case_id"], item["operator"] or ""))
    return {"groups": output, "excluded": excluded}


def l0_scaleout(rows):
    """L0 hardware scale-out: the same case/mechanism/size at each EP size."""
    groups = defaultdict(list)
    for row in rows:
        if row["benchmark_level"] != "L0":
            continue
        parameters = row.get("case_parameters") or {}
        key = (row["case_id"], row.get("category"),
               (row.get("observed") or {}).get("transport_protocol"),
               (row.get("observed") or {}).get("execution_resource"),
               parameters.get("message_bytes"), parameters.get("outstanding"),
               parameters.get("peer_count"), parameters.get("accelerator"))
        groups[key].append(row)
    output = []
    for (case_id, category, mechanism, resource, message_bytes, outstanding, peer_count,
         accelerator), group in sorted(groups.items(), key=lambda item: str(item[0])):
        points = []
        for row in sorted(group, key=_ep_size):
            latency = row.get("latency") or {}
            bandwidth = row.get("bandwidth") or {}
            points.append({
                "ep_size": _ep_size(row), "task_id": row["task_id"],
                "status": row["status"],
                "verification": ((row.get("verification") or {}).get("status") or "unverified"),
                "p50": latency.get("p50"),
                "bandwidth_GBps": bandwidth.get("payload_bandwidth_GBps"),
                "payload_bytes": bandwidth.get("payload_bytes"),
            })
        output.append({"case_id": case_id, "category": category, "mechanism": mechanism,
                       "execution_resource": resource, "message_bytes": message_bytes,
                       "outstanding": outstanding, "peer_count": peer_count,
                       "accelerator": accelerator, "points": points,
                       "comparable": len({point["ep_size"] for point in points
                                          if point["verification"] == "verified"}) >= 2})
    return output


def write(root, manifest, rows):
    """Write coverage_matrix.{md,csv,json}, operator_coverage.md and comparisons.

    The plan's ``report`` section may disable any of the three reports; disabled
    reports are not written and are not linked from the main report.
    """
    destination = root / "reports"
    destination.mkdir(exist_ok=True)
    options = _matrix(manifest).get("report") or {}
    want_matrix = options.get("generate_coverage_matrix", True)
    want_operators = options.get("compare_operators", True)
    want_comparison = options.get("compare_ep_sizes", True)
    sizes = _ep_sizes(rows, manifest)
    cases = case_matrix(rows, manifest)
    operators = operator_matrix(rows, manifest)
    cells = case_operator_matrix(rows)
    comparison = ep_size_comparison(rows, manifest) if want_comparison else {"groups": []}
    scaleout = l0_scaleout(rows)
    write_json(destination / "coverage_matrix.json", {
        "schema_version": 1, "run_id": manifest["run_id"], "ep_sizes": sizes,
        "cases": cases, "operators": operators, "case_operator": cells,
        "ep_size_comparison": comparison, "l0_scaleout": scaleout,
    })
    with (destination / "case_operator_coverage.csv").open("w", encoding="utf-8",
                                                          newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ep_size", "case_id", "operator", "status", "tasks", "reason"])
        for cell in cells:
            writer.writerow([cell["ep_size"], cell["case_id"], cell["operator"],
                             cell["status"], cell["tasks"], cell["reason"]])
    written = {"json": destination / "coverage_matrix.json",
               "case_operator": destination / "case_operator_coverage.csv"}

    status_counts = defaultdict(lambda: defaultdict(int))
    for record in cases:
        for size in sizes:
            status = record[str(size)]
            if status.startswith("alias"):
                status = "alias"
            status_counts[str(size)][status] += 1
    lines = [f"# Case coverage matrix {manifest['run_id']}", "",
             "`EP Size x Case` status. `passed` means at least one operator measured the case; "
             "a failed operator dominates and unsupported-only cells show the blocking status. "
             "A case that produced no task is `missing`, a case outside the plan is "
             "`out_of_scope`, and a legacy ID resolves to its canonical case as `alias`.",
             "",
             "| case | level | category | " + " | ".join(f"EP{size}" for size in sizes) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in sizes) + " |"]
    for record in cases:
        lines.append("| {case_id} | {level} | {category} | ".format(
            case_id=record["case_id"], level=record["level"], category=record["category"])
            + " | ".join(str(record[str(size)]) for size in sizes) + " |")
    lines += ["", "## Totals", ""]
    for size in sizes:
        counts = status_counts[str(size)]
        lines.append(f"- EP{size}: " + ", ".join(
            f"{status}={count}" for status, count in
            sorted(counts.items(), key=lambda item: _severity(item[0]), reverse=True)))
    lines.append("")
    (destination / "coverage_matrix.md").write_text("\n".join(lines), encoding="utf-8")
    if want_matrix:
        written["coverage"] = destination / "coverage_matrix.md"

    with (destination / "coverage_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "level", "category", *[f"EP{size}" for size in sizes]])
        for record in cases:
            writer.writerow([record["case_id"], record["level"], record["category"],
                             *[record[str(size)] for size in sizes]])

    operator_lines = [f"# Operator coverage {manifest['run_id']}", "",
                      "Per EP size and adapter: how many cases were supported, executed and passed.",
                      "",
                      "| EP | operator | cases | passed | failed | unsupported | dependency_missing "
                      "| environment_unavailable | not_applicable | not_run |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in operators:
        operator_lines.append(
            "| EP{ep_size} | {operator} | {cases} | {passed} | {failed} | {unsupported} | "
            "{dependency_missing} | {environment_unavailable} | {not_applicable} | {not_run} |".format(
                **entry))
    operator_lines += ["", "`passed` counts only measured tasks; unsupported and missing-"
                           "dependency tasks are never counted as passed.", ""]
    if want_operators:
        (destination / "operator_coverage.md").write_text("\n".join(operator_lines), encoding="utf-8")
        written["operators"] = destination / "operator_coverage.md"

    comparison_lines = [f"# EP size comparison {manifest['run_id']}", "",
                        "Only measurements with the same case, operator, parameter variant, model, "
                        "per-rank workload and timing boundary are grouped. L0 transport rows are "
                        "compared only when the execution path is `verified`.", ""]
    comparable = [group for group in comparison["groups"] if group["comparable"]]
    if comparable:
        comparison_lines += ["| case | operator | variant | timing | "
                             + " | ".join(f"EP{size} p50" for size in sizes) + " |",
                             "| --- | --- | --- | --- | " + " | ".join("---" for _ in sizes) + " |"]
        for group in comparable:
            points = {point["ep_size"]: point for point in group["points"]}
            comparison_lines.append(
                "| {case} | {operator} | {variant} | {timing} | ".format(
                    case=group["case_id"], operator=group["operator"],
                    variant=group["variant"], timing=group["timing"])
                + " | ".join(_format_point(points.get(size)) for size in sizes) + " |")
    else:
        comparison_lines.append("No measurement group is comparable across EP sizes yet.")
    excluded = [group for group in comparison["groups"] if not group["comparable"]]
    if excluded:
        comparison_lines += ["", "## Not comparable", "",
                             "| case | operator | reason |", "| --- | --- | --- |"]
        for group in excluded:
            comparison_lines.append(f"| {group['case_id']} | {group['operator']} | {group['reason']} |")
    comparison_lines += ["", "## L0 scale-out", "",
                         "| case | mechanism | bytes | peers | " +
                         " | ".join(f"EP{size}" for size in sizes) + " |",
                         "| --- | --- | --- | --- | " + " | ".join("---" for _ in sizes) + " |"]
    for entry in scaleout:
        points = {point["ep_size"]: point for point in entry["points"]}
        comparison_lines.append(
            "| {case} | {mechanism} | {size} | {peers} | ".format(
                case=entry["case_id"], mechanism=entry["mechanism"],
                size=entry["message_bytes"], peers=entry["peer_count"])
            + " | ".join(_format_scaleout(points.get(size)) for size in sizes) + " |")
    comparison_lines += ["", "`unverified` L0 rows are listed for visibility but are not used for "
                             "mechanism comparison.", ""]
    if want_comparison:
        (destination / "ep_size_comparison.md").write_text("\n".join(comparison_lines),
                                                           encoding="utf-8")
        (destination / "l0_scaleout.md").write_text(
            "\n".join(comparison_lines[comparison_lines.index("## L0 scale-out"):]) + "\n",
            encoding="utf-8")
        written["comparison"] = destination / "ep_size_comparison.md"
        written["l0_scaleout"] = destination / "l0_scaleout.md"
    return written


def case_operator_matrix(rows):
    """Every (EP size, case, operator) cell: the three-dimensional coverage.

    The Markdown matrix stays readable at ``case x EP``; this table is the
    machine-readable source that keeps the operator dimension.
    """
    cells = defaultdict(lambda: {"statuses": [], "tasks": 0, "reason": None})
    for row in rows:
        key = (_ep_size(row), row["case_id"], row["backend_id"])
        cell = cells[key]
        cell["statuses"].append(row["status"])
        cell["tasks"] += 1
        if row.get("error_reason") and cell["reason"] is None:
            cell["reason"] = row["error_reason"]
    output = []
    for (ep_size, case_id, operator), cell in sorted(cells.items()):
        output.append({
            "ep_size": ep_size, "case_id": case_id, "operator": operator,
            "status": aggregate(cell["statuses"]), "tasks": cell["tasks"],
            "reason": cell["reason"],
        })
    return output


def _format_point(point):
    if point is None:
        return "-"
    return f"{point['p50']:.4g} {point['unit'] or ''}".strip() if point.get("p50") is not None else "-"


def _format_scaleout(point):
    if point is None:
        return "-"
    if point["status"] != "passed":
        return point["status"]
    rate = point.get("bandwidth_GBps")
    return f"{rate:.3g} GB/s ({point['verification']})" if rate is not None else "passed"

