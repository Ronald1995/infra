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

#: L0 parameters that are displayed as their own column in the scale-out report.
#: They are a *presentation* subset: grouping always uses the full parameter
#: fingerprint, so a case-specific parameter can never be merged away.
SCALEOUT_COLUMNS = ("variant", "peer_policy", "peer_count", "outstanding", "message_bytes",
                    "accelerator", "shape_mode", "compute", "mode", "cores", "quiet_mode")


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


def _observed_mechanism(row):
    """``(mechanism, source)`` for one L0 row.

    ``observed`` wins: the runtime answer is the only evidence of what was
    actually exercised.  A row without an observation (for example a
    ``not_applicable`` cross-server point) falls back to the *requested*
    mechanism, which keeps the N/A next to the measurements of the same case.
    """
    observed = (row.get("observed") or {}).get("transport_protocol")
    if observed:
        return observed, "observed"
    requested = (row.get("case_parameters") or {}).get("mechanism")
    return (requested, "requested") if requested else (None, None)


def l0_scaleout(rows):
    """L0 hardware scale-out: one row per *complete* parameter variant.

    Grouping uses the whole case-parameter dictionary (fingerprint), plus the case,
    category and observed execution mechanism.  Nothing about a variant is
    inferred from a subset of keys: ``ring``/``same_server``/``cross_server``,
    ``device``/``host``, ``gmm``/``dgrad``/``wgrad`` and ``fixed``/``workload``
    therefore never collapse into one line.  Two different tasks at the same EP
    size are reported as ambiguous instead of one silently overwriting the other.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["benchmark_level"] != "L0":
            continue
        parameters = row.get("case_parameters") or {}
        mechanism, _source = _observed_mechanism(row)
        key = (row["case_id"], row.get("category"), mechanism,
               (row.get("observed") or {}).get("execution_resource"),
               fingerprint(parameters))
        groups[key].append(row)
    output = []
    for (case_id, category, mechanism, resource, parameter_fingerprint), group in sorted(
            groups.items(), key=lambda item: str(item[0])):
        parameters = dict(group[0].get("case_parameters") or {})
        mechanism_source = next((source for row in group
                                 for value, source in (_observed_mechanism(row),) if value == mechanism),
                                None)
        observed_mechanisms = sorted({value for row in group
                                      for value, _source in (_observed_mechanism(row),)
                                      if value and (row.get("observed") or {}).get("transport_protocol")})
        by_ep_size = defaultdict(list)
        for row in group:
            by_ep_size[_ep_size(row)].append(row)
        points, ambiguous = [], []
        for size, same in sorted(by_ep_size.items()):
            task_ids = sorted({row["task_id"] for row in same})
            if len(task_ids) > 1:
                ambiguous.append({
                    "ep_size": size, "task_ids": task_ids,
                    "reason": ("distinct tasks share the same case parameters but different "
                               "configurations; the point is not collapsed into one value")})
                continue
            row = min(same, key=lambda item: item["task_id"])
            latency = row.get("latency") or {}
            bandwidth = row.get("bandwidth") or {}
            points.append({
                "ep_size": size, "task_id": row["task_id"], "status": row["status"],
                "reason": row.get("error_reason"),
                "verification": ((row.get("verification") or {}).get("status") or "unverified"),
                "p50": latency.get("p50"),
                "bandwidth_GBps": bandwidth.get("payload_bandwidth_GBps"),
                "payload_bytes": bandwidth.get("payload_bytes"),
            })
        conflict = len(observed_mechanisms) > 1
        verified_sizes = {point["ep_size"] for point in points
                          if point["verification"] == "verified" and point["status"] == "passed"}
        entry = {
            "case_id": case_id, "category": category,
            "mechanism": mechanism, "mechanism_source": mechanism_source,
            "observed_mechanisms": observed_mechanisms, "mechanism_conflict": conflict,
            "execution_resource": resource,
            "parameter_fingerprint": parameter_fingerprint,
            "parameters": parameters,
            "points": points, "ambiguous": ambiguous,
            "comparable": bool(not conflict and not ambiguous and len(verified_sizes) >= 2),
        }
        for name in SCALEOUT_COLUMNS:
            entry[name] = parameters.get(name)
        if conflict:
            entry["reason"] = ("observed transport mechanism differs across EP sizes; the "
                               "rows are kept separate")
        elif ambiguous:
            entry["reason"] = "several tasks share these parameters at one EP size"
        else:
            entry["reason"] = None
        output.append(entry)
    output.sort(key=lambda item: (item["case_id"], str(item["variant"]),
                                  str(item["peer_policy"]), str(item["message_bytes"]),
                                  str(item["accelerator"]), item["parameter_fingerprint"]))
    return output


def task_matrix(rows, manifest):
    """``EP Size x Case x Operator x Variant``: planned tasks against results.

    The manifest is the single source of expected tasks, so a variant that never
    produced a result is reported as ``missing`` with its exact ``task_id``
    instead of disappearing from the report.  Results that are not in the manifest
    are reported as orphans, and a case the plan expected but that has no task at
    all is listed explicitly.
    """
    by_id = {row["task_id"]: row for row in rows}
    planned = manifest.get("tasks") or []
    cells, seen = [], set()
    for task in planned:
        seen.add(task["task_id"])
        parameters = task.get("parameters") or {}
        row = by_id.get(task["task_id"])
        cells.append({
            "ep_size": task.get("ep_size") or task["config"]["distributed"]["ep_size"],
            "level": task["level"], "case_id": task["case_id"], "operator": task["backend_id"],
            "variant": parameters.get("variant"),
            "parameter_fingerprint": fingerprint(parameters),
            "parameters": parameters,
            "suite_ids": list(task.get("suite_ids") or ()),
            "task_id": task["task_id"],
            "planned_status": task["status"],
            "planned_reason": task.get("reason"),
            "status": row["status"] if row else "missing",
            "reason": (row.get("error_reason") if row else
                       "no result record for a planned task"),
        })
    summary = defaultdict(lambda: {"planned": 0, "measured": 0, "missing": 0, "failed": 0,
                                   "unsupported": 0, "not_applicable": 0, "not_run": 0,
                                   "blocked": 0, "variants": set()})
    for cell in cells:
        entry = summary[(cell["ep_size"], cell["case_id"], cell["operator"])]
        entry["planned"] += 1
        entry["variants"].add(cell["variant"] or "default")
        status = cell["status"]
        if status == "passed":
            entry["measured"] += 1
        elif status == "missing":
            entry["missing"] += 1
        elif status in FAILED_STATUSES:
            entry["failed"] += 1
        elif status in ("dependency_missing", "environment_unavailable", "source_unavailable"):
            entry["blocked"] += 1
        elif status == "unsupported":
            entry["unsupported"] += 1
        elif status == "not_applicable":
            entry["not_applicable"] += 1
        else:
            entry["not_run"] += 1
    summary_rows = [{
        "ep_size": ep_size, "case_id": case_id, "operator": operator,
        "planned": entry["planned"], "measured": entry["measured"],
        "missing": entry["missing"], "failed": entry["failed"],
        "unsupported": entry["unsupported"], "not_applicable": entry["not_applicable"],
        "blocked": entry["blocked"], "not_run": entry["not_run"],
        "variants": sorted(entry["variants"]),
        "accounted": (entry["measured"] + entry["missing"] + entry["failed"]
                      + entry["unsupported"] + entry["not_applicable"] + entry["blocked"]
                      + entry["not_run"]),
    } for (ep_size, case_id, operator), entry in sorted(summary.items())]
    expected_cases = set((manifest.get("matrix") or {}).get("expected_cases") or ())
    present_cases = {task["case_id"] for task in planned}
    return {
        "cells": cells,
        "summary": summary_rows,
        "orphans": sorted(row["task_id"] for row in rows if row["task_id"] not in seen),
        "cases_without_tasks": sorted(expected_cases - present_cases),
        "totals": {
            "planned": len(planned), "results": len(rows),
            "matched": sum(1 for cell in cells if cell["status"] != "missing"),
            "missing": sum(1 for cell in cells if cell["status"] == "missing"),
            "measured": sum(1 for cell in cells if cell["status"] == "passed"),
        },
    }


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
    variants = task_matrix(rows, manifest)
    write_json(destination / "coverage_matrix.json", {
        "schema_version": 1, "run_id": manifest["run_id"], "ep_sizes": sizes,
        "topology": _matrix(manifest).get("topology") or {},
        "execution": _matrix(manifest).get("execution") or {},
        "cases": cases, "operators": operators, "case_operator": cells,
        "case_operator_variant": variants,
        "ep_size_comparison": comparison, "l0_scaleout": scaleout,
    })
    with (destination / "case_operator_coverage.csv").open("w", encoding="utf-8",
                                                          newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ep_size", "case_id", "operator", "status", "tasks", "reason"])
        for cell in cells:
            writer.writerow([cell["ep_size"], cell["case_id"], cell["operator"],
                             cell["status"], cell["tasks"], cell["reason"]])
    with (destination / "variant_coverage.csv").open("w", encoding="utf-8",
                                                    newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ep_size", "level", "case_id", "operator", "variant", "task_id",
                         "planned_status", "status", "reason", "parameter_fingerprint"])
        for cell in variants["cells"]:
            writer.writerow([cell["ep_size"], cell["level"], cell["case_id"], cell["operator"],
                             cell["variant"], cell["task_id"], cell["planned_status"],
                             cell["status"], cell["reason"], cell["parameter_fingerprint"]])
    written = {"json": destination / "coverage_matrix.json",
               "case_operator": destination / "case_operator_coverage.csv",
               "variant_csv": destination / "variant_coverage.csv"}

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
    comparison_lines += ["", "`unverified` L0 rows are listed for visibility but are not used for "
                             "mechanism comparison.", "",
                             "The per-variant L0 scale-out table is in `l0_scaleout.md`.", ""]
    if want_comparison:
        (destination / "ep_size_comparison.md").write_text("\n".join(comparison_lines),
                                                           encoding="utf-8")
        written["comparison"] = destination / "ep_size_comparison.md"
        (destination / "l0_scaleout.md").write_text(
            _scaleout_report(manifest, scaleout, sizes), encoding="utf-8")
        written["l0_scaleout"] = destination / "l0_scaleout.md"
    (destination / "variant_coverage.md").write_text(
        _variant_report(manifest, variants), encoding="utf-8")
    written["variants"] = destination / "variant_coverage.md"
    return written


def _scaleout_report(manifest, scaleout, sizes):
    """Markdown for ``l0_scaleout.md``: one line per complete parameter variant."""
    lines = [f"# L0 scale-out {manifest['run_id']}", "",
             "One row per case **and complete parameter variant**. The grouping key is the case, "
             "the observed transport mechanism, the execution resource and the fingerprint of "
             "every case parameter, so `ring`/`same_server`/`cross_server`, `device`/`host`, "
             "`gmm`/`dgrad`/`wgrad` and `fixed`/`workload` stay separate lines. "
             "`N/A` keeps the reason (for example an EP group without a cross-server peer) and "
             "`-` means the variant was not planned at that EP size.", "",
             "| case | variant | peer policy | mechanism | resource | message bytes | outstanding "
             "| peers | " + " | ".join(f"EP{size}" for size in sizes) + " |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | "
             + " | ".join("---" for _ in sizes) + " |"]
    for entry in scaleout:
        points = {point["ep_size"]: point for point in entry["points"]}
        ambiguous = {item["ep_size"] for item in entry["ambiguous"]}
        cells = []
        for size in sizes:
            if size in ambiguous:
                cells.append("ambiguous")
            else:
                cells.append(_format_scaleout(points.get(size)))
        lines.append(
            "| {case} | {variant} | {peer_policy} | {mechanism} | {resource} | {bytes} "
            "| {outstanding} | {peers} | ".format(
                case=entry["case_id"], variant=entry.get("variant"),
                peer_policy=entry.get("peer_policy"), mechanism=entry["mechanism"],
                resource=entry.get("execution_resource"), bytes=entry.get("message_bytes"),
                outstanding=entry.get("outstanding"), peers=entry.get("peer_count"))
            + " | ".join(cells) + " |")
    problems = [entry for entry in scaleout
                if entry["mechanism_conflict"] or entry["ambiguous"]]
    if problems:
        lines += ["", "## Unmatched variants", "",
                  "| case | variant | reason | EP sizes |", "| --- | --- | --- | --- |"]
        for entry in problems:
            sizes_text = ", ".join(f"EP{size}" for size in
                                   sorted({item["ep_size"] for item in entry["ambiguous"]}))
            lines.append("| {case} | {variant} | {reason} | {sizes} |".format(
                case=entry["case_id"], variant=entry.get("variant"),
                reason=entry["reason"], sizes=sizes_text or "all"))
    lines += ["", "`unverified` rows are kept for visibility but are not used for mechanism "
                  "comparison; only `verified` rows with at least two EP sizes are comparable.", ""]
    return "\n".join(lines)


def _variant_report(manifest, variants):
    """Markdown for ``variant_coverage.md``: planned x measured per variant."""
    lines = [f"# Case x operator x variant coverage {manifest['run_id']}", "",
             "The manifest is the single source of expected tasks. `missing` means a planned "
             "variant produced no result record; it is never counted as passed. The exact "
             "`task_id` of every gap is listed below.", ""]
    totals = variants["totals"]
    lines += [f"- planned tasks: {totals['planned']}",
              f"- result records: {totals['results']}",
              f"- matched: {totals['matched']}",
              f"- missing: {totals['missing']}",
              f"- measured (`passed`): {totals['measured']}", ""]
    if variants["cases_without_tasks"]:
        lines += ["Expected cases without any task: "
                  + ", ".join(variants["cases_without_tasks"]), ""]
    if variants["orphans"]:
        lines += ["Result records that are not in the manifest: "
                  + ", ".join(variants["orphans"][:50]), ""]
    lines += ["| EP | case | operator | variants | planned | measured | missing | failed "
              "| unsupported | not_applicable | blocked | not_run |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in variants["summary"]:
        lines.append("| EP{ep_size} | {case_id} | {operator} | {variants} | {planned} "
                     "| {measured} | {missing} | {failed} | {unsupported} | {not_applicable} "
                     "| {blocked} | {not_run} |".format(
                         **{**entry, "variants": len(entry["variants"])}))
    missing = {}
    for cell in variants["cells"]:
        if cell["status"] != "missing":
            continue
        key = (cell["ep_size"], cell["case_id"], cell["operator"])
        missing.setdefault(key, []).append((cell["variant"], cell["task_id"]))
    if missing:
        lines += ["", "## Missing variants", "",
                  "| EP | case | operator | variant | task_id |", "| --- | --- | --- | --- | --- |"]
        for (size, case_id, operator), items in sorted(missing.items()):
            for variant, task_id in items[:20]:
                lines.append(f"| EP{size} | {case_id} | {operator} | {variant} | `{task_id}` |")
    lines.append("")
    return "\n".join(lines)


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
    """Format one EP cell of the scale-out table, keeping the reason of a gap."""
    if point is None:
        return "-"
    if point["status"] != "passed":
        reason = point.get("reason")
        return f"{point['status']} ({reason})" if reason else point["status"]
    rate = point.get("bandwidth_GBps")
    return f"{rate:.3g} GB/s ({point['verification']})" if rate is not None else "passed"

