"""Section 2: Test Coverage Summary.

Coverage is counted per *planned task*, not per produced result, so a task that never
ran and a case the manifest expected but that has no record at all are both visible.
The raw status is always kept; the buckets only classify it.
"""
from collections import Counter, defaultdict

from analysis import comparability
from reporting import formatting as F
from reporting.sections import common
from schemas import STATUS_GROUPS, status_group

#: Buckets shown as their own column, in report order.
BUCKETS = tuple(label for label, _statuses in STATUS_GROUPS)


def render(ctx):
    lines = ["## 2. Test Coverage Summary", "",
             "Every planned task resolves to exactly one bucket. `Missing` means the manifest "
             "planned the task but no result record exists; the raw status is preserved in "
             "`coverage_matrix.json` and in every result file.", ""]
    lines += ["### 2.1 Status vocabulary", ""]
    lines += F.table(("Status category", "Raw statuses included"), common_status_rows())
    lines += ["### 2.2 Level coverage (L0-L4)", ""]
    lines += _dimension_table(ctx, "level")
    lines += ["### 2.3 EP size coverage", ""]
    lines += _dimension_table(ctx, "ep_size")
    lines += ["### 2.4 Backend coverage", ""]
    lines += _dimension_table(ctx, "backend")
    lines += ["### 2.5 Case x EP size matrix", ""]
    lines += _case_ep_matrix(ctx)
    lines += ["### 2.6 Case x backend matrix", ""]
    lines += _case_backend_matrix(ctx)
    lines += ["### 2.7 Variant coverage", ""]
    lines += _variant_coverage(ctx)
    return lines


def common_status_rows():
    return [(label, ", ".join(f"`{status}`" for status in statuses))
            for label, statuses in STATUS_GROUPS]


def planned_statuses(ctx):
    """``(task, status)`` for every planned task, with ``missing`` for a lost record."""
    by_id = {row.get("task_id"): row for row in ctx.rows}
    planned = set()
    for task in ctx.manifest.get("tasks") or ():
        task_id = task.get("task_id")
        planned.add(task_id)
        row = by_id.get(task_id)
        yield task, (row.get("status") if row is not None else "missing")
    for row in ctx.rows:
        if row.get("task_id") not in planned:
            # An orphan result is reported, never silently dropped from the totals.
            yield {"task_id": row.get("task_id"), "level": row.get("benchmark_level"),
                   "ep_size": comparability.ep_size(row),
                   "backend_id": row.get("backend_id"),
                   "case_id": row.get("case_id")}, row.get("status")


def _key(task, dimension):
    if dimension == "level":
        return task.get("level") or task.get("benchmark_level")
    if dimension == "ep_size":
        return task.get("ep_size") or ((task.get("config") or {}).get("distributed") or {}).get(
            "ep_size")
    return task.get("backend_id")


def _dimension_table(ctx, dimension):
    counts = defaultdict(Counter)
    for task, status in planned_statuses(ctx):
        counts[_key(task, dimension)][status_group(status) or str(status)] += 1
    rows = []
    for value in sorted(counts, key=lambda item: (item is None, str(item))):
        entry = counts[value]
        total = sum(entry.values())
        label = f"EP{value}" if dimension == "ep_size" else common.level_name(value) \
            if dimension == "level" else str(value)
        rows.append([label, *(F.count(entry.get(bucket, 0)) for bucket in BUCKETS),
                     F.count(total)])
    return F.table(("EP size" if dimension == "ep_size" else
                    "Level" if dimension == "level" else "Backend", *BUCKETS, "Total"), rows,
                   empty="No planned task exists for this dimension.")


def _aggregate_status(statuses):
    """Worst status in a cell: a failure dominates, then a pass, then the gap."""
    from reporting.coverage import aggregate
    return aggregate(list(statuses))


def _case_ep_matrix(ctx):
    cases = (ctx.coverage_data.get("cases") or [])
    sizes = (ctx.coverage_data.get("ep_sizes")
             or sorted({comparability.ep_size(row) for row in ctx.rows
                        if comparability.ep_size(row)}))
    if not cases:
        return ["_No case matrix was produced for this run "
                "(`generate_coverage_matrix` may be disabled)._", ""]
    rows = []
    for record in cases:
        alias = record.get("alias_of")
        if alias:
            continue
        rows.append([record.get("case_id"), F.text(record.get("level")),
                     F.text(record.get("category")),
                     *(F.text(record.get(str(size))) for size in sizes)])
    lines = F.table(("Case", "Level", "Category", *(f"EP{size}" for size in sizes)), rows)
    lines += ["`missing` = planned but no result record; `out_of_scope` = the case is not part of "
              "this plan; `alias->X` = a legacy ID that resolves to a canonical case.", ""]
    return lines


def _case_backend_matrix(ctx):
    cells = ctx.coverage_data.get("case_operator") or []
    if not cells:
        return ["_No case x operator cell was produced for this run._", ""]
    statuses = defaultdict(list)
    sizes = defaultdict(set)
    for cell in cells:
        statuses[(cell.get("case_id"), cell.get("operator"))].append(cell.get("status"))
        if cell.get("ep_size"):
            sizes[(cell.get("case_id"), cell.get("operator"))].add(cell.get("ep_size"))
    cases = sorted({cell.get("case_id") for cell in cells})
    operators = sorted({cell.get("operator") for cell in cells})
    rows = []
    for case_id in cases:
        row = [case_id]
        for operator in operators:
            key = (case_id, operator)
            if key not in statuses:
                row.append(F.MISSING)
                continue
            status = _aggregate_status(statuses[key])
            row.append(f"{status} ({F.join([f'EP{size}' for size in sorted(sizes[key])], separator=',')})"
                       if len(sizes[key]) > 1 else status)
        rows.append(row)
    lines = F.table(("Case", *(f"Backend `{operator}`" for operator in operators)), rows)
    lines += ["A cell is the worst status of that case/backend pair across EP sizes; the EP list "
              "is shown when more than one EP size contributed.", ""]
    return lines


def _variant_coverage(ctx):
    variants = ctx.coverage_data.get("case_operator_variant") or {}
    totals = variants.get("totals") or {}
    if not variants:
        return ["_No variant coverage was produced for this run._", ""]
    summary = [("Planned task variants", F.count(totals.get("planned"))),
               ("Result records", F.count(totals.get("results"))),
               ("Matched variants", F.count(totals.get("matched"))),
               ("Missing variants", F.count(totals.get("missing"))),
               ("Measured variants", F.count(totals.get("measured")))]
    lines = F.table(("Item", "Count"), summary)
    per_level = defaultdict(Counter)
    for cell in variants.get("cells") or ():
        per_level[cell.get("level")][status_group(cell.get("status")) or "Unknown"] += 1
    rows = []
    for level in sorted(per_level, key=lambda item: str(item)):
        entry = per_level[level]
        rows.append([common.level_name(level), F.count(sum(entry.values())),
                     *(F.count(entry.get(bucket, 0)) for bucket in BUCKETS)])
    lines += F.table(("Level", "Variants", *BUCKETS), rows)
    if variants.get("orphans"):
        lines += [f"Result records that are not in the manifest: "
                  f"{F.join(variants['orphans'][:20])}.", ""]
    if variants.get("cases_without_tasks"):
        lines += [f"Expected cases without any task: "
                  f"{F.join(variants['cases_without_tasks'])}.", ""]
    return lines


def artifacts(ctx):
    """Relative links to the coverage artifacts that were written."""
    links = []
    for label, key in (("Case coverage matrix", "coverage"),
                       ("Case x operator coverage", "case_operator"),
                       ("Case x operator x variant coverage", "variants"),
                       ("Operator coverage", "operators"),
                       ("EP size comparison", "comparison"),
                       ("L0 scale-out", "l0_scaleout"),
                       ("Coverage JSON", "json"),
                       ("Variant coverage CSV", "variant_csv")):
        if key in ctx.coverage:
            links.append(common.path_link(label, ctx.coverage[key], ctx.destination))
    return links
