"""Section 9: Failures and Limitations.

Every task that did not pass is listed with what was requested: the case, the level, the
EP size, the backend, the case category and the full parameter set are taken from the
manifest, so a failed L0 transport case still says which mechanism and message size it
asked for.  Section 9 never explains a failure without also showing the request.
"""
from analysis import comparability
from reporting import formatting as F
from reporting.sections import common
from schemas import SEVERITY, status_group

#: The buckets that belong in this section.
REPORTED_BUCKETS = ("Failed", "Unsupported", "Dependency Missing", "Environment Unavailable",
                    "Source Unavailable", "Not Run", "Not Applicable", "Missing")

#: Rows shown in the main report; the complete list is in ``failure_summary.md``.
MAX_ROWS = 200


TITLE = "Failures and Limitations"


def render(ctx):
    return [f"## 9. {TITLE}", "", *body(ctx)]


def body(ctx, limit=MAX_ROWS):
    rows = [row for row in ctx.rows if row.get("status") != "passed"]
    missing = _missing(ctx)
    lines = []
    total = len(rows) + len(missing)
    if not total:
        lines += ["_No task failed, was skipped or is missing a record in this run._", ""]
        return lines
    lines += [f"{total} task(s) did not produce a passing measurement "
              f"({len(rows)} with a result record, {len(missing)} planned but missing). "
              "A `Missing` row has the planned request from the manifest and no result at all.",
              ""]
    lines += ["### Summary by category", ""]
    counts = {}
    for row in rows:
        label = status_group(row.get("status")) or str(row.get("status"))
        counts[label] = counts.get(label, 0) + 1
    if missing:
        counts["Missing"] = counts.get("Missing", 0) + len(missing)
    lines += F.table(("Category", "Tasks"),
                     [(label, F.count(counts[label])) for label in REPORTED_BUCKETS
                      if counts.get(label)])
    lines += ["### Task detail", ""]
    ordered = ([("record", row) for row in _ordered(rows)]
               + [("missing", task) for task in missing])
    shown = ordered[:limit] if limit else ordered
    table_rows = [_detail_row(row) if kind == "record" else _missing_row(ctx, row)
                  for kind, row in shown]
    lines += F.table(("Task ID", "Case", "Level", "EP", "Backend", "Category", "Parameters",
                      "Status", "Error reason", "Log path"), table_rows)
    if len(ordered) > len(shown):
        lines += [f"_{len(shown)} of {len(ordered)} task(s) are shown; the complete list is in "
                  f"`failure_summary.md`._", ""]
    lines += ["`Parameters` is the planned request (`case_parameters`), so an unsupported or "
              "missing task still states which variant, mechanism, message size or peer count was "
              "asked for. `-` means the value was not recorded; it is never a zero.", ""]
    return lines


def _ordered(rows):
    return sorted(rows, key=lambda row: (row.get("benchmark_level") or "",
                                         row.get("case_id") or "",
                                         row.get("backend_id") or "",
                                         _severity(row.get("status")),
                                         str(row.get("task_id"))))


def _severity(status):
    return SEVERITY.index(status) if status in SEVERITY else -1


def _detail_row(row):
    return [f"`{row.get('task_id')}`", row.get("case_id"), F.text(row.get("benchmark_level")),
            _ep(row), F.text(row.get("backend_id")), F.text(comparability.category(row)),
            common.parameter_text(row), F.text(row.get("status")),
            F.text(row.get("error_reason")), _log_cell(row)]


def _log_cell(row):
    """The first two log paths plus a count: the full list stays in the result record."""
    logs = list(row.get("logs") or ())
    if not logs:
        return F.text(row.get("attempt")) if row.get("attempt") else F.MISSING
    shown = ", ".join(logs[:2])
    return shown + (f" (+{len(logs) - 2} more)" if len(logs) > 2 else "")


def _ep(row):
    size = comparability.ep_size(row)
    return f"EP{size}" if size else F.MISSING


def _missing(ctx):
    """Planned tasks that produced no record at all, with their manifest request."""
    present = {row.get("task_id") for row in ctx.rows}
    return [task for task in ctx.manifest.get("tasks") or ()
            if task.get("task_id") not in present]


def _missing_row(ctx, task):
    parameters = task.get("parameters") or {}
    return [f"`{task.get('task_id')}`", task.get("case_id"), F.text(task.get("level")),
            f"EP{task.get('ep_size')}" if task.get("ep_size") else F.MISSING,
            F.text(task.get("backend_id")), F.text(task.get("case_category")),
            F.join([f"{key}={value}" for key, value in sorted(parameters.items())
                    if value is not None], separator=", "),
            "missing", "no result record for a planned task", F.MISSING]
