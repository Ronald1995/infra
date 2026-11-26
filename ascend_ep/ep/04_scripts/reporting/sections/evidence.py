"""Section 10: Evidence and Raw Data.

Every claim in this report is traceable to a file, and every link is emitted only when
the target exists: a report must not contain a link that 404s.  Charts follow the same
rule -- either the image exists and is linked, or the reason it was not generated is
written out.
"""
from pathlib import Path

from reporting import formatting as F

#: Run-level artifacts, in reading order.
RUN_FILES = (
    ("Run manifest", "manifest.json"),
    ("Environment and versions", "environment.json"),
    ("Plan matrix metadata", "matrix.json"),
    ("Topology", "topology.json"),
    ("Resolved configuration", "resolved_config.yaml"),
    ("Capability matrix", "capabilities.json"),
    ("Per-EP completion summary", "ep_summary.json"),
    ("Analysis results", "analysis/results.json"),
)

#: Per-task artifacts, relative to one task directory.
TASK_FILES = ("metrics.json", "correctness.json", "status.json", "config.yaml",
              "samples.jsonl", "routing_trace.json")

#: At most this many per-task evidence rows are listed inline.
MAX_TASK_ROWS = 5


def render(ctx):
    lines = ["## 10. Evidence and Raw Data", ""]
    lines += ["### 10.1 Run-level artifacts", ""]
    lines += _run_files(ctx)
    lines += ["### 10.2 Per-task raw data", ""]
    lines += _task_files(ctx)
    lines += ["### 10.3 L0 evidence", ""]
    lines += _l0_evidence(ctx)
    lines += ["### 10.4 Profiler artifacts", ""]
    lines += _profiles(ctx)
    lines += ["### 10.5 Logs", ""]
    lines += _logs(ctx)
    lines += ["### 10.6 Charts", ""]
    lines += _charts(ctx)
    lines += ["### 10.7 Generated reports", ""]
    lines += _reports(ctx)
    lines += ["The report layer only reads these files. No report step rewrites `metrics.json`, "
              "`samples.jsonl` or any profiler artifact, so the raw measurement is always the "
              "authority over the Markdown.", ""]
    return lines


def _relative(ctx, path):
    """A link target relative to the report file's own directory.

    Everything that lives under the run root but outside ``reports/`` is reached with
    a ``../`` prefix, so the emitted Markdown link actually resolves from
    ``reports/benchmark_report.md``.
    """
    path = Path(path)
    try:
        return path.relative_to(ctx.destination).as_posix()
    except ValueError:
        pass
    try:
        return Path("..", *path.relative_to(ctx.root).parts).as_posix()
    except ValueError:
        return str(path)


def _exists(path):
    try:
        return path.is_file()
    except OSError:
        return False


def _run_files(ctx):
    rows = []
    for label, name in RUN_FILES:
        path = ctx.root / name
        if _exists(path):
            rows.append([label, f"[{name}]({_relative(ctx, path)})"])
        else:
            rows.append([label, f"{name} (not written by this run)"])
    return F.table(("Artifact", "Location"), rows)


def _task_files(ctx):
    rows = []
    for row in sorted(ctx.rows, key=lambda item: str(item.get("task_id"))):
        if len(rows) >= MAX_TASK_ROWS:
            break
        listing = _task_listing(ctx, row)
        if listing:
            rows.append([row.get("case_id"), f"`{row.get('task_id')}`", listing])
    lines = F.table(("Case", "Task", "Raw files"), rows,
                    empty="No per-task directory was found (the run produced no result record).")
    lines += ["Layout: `operators/<backend>/<task_id>/` (or `baseline_results/...` for an L0 "
              "task). `metrics.json` is the complete result record; `samples.jsonl` is one line "
              "per iteration per rank; `routing_trace.json` is the routing summary the byte model "
              "used.", ""]
    return lines


def _task_listing(ctx, row):
    attempt = row.get("attempt")
    folder = "baseline_results" if row.get("benchmark_level") == "L0" else "operators"
    base = ctx.root / folder / str(row.get("backend_id")) / str(row.get("task_id"))
    items = []
    for name in TASK_FILES:
        path = base / name
        if _exists(path):
            items.append(f"[{name}]({_relative(ctx, path)})")
    if not items and attempt:
        return f"`{attempt}/`"
    return ", ".join(items) if items else F.MISSING


def _l0_evidence(ctx):
    entries = []
    for label, name in (("L0 metrics", "l0_metrics.json"), ("L0 curves", "l0_curves.csv"),
                        ("L0 report", "l0_report.md")):
        path = ctx.destination / name
        if _exists(path):
            entries.append(f"[{label}]({name})")
    if not entries:
        return ["_No L0 evidence file was written (this run has no L0 result to report)._", ""]
    lines = ["- " + ", ".join(entries)]
    log_paths = sorted({path for row in ctx.rows
                        for path in ((row.get("evidence") or {}).get("log_paths") or ())})
    if log_paths:
        lines.append("- Verification log paths referenced by L0 records: "
                     + ", ".join(f"`{path}`" for path in log_paths[:10])
                     + (f" (+{len(log_paths) - 10} more)" if len(log_paths) > 10 else ""))
    else:
        lines.append("- No verification log path was referenced by any L0 record.")
    claims = sum(len((row.get("evidence") or {}).get("claims") or ()) for row in ctx.rows)
    lines += [f"- Verification evidence claims attached to records: {F.count(claims)}.", ""]
    return lines


def _profiles(ctx):
    rows = []
    for row in sorted(ctx.rows, key=lambda item: str(item.get("task_id"))):
        for artifact in row.get("profiling_artifacts") or ():
            path = ctx.root / artifact.get("manifest", "")
            link = (f"[{artifact.get('manifest')}]({_relative(ctx, path)})"
                    if _exists(path) else f"{artifact.get('manifest')} (missing)")
            rows.append([row.get("case_id"), f"`{row.get('task_id')}`", F.text(
                artifact.get("status")), link])
    if not rows:
        return ["_No profiler artifact was recorded for this run._", ""]
    return F.table(("Case", "Task", "Profile status", "Artifact"), rows)


def _logs(ctx):
    rows = [(f"`{row.get('task_id')}`", F.join(row.get("logs") or (), separator=", "))
            for row in sorted(ctx.rows, key=lambda item: str(item.get("task_id")))
            if row.get("logs")]
    control = ctx.root / "logs"
    lines = []
    if control.is_dir():
        lines += [f"Controller and worker logs are under `{_relative(ctx, control)}/`.", ""]
    lines += F.table(("Task", "Log files"), rows[:MAX_TASK_ROWS * 4],
                     empty="No task recorded a log path.")
    return lines


def _charts(ctx):
    report = ctx.charts
    if report is None:
        return ["Charts Not Generated: the chart stage did not run.", ""]
    lines = [report.status_line(), ""]
    links = report.links()
    if links:
        rows = [[label, f"[{label}]({path})"] for label, path in links]
        lines += F.table(("Chart", "File"), rows)
    if report.skipped:
        lines += ["Charts that were not generated, with the reason:", ""]
        lines += [f"- `{item.name}`: {item.reason}" for item in report.skipped]
        lines += [""]
    if report.missing_dependency:
        lines += [f"Install the optional chart dependency with "
                  f"`python -m pip install -e '.[charts]'` "
                  f"(missing: `{report.missing_dependency}`).", ""]
    return lines


def _reports(ctx):
    rows = []
    for label, key in (("Case coverage matrix", "coverage"),
                       ("Operator coverage", "operators"),
                       ("Case x operator x variant coverage", "variants"),
                       ("EP size comparison", "comparison"),
                       ("L0 scale-out", "l0_scaleout")):
        path = ctx.coverage.get(key)
        if path is not None and _exists(path):
            rows.append([label, f"[{path.name}]({path.name})"])
    for name in ("l0_report.md", "operator_comparison.md", "bandwidth_analysis.md",
                 "overlap_analysis.md", "scalability_analysis.md", "load_balance_analysis.md",
                 "planner_training.md", "dedup_comparison.md", "failure_summary.md"):
        path = ctx.destination / name
        if _exists(path):
            rows.append([name.replace(".md", "").replace("_", " ").title(),
                         f"[{name}]({name})"])
    return F.table(("Report", "File"), rows)
