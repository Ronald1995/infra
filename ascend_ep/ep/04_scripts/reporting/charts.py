"""Chart generation that can always report *why* a chart is missing.

A report must never link to an image that does not exist and must never fail
because an optional plotting dependency is absent.  Every call therefore returns a
:class:`ChartReport` describing what was created, what was skipped and why; the
report layer renders that description instead of guessing.

No chart is produced for a quantity that has no data -- an empty axis is noise, not
evidence.  Each skipped kind carries the concrete reason (no profiler data, one
backend only, one EP size, ...).
"""
from dataclasses import dataclass, field

#: The optional dependency charts need.
CHART_DEPENDENCY = "matplotlib"

#: Never generate more than this many charts of one kind.
MAX_PER_KIND = 6


@dataclass(frozen=True)
class ChartResult:
    """One chart: either it exists, or the reason it does not."""

    name: str
    path: str | None = None
    reason: str | None = None

    @property
    def created(self):
        return self.path is not None


@dataclass
class ChartReport:
    """The complete outcome of one chart-generation pass."""

    created: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    available: bool = True
    missing_dependency: str | None = None
    reason: str | None = None

    def add(self, result):
        (self.created if result.created else self.skipped).append(result)
        return result

    def skip(self, name, reason):
        return self.add(ChartResult(name=name, reason=reason))

    def links(self, prefix="charts"):
        """``[(label, relative path)]`` for the charts that exist."""
        return [(result.name, f"{prefix}/{result.path}") for result in self.created]

    def status_line(self):
        if not self.available:
            return (f"Charts Not Generated: {self.reason}. "
                    f"Missing dependency: {self.missing_dependency or 'unknown'}.")
        if not self.created:
            return ("Charts Not Generated: no chart had enough data "
                    f"({len(self.skipped)} plot(s) skipped, see below).")
        return (f"{len(self.created)} chart(s) generated"
                + (f", {len(self.skipped)} skipped" if self.skipped else "") + ".")

    def as_dict(self):
        return {
            "available": self.available,
            "missing_dependency": self.missing_dependency,
            "reason": self.reason,
            "created": [{"name": item.name, "path": item.path} for item in self.created],
            "skipped": [{"name": item.name, "reason": item.reason} for item in self.skipped],
        }


def _unavailable(reason, dependency=None):
    return ChartReport(available=False, missing_dependency=dependency, reason=reason)


def _plain(values):
    """Drop ``None`` while keeping the original index mapping."""
    return [value for value in values if value is not None]


def create_charts(destination, rows, analysis):
    """Render every chart that has data, and describe the rest.

    Never raises for a missing optional dependency or an unplottable series: the
    report degrades to text and says so.
    """
    report = ChartReport()
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except (ImportError, OSError, RuntimeError) as exc:
        return _unavailable(
            f"the optional `{CHART_DEPENDENCY}` dependency is unavailable ({exc})",
            CHART_DEPENDENCY)

    charts = destination / "charts"
    try:
        charts.mkdir(exist_ok=True)
    except OSError as exc:  # pragma: no cover - a read-only output tree
        return _unavailable(f"the charts directory could not be created ({exc})",
                            CHART_DEPENDENCY)
    plt.rcParams.update({"font.size": 9, "figure.autolayout": False})

    def plot(name, title, render, xlabel=None, ylabel=None):
        try:
            figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - backend failure
            return report.skip(name, f"the plotting backend could not create a figure ({exc})")
        try:
            axis.set_title(title)
            if xlabel:
                axis.set_xlabel(xlabel)
            if ylabel:
                axis.set_ylabel(ylabel)
            render(axis)
            figure.savefig(charts / (name + ".png"), dpi=130)
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            return report.skip(name, f"the series could not be rendered ({exc})")
        finally:
            plt.close(figure)
        return report.add(ChartResult(name=name, path=name + ".png"))

    details = (analysis.get("comparison_details") or {}).get("groups") or []
    comparisons = [group for group in details if group.get("kind") == "actual_comparison"]
    if comparisons:
        for group in comparisons[:MAX_PER_KIND]:
            members = [member for member in group["members"] if member.get("p50_ms") is not None]
            if len({member["backend"] for member in members}) < 2:
                continue
            label = f"latency-{group['case_id']}-{group['key'][:10]}"
            plot(label, f"{group['case_id']} · {group['device']} · P50 latency",
                 lambda axis, members=members: axis.barh(
                     [member["backend"] for member in members],
                     [member["p50_ms"] for member in members]),
                 xlabel="P50 latency (ms)")
            with_bandwidth = [member for member in members
                              if member.get("payload_GBps") is not None]
            if len({member["backend"] for member in with_bandwidth}) >= 2:
                plot(f"bandwidth-{group['case_id']}-{group['key'][:10]}",
                     f"{group['case_id']} · modelled payload bandwidth",
                     lambda axis, members=with_bandwidth: axis.barh(
                         [member["backend"] for member in members],
                         [member["payload_GBps"] for member in members]),
                     xlabel="payload bandwidth (GB/s)")
    else:
        report.skip("backend-comparison", "no group has two or more valid backends; "
                                          "one backend is a measurement, not a comparison")

    scaling_groups = analysis.get("scaling_groups") or {}
    if scaling_groups:
        for key, samples in sorted(scaling_groups.items())[:MAX_PER_KIND]:
            if len(samples) < 2:
                continue
            plot(f"scaling-{samples[0].get('mode', 'ep')}-{key[:10]}",
                 f"{samples[0]['mode'].capitalize()} scaling efficiency",
                 lambda axis, samples=samples: axis.plot(
                     [point["ep"] for point in samples],
                     [point["efficiency"] for point in samples], marker="o"),
                 xlabel="EP size", ylabel="scaling efficiency")
    else:
        report.skip("ep-scaling", "no case has two EP sizes with compatible conditions")

    pareto = analysis.get("pareto") or {}
    if pareto:
        for key, points in sorted(pareto.items())[:MAX_PER_KIND]:
            if len(points) < 2:
                continue
            plot(f"pareto-{key[:10]}", "Source tokens/s vs peak allocator",
                 lambda axis, points=points: axis.scatter(
                     [point["peak_allocator_bytes"] / (1 << 20) for point in points],
                     [point["tokens_per_second"] for point in points]),
                 xlabel="peak allocator (MiB)", ylabel="source tokens/s")
    else:
        report.skip("pareto", "no case has both a throughput and a memory measurement")

    fusion_points = analysis.get("fusion") or []
    if fusion_points:
        plot("fusion-speedup", "Measured serial latency / fused latency",
             lambda axis: axis.barh([point["fused_task"] for point in fusion_points],
                                    [point["fusion_speedup"] for point in fusion_points]),
             xlabel="speedup (serial / fused)")
    else:
        report.skip("fusion-speedup", "no serial/fused pair was measured under the same conditions")

    profiled = []
    for task, profiles in sorted((analysis.get("profiles") or {}).items()):
        for entry in profiles:
            for rank, profile in sorted(entry["ranks"].items()):
                idle = profile.get("device_idle_ratio")
                if idle is not None:
                    profiled.append((f"{task} · rank {rank}", idle))
    if profiled:
        plot("device-idle", "Idle fraction of the observed device span",
             lambda axis: axis.barh([name for name, _ in profiled],
                                    [idle for _, idle in profiled]),
             xlabel="device idle ratio")
    else:
        report.skip("device-idle", "no parsed profiler artifact contains a device idle ratio")

    load_rows = [row for row in rows if row.get("load_balance")]
    plotted_routes = set()
    drawn = 0
    for row in sorted(load_rows, key=lambda item: str(item["task_id"])):
        if drawn >= MAX_PER_KIND:
            break
        load = row["load_balance"]
        trace = row.get("routing_trace_hash")
        if trace in plotted_routes:
            continue
        plotted_routes.add(trace)
        name = row["task_id"]
        plot(f"expert-load-{name}", "Routes per expert",
             lambda axis, counts=load["expert_counts"]: axis.bar(range(len(counts)), counts),
             xlabel="expert index", ylabel="routed rows")
        plot(f"rank-load-{name}", "Routes per receiving rank",
             lambda axis, counts=load["rank_counts"]: axis.bar(range(len(counts)), counts),
             xlabel="rank", ylabel="routed rows")
        matrix = load.get("source_destination_matrix")
        if matrix:
            plot(f"traffic-{name}", "Source rank x destination rank · routed rows",
                 lambda axis, matrix=matrix: axis.imshow(matrix, aspect="auto"),
                 xlabel="destination rank", ylabel="source rank")
        drawn += 1
    if not load_rows:
        report.skip("expert-load", "no result carries a routing-trace load-balance summary")

    memory = []
    for row in sorted(rows, key=lambda item: str(item["task_id"])):
        available = [item["peak_allocated"] for item in (row.get("memory") or {}).get("ranks", ())
                     if item.get("peak_allocated") is not None]
        if available:
            memory.append((row["task_id"], max(available)))
    if memory:
        plot("peak-allocator", "Peak framework allocator",
             lambda axis: axis.barh([task for task, _ in memory[:MAX_PER_KIND * 3]],
                                    [size / (1 << 20) for _, size in memory[:MAX_PER_KIND * 3]]),
             xlabel="peak allocated (MiB)")
    else:
        report.skip("peak-allocator", "no result carries a framework memory snapshot")

    return report
