"""The common Report Context every section renders from.

A section must not have to re-read the manifest, re-count the statuses or re-run the
L0 analysis, because two sections that each derive their own numbers is exactly how
a report ends up disagreeing with itself.  :func:`build` computes the shared state
once and the section modules only format it.
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from analysis import comparability, l0 as l0_analysis
from schemas import group_totals, status_group

#: Device wording per device type.  A CPU run validates framework behaviour; only an
#: Ascend run measures device performance, and the report must not blur the two.
DEVICE_WORDING = {
    "cpu": "CPU (framework validation, not a device performance measurement)",
    "npu": "Ascend NPU",
}


@dataclass
class StatusSummary:
    """One consistent reading of "what did this run actually produce"."""

    planned: int = 0
    records: int = 0
    missing: list = field(default_factory=list)
    orphans: list = field(default_factory=list)
    statuses: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    correctness_checked: int = 0
    correctness_passed: int = 0

    @property
    def passed(self):
        return self.groups.get("Passed", 0)

    @property
    def failed(self):
        return self.groups.get("Failed", 0)

    @property
    def unsupported(self):
        return self.groups.get("Unsupported", 0)

    @property
    def blocked(self):
        return (self.groups.get("Dependency Missing", 0)
                + self.groups.get("Environment Unavailable", 0)
                + self.groups.get("Source Unavailable", 0))

    @property
    def not_run(self):
        return self.groups.get("Not Run", 0)

    @property
    def not_applicable(self):
        return self.groups.get("Not Applicable", 0)

    @property
    def lost(self):
        return self.groups.get("Missing", 0)

    @property
    def executed(self):
        """Tasks that actually ran: a pass or a genuine failure."""
        return self.passed + self.failed

    @property
    def terminal(self):
        return self.planned - self.not_run - self.lost

    @property
    def correctness_rate(self):
        return (self.correctness_passed / self.correctness_checked
                if self.correctness_checked else None)

    @property
    def completion_rate(self):
        return self.terminal / self.planned if self.planned else None

    @property
    def consistent(self):
        return sum(self.groups.values()) == self.planned + len(self.orphans)


def summarise(manifest, rows):
    """Count planned tasks against results without losing a single one."""
    by_id = {row.get("task_id"): row for row in rows}
    summary = StatusSummary()
    statuses = []
    planned_ids = set()
    for task in manifest.get("tasks") or ():
        task_id = task.get("task_id")
        planned_ids.add(task_id)
        row = by_id.get(task_id)
        if row is None:
            status = "missing"
            summary.missing.append(task_id)
        else:
            status = row.get("status")
        statuses.append(status)
        if row is not None and status == "passed":
            summary.correctness_checked += 1
            if (row.get("correctness") or {}).get("passed"):
                summary.correctness_passed += 1
    summary.planned = len(planned_ids)
    summary.records = len(rows)
    summary.orphans = sorted(row.get("task_id") for row in rows
                             if row.get("task_id") not in planned_ids)
    for task_id in summary.orphans:
        row = by_id[task_id]
        statuses.append(row.get("status"))
    summary.statuses = dict(sorted(Counter(statuses).items()))
    summary.groups = group_totals([status for status in statuses if status_group(status)])
    unknown = {status for status in statuses if status and not status_group(status)}
    summary.groups.update({status: statuses.count(status) for status in unknown})
    return summary


def device_kind(manifest, rows):
    """``npu`` when the run touched a device, else ``cpu``."""
    for row in rows:
        device = ((row.get("config") or {}).get("device"))
        if device:
            return device
    for task in manifest.get("tasks") or ():
        device = ((task.get("config") or {}).get("device"))
        if device:
            return device
    return "cpu"


def device_wording(kind):
    return DEVICE_WORDING.get(kind, kind)


def environment_view(manifest, rows, raw_capabilities=None):
    """Everything section 1 shows, with the source of each value kept."""
    environment = manifest.get("environment") or {}
    packages = environment.get("packages") or {}
    npu_row = next((row for row in rows
                    if ((row.get("config") or {}).get("device") == "npu")), None)
    row_environment = (npu_row or {}).get("environment") or {}
    topology = (manifest.get("matrix") or {}).get("topology") or {}
    mapping = None
    for row in rows:
        candidate = (row.get("topology") or {}).get("rank_mapping")
        if candidate:
            mapping = candidate
            break
    sizes = sorted({comparability.ep_size(row) for row in rows
                    if comparability.ep_size(row)})
    declared = (manifest.get("matrix") or {}).get("selected_ep_sizes") \
        or (manifest.get("matrix") or {}).get("ep_sizes") or []
    tp_sizes = sorted({comparability.tp_size(row) for row in rows
                       if comparability.tp_size(row)})
    return {
        "device": device_kind(manifest, rows),
        "host": environment.get("host"),
        "platform": environment.get("platform"),
        "os": environment.get("platform"),
        "python": _first_line(environment.get("python")),
        "npu_models": row_environment.get("npu_models") or environment.get("npu_models"),
        "driver_version": environment.get("driver_version"),
        "driver_version_source": environment.get("driver_version_source"),
        "npu_smi_info": environment.get("npu_smi_info"),
        "cann_version": row_environment.get("cann_version") or environment.get("cann_version"),
        "cann_version_source": environment.get("cann_version_source"),
        "cann_version_error": environment.get("cann_version_error"),
        "cann_root": environment.get("cann_root") or row_environment.get("cann_root"),
        "torch": packages.get("torch"),
        "torch_npu": packages.get("torch-npu"),
        "python_packages": packages,
        "git_commit": environment.get("git_commit"),
        "git_changes": environment.get("git_changes"),
        "ep_sizes": sizes or declared,
        "declared_ep_sizes": declared,
        "tp_sizes": tp_sizes or [1],
        "topology": topology,
        "rank_mapping": mapping,
        "device_visibility": environment.get("device_visibility"),
        "capabilities": raw_capabilities,
        "capability_source": (raw_capabilities or {}).get("source"),
        "capability_error": (raw_capabilities or {}).get("error"),
    }


def _first_line(value):
    if not value:
        return None
    return str(value).splitlines()[0].strip()


@dataclass
class ReportContext:
    """Everything one report pass knows, computed once."""

    root: Path
    destination: Path
    manifest: dict
    rows: list
    analysis: dict
    status: StatusSummary
    environment: dict
    l0: dict
    coverage: dict = field(default_factory=dict)
    coverage_data: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    charts: object = None

    # -- convenience ------------------------------------------------------
    @property
    def run_id(self):
        return self.manifest.get("run_id")

    @property
    def device(self):
        return self.environment["device"]

    @property
    def device_wording(self):
        return device_wording(self.device)

    @property
    def matrix(self):
        return self.manifest.get("matrix") or {}

    def comparison_groups(self, kind=None):
        """Comparison groups, optionally filtered by verdict."""
        groups = ((self.analysis.get("comparison_details") or {}).get("groups")) or []
        if kind is None:
            return groups
        return [group for group in groups if group.get("kind") == kind]

    def rows_at_level(self, level):
        return [row for row in self.rows if row.get("benchmark_level") == level]

    def levels(self):
        present = {row.get("benchmark_level") for row in self.rows}
        declared = set(self.matrix.get("levels") or ())
        return [level for level in ("L0", "L1", "L2", "L3", "L4")
                if level in present or level in declared]

    def ep_sizes(self):
        observed = {comparability.ep_size(row) for row in self.rows
                    if comparability.ep_size(row)}
        declared = set(self.matrix.get("selected_ep_sizes") or ())
        return sorted(observed or declared)


def build(root, manifest, rows, analysis, capabilities=None, coverage=None,
          coverage_data=None, destination=None, charts=None):
    """Assemble the context for one report pass."""
    rows = list(rows)
    root = Path(root)
    destination = Path(destination or (root / "reports"))
    coverage = coverage or {}
    if not coverage_data:
        coverage_data = {}
    return ReportContext(
        root=root, destination=destination, manifest=manifest, rows=rows,
        analysis=analysis or {},
        status=summarise(manifest, rows),
        environment=environment_view(manifest, rows, capabilities),
        l0=l0_analysis.analyse(rows),
        coverage=coverage, coverage_data=coverage_data,
        capabilities=capabilities or {}, charts=charts,
    )
