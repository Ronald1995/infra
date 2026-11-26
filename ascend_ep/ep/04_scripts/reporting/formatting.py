"""One number format, one missing-value rule, one Markdown table convention.

Reports are read side by side, so a value must look the same everywhere and an
absent value must never look like a measurement:

* every measured number is shown with **3 significant digits** (``1.011065`` ->
  ``1.01``, ``126599.18`` -> ``1.27e+05``, ``0.000123456`` -> ``0.000123``);
* percentages are percentages (``87.3%``), latency is in ms, bandwidth in GB/s,
  compute in TFLOPS and memory in MiB/GiB, always named in the column header;
* ``None`` renders as ``-`` (unknown) and a structurally inapplicable metric
  renders as ``N/A (<reason>)`` -- never as ``0``;
* exact counts (bytes, iterations, tasks) keep their digits instead of being
  rounded to three significant ones, because rounding a count is a lie about it.
"""
import math

#: Unknown value: nothing was recorded.
MISSING = "-"
#: The metric does not apply to this kind of case.
NA = "N/A"

#: The reason strings a report uses for a structurally inapplicable metric.
REASON_NOT_COMMUNICATION = "not a communication payload"
REASON_NO_MODEL = "no bandwidth model for this case"
REASON_NOT_APPLICABLE = "not applicable to this case"
REASON_NO_PROFILE = "no profiler data"
REASON_UNMEASURED = "not measured"


def escape(value):
    """Make one value safe inside a Markdown table cell."""
    text = "" if value is None else str(value)
    if text == "":
        return MISSING
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip() or MISSING


def number(value, digits=3, missing=MISSING):
    """Three significant digits, or the missing marker."""
    if value is None:
        return missing
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if not math.isfinite(numeric):
        return missing
    return f"{numeric:.{digits}g}"


def count(value, missing=MISSING):
    """An exact count: integers keep their digits, large values fall back to 3 s.f."""
    if value is None:
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if not math.isfinite(numeric):
        return missing
    if numeric == int(numeric) and abs(numeric) < 1e6:
        return str(int(numeric))
    return f"{numeric:.3g}"


def percent(value, digits=3, missing=MISSING):
    """A ratio shown as a percentage (``0.873`` -> ``87.3%``)."""
    if value is None:
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if not math.isfinite(numeric):
        return missing
    return f"{numeric * 100:.{digits}g}%"


def milliseconds(value, missing=MISSING):
    """Latency in ms, unit named in the header (not repeated in the cell)."""
    return number(value, missing=missing)


def bandwidth(value, missing=MISSING):
    """Bandwidth in GB/s, unit named in the header."""
    return number(value, missing=missing)


def tflops(value, missing=MISSING):
    """Effective TFLOPS, unit named in the header."""
    return number(value, missing=missing)


def ratio(value, missing=MISSING):
    """A dimensionless ratio, 3 significant digits."""
    return number(value, missing=missing)


def binary(value, missing=MISSING):
    """A byte count in MiB or GiB, with the unit in the cell."""
    if value is None:
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if not math.isfinite(numeric):
        return missing
    if abs(numeric) >= 1 << 30:
        return f"{numeric / (1 << 30):.3g} GiB"
    return f"{numeric / (1 << 20):.3g} MiB"


def na(reason=REASON_NOT_APPLICABLE):
    """An explicit "does not apply", with the reason kept in the cell."""
    return f"{NA} ({reason})"


def text(value, missing=MISSING):
    """Free text: the missing marker when empty, otherwise the value itself."""
    return escape(value) if value not in (None, "") else missing


def boolean(value, missing=MISSING):
    if value is None:
        return missing
    return "yes" if value else "no"


def join(values, separator=", ", missing=MISSING):
    """Join a list for one table cell."""
    items = [str(value) for value in (values or ()) if value not in (None, "")]
    return escape(separator.join(items)) if items else missing


def status_counts(counts, labels=None):
    """``passed=4, unsupported=2`` in a stable order."""
    labels = labels or {}
    items = sorted(counts.items(), key=lambda item: (labels.get(item[0], 99), item[0]))
    return join([f"{label}={value}" for label, value in items if value], separator=", ")


def table(columns, rows, empty=None):
    """Render a Markdown table, or one explicit line when there is nothing to show."""
    rows = [[escape(cell) for cell in row] for row in rows]
    if not rows:
        return [f"_{empty}_" if empty else "_No rows._", ""]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return [*lines, ""]


def link(label, target):
    return f"[{label}]({target})"


def bullet_list(items, empty="_None._"):
    items = [item for item in items if item]
    if not items:
        return [empty, ""]
    return [*(f"- {item}" for item in items), ""]
