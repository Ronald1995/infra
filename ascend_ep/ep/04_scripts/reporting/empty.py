"""The one shape an analysis report takes when it has no data.

A report that cannot analyse anything is still a deliverable: it has to name the
missing input, say which conditions would produce it, and offer whatever partial
evidence *does* exist.  A file containing only a heading is not acceptable, and
neither is a silently omitted report.

Every specialty report calls :func:`unavailable` instead of writing an empty table.
"""

#: The status word used by an analysis that cannot be performed.
UNAVAILABLE = "Not Available"
#: The status word used when an analysis ran but produced no comparison.
NO_COMPARISON = "No Comparison"


def _blocks(level, status, findings, causes, required, alternatives, conclusion):
    """The body shared by the full document and an embedded section."""
    marks = "#" * (level + 1)
    lines = [f"{marks} Status", "", status, "",
             f"{marks} Current Findings", "", findings.strip(), "",
             f"{marks} Why Analysis Is Unavailable", ""]
    lines += [f"- {item}" for item in causes] or ["- No data was collected for this analysis."]
    lines += ["", f"{marks} Required Conditions", ""]
    lines += [f"- {item}" for item in required] or ["- Not specified."]
    lines += ["", f"{marks} Available Alternative Evidence", ""]
    lines += [f"- {item}" for item in alternatives] or [
        "- No alternative evidence exists in this run."]
    lines += ["", f"{marks} Conclusion", "", conclusion.strip(), ""]
    return lines


def unavailable(title, *, findings, causes, required, alternatives=(), conclusion,
                status=UNAVAILABLE):
    """Render the standard "cannot analyse" document.

    ``findings``      what the run does show (one paragraph),
    ``causes``        the concrete reasons the analysis is unavailable (bullets),
    ``required``      the conditions that would make it available (bullets),
    ``alternatives``  evidence that *is* available and where to find it (bullets),
    ``conclusion``    the claim the report is not allowed to make (one paragraph).
    """
    return [f"# {title}", "",
            *_blocks(1, status, findings, causes, required, alternatives, conclusion)]


def section(title, *, findings, causes, required, alternatives=(), conclusion,
            level=2, status=UNAVAILABLE):
    """The same structure as a subsection of a larger report."""
    return [f"{'#' * level} {title}", "",
            *_blocks(level, status, findings, causes, required, alternatives, conclusion)]
