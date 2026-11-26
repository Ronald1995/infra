"""Coverage metrics: planned -> executed -> correct -> verified.

The four numbers answer different questions and must never be collapsed into one
"pass rate":

* **planning**  -- did the plan contain every requirement? 100% is expected and means
  the matrix expansion is complete, not that anything ran.
* **execution** -- of the requirements applicable to this deployment, how many actually
  ran? A ``not_applicable`` requirement leaves the denominator (it cannot run here); a
  ``not_run`` or ``blocked_topology`` one stays in it and lowers this number.
* **correctness** -- of the executed requirements, how many passed their correctness
  checks?
* **verified** -- of the executed requirements, how many produced a measurement whose
  mechanism was actually proven? A result marked ``unverified``, or one that used a
  fallback path, does not count even when the numbers look good.

The distinction matters most where a mechanism has no implementation: planning can be
100% while verified coverage cannot, and a single "coverage" number would hide exactly
that.
"""
from schemas import FAILED, NOT_RUN

MEASURED = {"passed"}
#: Statuses that cannot run on this deployment; they leave the execution denominator.
EXCLUDED = {"not_applicable"}
#: Order used when one requirement has several rows: the worst one wins.
_SEVERITY = {"passed": 0, "unverified": 1, "not_applicable": 2, **{
    status: 3 for status in NOT_RUN if status != "not_applicable"}}

_GROUP_LABELS = ("passed", "unverified", "failed", "not_run", "not_applicable")


def requirement(row):
    """The identity of one requirement: case, level, operator, EP size."""
    return (row.get("case_id"), row.get("benchmark_level") or row.get("level"),
            row.get("backend_id") or row.get("operator"), row.get("ep_size"))


def _status(row):
    return row.get("status") or "not_run"


def _verified(row):
    """Whether a passed row actually proves the mechanism it claims."""
    if _status(row) != "passed":
        return False
    observation = row.get("observation") or {}
    if observation.get("fallback") or observation.get("compatibility_fallback"):
        return False
    verification = row.get("verification")
    if isinstance(verification, dict) and verification.get("status") not in (None, "verified"):
        return False
    correctness = row.get("correctness")
    if isinstance(correctness, dict) and correctness.get("verified") is False:
        return False
    return True


def _worse(left, right):
    if left is None:
        return right
    return left if _SEVERITY.get(left, 9) >= _SEVERITY.get(right, 9) else right


def _group(status):
    if status == "passed":
        return "passed"
    if status == "unverified":
        return "unverified"
    if status in FAILED:
        return "failed"
    if status in EXCLUDED:
        return "not_applicable"
    return "not_run"


def metrics(manifest, rows):
    """The four coverage numbers plus the per-status breakdown."""
    tasks = manifest.get("tasks") or ()
    expected = {requirement(task) for task in tasks}
    status = {}
    for row in rows:
        key = requirement(row)
        if key not in expected:
            expected.add(key)  # a row outside the manifest is still an executed point
        status[key] = _worse(status.get(key), _status(row))
    present = {key for key in status}
    planned = len(expected)
    missing = sorted(expected - present, key=str)
    applicable = {key for key in expected if status.get(key) not in EXCLUDED}
    executed = {key for key in applicable
                if status.get(key) in MEASURED or status.get(key) in FAILED
                or status.get(key) == "unverified"}
    correct = {key for key in executed if status.get(key) == "passed"}
    verified = {key for key in correct
                if _verified(next(row for row in rows if requirement(row) == key))}
    breakdown = {label: 0 for label in _GROUP_LABELS}
    for key in expected:
        breakdown[_group(status.get(key, "not_run"))] += 1
    return {
        "schema_version": 1,
        "planned": planned,
        "present": len(present),
        "applicable": len(applicable),
        "executed": len(executed),
        "correct": len(correct),
        "verified": len(verified),
        "planning_coverage": len(present) / planned if planned else None,
        "execution_coverage": len(executed) / len(applicable) if applicable else None,
        "correctness_pass_rate": len(correct) / len(executed) if executed else None,
        "verified_performance_coverage": len(verified) / len(applicable) if applicable else None,
        "status_breakdown": breakdown,
        "missing": [list(key) for key in missing[:20]],
        "missing_count": len(missing),
        "not_applicable": breakdown["not_applicable"],
        "blocked": sorted({status[key] for key in expected
                           if status.get(key) in NOT_RUN
                           and status.get(key) not in EXCLUDED}),
    }
