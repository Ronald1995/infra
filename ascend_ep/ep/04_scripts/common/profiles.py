"""Execution profiles: which suites, levels and dimensions a run expands.

The requirement matrix is identical for every profile; a profile only decides how much
of it one invocation *executes*.  That distinction is what keeps a `smoke` run from
being reported as benchmark coverage:

* ``smoke``       -- one case per level (L0, L1) and one value per swept dimension;
* ``core``        -- every EP size, every standard case, every applicable operator, all
  declared dimension values, but only the standard suites;
* ``full``        -- everything the plan lists, including the specialised suites
  (routing targets, stragglers, combined long tail, topology, scaling, dedup,
  quantized, training tail);
* ``exhaustive``  -- full plus the legacy alias identifiers expanded as their own tasks.

A profile never invents a suite the plan did not configure: `core` uses the intersection
of its suite list with the plan's, and reports which of its suites were unavailable.
"""
PROFILE_NAMES = ("smoke", "core", "full", "exhaustive")

#: Suites every EP size must cover in the standard baseline (spec section 12.2).
CORE_SUITES = ("all", "hardware", "memory", "transport", "collectives", "overlap",
               "decode", "prefill", "training_tail", "high_throughput")

PROFILES = {
    "smoke": {"suites": ("all",), "levels": ("L0", "L1"), "cases_per_level": 1,
              "dimension_policy": "first", "aliases": "dedupe"},
    "core": {"suites": CORE_SUITES, "levels": None, "dimension_policy": "all",
             "aliases": "dedupe"},
    "full": {"suites": None, "levels": None, "dimension_policy": "all", "aliases": "dedupe"},
    "exhaustive": {"suites": None, "levels": None, "dimension_policy": "all",
                   "aliases": "run"},
}


def resolve(profile, plan_suites, plan_levels, cases=None):
    """Resolve one profile against a plan's declared scope.

    Returns ``{suites, levels, cases, aliases, dimension_policy, unavailable_suites}``.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown execution profile {profile!r}; "
                         f"choose from {PROFILE_NAMES}")
    spec = PROFILES[profile]
    suites = list(dict.fromkeys(plan_suites))
    unavailable = []
    if spec["suites"] is not None:
        wanted = [suite for suite in spec["suites"] if suite in suites]
        unavailable = [suite for suite in spec["suites"] if suite not in suites]
        suites = wanted
    levels = list(spec["levels"]) if spec["levels"] else list(plan_levels)
    return {"profile": profile, "suites": suites, "levels": levels,
            "cases": cases, "aliases": spec["aliases"],
            "dimension_policy": spec["dimension_policy"],
            "cases_per_level": spec.get("cases_per_level"),
            "unavailable_suites": unavailable}


def summary(profile):
    if profile not in PROFILES:
        raise ValueError(f"unknown execution profile {profile!r}")
    spec = PROFILES[profile]
    return {"profile": profile, "suites": list(spec["suites"] or ()),
            "levels": list(spec["levels"] or ()),
            "dimension_policy": spec["dimension_policy"], "aliases": spec["aliases"]}
