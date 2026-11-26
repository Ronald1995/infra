"""Case catalog facade.

The definitions live in :mod:`common.case_spec`; this module keeps the historic
import surface (``catalog()``, ``canonical_cases()``, ``alias_map()``,
``case_counts()``, ``l0_family()``, ``resolve()``, ``L0_PARAMETERS``) working so
existing plans, result files and tests stay valid.

``Case`` is an alias of :class:`common.case_spec.CaseSpec`, not a second type,
because a second type is exactly how the two definitions drifted apart before.
"""
from common.case_spec import (  # noqa: F401 - re-exported compatibility surface
    APPLICABILITY_RULES, CATEGORIES, CORRECTNESS_TOKENS, FIXED_OWNER, GENERIC_RULES,
    LEGACY_ALIASES, LEVELS, L0_PARAMETERS, PHASES, SIZE_WINDOWS, SWEEPS, TRANSFORMS,
    ApplicabilityContext, CaseSpec, alias_map, canonical_cases, canonical_specs,
    case_counts, catalog, handler_for, l0_family, owned_by, resolve, rule, spec, specs,
    transform, validate_catalog, variants_for,
)

#: Historic name for the catalog entry type.
Case = CaseSpec


def l0_family_of(case):
    """Alias kept for callers that used the module-level helper."""
    return l0_family(case)


__all__ = [
    "APPLICABILITY_RULES", "CATEGORIES", "CORRECTNESS_TOKENS", "CaseSpec", "Case",
    "FIXED_OWNER", "GENERIC_RULES", "LEGACY_ALIASES", "LEVELS", "L0_PARAMETERS",
    "PHASES", "SIZE_WINDOWS", "SWEEPS", "TRANSFORMS", "ApplicabilityContext",
    "alias_map", "canonical_cases", "canonical_specs", "case_counts", "catalog",
    "handler_for", "l0_family", "l0_family_of", "owned_by", "resolve", "rule", "spec",
    "specs", "transform", "validate_catalog", "variants_for",
]
