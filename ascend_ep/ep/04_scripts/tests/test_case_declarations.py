"""Declaration consistency: the catalog, the registry and the generated docs agree.

These are the checks that stop a case declaration from drifting away from the code
that runs it.  They cover the four defects this refactor fixed:

* a case with no requirement at all (COMM-010) looking unconditionally supported;
* a case whose *preparation* needs a capability it never declared (COMM-011 dispatch,
  MOE-002 forward graph);
* a level read off a case ID prefix (LAYER-007 is L3, family trace_replay);
* L2 serial reference and native fusion collapsing into one "L2" capability.
"""
from pathlib import Path

import pytest

from common import case_spec, docgen, scheduler
from common.case_spec import catalog
from common.registry import registry

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ROOT / "operators"


@pytest.fixture(scope="module")
def cases():
    return catalog()


@pytest.fixture(scope="module")
def specs():
    return registry()


def test_no_operator_case_is_unconditionally_supported(cases):
    """An empty ``requires`` must be backed by conditional requirements."""
    offenders = [case.id for case in cases.values()
                 if case.level != "L0" and case.alias_of is None
                 and not case.requires and not case.conditional_requires]
    assert offenders == [], f"cases with no declared requirement: {offenders}"


def test_comm_010_declares_the_capability_its_rule_checks(cases):
    case = cases["COMM-010"]
    declared = {capability for _, capability in case.conditional_requires}
    assert declared == {"rank_dedup", "server_dedup"}
    # The execution decision lives in the rule; both capabilities must be named
    # there, otherwise the declaration and the scheduler disagree.
    source = Path(case_spec.__file__).read_text(encoding="utf-8")
    for capability in declared:
        assert f'"{capability}"' in source or f"{capability}_dedup" in source
    assert "dedup_mode" in case.applicability


def test_comm_011_declares_the_dispatch_it_prepares_with(cases):
    case = cases["COMM-011"]
    assert case.requires == ("combine_add_rmsnorm",)
    assert case.setup_requires == ("dispatch",), (
        "the handler dispatches before it times rmsnorm_combine; that is a setup "
        "dependency and must not be silently assumed")


def test_backward_cases_declare_their_forward_graph(cases):
    assert cases["MOE-002"].setup_requires == ("moe_forward",)
    assert cases["MOE-002"].requires == ("moe_backward",)
    # forward+backward runs both inside the same timed window, so both are requires.
    assert set(cases["MOE-003"].requires) == {"moe_forward", "moe_backward"}
    assert "moe_forward" in cases["MOE-004"].setup_requires


def test_layer_007_is_l3_by_metadata_not_by_id_prefix(cases):
    case = cases["LAYER-007"]
    assert case.level == "L3"
    assert case.family == "trace_replay"
    assert case.id.startswith("LAYER-"), "the id is kept for historical compatibility"


def test_l2_serial_and_native_are_different_capabilities(cases):
    serial = cases["FUSE-001"]
    native = cases["FUSE-002"]
    assert serial.requires == ("dispatch",)
    assert native.requires == ("dispatch_gmm",)
    assert serial.baseline_reference and not native.baseline_reference
    assert cases["FUSE-003"].requires == ("dispatch", "combine")
    assert cases["FUSE-004"].requires == ("gmm_combine",)


class _Spec:
    """A minimal adapter stand-in for the scheduler decision."""

    def __init__(self, capabilities):
        self.id = "fake"
        self.capabilities = set(capabilities)
        self.status = "integrated"
        self.reason = None
        self.constraints = {}
        self.source = "test"
        self.commit = "0" * 40


def test_the_scheduler_rejects_a_missing_setup_capability(cases):
    import copy
    base = copy.deepcopy(docgen._configs()[0])
    spec = _Spec({"combine_add_rmsnorm"})
    status, reason = scheduler.status_for(spec, cases["COMM-011"], base)
    assert status == "unsupported"
    assert "setup" in reason and "dispatch" in reason


def test_the_scheduler_accepts_a_case_once_its_setup_capability_exists(cases):
    import copy
    base = copy.deepcopy(docgen._configs()[0])
    spec = _Spec({"combine_add_rmsnorm", "dispatch"})
    status, _ = scheduler.status_for(spec, cases["COMM-011"], base)
    assert status == "not_run"


def test_every_operator_declares_capabilities_that_are_used_by_a_case(specs, cases):
    """A capability nobody requires is either documentation debt or a typo."""
    used = set()
    for case in cases.values():
        used.update(case.requires)
        used.update(capability for _, capability in case.conditional_requires)
        used.update(case.setup_requires)
    for operator, spec in specs.items():
        unused = set(spec.capabilities) - used
        # training-only gradients are consumed by correctness handlers, not requires
        unused -= {"router_weight_grad", "expert_weight_grad", "dedup_training"}
        assert not unused, f"{operator} declares capabilities no case asks for: {unused}"


def test_generated_readmes_are_current():
    """The committed READMEs must equal what the generator produces now."""
    plans = {name: docgen.plan_for(name) for name in registry()}
    stale = []
    for operator, spec in registry().items():
        path = OPERATORS / operator / "README.md"
        original = path.read_text(encoding="utf-8")
        updated = docgen.splice_contract(original, docgen.contract_section(
            operator, spec, plans[operator]))
        if updated != original:
            stale.append(operator)
    assert stale == [], f"run `python scripts/docgen.py --write`: {stale}"


def test_the_capability_matrix_covers_every_operator():
    from scripts import docgen as cli

    plans = {name: docgen.plan_for(name) for name in registry()}
    text = cli.matrix_text(plans)
    for operator in registry():
        assert f"`{operator}`" in text, operator
    assert "## 不支持原因" in text


def test_comm_010_is_not_reported_as_universally_supported():
    """Its requirement is conditional, so no row may claim it unconditionally."""
    plans = {name: docgen.plan_for(name) for name in registry()}
    for operator, planned in plans.items():
        status, reason = planned.get("COMM-010", ("unsupported", None))
        if status in docgen.EXECUTABLE and status != "conditional":
            capabilities = set(registry()[operator].capabilities)
            assert capabilities & {"rank_dedup", "server_dedup"}, (
                f"{operator} claims COMM-010 without a dedup capability")
