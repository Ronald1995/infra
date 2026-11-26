"""The transport registry is the single declaration, and the long-tail chart exists.

Two properties matter:

* mechanism metadata (which mechanisms exist, which dimensions they sweep) has exactly
  one source -- ``common/transport.py`` -- and the configuration validator and the
  variant expander both read it;
* the long-tail chart is produced from measured pairs and degrades to a link-less
  report when there is nothing to plot.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import transport  # noqa: E402
from common.case_spec import L0_PARAMETERS, spec  # noqa: E402
from common.config import L0_MECHANISMS, load_configs, validate  # noqa: E402
from common.variants import MECHANISM_DIMENSIONS, case_dimensions  # noqa: E402


def test_the_registry_is_the_only_mechanism_list():
    assert L0_MECHANISMS == transport.IDS
    assert set(MECHANISM_DIMENSIONS) == set(transport.IDS)


def test_every_mechanism_has_a_case_and_every_case_a_mechanism():
    for identifier, item in transport.MECHANISMS.items():
        assert item["link_cases"], identifier
        for case_id in (*item["link_cases"], *item["overlap_cases"]):
            assert spec(case_id).id == case_id
            assert transport.case_mechanism(case_id) == identifier
    declared = {value.get("mechanism") for value in L0_PARAMETERS.values()} - {None}
    assert declared <= set(transport.IDS) | {"auto"}
    assert set(transport.IDS) <= declared


def test_each_mechanism_describes_four_axes():
    for identifier, item in transport.MECHANISMS.items():
        assert item["programming_model"], identifier
        assert item["physical_link"], identifier
        assert item["execution_engines"], identifier
        assert item["claim"], identifier
        # A protocol is optional (SHMEM has none), but when present it is a string.
        assert item["protocol"] is None or isinstance(item["protocol"], str)


def test_the_dimension_table_is_derived_from_the_registry():
    for identifier in transport.IDS:
        for name in ("cores", "quiet_mode"):
            assert (MECHANISM_DIMENSIONS[identifier][name]
                    == transport.dimension(identifier, name))


def test_only_shmem_udma_device_sweeps_the_runtime_controls():
    assert transport.dimension("shmem_udma", "cores", "device") == "sweep"
    assert transport.dimension("shmem_udma", "quiet_mode", "device") == "sweep"
    assert transport.dimension("shmem_udma", "cores", "host") == "fixed"
    assert transport.dimension("shmem_udma", "quiet_mode", "host") == "fixed"
    for identifier in transport.IDS:
        if identifier == "shmem_udma":
            continue
        assert transport.dimension(identifier, "cores") == "not_applicable"
        assert transport.dimension(identifier, "quiet_mode") == "not_applicable"


def test_an_unknown_mechanism_is_rejected():
    with pytest.raises(ValueError, match="unknown transport mechanism"):
        transport.mechanism("teleport")
    config = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]
    config["benchmark"]["transport"]["enabled"] = ["teleport"]
    with pytest.raises(ValueError, match="transport.enabled"):
        validate(config)


def test_the_variant_expander_reads_the_same_table():
    settings = load_configs(ROOT / "configs" / "ep8_bf16.yaml")[0]["benchmark"]
    device = case_dimensions(spec("LINK-013"), settings, 8, "device")
    host = case_dimensions(spec("LINK-013"), settings, 8, "host")
    assert device["cores"]["type"] == "sweep"
    assert host["cores"]["type"] == "fixed"
    for case_id in ("LINK-001", "LINK-009", "LINK-012"):
        dimensions = case_dimensions(spec(case_id), settings, 8, "default")
        assert dimensions["cores"]["type"] == "not_applicable"


# --------------------------------------------------------------------- chart ---

def test_the_long_tail_chart_is_generated_from_pairs(tmp_path):
    from reporting.charts import create_charts

    analysis = {"long_tail": {"pairs": [
        {"case_id": "COMM-001", "operator": "a", "ep_size": 8, "slowdown_ratio": 1.2},
        {"case_id": "MOE-001", "operator": "b", "ep_size": 16, "slowdown_ratio": 1.8}]}}
    report = create_charts(tmp_path, [], analysis)
    names = [item.name for item in report.created]
    if not report.available:
        pytest.skip(f"chart backend unavailable: {report.reason}")
    assert "long-tail-slowdown" in names
    assert (tmp_path / "charts" / "long-tail-slowdown.png").is_file()
    assert ("long-tail-slowdown", "charts/long-tail-slowdown.png") in report.links()


def test_no_pairs_means_no_long_tail_chart(tmp_path):
    from reporting.charts import create_charts

    report = create_charts(tmp_path, [], {"long_tail": {"pairs": []}})
    if not report.available:
        pytest.skip(f"chart backend unavailable: {report.reason}")
    assert "long-tail-slowdown" not in [item.name for item in report.created]


def test_the_section_links_the_chart_when_present():
    from reporting.sections import long_tail as section

    class Charts:
        available = True

        def links(self, prefix="charts"):
            return [("long-tail-slowdown", "charts/long-tail-slowdown.png")]

    class Context:
        charts = Charts()
        analysis = {"long_tail": {"pairs": [
            {"case_id": "COMM-001", "operator": "a", "ep_size": 8,
             "baseline_ms": 10.0, "stressed_ms": 12.0, "slowdown_ratio": 1.2,
             "throughput_retention": 0.833, "amplification": 2.0,
             "stress": {"kind": "straggler"}, "calibration": None}]}}

    lines = "\n".join(section.render(Context()))
    assert "![long-tail-slowdown](charts/long-tail-slowdown.png)" in lines
    assert "Measured cost per stressed point" in lines
