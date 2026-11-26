"""Evidence and capability: what a transport result may claim.

The rule these tests protect: a requested mechanism is never evidence of an observed
physical link.  A descriptor, a configuration value or a compiled channel cannot
promote a link to verified; only runtime answers can, and two independent ones must
agree.
"""
import pytest

from baselines.l0.topology import runtime_probe
from baselines.l0.verification import evidence, path_verifier
from baselines.l0.verification.vocabulary import requested


def make_collector(**kwargs):
    collector = evidence.EvidenceCollector(
        requested=requested(backend="hcomm", operation="put", execution_resource="AIV",
                            transport_protocol="urma"), **kwargs)
    return collector


# ------------------------------------------------------------- physical link ---

def test_configuration_alone_never_verifies_a_link():
    collector = make_collector()
    link, confidence, sources = collector.resolve_link()
    assert link == "unverified"
    assert confidence == "unknown"
    assert sources == []


def test_a_channel_kind_alone_is_inferred_not_verified():
    collector = make_collector(channel_kind="AivUrmaChannel", engine="AIV")
    link, confidence, sources = collector.resolve_link()
    assert link == "urma"
    assert confidence == "inferred"
    assert sources == ["channel_kind"]


def test_channel_kind_and_rank_graph_protocol_together_verify_the_link():
    collector = make_collector(channel_kind="AivUrmaChannel", engine="AIV",
                               protocol="UB_CTP")
    link, confidence, sources = collector.resolve_link()
    assert link == "urma"
    assert confidence == "verified"
    assert set(sources) == {"channel_kind", "rank_graph"}


def test_disagreeing_evidence_is_not_verified():
    collector = make_collector(channel_kind="AicpuTsRoceChannel", engine="AICPU_TS",
                               protocol="HCCS")
    link, confidence, _sources = collector.resolve_link()
    assert confidence == "inferred"
    assert link == "roce"


def test_the_unknown_answer_is_explicit():
    collector = make_collector()
    observed = collector.observation().observed
    assert observed["physical_link"] is None
    assert observed["physical_link_confidence"] == "unknown"


# ------------------------------------------------------------------- ladder ---

def test_ladder_starts_configured_and_stops_at_the_first_missing_evidence():
    collector = make_collector()
    levels = collector.levels()
    assert levels["configured"] is True
    assert levels["endpoint_verified"] is False
    assert evidence.strongest_level(levels) == "configured"


def test_ladder_reaches_channel_acquired_without_claiming_a_kernel():
    collector = make_collector(endpoint_created=True,
                               channel_kind="AivUrmaChannel", engine="AIV")
    levels = collector.levels()
    assert levels["endpoint_verified"] and levels["channel_acquired"]
    assert levels["kernel_verified"] is False
    assert levels["physical_link_verified"] is False
    assert evidence.strongest_level(levels) == "channel_acquired"


def test_ladder_reaches_the_top_only_with_all_evidence():
    collector = make_collector(endpoint_created=True, channel_kind="AivUrmaChannel",
                               engine="AIV", protocol="UB_CTP", kernel_name="aiv_write_nbi",
                               profiled_kernel="asc_comm_aiv_write_nbi", execution_verified=True)
    levels = collector.levels()
    assert all(levels[rung] for rung in evidence.LADDER)
    assert evidence.strongest_level(levels) == "physical_link_verified"


def test_kernel_evidence_is_required_for_kernel_verified():
    collector = make_collector(endpoint_created=True, channel_kind="AivUbMemChannel")
    assert collector.levels()["kernel_verified"] is False
    collector.kernel("aiv_write_nbi")
    assert collector.levels()["kernel_verified"] is True


def test_collector_records_each_kind_as_strong_evidence():
    collector = make_collector()
    collector.descriptor("bridge description", channel="AivUrmaChannel")
    collector.rank_graph(3, "UB_CTP")
    collector.endpoint("endpoint created")
    collector.channel("AivUrmaChannel", engine="AIV")
    collector.kernel("aiv_write_nbi")
    collector.msprof("asc_comm_aiv_write_nbi")
    collector.execution("peer payload matched")
    sources = {item.source for item in collector.items}
    assert sources == {"native_descriptor", "rank_graph", "endpoint", "channel_kind",
                       "kernel", "msprof_kernel", "runtime_execution"}
    for item in collector.items:
        assert item.confidence == "strong"
        assert item.source in path_verifier.STRONG_SOURCES


def test_unknown_evidence_kind_is_refused():
    collector = make_collector()
    with pytest.raises(ValueError, match="unknown evidence kind"):
        collector.add("vibes", {})


def test_annotate_record_attaches_the_ladder():
    collector = make_collector(endpoint_created=True, channel_kind="AivUrmaChannel",
                              engine="AIV")
    record = {}
    evidence.annotate_record(record, collector)
    assert record["verification_level"] == "channel_acquired"
    assert record["verification_levels"]["physical_link_verified"] is False
    assert record["verification"]["status"] in {"unverified", "fallback"}


# -------------------------------------------------------------- runtime probe ---

DESCRIPTOR = {
    "channels": [
        {"channel": "AivUbMemChannel", "execution_resource": "AIV",
         "mechanisms": ["UB_MEM"]},
    ]
}
DEVICE = {"soc": "Ascend950"}


def test_probe_reports_environment_unavailable_when_no_channel_was_acquired():
    records = runtime_probe.probe({}, channels=[
        {"case_id": "LINK-001", "channel": "AivUbMemChannel",
         "execution_resource": "AIV", "mechanisms": ["ub_memory"]}],
        descriptor=DESCRIPTOR, device=DEVICE)
    record = records[0]
    assert record.supported is False
    assert record.status in {"environment_unavailable", "dependency_missing",
                             "not_implemented"}
    stage = record.stage("channel_acquired")
    assert stage.state == "unknown"


def test_probe_distinguishes_missing_kernel_from_missing_channel():
    records = runtime_probe.probe({}, channels=[
        {"case_id": "LINK-001", "channel": "AivUbMemChannel",
         "execution_resource": "AIV", "mechanisms": ["ub_memory"]}],
        descriptor=DESCRIPTOR, device=DEVICE)
    stages = {item.stage: item.state for item in records[0].stages}
    assert stages["compiled"] == "pass"
    assert stages["channel_acquired"] == "unknown"
    assert stages["kernel_available"] in {"fail", "pass"}


def test_probe_refuses_an_uncompiled_channel():
    records = runtime_probe.probe({}, channels=[
        {"case_id": "LINK-010", "channel": "AicpuTsRoceChannel",
         "execution_resource": "AICPU_TS", "mechanisms": ["roce"]}],
        descriptor=DESCRIPTOR, device=DEVICE)
    record = records[0]
    assert record.supported is False
    assert record.stage("compiled").state == "fail"
    assert record.status == "unsupported"


def test_probe_without_a_bridge_is_a_dependency_not_a_pass():
    records = runtime_probe.probe({}, channels=[
        {"case_id": "LINK-001", "channel": "AivUbMemChannel",
         "execution_resource": "AIV", "mechanisms": ["ub_memory"]}],
        descriptor={}, device=DEVICE)
    assert records[0].status == "dependency_missing"
    assert records[0].supported is False


def test_probe_output_is_serialisable():
    records = runtime_probe.probe({}, channels=[], descriptor=DESCRIPTOR, device=DEVICE)
    payload = runtime_probe.as_dict(records)
    assert payload["stages"] == list(runtime_probe.STAGES)
    assert payload["records"] == []
    assert payload["summary"] == {}


def test_a_busy_resource_is_not_reported_as_unsupported():
    assert runtime_probe.STAGE_STATUS[("resource_available", False)] == "environment_unavailable"
    assert runtime_probe.STAGE_STATUS[("channel_acquired", False)] == "environment_unavailable"
    assert runtime_probe.STAGE_STATUS[("soc", False)] == "unsupported"
