import json
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from analysis.comparison import comparisons, dedup_comparisons, scaling
from common.config import DEFAULT, load_configs, merge, validate
from common.controller import create_manifest, execute, results
from common.registry import ROOT, registry
from common.routing import generate, summarize, summarize_local
from common.scheduler import make_plan
from common.storage import write_json
from metrics.summary import compute, latency, traffic
from profiling.analyze_overlap import analyze
from profiling.parse_profile import parse_directory
from run_benchmark import main
from schemas import result


def test_research_registry_and_directory_coverage():
    import re

    survey = (ROOT.parents[1] / "ep.md").read_text(encoding="utf-8")
    rows = [line for line in survey.splitlines() if line.startswith("| ")][1:]
    ids = set()
    for row in rows:
        column = row.split("|")[1].strip()
        match = re.match(r'(?:<span title="([a-z][a-z0-9_]+)"|\x60([a-z][a-z0-9_]+)\x60)', column)
        assert match, column
        ids.add(match.group(1) or match.group(2))
    directories = {p.parent.name for p in (ROOT / "operators").glob("*/config.yaml")}
    assert len(rows) == len(ids) == 36
    assert ids == set(registry()) == directories


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_config_files(path):
    schema = json.loads((ROOT / "schemas/config-v1.json").read_text(encoding="utf-8"))
    for cfg in load_configs(path):
        jsonschema.validate(cfg, schema)
        assert make_plan([cfg])


def test_invalid_config():
    with pytest.raises(ValueError, match="top_k"):
        validate(merge(DEFAULT, {"model": {"top_k": 999}}))
    with pytest.raises(ValueError, match="expert_placement"):
        validate(merge(DEFAULT, {"workload": {"expert_placement": [2] * 8}}))
    with pytest.raises(ValueError, match="token"):
        validate(merge(DEFAULT, {"distributed": {"ep_size": 2}, "workload": {"tokens_per_rank": [1]}}))


@pytest.mark.parametrize("strategy", ["balanced", "uniform_random", "expert_hotspot", "rank_hotspot",
    "server_hotspot", "high_locality", "cross_server_heavy", "dedup_controlled", "dynamic_trace", "real_trace_replay"])
def test_routing_placement_and_dedup(strategy, tmp_path):
    cfg = validate(merge(DEFAULT, {"distributed": {"ep_size": 4},
        "workload": {"tokens_per_rank": [16, 0, 5, 7], "routing": strategy,
                     "expert_placement": [1, 0, 3, 2, 2, 0, 3, 1], "target_rank_max_mean": 10,
                     "trace_path": str(tmp_path / "trace.json")},
        "topology": {"server_ids": ["a", "b", "a", "b"]}}))
    if strategy == "real_trace_replay":
        write_json(tmp_path / "trace.json", {"steps": [{"ranks": [
            {"topk_ids": [[0, 2]] * n, "topk_weights": [[.3, .7]] * n}
            for n in (16, 0, 5, 7)
        ]}]})
    routes = [generate(cfg, r) for r in range(4)]
    assert np.array_equal(routes[0].ids, generate(cfg, 0).ids)
    records = [summarize_local(route, idx, cfg["topology"]["server_ids"]) for idx, route in enumerate(routes)]
    summary = summarize(routes[0], records, cfg)
    assert sum(summary["expert_counts"]) == 56
    assert summary["source_destination_matrix"][1] == [0] * 4
    assert 0 <= summary["cross_server_ratio"] <= 1
    assert summary["targets"]["rank_max_mean"]["met"] is False
    volumes = traffic(summary, cfg, routes[0].tokens, "rank_dedup")
    assert volumes["rank_dedup_payload_bytes"] <= volumes["remote_expanded_bytes"]
    assert volumes["logical_bytes"] == 56 * 128 * 2


def test_rank_max_percentiles_and_flops():
    def record(values):
        return {"samples": [{"latency_ns": v * 1e6, "host_issue_ns": 1,
                             "host_wait_ns": 1, "output_ready_ns": v * 1e6}
                            for v in values]}
    summary = latency([record([1, 4, 2]), record([2, 2, 6])])
    assert summary["rank_max"] == [2, 4, 6]
    assert summary["p50"] == 4
    assert summary["p95"] == pytest.approx(5.8)
    values = compute(DEFAULT, 10, "forward_backward", 1)
    assert values["effective_flops"] == 18 * 10 * 128 * 64


def test_specialized_cases_plan_native_and_control_tasks():
    load = load_configs(ROOT / "configs/load_balance.yaml")
    tasks = make_plan(load, cases=["LAYER-003"])
    assert {task["parameters"]["variant"] for task in tasks if not task["reason"]} == {
        "fresh_plan", "cached_plan", "fixed_owner",
    }
    dedup = load_configs(ROOT / "configs/dedup.yaml")
    tasks = make_plan(dedup, cases=["COMM-010", "LAYER-005", "LAYER-006"])
    active = [(t["case_id"], t["backend_id"], t["config"]["dedup"]["mode"])
              for t in tasks if not t["reason"]]
    assert len(active) == 6
    assert ("LAYER-006", "moonep_server_dedup", "server") in active
    overlap = load_configs(ROOT / "configs/overlap.yaml")
    tasks = make_plan(overlap, cases=["FUSE-005", "FUSE-006", "FUSE-007", "MOE-007"])
    assert len([t for t in tasks if not t["reason"]]) == 11


def test_dedup_comparison_joins_only_matching_traces():
    configs = load_configs(ROOT / "configs/dedup.yaml")
    tasks = [task for task in make_plan(configs, cases=["COMM-010"]) if not task["reason"]]
    rows = [result({"run_id": "d", "environment": {}}, task, "passed") for task in tasks]
    for row, milliseconds, payload in zip(rows, (4, 2), (1024, 256)):
        row.update(latency={"p50": milliseconds}, correctness={"passed": True},
                   routing_trace_hash="same", bandwidth={"cross_server_payload_bytes": payload})
    matched, = dedup_comparisons(rows)
    assert matched["speedup"] == 2
    assert matched["modeled_payload_saving"] == .75
    rows[0]["routing_trace_hash"] = "other"
    assert dedup_comparisons(rows) == []


def test_profile_formats_and_interval_union(tmp_path):
    rank = tmp_path / "rank-2"
    rank.mkdir()
    (rank / "kernel_details.csv").write_text(
        "Kernel Name,Stream ID,Task Start Time(us),Task Duration(us)\n"
        "HcclSend,1,0,10\nMatMul,2,5,20\nMatMul,3,6,3\n", encoding="utf-8"
    )
    parsed = parse_directory(tmp_path)
    report = analyze(parsed)["2"]
    assert report["visible_overlap_ns"] == 5000
    assert report["visible_compute_ns"] == 20000
    assert report["visible_overlap_ratio"] == .5
    assert report["device_idle_ns"] == 0
    assert report["internal_overlap_ratio"] is None


def passing(task, run, rate, ep, scaling_mode, tokens, experts):
    cfg = merge(task["config"], {"device": "cpu", "scaling": {"name": "test", "mode": scaling_mode},
        "distributed": {"ep_size": ep}, "model": {"num_experts": experts}})
    row = result(run, {**task, "config": cfg}, "passed")
    row.update(latency={"p50": 1}, correctness={"passed": True},
               throughput={"global_source_tokens_per_second": rate},
               workload={"global_source_tokens": tokens, "tokens_per_rank": [tokens // ep] * ep},
               routing_trace_hash=f"trace-{ep}")
    return row


@pytest.mark.parametrize(("mode", "tokens", "experts"), [
    ("weak", (8,16), (8,16)), ("strong", (8,8), (8,8)),
])
def test_scaling_requires_compatible_groups(mode, tokens, experts):
    task = make_plan([DEFAULT], cases=["COMM-001"])[0]
    run = {"run_id": "r", "environment": {}}
    first = passing(task, run, 100, 2, mode, tokens[0], experts[0])
    second = passing(task, run, 150, 4, mode, tokens[1], experts[1])
    points = next(iter(scaling([first, second]).values()))
    assert points[1]["efficiency"] == .75
    invalid = {**second, "config": merge(second["config"], {"model": {"dtype": "float32"}})}
    assert scaling([first, invalid]) == {}
    assert len(comparisons([first, second])) == 2
    second["status"] = "correctness_failed"
    assert scaling([first, second]) == {}


def test_resume_retains_attempts(tmp_path, monkeypatch):
    task = make_plan([merge(DEFAULT, {"device": "cpu", "model": {"dtype": "float32"}})],
                     cases=["COMM-001"])[0]
    root, manifest = create_manifest(tmp_path, [task], "resume")
    launch_count = 0

    def launch(task, manifest_path, attempt, profile, args):
        nonlocal launch_count
        launch_count += 1
        attempt.mkdir(parents=True)
        if launch_count == 1:
            write_json(attempt / "rank-1-error.json", {"rank": 1, "status": "oom", "reason": "test OOM"})
            return "runtime_failed"
        record = result(manifest, task, "passed")
        record["correctness"] = {"passed": True}
        write_json(attempt / "result.json", record)
        return None

    import common.controller
    monkeypatch.setattr(common.controller, "launch_worker", launch)
    args = SimpleNamespace(nnodes=1, node_rank=0)
    assert execute(root, manifest, args) == 1
    assert list(results(root, manifest))[0]["status"] == "oom"
    assert execute(root, manifest, args, resume=True) == 0
    assert execute(root, manifest, args, resume=True) == 0
    assert launch_count == 2
    assert len(list((root / "logs").rglob("status.json"))) == 2
    schema = json.loads((ROOT / "schemas/result-v1.json").read_text(encoding="utf-8"))
    jsonschema.validate(list(results(root, manifest))[0], schema)


def test_legacy_list_and_plan(tmp_path, capsys):
    assert main(["--list-backends"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 36
    assert main(["plan", "--config", str(ROOT / "configs/weak_scaling.yaml"), "--ep-size", "128",
                 "--operators", "all", "--cases", "COMM-001", "--output", str(tmp_path),
                 "--run-id", "test-plan"]) == 0
    manifest = json.loads((tmp_path / "test-plan/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tasks"]) == 36
    assert all(t["config"]["distributed"]["ep_size"] == 128 for t in manifest["tasks"])
