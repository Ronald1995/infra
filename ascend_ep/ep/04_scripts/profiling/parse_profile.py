"""Normalize CANN CSV and Chrome trace exports with explicit time units."""
import csv
from dataclasses import dataclass, asdict
import gzip
import json
from pathlib import Path
import re

from profiling.kernel_classifier import classify


@dataclass(frozen=True)
class Event:
    rank: int
    stream: str
    kernel_name: str
    kernel_category: str
    start_ns: int
    duration_ns: int
    correlation_id: str | None

    def as_dict(self):
        return asdict(self)


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold().replace("μ", "u").replace("µ", "u"))


def parse_csv(path, rank=0):
    with path.open(encoding="utf-8-sig", newline="") as data:
        for row in csv.DictReader(data):
            fields = {normalized(k): v for k, v in row.items() if k and v}
            name = next((fields[key] for key in ("name", "opname", "kernelname", "taskname", "type")
                         if key in fields), None)
            start, duration = None, None
            for key, value in fields.items():
                unit = next((factor for unit, factor in (("ns", 1), ("us", 1000), ("ms", 1_000_000))
                             if key.endswith(unit)), None)
                if unit is None:
                    continue
                try:
                    ns = round(float(value.replace(",", "")) * unit)
                except ValueError:
                    continue
                if key.startswith(("start", "taskstart", "opstart")):
                    start = ns
                if key.startswith(("duration", "taskduration", "opduration")):
                    duration = ns
            if name is None or start is None or duration is None or duration <= 0:
                continue
            yield Event(rank, fields.get("streamid", fields.get("stream", "")), name,
                        classify(name, fields.get("tasktype", "")), start, duration,
                        fields.get("correlationid"))


def parse_trace(path, rank=0):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt" if path.suffix == ".gz" else "r", encoding="utf-8-sig") as source:
        body = json.load(source)
    items = body if isinstance(body, list) else body.get("traceEvents", [])
    for item in items:
        if item.get("ph") != "X" or not item.get("dur") or "ts" not in item:
            continue
        category = str(item.get("cat") or "")
        name = str(item.get("name") or "")
        args = item.get("args") or {}
        lower = category.casefold()
        category_id = ("host" if lower in {"cpu_op", "user_annotation", "python_function", "runtime"}
                       or name.startswith("ep_") else classify(name, category))
        if not category and not args.get("Task Type"):
            continue
        yield Event(rank, str(args.get("Stream Id", args.get("stream", item.get("tid", "")))),
                    name, category_id, round(float(item["ts"]) * 1000),
                    round(float(item["dur"]) * 1000),
                    str(args.get("correlation", args.get("Correlation ID", ""))) or None)


def parse_directory(directory):
    directory = Path(directory)
    output = []
    for rank_folder in sorted(directory.glob("rank-*")):
        if not rank_folder.is_dir():
            continue
        rank = int(rank_folder.name.split("-")[-1])
        traces = sorted(rank_folder.rglob("trace_view.json")) or sorted(rank_folder.rglob("*.pt.trace.json*"))
        if traces:
            output.extend(event for trace in traces for event in parse_trace(trace, rank))
        else:
            for pattern in ("kernel_details*.csv", "operator_details*.csv", "op_summary*.csv"):
                files = sorted(rank_folder.rglob(pattern))
                if files:
                    output.extend(event for file in files for event in parse_csv(file, rank))
                    break
    return output
