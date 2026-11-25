"""Import historical ascendep records without assigning local validation evidence."""
import re

from common.storage import fingerprint, load_json
from schemas import result

REPOSITORY = "https://github.com/penn513/ascendep"
REFERENCE_CONFIG = {"ep": 8, "dtype": "bfloat16", "shape": [4096, 7168, 3072],
                    "experts": 24, "top_k": 6, "planner": "off"}


def import_results(path, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full Git SHA")
    if path.suffix.lower() == ".md":
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 6 or not re.fullmatch(r"[a-z][a-z0-9_]+", cells[0]):
                continue
            if not re.fullmatch(r"\d+\.?\d*\s*/\s*\d+\.?\d*", cells[3]):
                continue
            timing = lambda value: dict(zip(("p50", "p95"), map(float, value.split("/"))))
            entries.append({"backend": cells[0], "status": "pass", **REFERENCE_CONFIG,
                            "metrics": {key: timing(cells[index]) for index, key in
                                        enumerate(("fwd_ms", "bwd_ms", "fwd_bwd_ms"), 1)},
                            "source_tokens_per_second": float(cells[4]), "archive": cells[5]})
    else:
        body = load_json(path)
        entries = body if isinstance(body, list) else body.get("results", [body])
    records = []
    for original in entries:
        task = {"task_id": fingerprint(original)[:20], "case_id": "MOE-003", "level": "L3",
                "operation": "forward_backward", "backend_id": original.get("backend", original.get("backend_id")),
                "config": {"suite_id": "historical_ep8", "device": "npu", "topology": {}},
                "source_versions": {"source_repository": REPOSITORY, "source_commit": commit},
                "parameters": {}}
        row = result({"run_id": f"ascendep-{commit[:12]}", "environment": {}}, task)
        row.update(source="imported", source_repository=REPOSITORY, source_commit=commit,
                   measurement_scope="as reported by ascendep; see original fields",
                   original_config={k: original[k] for k in REFERENCE_CONFIG if k in original},
                   original_record=original)
        records.append(row)
    if not records:
        raise ValueError(f"no ascendep records recognized in {path}")
    return {"schema_version": 1, "source": "imported", "results": records}
