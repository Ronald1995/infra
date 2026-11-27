"""Generate every capability statement from the registry and the case catalog.

A README that repeats the capability matrix by hand drifts the moment a case or an
adapter changes.  This module is the single generator: it asks the scheduler what
each (operator, case) point actually is, then renders

* the per-operator ``## Benchmark contract`` bullets,
* the L1 / L2-serial / L2-native / L3 / L4 summary matrix,
* the full operator x case support table and the unsupported-reason table.

Nothing here decides support.  ``scheduler.status_for`` does; this only formats its
answer, so a README cannot disagree with the plan.
"""
from pathlib import Path

from common import scheduler
from common.case_spec import catalog
from common.registry import registry

ROOT = Path(__file__).resolve().parents[1]

#: The configuration the documentation describes.  EP8 BF16 is the reference point
#: the READMEs already quote, so a reader can reproduce any row with
#: `python run_benchmark.py --config configs/ep8_bf16.yaml`.
REFERENCE_CONFIG = ROOT / "configs" / "ep8_bf16.yaml"

#: Statuses that mean "this point will be measured".
EXECUTABLE = ("not_run", "conditional", "executed", "running", "completed", "passed",
              "failed")

#: Documentation only: when a case is `not_applicable` purely because the reference
#: configuration does not select its scenario, re-evaluate it under a configuration
#: that does and report it as *conditional*.  This is what stops a dedup case from
#: looking unsupported on the operator that exists to run it.
CONDITION_PATCHES = {
    "dedup_mode": {"dedup": {"mode": "rank"}},
    "layer_dedup_mode": None,                     # resolved per case below
    "planner_enabled": {"load_balance": {"enabled": True}},
    "trace_replay_workload": {"workload": {"routing": "real_trace_replay"}},
}

_DEDUP_LAYER = {"rank_dedup_layer": "rank", "server_dedup_layer": "server"}


def _patch_for(case, spec):
    for name in case.applicability:
        if name == "layer_dedup_mode":
            layer = _DEDUP_LAYER.get(case.operation)
            return {"dedup": {"mode": layer}} if layer else None
        if name == "dedup_mode":
            # Re-evaluate under the mode this adapter actually implements, so a
            # server-dedup operator is documented against server dedup.
            capabilities = set(spec.capabilities)
            for mode in ("server", "rank"):
                if f"{mode}_dedup" in capabilities:
                    return {"dedup": {"mode": mode}}
            return {"dedup": {"mode": "rank"}}
        if name in CONDITION_PATCHES and CONDITION_PATCHES[name]:
            return CONDITION_PATCHES[name]
    return None


def _configs():
    from common.config import load_configs

    return load_configs(REFERENCE_CONFIG)


def _merge(base, patch):
    merged = dict(base)
    for key, value in patch.items():
        current = merged.get(key)
        merged[key] = {**current, **value} if isinstance(current, dict) else value
    return merged


def plan_for(operator, configs=None):
    """Every case for one operator, as ``{case_id: (status, reason)}``."""
    configs = configs or _configs()
    base = configs[0]
    spec = registry()[operator]
    planned = {}
    for case_id, case in catalog().items():
        if case.alias_of is not None:
            continue
        cfg = scheduler.prepare_case(base, case)
        status, reason = scheduler.status_for(spec, case, cfg)
        if status == "not_applicable":
            patch = _patch_for(case, spec)
            if patch:
                patched = scheduler.prepare_case(_merge(base, patch), case)
                retry, retry_reason = scheduler.status_for(spec, case, patched)
                if retry in EXECUTABLE:
                    status, reason = "conditional", retry_reason or (
                        "only when " + ", ".join(f"{key}={value}"
                                                 for key, value in patch.items()))
        planned[case_id] = (status, reason)
    return planned


def plan_all(configs=None):
    """``{operator: {case_id: (status, reason)}}`` for every registered operator."""
    configs = configs or _configs()
    return {operator: plan_for(operator, configs) for operator in registry()}


BUCKETS = (
    ("L1", "L1 独立通信", lambda case: case.level == "L1"),
    ("L2S", "L2 串行组合", lambda case: case.level == "L2" and case.baseline_reference),
    ("L2N", "L2 原生融合", lambda case: case.level == "L2" and not case.baseline_reference),
    ("L3", "L3 完整 MoE", lambda case: case.level == "L3"),
    ("L4", "L4 路由器/规划/去重", lambda case: case.level == "L4"),
)

_ORDER = ("HW", "LINK", "EXEC", "MEM", "OVERLAP", "COMM", "FUSE", "MOE", "LAYER")


def _case_order(case_id):
    prefix, _, number = case_id.partition("-")
    return (_ORDER.index(prefix) if prefix in _ORDER else 99,
            int(number) if number.isdigit() else 0)


def _bucket_cases(bucket):
    return [case_id for case_id in sorted(catalog(), key=_case_order)
            if bucket(catalog()[case_id]) and catalog()[case_id].alias_of is None]


def runnable(planned, bucket):
    return [case_id for case_id in _bucket_cases(bucket)
            if planned.get(case_id, ("unsupported", None))[0] in EXECUTABLE
            and planned.get(case_id, ("", None))[0] != "conditional"]


def conditional(planned, bucket):
    return [case_id for case_id in _bucket_cases(bucket)
            if planned.get(case_id, ("", None))[0] == "conditional"]


def blocked(planned, bucket):
    return [(case_id, planned.get(case_id, ("unsupported", "not planned"))[1])
            for case_id in _bucket_cases(bucket)
            if planned.get(case_id, ("unsupported", None))[0] not in EXECUTABLE]


def summary_row(operator, planned):
    row = {"operator": operator}
    for key, _, bucket in BUCKETS:
        row[key] = runnable(planned, bucket)
        row[key + "_conditional"] = conditional(planned, bucket)
    return row


def _levels(case_ids):
    """``COMM-001, COMM-003`` or ``COMM-001..003`` only for a real run of three."""
    if not case_ids:
        return "—"
    groups = {}
    for case_id in case_ids:
        prefix, _, number = case_id.partition("-")
        groups.setdefault(prefix, []).append(int(number))
    parts = []
    for prefix in sorted(groups, key=lambda item: _ORDER.index(item) if item in _ORDER else 99):
        numbers = sorted(groups[prefix])
        run = [numbers[0]]
        pieces = []
        for number in numbers[1:]:
            if number == run[-1] + 1:
                run.append(number)
            else:
                pieces.append(run)
                run = [number]
        pieces.append(run)
        for piece in pieces:
            if len(piece) >= 3:
                parts.append(f"{prefix}-{piece[0]:03d}..{piece[-1]:03d}")
            else:
                parts.extend(f"{prefix}-{number:03d}" for number in piece)
    return ", ".join(parts)


def contract_bullets(operator, spec, planned):
    """The generated part of an operator README's benchmark contract."""
    bullets = []
    can = {key: runnable(planned, bucket) for key, _, bucket in BUCKETS}
    lines = [f"{label} {_levels(can[key])}" for key, label, _ in BUCKETS if can[key]]
    bullets.append("- **Cases it can run** (capability-level): "
                   + ("; ".join(lines) if lines else "none") + ".")

    cond = []
    for key, label, bucket in BUCKETS:
        for case_id in conditional(planned, bucket):
            reason = planned.get(case_id, (None, None))[1]
            cond.append(f"{case_id} ({label}): {reason}" if reason else f"{case_id} ({label})")
    if cond:
        bullets.append("- **Conditional cases** (runnable only under a matching "
                       "configuration): " + "; ".join(cond) + ".")

    gaps = []
    for key, label, bucket in BUCKETS:
        entries = blocked(planned, bucket)
        if entries:
            gaps.append(f"{label} " + ", ".join(case_id for case_id, _ in entries))
    bullets.append("- **Cases it cannot run**: " + ("; ".join(gaps) if gaps else "none") + ".")

    reasons = sorted({reason for key, _, bucket in BUCKETS
                      for _, reason in blocked(planned, bucket) if reason})
    if reasons:
        bullets.append("- **Why not**: " + "; ".join(reasons[:6])
                       + (" …" if len(reasons) > 6 else "") + ".")
    return bullets


#: Labels whose bullet is generated, and therefore replaced on every run.
GENERATED_LABELS = ("**Cases it can run**", "**Cases it cannot run**",
                    "**Conditional cases**", "**Why not**")


def contract_section(operator, spec, planned):
    return ["## Benchmark contract", ""] + contract_bullets(operator, spec, planned) + [""]


def splice_contract(text, section_lines):
    """Replace the generated bullets of ``## Benchmark contract``, keep the rest.

    Operator-specific prose (correctness, timing, registry constraints, source) is
    hand-written and stays; only the capability statements are generated.
    """
    lines = text.splitlines()
    try:
        start = lines.index("## Benchmark contract")
    except ValueError:
        return text.rstrip() + "\n\n" + "\n".join(section_lines) + "\n"
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    kept = [line for line in lines[start + 1:end]
            if not (line.startswith("- ") and any(label in line for label in GENERATED_LABELS))]
    block = section_lines[1:]                     # drop the heading, it already exists
    while block and not block[-1].strip():
        block.pop()
    while kept and not kept[0].strip():
        kept.pop(0)
    merged = block + [""] + kept
    return "\n".join(lines[:start + 1] + merged + lines[end:]).rstrip() + "\n"


def render_matrix(plans):
    """The operator x level summary table."""
    lines = ["| Operator | L1 | L2 Serial | L2 Native | L3 | L4 |",
             "|---|---|---|---|---|---|"]
    for operator in sorted(plans):
        row = summary_row(operator, plans[operator])
        cells = []
        for key in ("L1", "L2S", "L2N", "L3", "L4"):
            cell = _levels(row[key])
            if row[key + "_conditional"]:
                extra = _levels(row[key + "_conditional"])
                cell = f"{cell}（条件：{extra}）" if cell != "—" else f"（条件：{extra}）"
            cells.append(cell)
        lines.append(f"| `{operator}` | " + " | ".join(cells) + " |")
    return lines


def render_case_matrix(plans):
    """The full operator x case table with the support verdict."""
    lines = ["| Operator | Case | Level | Requires | Setup | Conditional | Support |",
             "|---|---|---|---|---|---|---|"]
    cases = catalog()
    for operator in sorted(plans):
        planned = plans[operator]
        for case_id in sorted(cases, key=_case_order):
            case = cases[case_id]
            if case.alias_of is not None or case.level == "L0":
                continue
            status, reason = planned.get(case_id, ("unsupported", "not planned"))
            support = ("yes" if status in EXECUTABLE and status != "conditional"
                       else "conditional" if status == "conditional" else
                       "no" if status == "unsupported" else status)
            requires = ", ".join(case.requires) or "—"
            setup = ", ".join(case.setup_requires) or "—"
            conditional = "; ".join(f"{cond} → {cap}"
                                    for cond, cap in case.conditional_requires) or "—"
            detail = support + (f"（{reason}）" if reason and support != "yes" else "")
            lines.append(f"| `{operator}` | {case_id} | {case.level} | {requires} | "
                         f"{setup} | {conditional} | {detail} |")
    return lines


def render_unsupported(plans):
    lines = ["| Operator | Case | Level | Reason |", "|---|---|---|---|"]
    cases = catalog()
    for operator in sorted(plans):
        for case_id in sorted(plans[operator], key=_case_order):
            case = cases.get(case_id)
            if case is None or case.level == "L0":
                continue
            status, reason = plans[operator][case_id]
            if status in EXECUTABLE:
                continue
            lines.append(f"| `{operator}` | {case_id} | {case.level} | {reason or '—'} |")
    return lines
