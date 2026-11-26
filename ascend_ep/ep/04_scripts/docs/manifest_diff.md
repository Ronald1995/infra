# Manifest diff

`scripts/manifest_diff.py` compares two manifests and classifies every difference, so a
refactor can be proved not to have moved the benchmark matrix — or to have moved it in
exactly the documented way.

## 1. Usage

```sh
python scripts/manifest_diff.py OLD_MANIFEST NEW_MANIFEST
python scripts/manifest_diff.py OLD NEW --json reports/manifest_diff.json
python scripts/manifest_diff.py OLD NEW --allow-executable-change
```

Both arguments are `manifest.json` files as written by `run_benchmark.py plan` (a JSON
object with a `tasks` list, or a bare task list). `--json` writes the structured result;
`--allow-executable-change` downgrades a changed executable set to a pass.

Exit codes: `0` when the executable point set is identical (or the flag is given), `1`
otherwise. The gate is deliberately about *executable* points only, so an intentional,
documented status change does not fail a refactor that changed no measured work.

## 2. What one point is

`task_signature` keeps the identity a reviewer cares about:

```
(ep_size, case_id, level, operator, variant, message_bytes)
```

A task's `task_id` is a fingerprint of the effective config, the operator, the case, the
parameters and the EP size (`common/scheduler.py::_identity`); the diff ignores it and uses
the signature above, so a task that was re-expanded from the same point still matches.

## 3. Classifications

| Label in the output | Field | Meaning |
|---|---|---|
| `identical` | (not reported) | same signature and same `status`/`reason` |
| `status` | `status_changes` | the same point now resolves to a different status or reason — a capability or rule change |
| `parameter` | `parameter_changes` | the same `(case, operator, variant)` now carries different parameters |
| `scope` | `scope.removed` / `scope.added` | a point exists in one manifest and not the other — a real coverage change |
| `count` | `count_changes` | the same signature appears a different number of times — a duplicate/expansion change |
| `executable` | `executable` | the set of points whose status is in `("not_run", "passed")`, compared as a set |

`per_ep` reports the task count per EP size; `totals` reports raw task counts and distinct
points.

## 4. The refactor's own numbers

Before: the manifest pair recorded during the refactor (old `common/cases.py` /
`common/variants.py` / `common/scheduler.py` / `benchmarks/operator_cases.py`).
After: this tree, `configs/benchmark_plan.yaml` with EP8/EP64/EP128 (`mode: full_matrix`,
16 suites). The structured diff of that pair is kept at
`docs/manifest_diff_refactor.json`; the figures below are read from it.

| Metric | Before | After |
|---|---|---|
| Tasks at EP8 | 6159 | 6753 |
| Tasks at EP64 | 6145 | 6753 |
| Tasks at EP128 | 6145 | 6753 |
| Tasks, total | 18449 | 20259 |
| Distinct executable points | 719 | 719 |

The executable *count* is unchanged, but the executable *set* is not identical, and the tool
says so (`executable.identical: false`). The difference is exactly the label fix of §5.3:
nine points are removed with variant `reduce_scatter` (3 message sizes x 3 EP sizes) and nine
are added with `reducescatter`. No other executable point appears or disappears. A strict run
of `manifest_diff.py` therefore exits 1 for this pair, which is the intended behaviour for a
documented executable-set change.

Reproduce the after column (writes a run directory):

```sh
python run_benchmark.py plan --plan configs/benchmark_plan.yaml
# EP8: 6753 tasks, EP64: 6753 tasks, EP128: 6753 tasks
```

## 5. What each difference is

### 5.1 Parameters are expanded before the capability verdict

Before, an unsupported point collapsed to a single parameter set, which made EP8 produce
more tasks than EP64 (6159 vs 6145) for the same case set. After,
`common/variants.py::task_parameters` never consults a status and
`common/scheduler.py::make_plan` attaches the verdict afterwards. The EP sizes now agree,
and `common/scheduler.py::check_parity` raises when a future change breaks that.

This is what moves `FUSE-005` (3 variants), `FUSE-006` (3) and `FUSE-007` (4) from a
collapsed expansion to their full one even while their status is `unsupported`. Verified in
this tree: each of the three cases has all of its variants at EP8, EP64 and EP128.

### 5.2 The fixed-owner variant is declared on the case

`LAYER-003`, `LAYER-005` and `LAYER-006` name `owner: alltoall_hccl` on the `CaseSpec`
(`_CASE_OWNER`). `common/variants.py::operator_variants` emits one extra parameter set with
`variant: "fixed_owner"` for that adapter, and the scheduler relaxes the scenario contract
for it (`common/scheduler.py::_contract`) because the owning adapter implements only the
plain MoE path. Before, `LAYER-005`/`LAYER-006` reported `default` where they should report
`fixed_owner`; that is a `parameter`/`scope` diff, not a status change.

### 5.3 One deliberate label fix: HW-006

`HW-006` is the framework collective case with two variants. Its variants are now the
canonical operation names: `("allgather", "reducescatter")`.

Before, the second variant was named `reduce_scatter`. The code always ran ReduceScatter
for any variant that was not `allgather`
(`baselines/l0/collectives/torch_hccl.py::_collective`, final branch), but the *request
block* named the operation through `_operation_name`, which only recognised
`("allgather", "reducescatter")`. A `reduce_scatter` variant therefore ran ReduceScatter and
recorded `allgather` in `requested.operation`. Renaming the variant to `reducescatter` makes
the label and the code path agree; `validate_catalog` and
`tests/test_case_spec.py::test_variant_owners_are_declared_once` pin the new tokens.

This is the only executable-set change in the refactor: nine points are renamed and no other
executable point is added or removed.

## 6. Reading a diff

```
tasks: 18449 -> 20259
distinct points: ... -> ...
tasks per EP size: {8: 6159, 64: 6145, 128: 6145} -> {8: 6753, 64: 6753, 128: 6753}
executable points: 719 -> 719 (count equal; set differs by the nine renamed points)
status changes: N
parameter changes: N
count changes: N
scope removed: N ({...})
scope added:   N ({...})
```

Order of review:

1. `executable.identical` — must be `true` unless the change is meant to alter the measured
   work;
2. `executable.removed` / `added` — each entry needs a written justification;
3. `status_changes` — a status change is either a capability fix or a regression; the reason
   string tells which;
4. `parameter_changes` — the only legitimate ones are a declared label fix (§5.3);
5. `scope` / `count` — coverage changes; they should match the intended variant expansion.

## 7. What the tool does not prove

- It compares manifests, not results: it cannot show that a case still measures the same
  code path, only that the same point is planned.
- It does not check `reason` text quality, only equality.
- It does not detect a change in a *handler*: two manifests generated by different handler
  bodies with identical parameters and statuses are `identical` here. The handler-level
  guarantees are covered by `tests/test_case_spec.py` and the case tests, not by this tool.
- Within one signature, tasks are paired by sorted `task_id`
  (`scripts/manifest_diff.py::compare`). When a signature's task count changes, the pairing
  can attribute an unrelated parameter set to a point, so `parameter_changes` must be read
  together with `count_changes` and the raw `parameters` in the JSON output before any change
  is accepted as real.
