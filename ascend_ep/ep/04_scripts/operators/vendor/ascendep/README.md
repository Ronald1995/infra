# Vendored ascendep native adapters

`ascendep/ep/` is a byte-exact copy of the adapter modules that `operators/ascendep.py`
imports at runtime. A benchmark run therefore needs **no** external
`penn513/ascendep` checkout: the framework imports `ep.<operator>.demo` straight from
this directory.

| | |
|---|---|
| Upstream | <https://github.com/penn513/ascendep> |
| Commit | `24ff84d3ab8c01e1e2071802465173e624e1f89d` (pinned by `operators/ascendep.py:UPSTREAM_COMMIT`) |
| Subdirectory | `ep/` |
| Vendored | 23 files / 47 376 bytes |
| Vendored at | 2026-11-26 |
| License | **none declared upstream** — see `../THIRD_PARTY_NOTICES.md` |
| Machine-readable record | `VENDOR.json` |

`VENDOR.json` is the authority on content: it lists every vendored file with its upstream
path, byte size and `sha256`, the reason that file is needed, and every upstream file that
was deliberately left out together with the reason. Do not edit the tree by hand.

## Layout

The upstream layout is preserved, so `sys.path.insert(0, "<this directory>")` makes
`import ep.<name>.demo` resolve exactly as it does in a full checkout:

```
operators/vendor/ascendep/
├── VENDOR.json                     provenance + per-file sha256
├── README.md                       this file
└── ep/                             copy of upstream ep/
    ├── runner.py                   support module
    ├── megatron.py                 support module
    └── <operator>/demo.py          21 adapters
```

`operators/vendor/.gitattributes` marks the tree `-text`, so git stores these bytes
verbatim instead of normalising line endings (the pinned checkout uses CRLF in the working
tree). Without it the recorded digests would change on checkout and `--check` would fail.

## Why each file is here

Every vendored file is reachable from `operators/ascendep.py`; the `reason` field in
`VENDOR.json` repeats this per file.

**Support modules** — imported at module level by the adapters, so they must exist before
any adapter can be loaded:

| file | why |
|---|---|
| `ep/runner.py` | every adapter ends with `from ep.runner import run`; supplies the upstream CLI entry point plus `sample()` / `oracle()` |
| `ep/megatron.py` | `from ep.megatron import dispatch` in `ep/allgather_megatron/demo.py` and `ep/alltoall_megatron/demo.py` |

**Operator adapters** — one `demo.py` per operator the registry routes through
`AscendEPBackend`, `MC2V2Backend` or `FusedGMMBackend`. The module is
`ep.<registry upstream or operator id>.demo`:

| operator id | backend | adapter |
|---|---|---|
| `allgather_fsdpturbo` | `AscendEPBackend` | `ep/allgather_fsdpturbo/demo.py` |
| `allgather_megatron` | `AscendEPBackend` | `ep/allgather_megatron/demo.py` |
| `allgather_overlap_mindspeed` | `AscendEPBackend` | `ep/allgather_overlap_mindspeed/demo.py` |
| `alltoall_fsdpturbo` | `AscendEPBackend` | `ep/alltoall_fsdpturbo/demo.py` |
| `alltoall_megatron` | `AscendEPBackend` | `ep/alltoall_megatron/demo.py` |
| `alltoallv_gmm_cann` | `FusedGMMBackend` | `ep/alltoallv_gmm_cann/demo.py` |
| `chunkmoe_fsdpturbo` | `AscendEPBackend` | `ep/chunkmoe_fsdpturbo/demo.py` |
| `deepep_ascend` | `ExpandedDeepEPBackend` | `ep/deepep_ascend/demo.py` |
| `deepep_cann` | `AscendEPBackend` | `ep/deepep_cann/demo.py` |
| `deepep_lueelu` | `ExpandedDeepEPBackend` | `ep/deepep_lueelu/demo.py` |
| `deepep_sglang` | `AscendEPBackend` | `ep/deepep_sglang/demo.py` |
| `domino_fsdpturbo` | `AscendEPBackend` | `ep/domino_fsdpturbo/demo.py` |
| `ep_gmm_fused_lueelu` | `FusedGMMBackend` | `ep/ep_gmm_fused_lueelu/demo.py` |
| `fused_fsdpturbo` | `AscendEPBackend` | `ep/fused_fsdpturbo/demo.py` |
| `mc2_dispatch_cann` | `MC2V2Backend` | `ep/mc2_dispatch_cann/demo.py` |
| `mc2_fsdpturbo` | `AscendEPBackend` | `ep/mc2_fsdpturbo/demo.py` |
| `megamoe_fsdpturbo` | `AscendEPBackend` | `ep/megamoe_fsdpturbo/demo.py` |
| `megamoe_lueelu` | `AscendEPBackend` | `ep/megamoe_lueelu/demo.py` |
| `megamoe_td` | `AscendEPBackend` | `ep/megamoe_td/demo.py` |
| `moonep` | `AscendEPBackend` | `ep/moonep/demo.py` |

`deepep_direct_lueelu` has no `demo.py` of its own: the registry sets
`upstream: deepep_ascend`, so it loads `ep/deepep_ascend/demo.py`.

**Transitive adapter** — not an ascendep-backed operator, but imported by two adapters at
module level, so it has to be present or those operators cannot initialise:

| file | why |
|---|---|
| `ep/allgather_hccl/demo.py` | `from ep.allgather_hccl.demo import Gather` in `ep/deepep_sglang/demo.py` and `ep/mc2_dispatch_cann/demo.py` |

## What is NOT included

Nothing else from upstream is copied. `VENDOR.json` → `excluded` records all 70 remaining
files with a reason; the groups are:

* **Upstream CLI and harness** — `ascendep/` (`python -m ascendep`), `ep/suite.py`,
  `ep/report_ep2.py`, `ep/report_profile.py`, `ep/archive_profile.py`,
  `ep/backends.json`. No module under `ep/` imports them; infra replaces this layer with
  `common/` and `run_benchmark.py`.
* **`ep/alltoall_hccl/demo.py`** — reachable only through the lazy
  `from ep.alltoall_hccl.demo import forward as baseline` inside `ep.runner.run()`, i.e.
  upstream's own torchrun CLI. `operators/ascendep.py` imports the adapter modules
  directly and never calls `run()`, and infra's `alltoall_hccl` operator is a
  `CollectiveBackend` that does not touch ascendep at all.
* **Adapters infra implements natively** — `ep/megamoe_cann/demo.py` (`CANNMegaMoEBackend`)
  and `ep/hierarchical_alltoallv_mindspeed/demo.py` (`MindSpeedHierarchicalBackend`).
* **Tests, configs, cluster scripts, examples, docs, reports, READMEs, and the SHMEM
  template build helper** — none of them is importable code needed at runtime.
* **Every native library** — see the next section. Native code is not vendored here.

## Runtime dependencies that are still native

The vendored Python is only a thin adapter: every `forward`/`dispatch` call lands in a
native extension or an external framework. Those are **not** vendored; they are managed by
the dependency manifest owned by another workstream
(`dependencies/manifest.yaml`) and installed through `scripts/doctor.py`. A missing one
surfaces from `AscendEPBackend.initialize` as `BenchmarkUnavailable("dependency_missing",
"ascendep adapter <module> needs the native module <name> ...")`.

| package | required by (vendored adapter) | import site |
|---|---|---|
| DeepEP (`deep_ep`) | `deepep_ascend`, `deepep_lueelu`, `deepep_sglang` | `deepep_lueelu` at module level; the other two inside `forward()` |
| DeepEP experimental (`deep_ep_experimental`) | `ep_gmm_fused_lueelu`, `megamoe_lueelu` | module level |
| SHMEM (`shmem`, `_moe_testkit.init_aclshmem`) | `megamoe_td`, and `moonep` through `npu_patch` | inside `forward()` |
| `npu_patch` | `moonep` | inside `forward()` |
| `fsdp_turbo` | `allgather_fsdpturbo`, `chunkmoe_fsdpturbo`, `domino_fsdpturbo`, `megamoe_fsdpturbo`, `alltoall_fsdpturbo`, `fused_fsdpturbo`, `mc2_fsdpturbo` | module level for the first four, inside `forward()` for the last three |
| `mega_moe` | `megamoe_td` | inside `forward()` |
| `mindspeed` | `allgather_overlap_mindspeed` | module level |
| `megatron` | `allgather_megatron`, `alltoall_megatron`, `allgather_overlap_mindspeed` | module level |
| `cann_ops_transformer` | `deepep_cann` | inside `forward()` |
| `torch_npu` | `allgather_megatron`, `alltoall_megatron` (`torch_npu.contrib.transfer_to_npu`), `alltoallv_gmm_cann`, `mc2_dispatch_cann`, and every NPU run | module level for those four |

The adapters are imported lazily from `AscendEPBackend.initialize`, so importing
`operators.ascendep` itself stays CPU-only and never pulls in `torch_npu` or any native
library.

## Checking and refreshing

```sh
python scripts/vendor_ascendep.py --check     # exit 0 only if every digest matches
python scripts/vendor_ascendep.py --list      # print the recorded manifest
python scripts/vendor_ascendep.py --refresh --source /path/to/ascendep
```

`--check` recomputes `sha256` for every vendored file against `VENDOR.json` and fails,
non-zero, on a mismatch, a missing file, an unlisted extra file, or a manifest whose
pinned commit no longer matches `operators/ascendep.py:UPSTREAM_COMMIT`. It is the
integrity gate for this tree; `operators/ascendep.py` only verifies the recorded commit and
that the manifest describes files that actually exist, so the digests are not recomputed on
every benchmark start.

`--refresh --source <dir>` re-copies the selected files from a git checkout and rewrites
`VENDOR.json`. It **refuses** a checkout whose `HEAD` is not the pinned commit unless
`--allow-other-commit` is passed, and it refuses if a newly selected file is missing from
the checkout or if an upstream file has no vendoring decision recorded. Files that are no
longer selected are pruned. To vendor a new upstream commit, update
`operators/ascendep.py:UPSTREAM_COMMIT` first, then refresh; `--check` fails until the two
agree.

The selection itself lives in `scripts/vendor_ascendep.py` (`OPERATORS`, `TRANSITIVE`,
`EXCLUSIONS`), next to the reason recorded for each file, so refreshing is reproducible
rather than a manual copy.

## Using an external checkout instead

`EP_ASCENDEP_ROOT=/path/to/ascendep` overrides this copy for development. The override is
validated exactly like the vendored tree — `ep/runner.py` must exist, the checkout must be
clean under `ep/`, and `HEAD` must be the pinned commit — and it is reported as a separate
origin so a result can never be mistaken for a vendored run:

```python
from operators.ascendep import source_info

source_info()
# {'origin': 'vendored',
#  'commit': '24ff84d3ab8c01e1e2071802465173e624e1f89d',
#  'pinned': '24ff84d3ab8c01e1e2071802465173e624e1f89d',
#  'path': '.../operators/vendor/ascendep',
#  'upstream': 'https://github.com/penn513/ascendep',
#  'manifest': '.../operators/vendor/ascendep/VENDOR.json'}

# with EP_ASCENDEP_ROOT set, 'origin' becomes 'override', 'manifest' becomes None and
# 'path' is the external checkout.
```

## Licensing

Upstream declares no license file at the pinned commit. Read
`operators/vendor/THIRD_PARTY_NOTICES.md` before redistributing this tree; an owner
sign-off is required. That is a compliance item, not a code issue.
