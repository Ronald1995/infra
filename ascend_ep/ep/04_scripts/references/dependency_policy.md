# Dependency policy

`dependencies/manifest.yaml` is the only source of truth. The workflow adds policy, not a
second catalog, and never guesses a URL, a commit or a build command.

## The ownership model decides what automation may touch

| Ownership | Example | Workflow action |
|---|---|---|
| `in_repo` | `baselines/l0/*/native`, `kernels/asc_comm` | build it here; never fetch it |
| `vendored` | `operators/vendor/ascendep` (23 files, hash-verified) | verify it (`scripts/vendor_ascendep.py --check`); never fetch it |
| `external` | torch, torch_npu, SHMEM, DeepEP, MoonEP, ops-transformer, FSDPTurbo, Megatron, MindSpeed, DeepGEMM, vllm-ascend, MPI | install only from the declared pin |
| `host` | driver, firmware, CANN, HCCL/HCOMM, npu-smi, device nodes, deployment env | **never**; report with `manual_step` |

`EP_ASCENDEP_ROOT` is a development override for adapter work, not a dependency. A run
records `origin: vendored`; nothing in the install path reads the variable.

## What the workflow may do

| Action | Allowed automatically? |
|---|---|
| read-only checks (`doctor`, `dependency_report`, `--dry-run`, `--check-only`) | yes |
| install a `pip` dependency listed as `installable: true` | only with `environment.dependency_policy: install_missing` **and** (`allow_network: true` or a `wheelhouse`) |
| fetch + build a pinned `source` dependency | only with `allow_network: true`; the installer clones the declared URL at the declared commit/tag and verifies `HEAD` |
| reuse a pinned checkout already in `$EP_SOURCE_CACHE` | yes |
| fetch anything with `--offline` | **never**: a missing or drifted checkout is `BLOCKED` |
| install a system package (mode `system`) | needs `allow_sudo`, off unless stated |
| install/upgrade driver, firmware, CANN, device nodes | **never** — host-provided, with a `manual_step` |
| follow a branch head, `latest`, or an unpinned clone | **never** — an unpinned entry is not installable at all |
| uninstall, downgrade or `--clean` an existing package/checkout | **never** |

The installer is invoked with the authorization the request granted, so nothing happens
behind the request's back:

```sh
bash scripts/install_deps.sh --profile all --allow-network          # allow_network: true
bash scripts/install_deps.sh --profile all --offline \
    --source-cache "$EP_SOURCE_CACHE"                                # allow_network: false
```

A dependency that only blocks some operators blocks those operators. The rest of the matrix
still runs; the blocked tasks keep `dependency_missing` and the report names them.

## Classification

`scripts/workflow/dependency.py::classify` splits the doctor's failures by `install_spec.mode`
(the same word the installer reads):

| Bucket | Meaning | Effect |
|---|---|---|
| `ok` | present (or optionally missing) | none |
| `installable` | `mode: pip`, permitted by the policy | candidate for `dependency_install` |
| `buildable` | `mode: source`, pinned; the checkout may or may not be cached yet | candidate for fetch+build |
| `host` | `ownership: host/in_repo/vendored`, `mode: manual/none/system`, or `installable: false` | blocked, with `manual_step` |
| `unsatisfied` | reported by the doctor but absent from the catalog | treated as blocked; a catalog bug to fix |

Each record keeps `ownership`, `environment`, `install_mode`, `pin`, `install_reason` and
`manual_step`, so the report can say *who* owns the gap, not only that it exists.

## Environments

Two families install the same module name, so the environment is part of the operator
contract, not an installation detail:

| Environment | Provides | Conflicts with |
|---|---|---|
| `base` | framework, CANN extensions, SHMEM, FSDPTurbo, MegaMoE, Megatron, MindSpeed, DeepGEMM, vllm-ascend, vendored ascendep | -- |
| `deep_ep_lueelu` | `deep_ep`, `deep_ep_experimental` (lueelu/DeepEP @83252a5) | `deep_ep_v2` |
| `deep_ep_v2` | `deep_ep` (DeepEP_9283 @69090a1) | `deep_ep_lueelu` |
| `moonep_npu` | `moonep`, `npu_patch` (MoonEP `dev_npu` @b4127ba) | `moonep_dedup` |
| `moonep_dedup` | `moonep`, `npu_patch` (MoonEP `dev_server_dedup` @d8495c4) | `moonep_npu` |

`dependencies/environments/map.yaml` is the operator -> environment map, surfaced as
`Operator.environment`. The doctor's `environment.isolation` check fails when two
conflicting families are importable at once; a run in the wrong environment is a wrong
measurement, not a warning.

## Gate

`dependency_install` passes only after the doctor has been **re-run** and the installable
set is empty. The installer's exit code alone is not evidence: `install_deps.sh` can succeed
while a pin is still unsatisfied (a build that linked against the wrong CANN, a wheel from
the wrong index).

If the request allowed installation but not the network, the phase returns `blocked` with
that exact reason instead of silently doing nothing. That distinction is what makes
"缺少依赖就安装" honest on an offline host.
