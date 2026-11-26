# Dependency policy

`dependencies/manifest.yaml` is the only source of truth. The workflow adds policy, not a
second catalog.

## What the workflow may do

| Action | Allowed automatically? |
|---|---|
| read-only checks (`doctor`, `--dry-run`, `--check-only`) | yes |
| install a `pip` / `python_import` dependency listed as `installable: true` | only with `environment.dependency_policy: install_missing` **and** (`allow_network: true` or a `wheelhouse`) |
| build a `native_build` dependency whose pinned checkout is already present | yes, when the toolchain is complete |
| download a source checkout | no: the pinned checkout must already exist, otherwise the item is reported as blocked with its `install_reason` |
| install a system package (`apt`/`dnf`/`zypper`) | no: needs `allow_sudo`, which is off unless stated |
| install/upgrade driver, firmware, CANN, device nodes | **never** — host-provided, with a `manual_step` |
| uninstall or downgrade an existing package | **never** |

A dependency that only blocks some operators blocks those operators. The rest of the
matrix still runs; the blocked tasks keep `dependency_missing` and the report names them.

## Classification

`scripts/workflow/dependency.py::classify` splits the doctor's failures:

| Bucket | Meaning | Effect |
|---|---|---|
| `ok` | present (or optionally missing) | none |
| `installable` | a wheel/module the policy permits installing | candidate for `dependency_install` |
| `buildable` | a source build whose checkout exists | candidate for a build, not a download |
| `host` | `kind: host_provided`, `installable: false`, or a system action | blocked, with `manual_step` |
| `unsatisfied` | reported by the doctor but absent from the catalog | treated as blocked; a catalog bug to fix |

## Gate

`dependency_install` passes only after the doctor has been **re-run** and the installable
set is empty. The installer's exit code alone is not evidence: `install_deps.sh` can
succeed while a pin is still unsatisfied.

If the request allowed installation but not the network, the phase returns `blocked` with
that exact reason instead of silently doing nothing. That distinction is what makes
"缺少依赖就安装" honest on an offline host.
