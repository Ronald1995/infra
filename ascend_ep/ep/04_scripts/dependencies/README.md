# Third-party dependencies

This directory is the **single machine-readable catalog** of everything the EP
benchmark needs from outside this repository, plus the descriptors and the tools that turn
it into an executable installation plan.

**The rule this catalog exists to enforce:** `infra/ascend_ep` is an independent,
reproducible, deliverable benchmark project. Another repository may be a technical
reference, but it is never an implicit runtime dependency. Code that must be reused is
vendored here, reviewed, pinned and hash-verified; everything else is a *declared* external
dependency with an exact pin and an installation path, or explicit host state.

```
dependencies/
  manifest.yaml       the catalog: every dependency's owner, pin, install recipe,
                      verification, environment and conflicts (schema version 2)
  python/*.txt        pinned requirement files (base / npu / deep_ep / shmem / profiling)
  system/*.yaml       per-host-dependency descriptors: how to detect, how to install,
                      which environment variables matter, what evidence is collected
  environments/       the isolated dependency environments and the operator -> environment map
  README.md           this file
scripts/
  dependency_plan.py  manifest -> ordered fetch/build/install/verify plan (+ the bash to run)
  dependency_closure.py  which third-party modules an operator really imports
  dependency_report.py   per-operator dependency readiness (the coverage report)
  doctor.py           the probe: per-dependency records + availability matrix
  install_deps.sh     the executor: plan, authorize, run the steps, re-verify with the doctor
  check_cann.py / check_communication.py / check_environment.py   focused views
  vendor_ascendep.py  verify (or refresh) the vendored ascendep adapter tree
```

Start with the probe, not the installer:

```sh
cd ascend_ep/ep/04_scripts
python -m scripts.doctor --profile all          # human table + matrix, exit 1 if not ready
python -m scripts.doctor --profile all --json   # the same, machine-readable
python -m scripts.doctor --profile cpu          # what a CPU-only box is expected to have
python scripts/dependency_report.py             # per-operator readiness + environments
```

## Ownership model

Every entry declares who provides it (`ownership`), and the doctor, the installer and the
tests all agree on that one word:

| Ownership | Meaning | Installable here? |
|---|---|---|
| `in_repo` | implemented in this repository (`kernels/asc_comm`, `baselines/l0/*/native`, `launchers/`) | no: build it, never fetch it |
| `vendored` | third-party source copied here, pinned and hash-verified (`operators/vendor/ascendep`, 23 files) | no: verify it, never fetch it |
| `external` | a real third-party package or source build this repository installs | yes, from an exact pin |
| `host` | driver, firmware, CANN toolkit, HCCL/HCOMM, device nodes, deployment state | never: manual step |

`penn513/ascendep` is no longer a dependency. Its adapter tree is vendored, and the
catalog records it as `vendored-ascendep` with `origin=vendored` in every result
(`operators/ascendep.py::source_info`). `EP_ASCENDEP_ROOT` survives only as a documented
development override for adapter development; nothing in the normal run or installation
path reads it.

Each entry also declares:

| Field | Meaning |
|---|---|
| `pin` | `commit` (40-hex) / `release_tag` (+ how the commit is resolved) / `pip` / `system` / `in_repo` / `none` |
| `install_spec` | `pip` (requirement file, wheel source), `source` (URL, directory, pin, steps), `system` (manager packages, sudo), `manual`, `none` |
| `modules` | the Python module names it provides (this is how the operator closure is mapped) |
| `environment` | which isolated environment installs it (`environments/*.yaml`) |
| `conflicts` | the dependencies that install the same module name |
| `check` | the doctor probe (never a claim) |

## Categories and profiles

| Category | Content | Installable here? |
|---|---|---|
| **A** `python_base` | python, torch, numpy, PyYAML, jsonschema, pandas*, matplotlib* | yes, offline |
| **B** `cann` | driver*, CANN toolkit*, torch_npu, ops-transformer, HCCL*, HCOMM*, npu-smi*, AscendC toolchain* | partly (wheels + source builds) |
| **C** `communication` | HCOMM*, SHMEM, npu_patch, DeepEP, deep_ep_experimental, MoonEP, fsdp_turbo, mega_moe, ascendep, ASC-COMM kernels, L0 bridges | partly (source builds) |
| **D** `multinode` | MPI, deployment env*, HCCL config*, device permissions* | MPI only |
| **E** `profiling` | msprof*, hccl_test*, ASC-COMM build toolchain* | no |

`*` = host-provided: see [The host boundary](#the-host-boundary-never-installed-here).
`pandas` / `matplotlib` are optional (absence degrades `reporting`, it does not fail a run).

| Profile | Categories | Use it for |
|---|---|---|
| `cpu` | A | an offline CPU-only machine; this must pass with no NPU |
| `npu` | A B | a node with CANN + torch_npu + devices |
| `communication` | A B C | SHMEM/DeepEP/MoonEP/ASC-COMM work |
| `profiling` | A B E | msprof captures and the toolchain |
| `multinode` | A B D | EP64/EP128 across servers |
| `all` | A B C D E | the full environment |

A profile only *proves* the categories it names. The doctor prints the groups it
could not evaluate under `out of scope for this profile` instead of pretending
they are available.

## The CPU / offline path

The CPU base stack is deliberately minimal so a wheelhouse can carry it.

```sh
# once, on a machine that has the network
pip download --only-binary=:all: -r dependencies/python/base.txt -d /data/wheels

# on the offline machine
bash scripts/install_deps.sh --profile cpu --wheelhouse /data/wheels
# equivalently, without the installer:
pip install --no-index --find-links /data/wheels -r dependencies/python/base.txt
```

`--offline-wheelhouse DIR` makes the doctor *prove* the wheelhouse instead of
assuming it:

```sh
python -m scripts.doctor --profile cpu --offline-wheelhouse /data/wheels
# wheelhouse:<name> pass|fail per pinned requirement, and exit 1 when a pin is missing
```

Two deliberate omissions: `python` is not a wheel (install a CPython 3.10-3.12),
and the two reporting extras (`pandas`, `matplotlib`) are marked optional in the
catalog, so a small wheelhouse still installs and the doctor reports `warning`
rather than a false failure.

## The NPU path

```sh
# CANN and the driver are host state: install them on the host first (see below).
bash scripts/install_deps.sh --profile npu --wheelhouse /data/ascend-wheels
python -m scripts.doctor --profile npu
```

`torch_npu==2.6.0.post3` must match the installed `torch==2.6.0` **and** the CANN
release. The catalog records the triples under `compatibility.rows`; the doctor
compares the observed triple and reports an unknown pairing as `skip` (never
`pass`), because "works until it silently does not" is exactly what an unlisted
pairing looks like:

```sh
python scripts/check_cann.py            # prints the pairing table + the B-category checks
python scripts/check_cann.py --json     # same, for automation
```

## Communication runtimes

```sh
bash scripts/install_deps.sh --profile communication --dry-run   # print every step first
bash scripts/install_deps.sh --profile communication --allow-network
bash scripts/install_deps.sh --profile communication --offline \
    --source-cache /data/ep-sources --wheelhouse /data/ep-wheels
python scripts/check_communication.py
```

Every C-category runtime is a **pinned source build** and the installer is
manifest-driven (`scripts/dependency_plan.py` produces the plan; the shell only runs it):

1. read `install_spec.source.url` and `install_spec.source.pin` from the catalog;
2. reuse the checkout in `$EP_SOURCE_CACHE` (default `~/.cache/ep-benchmark/deps`) if it is
   already at the pin, otherwise `git clone` + `git checkout <commit>` (or `tags/<tag>`);
3. verify the resulting `HEAD` (or the resolved tag commit) and **record it as evidence**;
4. run the declared build/install steps in the dependency's environment;
5. run the declared verify commands;
6. re-run the doctor, which decides whether the profile is satisfied.

Rules it never breaks:

* no unpinned clone, no branch head, no `latest`;
* no fetch at all with `--offline`; a missing or drifted checkout is a `BLOCKED` line;
* no `--clean`, no deletion of an existing checkout or build artifact;
* no sudo, no driver/firmware/CANN install: those are host blockers with a manual step;
* no URL, commit or build command in the shell: all of it comes from the catalog.

`--dry-run` prints the plan and the exact commands. It exits non-zero only when the plan has
an *actionable* blocker (an unpinned installable package, a missing wheelhouse); host
blockers are expected on a machine without CANN and are printed with their manual step.

### Isolated environments

Two dependency families install the **same module name**, so they cannot share an
interpreter:

| Environment | Provides | Conflicts with |
|---|---|---|
| `base` | framework, CANN extensions, SHMEM, FSDPTurbo, MegaMoE, Megatron, MindSpeed, DeepGEMM, vllm-ascend, vendored ascendep | - |
| `deep_ep_lueelu` | `deep_ep` + `deep_ep_experimental` from lueelu/DeepEP @83252a5 | `deep_ep_v2` |
| `deep_ep_v2` | `deep_ep` from DeepEP_9283 @69090a1 | `deep_ep_lueelu` |
| `moonep_npu` | `moonep` + `npu_patch` from MoonEP `dev_npu` @b4127ba | `moonep_dedup` |
| `moonep_dedup` | `moonep` + `npu_patch` from MoonEP `dev_server_dedup` @d8495c4 | `moonep_npu` |

`environments/map.yaml` assigns every operator to one environment
(`common/registry.py::Operator.environment`). The installer creates an isolated
environment with its own venv under `$EP_ENV_ROOT` and installs into *that* interpreter;
the doctor's `environment.isolation` check fails when two conflicting families are
importable in one interpreter, because that is how a run silently measures the wrong
implementation.


The L0 native bridges (`libl0_transport_bridge.so`, `libl0_mem.so`,
`libep_l0_udma.so`) are built from `baselines/l0/*/native/` in this repository;
`libl0_transport_bridge.so` is what turns `LINK-001..011` from
`dependency_missing` into measurements.

### BLOCKED: ASC-COMM kernels

`kernels/asc_comm/` exists, but its data planes cannot be built here:

1. the device programs are not written (`aiv/write_nbi/kernel.cpp`,
   `ccu/dsl/kernel.cpp`); `build.sh` warns and the manifest records
   `built: false`, or fails outright with `--strict`;
2. the CANN **ASC-COMM toolchain** (toolkit headers plus the CCU DSL frontend) is
   not available on any reachable host, and `ASCEND_HOME_PATH` is unset.

That is recorded three times on purpose: in `manifest.yaml`
(`asc-comm-kernels.install_reason`), in `system/asc_comm.yaml` (`status: blocked`
plus a `blockers:` list with `clears_when` for each), and in the doctor's matrix,
where `l0_transport` and `l0_overlap` list `asc-comm-kernels` as a missing
dependency. The installer exits non-zero with the same text.

The project owns its artifact contract, and the doctor reads it rather than
guessing filenames: `build/manifest.json` (written only after the link step) names
the library per engine, `build/libasc_comm_kernels.so` is the artifact, and
`baselines/l0/transport/kernel_artifacts.py` additionally resolves the ABI symbols
from `common/abi.h` before a transport case may use it.

## Multinode

```sh
bash scripts/install_deps.sh --profile multinode --yes   # --yes allows sudo for openmpi-bin
python scripts/check_environment.py
```

MPI is a package; the rest of the D category is deployment state that this
repository must not set from an installer: `MASTER_ADDR`, `MASTER_PORT`,
`ASCEND_RT_VISIBLE_DEVICES`, `HCCL_IF_BASE_PORT`, and the `HCCL_*` configuration.
`check_environment.py` prints the environment the launcher will actually inherit
and the `LD_LIBRARY_PATH` coverage that decides whether `libhccl.so` resolves.

The doctor identifies the MPI *implementation* (`Open MPI` / `MPICH/Hydra` /
`Intel MPI`) and its version. `command -v mpirun` is not treated as evidence: the
hostfile syntax and the binding flags differ per implementation.

## Profiling

```sh
bash scripts/install_deps.sh --profile profiling
python -m scripts.doctor --profile profiling
```

`msprof` and `hccl_test` ship inside a CANN release and are **not installable from
a container**. The installer prints the exact host step and only installs the
build toolchain (`cmake`, `g++`) -- and only with `--yes` for the root step.

## The host boundary (never installed here)

A driver is kernel code; a device node is a kernel object; HCCL/HCOMM/msprof/hccl_test
are toolkit components compiled against that driver. None of them can be installed
from inside a container, and a copy inside the container shadows the host's
libraries -- which is worse than a clear failure.

| Host component | Manual step (printed by the installer, never run) |
|---|---|
| NPU driver + firmware | `Ascend-hdk-<soc>-npu-driver_<ver>_linux-<arch>.run --full`, then firmware, then reset; `npu-smi info` must list every chip |
| CANN toolkit (+ kernels package) | `Ascend-cann-toolkit_<ver>_linux-<arch>.run --install`, `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Device permissions | on the host: `usermod -aG HwHiAiUser <user>`; container: `--device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/devmm_svm` |
| npu-smi / ccec / msprof | come with the driver and toolkit under `/usr/local/Ascend` |
| hccl_test | build CANN oam-tools on the host, then `export EP_HCCL_TEST_BINARY=.../alltoall_test` |
| `MASTER_*` / `HCCL_*` | set per deployment in the launch script |

`LD_LIBRARY_PATH` is the only mechanism that makes `libhccl.so` and `libhcomm.so`
visible to `ctypes` and to torch. `ctypes.util.find_library` does not consult it on
every platform, which is why the doctor scans `LD_LIBRARY_PATH` explicitly and
reports the coverage rather than the value.

## Availability matrix

The doctor computes this per host. This is the **static dependency map** (who needs
what) plus the expected verdict on the CPU-only machine this module was developed
on, so a reader can tell "expected" from "broken".

| Capability group | Needs (beyond category A) | CPU-only host | On a documented Ascend node |
|---|---|---|---|
| `l0_memory` (MEM-001..004) | L0 memory bridge | **blocked** (`l0-native-bridges`) | available once `libl0_mem.so` is built |
| `l0_transport` (LINK-001..013) | driver, CANN, torch_npu, HCCL, HCOMM, AscendC, SHMEM, ASC-COMM kernels, bridges, MPI, RDMA, /dev/shm | **blocked** (13 deps) | **degraded** until the ASC-COMM toolchain exists |
| `l0_collectives` (EXEC-001..006) | driver, CANN, torch_npu, HCCL, HCOMM, hccl_test, device permissions | **blocked** | available |
| `l0_hardware` (HW-001..006) | driver, CANN, torch_npu, HCCL, device permissions | **blocked** | available |
| `l0_compute` (HW-010..011) | driver, CANN, torch_npu, device permissions | **blocked** | available |
| `l0_overlap` (OVERLAP-001..009) | as transport + msprof | **blocked** | **degraded** (ASC-COMM) |
| `l0_topology` | driver, CANN, torch_npu, npu-smi, device permissions | **blocked** | available |
| `l0_verification` | -- (category A only) | available | available |
| `profiling` | driver, CANN, torch_npu, npu-smi, msprof, device permissions | **blocked** | available |
| `reporting` | pandas*, matplotlib* | **degraded** (pandas optional) | degraded or available |
| `multinode` | as l0_collectives + MPI, deployment env, HCCL config, RDMA | **blocked** | available after the deployment exports its variables |
| `operator:allgather_hccl`, `operator:alltoall_hccl` | -- (category A only; CPU/Gloo path) | available | available |
| the other 34 `operator:*` groups | driver, CANN, torch_npu, HCCL, device permissions, plus their runtime (ascendep checkout, DeepEP fork, MoonEP, ops-transformer, sgl-kernel-npu, ...) | **blocked** | available when that runtime is installed |

Read it as: **`blocked` on the CPU-only host is the expected result**, not a
regression. `--profile cpu` is the profile that must come back green there.

## Status vocabulary

| Status | Meaning | Effect on the matrix |
|---|---|---|
| `pass` | present and usable; the evidence quotes what was read | `available` |
| `warn` | present but not ideal (version outside the pin, optional dependency absent, partial build) | `degraded` |
| `fail` | absent or unusable | `blocked` |
| `skip` | the probe cannot decide (unknown check type, or the pairing cannot be compared because nothing is installed) | `degraded` |

A missing dependency is **never** `pass` and **never** `skip`; `skip` is not a way
to be green. Matrix statuses are only `available`, `degraded` and `blocked`.

## Keeping it correct

```sh
python -m pytest tests/dependencies tests/operators_vendor -q
python scripts/vendor_ascendep.py --check
python scripts/dependency_report.py            # per-operator readiness
```

* `test_doctor.py` asserts host-independent invariants: a dependency the probe
  could not find is never `pass`, `available` means every listed dependency
  passed, the status vocabulary is closed, and the CLI JSON round-trips.
* `test_manifest_consistency.py` asserts the three representations agree: every
  requirement pin appears in the catalog with the same version, `pip` entries are
  installable while host-provided ones never are (and always carry the manual
  step), every check type has a prober, and the system descriptors only reference
  dependency ids that exist.
* `test_dependency_catalog.py` asserts the self-contained contract: the ownership model,
  an immutable pin for every installable dependency, no `ascendep-checkout`, symmetric
  conflicts across isolated environments, no undeclared module in any operator's import
  closure, an installable plan for every profile, `--offline` producing no fetch step, the
  installer owning no URL, and the vendored tree failing closed when tampered with.
* `test_vendored_ascendep.py` proves the adapter tree resolves without
  `EP_ASCENDEP_ROOT`, and that the manifest digests match every vendored file.

When adding a dependency:

1. add it to `manifest.yaml` with `ownership`, `pin`, `install_spec`, `modules`,
   `environment`, `conflicts`, `check` and `required_by`;
2. pin a requirement in `python/*.txt` when it is a wheel, or a commit/tag when it is a
   source build -- an unpinned installable entry is a catalog error;
3. put it in an environment: `base` if its modules are unique, a new isolated environment
   if it collides with another family (and add the operator mapping in
   `environments/map.yaml`);
4. run `python scripts/dependency_plan.py --profile all` and the test suite; the plan is
   the proof that the entry is actually installable.

When adding an operator: add it to `operators/registry.yaml` **and** to
`environments/map.yaml`. The closure test then requires every module it imports to be
declared by some dependency; an undeclared import is a catalog error, not a warning.

When adding a benchmark family: add a `capability_groups` entry and list it in the
`required_by` of every dependency it needs -- an unlisted token is a catalog error that the
tests catch.
