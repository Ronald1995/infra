# Installing on a fresh Ascend host

This is the deployment procedure for `ascend_ep/ep/04_scripts`. It assumes a bare Ascend
server (driver and CANN may not be installed yet) and ends with a doctor run that proves
every dependency of the requested profile is present.

Two rules make the procedure reproducible:

* **the catalog is the only source of truth** -- `dependencies/manifest.yaml` declares every
  dependency, its owner, its exact pin, its build/install steps and its verification. The
  installer reads it; it has no URLs or commits of its own.
* **host state is never installed from inside the project** -- the driver, firmware, CANN
  toolkit, HCCL/HCOMM and device nodes are provisioned on the host. The project detects them,
  reports them as blocked with the exact manual step, and refuses to pretend otherwise.

---

## 0. What you need before you start

| Item | Why | Check |
|---|---|---|
| Ascend NPU + driver + firmware | the device is host state | `npu-smi info` lists the chips |
| CANN toolkit (8.2.RC1 or 8.3.RC1) | `libascendcl`, `libhccl`, `ccec`, `msprof` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh && ccec --version` |
| Python 3.10-3.12 | the framework pins | `python -V` |
| git, a C++17 compiler, cmake, make | source builds (SHMEM, DeepEP, DeepGEMM, ...) | `git --version && g++ --version` |
| network access **or** a source cache + wheelhouse | step 3 | see below |
| the Ascend `torch_npu` wheel | it is **not** on PyPI | `ls /data/ascend-wheels | grep torch_npu` |

Record the observed triple -- it determines every wheel and every source pin:

```sh
python -V
ls /usr/local/Ascend/ascend-toolkit/            # the CANN version directory
npu-smi info -t board -i 0 | head -20           # SoC name, chip count
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
```

The catalog's known-good triples live in `dependencies/manifest.yaml` under
`compatibility.rows` (CANN, torch, torch_npu, SoC). If your triple is not listed, the doctor
reports the pairing as `skip` -- an unknown pairing is never a pass.

## 1. Get the code

```sh
git clone <this repository>
cd infra/ascend_ep/ep/04_scripts
python -m venv .venv && . .venv/bin/activate     # or use the host's python
python -m pip install -e '.[test,charts]'        # the framework itself
```

The framework install is separate from the dependency install: `pip install -e .` gives you
`run_benchmark.py`, `common/`, `operators/`, `baselines/`, and the vendored adapter tree.
Nothing else is required to run the CPU smoke path.

## 2. Look before you install

```sh
python -m scripts.doctor --profile all           # what is missing and why
python scripts/dependency_report.py              # per-operator readiness, per environment
python scripts/dependency_plan.py --profile all  # the exact plan, human readable
```

The plan is the contract. It lists, per dependency: the environment, the fetch (URL +
pinned commit/tag), the build/install steps, the verify commands, and a `blocked:` section
for everything the project may not install. `--json` gives the same as data;
`--script` gives the bash that will run.

Blockers fall into two kinds, and only one of them is yours to clear:

* **actionable** -- an installable dependency without a wheel source or without a pin. The
  plan exits non-zero and names it. Fix the catalog or supply the wheelhouse.
* **host** -- driver, firmware, CANN, HCCL/HCOMM, device nodes, system packages. Expected on
  a machine that has not been provisioned; each carries its `manual_step`.

## 3. Prepare the inputs (offline or online)

**Wheels.** `torch_npu` is not published on PyPI. Either point the installer at a wheelhouse
or set the index that serves the pinned triple:

```sh
export EP_ASCEND_WHEEL_INDEX=https://<your-ascend-wheel-index>/simple   # or:
python -m pip download -d /data/ep-wheels -r dependencies/python/base.txt \
    -r dependencies/python/npu.txt --extra-index-url "$EP_ASCEND_WHEEL_INDEX"
```

The installer refuses to fall back to PyPI for `torch_npu`: a silent substitution would
change the measurement.

**Sources.** Source builds are cloned once into `$EP_SOURCE_CACHE`
(default `~/.cache/ep-benchmark/deps`) and reused afterwards. For an air-gapped host, copy
that directory (or clone the repos there yourself) and pass `--offline`:

```sh
export EP_SOURCE_CACHE=/data/ep-sources
ls $EP_SOURCE_CACHE      # DeepEP/ MoonEP/ ops-transformer/ shmem/ Megatron-LM/ ...
```

`--offline` never runs `git fetch` or `git clone`: a missing or drifted checkout becomes a
`BLOCKED` line naming the expected commit.

## 4. Install the declared dependencies

```sh
bash scripts/install_deps.sh --profile all --dry-run                 # 1. show the plan
bash scripts/install_deps.sh --profile all --allow-network           # 2a. online
bash scripts/install_deps.sh --profile all --offline \
    --source-cache /data/ep-sources --wheelhouse /data/ep-wheels     # 2b. offline
bash scripts/install_deps.sh --check-only --profile all              # 3. verify only
```

What it does, in order:

1. builds the plan from the catalog (`scripts/dependency_plan.py`);
2. creates each isolated environment that the plan needs (`$EP_ENV_ROOT`, default
   `.envs/`) -- a venv per conflicting dependency family;
3. for every dependency: reuse the pinned checkout or fetch it, verify `HEAD` against the
   pin (or resolve the tag and record the commit), run the declared build/install steps in
   the environment's interpreter, run the declared verify commands;
4. re-runs the doctor and exits with its verdict.

What it will never do: fetch an unpinned branch, follow `latest`, install a driver, firmware,
CANN or a system package, run `sudo`, delete an existing checkout or build output, or claim
success when a step failed. Every execution is logged under `.install-logs/`.

## 5. Prove it

```sh
python -m scripts.doctor --profile all --json | python -m json.tool | head -40
python scripts/dependency_report.py --out /tmp/dependency-coverage.md
python -m pytest tests/dependencies tests/operators_vendor -q
```

The doctor's exit code is the installation verdict: `0` when every dependency the profile
needs passed. `warn` means "present but not the pinned version" (for example a newer torch);
`fail` means missing or unusable. The matrix at the end groups the failures by capability so
you can see which benchmark families are blocked and by what.

Then run the CPU smoke test, which needs no NPU:

```sh
python run_benchmark.py pipeline --config configs/smoke_cpu.yaml \
  --cases COMM-001,COMM-004,COMM-005,MOE-003,LAYER-002 --run-id install-check
```

## 6. Isolated environments

Two families install the **same** Python module name, so they cannot share an interpreter:

| Environment | Installs | Conflicts with |
|---|---|---|
| `base` | framework, CANN extensions, SHMEM, FSDPTurbo, MegaMoE, Megatron, MindSpeed, DeepGEMM, vllm-ascend, vendored ascendep | -- |
| `deep_ep_lueelu` | `deep_ep`, `deep_ep_experimental` (lueelu/DeepEP @83252a5) | `deep_ep_v2` |
| `deep_ep_v2` | `deep_ep` (DeepEP_9283 @69090a1) | `deep_ep_lueelu` |
| `moonep_npu` | `moonep`, `npu_patch` (MoonEP `dev_npu` @b4127ba) | `moonep_dedup` |
| `moonep_dedup` | `moonep`, `npu_patch` (MoonEP `dev_server_dedup` @d8495c4) | `moonep_npu` |

`dependencies/environments/map.yaml` assigns every operator to an environment and
`common/registry.py` exposes it as `Operator.environment`. To run one operator family in its
own interpreter:

```sh
.envs/deep_ep_lueelu/bin/python run_benchmark.py run --plan configs/benchmark_plan.yaml \
    --operators deepep_lueelu
```

The doctor's `environment.isolation` check fails when two conflicting families are
importable in one interpreter, because that is how a run silently measures the wrong
implementation.

## Common blockers

| Symptom | Meaning | What to do |
|---|---|---|
| `npu-driver: ... /dev/davinci* missing` | host driver not installed | install the driver + firmware on the host (never in a container) |
| `cann-toolkit: ASCEND_HOME_PATH is unset` | `set_env.sh` not sourced | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` on every rank |
| `torch-npu: the wheel is not on PyPI` | no wheelhouse / index | pass `--wheelhouse` or set `EP_ASCEND_WHEEL_INDEX` |
| `fsdp-turbo: ... not recorded the commit` | upstream pin unresolved | install the FSDPTurbo revision matching your MindSpeed, then record its commit |
| `mega-moe: ... test tree is not identified` | the `mega_moe` test tree is deployment-owned | provide it on `PYTHONPATH`, or use `megamoe_cann` |
| `environment.isolation: deep-ep and deep-ep-v2 are both importable` | two forks in one interpreter | run each family in its own environment |
| `asc-comm-kernels: device programs are not written` | the ASC-COMM kernels are unimplemented | documented in `docs/blocked.md`; the L0 transport cases stay `dependency_missing` |

Everything that is *not* installable here is listed with its reason in
`dependencies/manifest.yaml` (`install_reason`, `manual_step`) and summarised by the doctor.
