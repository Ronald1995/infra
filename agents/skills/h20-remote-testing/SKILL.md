---
name: h20-remote-testing
description: Run vLLM tests on the remote H20 GPU cluster (Alibaba Cloud Singapore, 6 nodes x 8 H20e, Slurm), covering jump-host SSH access, persistent environment reuse, GPU allocation, containers, pre-commit and DCO sign-off checks, and job diagnostics. Use when the user asks to verify on H20 or the remote GPU cluster - including Chinese phrasings such as "使用远程H20验证", "H20 验证", "远程验证", "上 H20 跑一下", "在 H20 集群测试" - and when reusing or rebuilding the shared H20 environment. Not for local-only tests.
---

# H20 Remote Testing

Use the H20 cluster to validate vLLM changes only when remote GPU testing is requested or is necessary.
This is the SGP H20e cluster (6 nodes x 8 H20e). The old 2x8 H20 cluster
(`vllm-h20-jump` / `vllm-h20-head` / `vllm-h20-01/02`) is decommissioned after
2026/10/16 - do not start new work there.

## Access and safety

- Obtain the cluster account and the `vllm_h20` jump-host private key from an administrator through an approved private channel. Never put a private key, a user roster, or another person's credentials in the repository, command output, job script, or this skill.
- SSH aliases: `vllm-jumper-sgp` (jump host, `jump` account, shell disabled), `vllm-sgp-h20-head` (Slurm head node, no GPU), and `vllm-sgp-h20-0[1-6]` (the GPU nodes). Head and workers must use the requesting user's own cluster username.
- This network cannot use the documented plain `ProxyJump`: the wired egress blocks outbound port 22, so the hop must leave through the wireless NIC, whose client address comes from DHCP. `~/.ssh/config` therefore uses
  `ProxyCommand C:\Users\liurong\.ssh\h20-wlan-proxy.cmd %h %p <hop-alias>`,
  which resolves the current WLAN IPv4 on every connection and passes it to `ssh -b`. The head entry hops via `vllm-jumper-sgp`; the worker entries hop via `vllm-sgp-h20-head`, because the jump host has **no route to the 172.16.2.0/24 Slurm network** and a worker reached through it fails with `Connection timed out during banner exchange` plus `channel 0: open failed: connect failed`. Failure modes to recognise: `bind <ip>: Unknown error` from a stale literal `BindAddress`, `Connection timed out during banner exchange` from `-W "[host]:port"` quoting, and `h20-wlan-proxy: WLAN adapter has no IPv4 address`.
- The nested jump ssh runs with `BatchMode=yes` and the default host-key policy, so every host key must already be in `known_hosts`. An unrecorded node fails as `Host key verification failed` followed by `Connection closed by UNKNOWN port 65535`; record it once with `ssh -o StrictHostKeyChecking=accept-new <alias> hostname`. The jumper (47.237.183.209) and head (43.106.48.188) keys are already recorded.
- Confirm access with `ssh vllm-sgp-h20-head`. Do not SSH to a GPU node unless you hold a Slurm allocation on that exact node.
- Driving the cluster from PowerShell is where quoting breaks: quotes are stripped before the remote `bash -c` sees them, so inlined quoted commands arrive mangled (and can hang a session waiting on stdin). Write the script to a file and pipe it as base64 (`ssh <host> "base64 -d | bash -s"`), or upload it and run it by path. Verify an upload by comparing `md5sum` when the script matters.
- Keep source, persistent data, and logs under `/home`. Node-local storage is volatile - see "Where code, models and caches live".

## Two machines, two filesystem views

| | head node (`vllm-sgp-h20-head`, no GPU) | job nodes (`vllm-sgp-h20-0[1-6]`, 8 H20e each) |
| --- | --- | --- |
| SSH | through the jump host | through the head node, and only while you hold a Slurm allocation on that exact node; use `srun --pty bash` |
| Role | prepare code and environment, submit and inspect jobs | build and run tests inside allocations and containers |
| CPU / memory | 8 cores, 29 GB, local disk 99 GB with ~54 GB free | 192 cores, ~2 TB RAM, 14 TB node-local NVMe |
| Image | Alibaba Cloud Linux 4, `python3.11`, `gcc`, `git`, no `/mnt/scratch` | Alibaba Cloud Linux 3 kernel image, `python3.6` **only**, no `ninja`/`cmake`, `nvcc` 13.0 at `/usr/local/cuda/bin` (not on `PATH`) |
| Container | enroot absent | enroot + Pyxis, images cached per node under `/mnt/scratch/enroot/$USER` |
| Network egress | **throttled to tens of KB/s** | 0.6-7 MB/s (GitHub direct ~0.85 MB/s, `gh-proxy.com` 6.8 MB/s, Hugging Face ~1.3 MB/s) |

`/home` is the CPFS-backed shared home (`/mnt/cpfs/home` bind-mounted over `/home` on all seven machines), so anything prepared on the head node is visible on the job nodes. `sbatch`, `srun`, `salloc`, `squeue`, and `scancel` run on the head node.

The egress asymmetry is the single most important operational fact: **do every large download or build inside a GPU allocation**, not on the head node. Measured 2026-10-15 for the same torch wheel: 26 KB/s on the head node, 577 KB/s on `vllm-sgp-h20-01`; 8 parallel streams on a GPU node aggregate ~5.4 MB/s. Writing to `$HOME` (CPFS) from the job node is what keeps the result reusable everywhere.

## Where code, models and caches live

| Path | Scope | What belongs here |
| --- | --- | --- |
| `/home` (= `/mnt/cpfs/home`) | CPFS, all 7 machines | code, the shared environment, logs, results - anything that must survive and follow you to every node |
| `/mnt/cpfs` | CPFS, cluster-wide, 10 TiB | the documented model/dataset distribution tier (put it once, every node sees it) |
| `/mnt/scratch` | node-local NVMe RAID0, 14 TB, ~43 GB/s, **volatile, no backups, no cross-node sharing** | regenerable caches and intermediates only |
| `/mnt/scratch/models` | node-local, 0777+ACL | **real / hand-staged / converted weights - check here first** |
| `/mnt/scratch/hf-hub` | node-local, 0777+ACL | `HF_HUB_CACHE`: Hugging Face blobs, one download **per node**, shared with everyone on that node |
| `/mnt/scratch/vllm-cache` | node-local, 0777+ACL | `VLLM_CACHE_ROOT`: content-hashed torch.compile artifacts |
| `/mnt/scratch/users/$USER` | node-local, private | your own per-node data: intermediates, in-flight datasets, traces |
| `/mnt/scratch/shared` | node-local, 0777+ACL | ad-hoc sharing with other users **on the same node** (never secrets) |

Choosing where something goes:

- **Running a real model: look before you download.** On the node the job will use, check `/mnt/scratch/models/<name>` and `/mnt/scratch/hf-hub/models--*` first - another user may already have staged or downloaded it. On a hit, just use it (`LLM(model="/mnt/scratch/models/<name>")`, or the plain HF id when it is already in `hf-hub`). Only on a miss stage it yourself (recipe below).
- **"Already cached" is per node**, because both directories are node-local: a model on `vllm-sgp-h20-01` is invisible from `-02`. When a job must reuse a staged model, pin the node (`sbatch -w vllm-sgp-h20-01`); otherwise expect one download per node.
- **Your own per-node data goes to `/mnt/scratch/users/$USER`** (private, fast, 14 TB) - intermediates, datasets in flight, profiling traces. Results that matter go to `/home`, not here.
- **Handing files to another user on the same node** is what `/mnt/scratch/shared` is for; keep tokens, keys and private data out of it.
- The three shared caches are explicitly prunable ("anyone may clean stale entries"): treat everything under `/mnt/scratch/{models,hf-hub,vllm-cache}` as regenerable, never as the only copy.

`/mnt/scratch/README.md` (the administrator's contract, readable on every GPU node) names the shared model directory: **`/mnt/cpfs/models` is the cluster-wide tier for model and dataset distribution**, while the `/mnt/scratch` caches are per-node and prunable. Measured 2026-10-15: `/mnt/cpfs` is root-owned `0755`, `/mnt/cpfs/models` does not exist, and `mkdir /mnt/cpfs/models` fails with `Permission denied`; the only CPFS path a non-root user can write is `$HOME`. So:

- For tests, point Hugging Face at the writable shared per-node cache and let the durable copy wait for an admin:
  ```bash
  export HF_HOME=/mnt/scratch/users/$USER/.hf   # holds your HF token - never share it
  export HF_HUB_CACHE=/mnt/scratch/hf-hub
  export VLLM_CACHE_ROOT=/mnt/scratch/vllm-cache
  ```
- The directory to *store* a model in is `/mnt/scratch/models` (shared `0777`); Hugging Face blobs land in `/mnt/scratch/hf-hub` automatically once `HF_HUB_CACHE` is exported.
- **Hugging Face downloads are fragile here.** `snapshot_download` dies on the multi-hundred-MB weights (`HF_HUB_DISABLE_XET=1` avoids `CAS Client Error: ... error decoding response body`, but plain HTTPS then reports `The read operation timed out` and stalls past ~300 MB), and the API no longer exposes `lfs.oid`, so blob names cannot be reconstructed by hand. Stage the heavy files with a resumable curl loop and point vLLM at the directory:
  ```bash
  DEST=/mnt/scratch/models/Qwen3-0.6B; mkdir -p "$DEST"
  for f in $(curl -s "https://huggingface.co/api/models/Qwen/Qwen3-0.6B" \
      | python3 -c 'import json,sys;print("\n".join(s["rfilename"] for s in json.load(sys.stdin)["siblings"]))'); do
    mkdir -p "$DEST/$(dirname "$f")"
    for attempt in $(seq 1 60); do
      curl -L -C - -s -o "$DEST/$f" --speed-limit 1000 --speed-time 60 --max-time 900 \
        "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/$f" && break
      sleep 5
    done
  done
  # then LLM(model="$DEST")   -- measured 0.3-3 MB/s, 1.5 GB weights in ~25 min
  ```
  The job nodes run curl 7.61.1, which rejects `--retry-all-errors`; resume with a shell
  retry loop around `curl -C -` instead (each attempt continues the partial file, so a
  stalled connection costs nothing but a retry). Small configs and tokenizers download
  fine through `snapshot_download(..., ignore_patterns=["*.safetensors"])`.
- Cross-node reuse needs a durable home-tier copy: ask an admin to create a writable shared directory such as `/mnt/cpfs/models` (`0777` + ACL, like the scratch caches). Until then `$HOME/hf-cache` is the only cluster-wide store, and the first job on each new node re-downloads from `/mnt/scratch/hf-hub`.
- `/mnt/scratch` is absent on the head node (and on any node with a missing array): a head-node download step cannot use the scratch cache, so fall back to `$HOME/hf-cache`.

## Reuse a persistent environment

Prepare once, reuse across sessions and nodes. An environment is reusable only if **every interpreter and library it references lives under `/home`**:

```text
$HOME/vllm-shared/           # on CPFS /home: prepared once, reused from every node
├── base-venv/               # torch (2.14.1+cu130 here): logic/route/metric tests
├── prebuilt-venv/           # vllm 0.31.0 + torch 2.13.0+cu130: use this for end-to-end
├── wheels/                  # per-commit vllm wheels fetched from wheels.vllm.ai
├── scripts/run-precommit.sh # shared pre-commit hooks + sign-off check
├── make-overlay.sh          # overlay changed Python files onto the installed package
├── openssl-ctr/             # OpenSSL 3.0.2 for Rust builds in containers
├── uv-tools/                # uv-installed pre-commit
├── env-build.slurm          # rebuild/extend the environment (GPU job)
└── smoke.slurm              # stage weights + one generation (GPU job)
$HOME/.local/bin/uv          # uv 0.13.1, installed from the PyPI wheel
$HOME/vllm-shared/uv-python/ # uv-managed CPython 3.12.15 (UV_PYTHON_INSTALL_DIR)
```

Bootstrap facts, all measured on 2026-10-15 (the new cluster starts from an empty home - no user files were migrated):

- **Build the environment inside a GPU allocation, not on the head node.** The head node's egress is throttled to tens of KB/s (a torch wheel moves at ~26 KB/s, GitHub at ~2 KB/s), while a job node reaches 0.6-7 MB/s. A `sbatch` script that does the `uv python install` / `uv pip install` / model download and writes into `$HOME` is the difference between minutes and hours.
- Even from a job node, the upstream hosts are uneven: `files.pythonhosted.org` crawls at ~24 KB/s (a `pip install` of a few hundred MB never finishes), while **`https://mirrors.aliyun.com/pypi/simple/` serves the same wheels at 2.7-3.3 MB/s** and `download.pytorch.org` gives ~0.3 MB/s. Export `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/` (or `uv pip install -i <mirror>`, `pip install -i <mirror>`) before any install. GitHub release downloads are also slow (~0.85 MB/s direct, `gh-proxy.com` ~6.8 MB/s, `releases.astral.sh` ~1.7 MB/s), so install uv from the PyPI wheel rather than `curl https://astral.sh/uv/install.sh | sh`: `pip3 download uv`, `python3 -m zipfile -e <whl> <dir>`, copy `uv-<ver>.data/scripts/uv` to `$HOME/.local/bin/uv` (the wheel ships the same binary).
- Set `UV_PYTHON_INSTALL_DIR=$HOME/vllm-shared/uv-python`, `UV_TOOL_DIR=$HOME/vllm-shared/uv-tools`, `UV_TOOL_BIN_DIR=$HOME/vllm-shared/uv-tools/bin` so the interpreter and tools stay on `/home`, and `UV_CACHE_DIR=/mnt/scratch/users/$USER/uv-cache` on a job node (`/var/tmp/uv-cache` on the head node): the wheel cache is large and disposable.
- The job-node image ships only `python3.6`, so a venv built from the node's own interpreter is not portable; always create venvs from the `/home` CPython (`uv venv --python 3.12 "$HOME/vllm-shared/<venv>"`). On the head node `uv python install` needs GitHub, so run it in an allocation too.
- A CPU-only venv is not cheap here: with `UV_DEFAULT_INDEX` set to the mirror, `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` still resolved the CUDA build (torch 528 MB plus ~2.4 GB of `nvidia-*` and triton wheels). Unset `UV_DEFAULT_INDEX` for that one command if a small CPU torch matters; otherwise accept the CUDA stack, which then makes the vllm install in the next venv nearly free (uv reuses its cache).
- The old cluster's venvs were not migrated: its `/home` was not carried over, it is unreachable from the new cluster (which blocks its own outbound port 22 to the old head), and the only cross-cluster path is the local workstation at ~40 KB/s.

```bash
export PATH="$HOME/.local/bin:$HOME/vllm-shared/uv-tools/bin:$PATH"
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_PYTHON_INSTALL_DIR=$HOME/vllm-shared/uv-python
uv python install 3.12                                  # 39 s from a GPU node
uv venv --python 3.12 "$HOME/vllm-shared/base-venv"
uv pip install --python "$HOME/vllm-shared/base-venv/bin/python" torch
uv venv --python 3.12 "$HOME/vllm-shared/prebuilt-venv"
uv pip install --python "$HOME/vllm-shared/prebuilt-venv/bin/python" vllm pytest
```

Run this from a GPU allocation as `sbatch` (`$HOME/vllm-shared/env-build.slurm` is the
committed version: uv install, both venvs, pre-commit, weight staging). `base-venv` exists
so tests that never touch vllm's extensions can use a separate interpreter; here it is not
a small CPU tree (see the mirror trap below).

### Start by checking what already exists

Environment setup is the expensive part, so spend the first commands looking for
work that is already done rather than rebuilding it:

- `ls $HOME/vllm-shared` and read `$HOME/vllm-shared/README.md` - an earlier session
  may already provide the venvs, the compiled vllm, and cached weights.
- `ls $HOME/vllm-tasks` - a task directory may already hold a worktree pinned to the
  commit under test, with its `run.sh` / `run-cuda.sh`.
- `git -C <worktree> rev-parse HEAD` against the target commit - re-pinning is cheap,
  rebuilding is not.
- On a GPU node, `ls /mnt/scratch/models` and `ls /mnt/scratch/hf-hub/models--*` show which real models are already on **that node** - reuse them instead of downloading, and pin the node (`-w`) when the job depends on them. Your own leftovers from earlier sessions are in `/mnt/scratch/users/$USER`.
- Only then decide between: reuse the baseline, overlay the change, or rebuild.

### When setup starts to repeat, fix the environment or this skill

Treat repeated environment work as a defect, not as normal cost:

- If a step is being redone because a previous session's work was undiscoverable,
  leave the missing pointer in `$HOME/vllm-shared/README.md` (layout, reuse order,
  measured traps) and add the rule here so the next session finds it first.
- If the same build happens twice for the same fingerprint, promote the result once
  so later sessions reuse it instead of rebuilding, and say where it now lives.
- If the same command fails the same way twice, record the failure and the working
  alternative as a bullet here; a trap that costs time twice is a documentation bug.
- Keep this file lean (see the repository's guide for agent-instruction budgets):
  prefer folding a new fact into an existing bullet over appending a new section.

### Compiled extensions: reuse before building

`import vllm` from a bare checkout cannot run a model - the C++/CUDA extensions are
missing. **Try to reuse an installed vllm before compiling anything.** Builds from the
same period differ little in C/CUDA code, so the artifacts are usually interchangeable.

Order of preference:

1. **Reuse `$HOME/vllm-shared/prebuilt-venv/`.** vllm must be an *installed* package
   (package + `*.dist-info` inside the venv's `site-packages`): a bare `PYTHONPATH` to
   the checkout is not enough - platform discovery then resolves to
   `UnspecifiedPlatform`, whose empty `device_type` fails with
   `Device string must not be empty`. The venv's torch CUDA version must also match
   what vllm was built against (`libcudart.so.<major>`).
2. **Pure-Python change? Overlay instead of compiling.** Copy the installed package and
   symlink the task's changed Python files over it, then run with the overlay ahead of
   the installed tree. `$HOME/vllm-shared/make-overlay.sh` does this:
   ```bash
   cd "$OVERLAY"                       # NOT the worktree: cwd wins over PYTHONPATH
   export PYTHONPATH="$OVERLAY"
   "$HOME/vllm-shared/prebuilt-venv/bin/python" -m pytest -q --import-mode=importlib <absolute test paths>
   ```
   Without `--import-mode=importlib`, pytest inserts the tests' rootdir and the
   worktree's pure-Python `vllm` shadows the installed package, so `import vllm._C...`
   fails. A plain `PYTHONPATH=<worktree>:<site-packages>` fails the same way.
3. **Need the checkout's own extensions?** Prefer the per-commit wheel index over a
   source build: `https://wheels.vllm.ai/<upstream-commit>/vllm/` serves
   `vllm-<version>+g<commit>-cp38-abi3-manylinux_2_28_x86_64.whl` for commits that passed
   the wheel-build job (fork SHAs return 404). Extract it with
   `prebuilt-venv/bin/python -m zipfile -e <whl> <dir>`, put `<dir>` ahead on
   `PYTHONPATH`, symlink the task's changed files over it, and run with
   `prebuilt-venv/bin/python` (torch is reused). Cache the wheel under
   `$HOME/vllm-shared/wheels/`.
4. **Only when the compiled surface changed and no wheel exists**: build once in a GPU
   job, then promote the result into the shared venv (or into a new
   `$HOME/vllm-shared/prebuilt/` package directory) so later sessions reuse it instead of
   rebuilding. `cp -al` copies a venv instantly on the same filesystem (hardlinks) - never
   edit files through the copy.

End-to-end recipe (verified 2026-10-15 on `vllm-sgp-h20-01`, one GPU: vllm 0.31.0 generated
`' Paris. The capital of France is also the capital of the Republic of France.'`). The
prebuilt venv already carries torch and the CUDA runtime, so no container is needed; stage
the weights first (see above) and point vLLM at the directory:

```bash
srun --partition=h20 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:40:00 bash -lc '
  export HF_HOME=/mnt/scratch/users/$USER/.hf HF_HUB_CACHE=/mnt/scratch/hf-hub
  export VLLM_CACHE_ROOT=/mnt/scratch/vllm-cache HF_HUB_DISABLE_XET=1
  $HOME/vllm-shared/prebuilt-venv/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(model=\"/mnt/scratch/models/Qwen3-0.6B\", max_model_len=512, gpu_memory_utilization=0.30, enforce_eager=True)
print(llm.generate([\"The capital of France is\"], SamplingParams(max_tokens=16, temperature=0.0))[0].outputs[0].text)"'
```

`$HOME/vllm-shared/smoke.slurm` is that run as a batch job (weights + generation);
`env-build.slurm` is the equivalent for rebuilding the environment. Reuse them instead of
re-typing the setup.

Pass criteria for a smoke run: the engine starts, the generation returns text, and the
assertion on that output holds. Budget conservatively - one GPU and under ~40 minutes
for a small model; ask before requesting multi-GPU. Never claim an end-to-end result
without the command output that shows the model actually produced tokens.

Give each unit of work its own tree, and never let a job rewrite someone else's checkout:

```text
$HOME/vllm-tasks/<task>/
├── src/     # its own git worktree pinned to an explicit commit
├── .venv/   # only when extra deps are needed; otherwise use the shared venv
└── logs/
```

Rules that keep this working:

- A job script must **not** run `git reset --hard`, `git checkout`, or anything else that rewrites the worktree. Verify the commit and fail loudly instead, so parallel sessions cannot clobber each other. Run this check **before** entering a container on a GPU node: those images have no `git` binary.
  ```bash
  expected=<sha>; actual=$(git -C "$SRC" rev-parse HEAD)
  [ "$actual" = "$expected" ] || { echo "HEAD mismatch: $actual != $expected" >&2; exit 2; }
  ```
- Pin a worktree with `git worktree add --detach <dir> <sha>`, and resolve new commits on the head node (fetch, then `git worktree remove` and re-add, or add a fresh directory) instead of mutating a shared checkout in place.
- Prefer `uv` for Python: it provisions the interpreter itself and its wheel cache makes a rebuild cheap.

## Prepare the remote test

1. On the head node, use the intended vLLM commit or branch under `/home`; inspect the working tree before changing it. Do not overwrite existing remote work.
2. State the test objective, model and data locations, GPU count, time limit, and expected pass condition before submitting work.
3. Inspect capacity with `sinfo` and current jobs with `squeue`. Request only the GPUs, CPU, memory, and time needed.

## Run with Slurm

The single partition is `h20` with six nodes, each with eight H20e GPUs (141 GB, NVSwitch full mesh), 192 vCPUs and ~2 TB RAM. Every GPU command must request GPUs explicitly - changing `CUDA_VISIBLE_DEVICES` never grants access - but the GRES carries **no type**: `--gres=gpu:h20:1`, which the cluster's own getting-started page shows, fails with `srun: error: Unable to allocate resources: Requested node configuration is not available` (measured 2026-10-15). Use `--gres=gpu:<count>`.

For an interactive investigation, allocate a shell, then confirm the allocation and visible device:

```bash
srun --partition=h20 --gres=gpu:1 --time=01:00:00 --pty bash
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"
```

For reproducible or longer validation, use a batch script with at least the partition, GPU request, time limit, and an output file, then submit it with `sbatch`. Put the actual test command and any environment activation in that script. Use `squeue`, `scontrol show job <job_id>`, and the job's output file to diagnose it; use `scancel <job_id>` only for jobs that belong to the requester.

Slurm here is 24.11 but the head node's `squeue` rejects `-o` format strings outright (it prints the usage block) - use the default output. Slurm accounting is not deployed, so `sacct` and history are unavailable. CPU and memory are scheduled but not hard-enforced: keep requests modest. A direct `ssh` into a node that holds your allocation does **not** get the job's `CUDA_VISIBLE_DEVICES`, so run GPU work through `srun`/`sbatch` where Slurm sets the step environment.

Use `salloc` when multiple `srun` commands must share an allocation (`salloc --partition=h20 --gres=gpu:1 --time=02:00:00`, then `srun ...`), and exit the allocation shell when finished so the node is released.

## Containers

Pyxis/enroot is available on the job nodes; `/etc/enroot/enroot.conf.d/90-sgp-h20.conf` already redirects runtime, cache, and data to `/mnt/scratch/enroot/$USER/...` (node-local, root-owned `/run` is unusable in a batch job) and pins `-comp lz4` because the bundled `squashfuse` cannot mount the default. The home directory is mounted by default, which is what makes the shared environment above usable inside the container:

```bash
srun --partition=h20 --gres=gpu:1 --cpus-per-task=8 --mem=64G --pty \
  --container-image='nvcr.io#nvidia/cuda:12.4.1-base-ubuntu22.04' \
  --container-mounts="$HOME:/host-home" \
  --container-workdir=/host-home/vllm-tasks/<task>/src \
  bash -lc 'export PATH=/host-home/.local/bin:$PATH; /host-home/vllm-shared/prebuilt-venv/bin/python -m pytest -q tests/...'
```

Facts that cost time to rediscover:

- `nvcr.io#nvidia/cuda:*-devel` images ship the CUDA toolchain and a C compiler but **no** torch, vllm, pytest, `pkg-config`, `libssl-dev`, `curl`, or even `git`, and their rootfs is **read-only**. Install into a writable mounted directory (a venv under `/host-home`), never into the image, and do git work outside the container.
- Without `--container-image` you run on the node image, which has only `python3.6`, no `ninja`/`cmake`, and `nvcc` 13.0 that is not on `PATH`. Prefer the `/home` venvs; add `/usr/local/cuda/bin` to `PATH` only when something must JIT a kernel.
- An image that already bundles CUDA torch (an NGC PyTorch image) can host a full end-to-end model run without building a CUDA venv; create its venv under `/host-home` so the result persists for later sessions.
- `--container-env` mis-parses values containing `=` or `,` here. Export such variables inside the container's own script and keep only simple path values in `--container-env`.
- For Rust builds, point `OPENSSL_DIR` / `OPENSSL_LIB_DIR` / `OPENSSL_INCLUDE_DIR` at `$HOME/vllm-shared/openssl-ctr/usr` (Ubuntu 22.04 OpenSSL 3.0.2, matched to the container), because the node image has no development headers. The directory was carried over from the old cluster; recreating it from Ubuntu debs needs zstd (the image's `tar` cannot read `data.tar.zst`) and requires flattening `include/x86_64-linux-gnu/openssl/*.h` into `include/openssl/`, or `openssl-sys` reports "Failed to find OpenSSL development headers" because its single `-I` misses the arch-specific `opensslconf.h`.
- Compile in a Slurm allocation with a limited `CARGO_BUILD_JOBS`; building on the head node competes for its 29 GB of memory.
- A registry pull is the slowest part of a first container run on a node: `nvcr.io` delivered a CUDA base image at ~60 KB/s (62 MB in 17 minutes) and the cache is per user per node under `/mnt/scratch/enroot/$USER/cache`, so `--container-image` costs that pull again on every new node. Prefer the `/home` venvs unless the change needs an image's own toolchain.
- `$HOME/vllm-shared/README.md` records the shared layout, the reuse order, and the measured traps; keep it in sync when the environment changes.

Before running vLLM, verify `nvidia-smi`, the CUDA toolchain (`/usr/local/cuda` -> `cuda-13.0`), and that the code and test dependencies are available in the selected environment.

## Pre-commit before spending GPU time

Run the hooks on the changed files **before** submitting a verification job, and again
before reporting. Cheap static checks catch what CI would find, and a GPU job that dies
on a lint error wastes the allocation.

Reuse the shared hook environments instead of building them per session:

```bash
# every hook the diff touches + the DCO check; non-zero exit if either fails
bash "$HOME/vllm-shared/scripts/run-precommit.sh" <worktree> <base-ref>
```

Facts behind that script:

- `PRE_COMMIT_HOME` must be **node-local disk**. `/home` is a FUSE CPFS mount, and a hook environment created there is slow and fragile; `/var/tmp/vllm-shared-precommit` (head-node ext4, ~54 GB free) is the populated cache every session reuses.
- `pre-commit` itself comes from `uv tool install pre-commit` into `$HOME/vllm-shared/uv-tools`; put `$HOME/vllm-shared/uv-tools/bin` and `$HOME/.local/bin` on `PATH` before running the script. The script exports `PIP_INDEX_URL` pointing at the Alibaba mirror, because hook environments are pip-installed and `files.pythonhosted.org` runs at ~24 KB/s here.
- **Three hooks have no toolchain on this image**: `typos` (rust), `markdownlint-cli2` (node), and `actionlint` (golang) - `cargo`/`node`/`go` are absent, and pre-commit downloads them on demand. List them in `SKIP=` (or run hooks by id) so they cannot block the run; pre-commit still clones their repos, which costs a few minutes on a fresh `PRE_COMMIT_HOME`. Measured with `SKIP=typos,markdownlint-cli2,actionlint,rust-cargo-fmt`: `run-precommit.sh` exited 0 with 13 hooks passing and the DCO check green. Install the toolchains in an allocation if those three verdicts matter.
- `rust-cargo-fmt` shells out to `cargo fmt`, so a shared Rust toolchain needs `rustup component add rustfmt`. That hook is marked "only run locally as Buildkite will cover this" - treating it as optional because "CI does not run it" just moves the failure to Buildkite.
- `pre-commit run --files ...` covers the families a "run one hook" shortcut misses:
  `ruff-check`/`ruff-format`, `typos`, `clang-format`, `markdownlint-cli2`, `shellcheck`,
  `check-spdx-header`, `check-test-tethering`, the `pip-compile` locks, and the Rust hooks.
- **`Signed-off-by` is a required check, not a style nit.** GitHub's DCO app fails the PR
  when any commit in the range lacks the trailer or signs off with someone else's
  identity, and a commit made with `--no-verify` (or by an agent that skipped the
  `signoff-commit` hook) has none. `run-precommit.sh` checks this per commit and exits
  non-zero, so a green run is evidence rather than a printed log to eyeball. To fix:
  `git rebase --signoff <base>`, force-push, then confirm `DCO pass` in
  `gh pr checks <pr> --repo <owner>/<repo>`. The rewrite changes every SHA in the range
  and GitHub reports it as "pushed N commits", so when reporting say that the tree is
  unchanged (`git diff --stat <old-head> <new-head>`) and the count was the rebase.
- If a hook rewrites files, the commit was wrong even though the hook "failed" only on
  the first pass: review the fixups, re-stage, and re-run until clean.
- After re-pinning a worktree to a new commit, re-run pre-commit on the new diff; the
  previous run says nothing about the new code.

Report the exact pre-commit command and its result next to the test results, including
any hook skipped and why. The step is only complete when `run-precommit.sh` exits 0 -
hooks clean **and** every commit in the range signed off; a non-zero exit is a failure to
fix, not a log to summarise.

## Report results

After every verification run, report in two places:

1. **The session**: the commit, node allocation, GPU count, container image or environment, exact test command, result, relevant log path, and any failure summary. Do not include credentials or confidential model/data paths in a report intended for broader sharing.
2. **The GitHub PR under test**: post a comment with the same evidence. The PR is where reviewers look, and a result that only exists in the session is lost. Include at minimum:
   - the exact verified commit SHA (`<sha>` and whether it is a PR head, a rebase, or a local cherry-pick);
   - environment: cluster/node, container image, and how the compiled extensions were obtained (reused baseline, overlay, or a fresh build);
   - the exact command, and the observed result as a short log excerpt or test summary;
   - the pre-commit command and result (or the hook that was skipped and why);
   - the DCO state of the range: the sign-off lines from `run-precommit.sh` **and** the
     `DCO` conclusion from `gh pr checks` (a local pass with a red app check means the
     branch was pushed before the fix);
   - for anything end-to-end, the output that proves the model ran, not just "passed";
   - an explicit list of what was **not** verified, and why (missing dependency, unavailable toolchain, skipped test, time limit).
   Look for an existing comment from the same verification and reply or update it instead of stacking duplicates when the run is a re-verification of the same commit. Post the comment with `gh pr comment <number> --repo <owner>/<repo> --body-file <file>`, keeping the body in a file so quoting and formatting survive.

Never claim a verification passed without the command output that shows it. A partially verifying run is still worth posting as long as the unverified parts are named.
