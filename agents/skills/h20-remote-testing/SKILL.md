---
name: h20-remote-testing
description: Run vLLM tests on the remote H20 Slurm cluster (head node plus GPU job nodes), covering SSH access, persistent environment reuse, GPU allocation, containers, pre-commit and DCO sign-off checks, and job diagnostics. Use when the user asks to verify on H20 or the remote GPU cluster - including Chinese phrasings such as "使用远程H20验证", "H20 验证", "远程验证", "上 H20 跑一下", "在 H20 集群测试" - and when reusing or rebuilding the shared H20 environment. Not for local-only tests.
---

# H20 Remote Testing

Use the H20 cluster to validate vLLM changes only when remote GPU testing is requested or is necessary.

## Access and safety

- Obtain the H20 cluster account and the `vllm_h20` jump-host private key from an administrator through an approved private channel. Never put a private key, a user roster, or another person's credentials in the repository, command output, job script, or this skill.
- Configure SSH aliases locally: `vllm-h20-jump` for the jump host, `vllm-h20-head` for the Slurm head node, and `vllm-h20-01` / `vllm-h20-02` for the workers. The head and workers must use the requesting user's own cluster username and the workers must be reached through the jump host. The official configuration relies on `ProxyJump`; if connecting fails with `bind <ip>: Unknown error`, a stale `BindAddress` line is the cause - remove it instead of adding one.
- Confirm access with `ssh vllm-h20-head`. Do not directly SSH to a worker: a worker accepts access only while the user has an active Slurm allocation on that node.
- Keep source, persistent data, and logs under `/home`. `/mnt/nvme` is node-local and volatile; use it only for disposable, performance-sensitive files. `/mnt/nvme/shared` is writable by all users, so do not store secrets there.
- Driving the cluster from PowerShell is where quoting breaks: quotes are stripped before the remote `bash -c` sees them. Write the script to a file and pipe it as base64 (`ssh <host> "base64 -d | bash -s"`) instead of inlining quoted commands.

## Two machines, two filesystem views

| | head node (`vllm-h20-head`, no GPU) | job nodes (`vllm-h20-01/02`, 8 H20 each) |
| --- | --- | --- |
| SSH | reachable through the jump host | only with an active Slurm allocation; use `srun --pty bash` |
| Role | prepare code and environment, submit and inspect jobs | build and run tests inside containers |
| CPU / memory | 2 cores, low memory - a single `cargo test` can be OOM-killed | 192 cores, ~2 TB |
| Image | Alibaba Cloud Linux with OpenSSL 1.1 and `pkg-config` | different image with no OpenSSL headers or `pkg-config` |

`/home` is shared (`/cpfs/home`), so anything prepared on the head node is visible on the job nodes. `sbatch`, `srun`, `salloc`, `squeue`, and `scancel` run on the head node.

## Reuse a persistent environment

Prepare once, reuse across sessions and nodes. An environment is reusable only if **every interpreter and library it references lives under `/home`**:

```bash
# portable: uv's own CPython lives under the shared home
uv venv --python 3.12 "$HOME/vllm-shared/base-venv"
# not portable: the container / job node may not have this interpreter path
uv venv --python /usr/bin/python3.12 ...
```

Keep shared assets in one place and reference them instead of copying:

```text
$HOME/vllm-shared/           # prepared once, read-only afterwards
├── base-venv/               # CPU torch: logic/route/metric tests, fast to rebuild
├── base-venv-cuda/          # CUDA torch: anything that needs real GPU compute
├── enroot/                  # container image cache (large; never copy it)
├── scripts/run-precommit.sh # shared pre-commit hooks + sign-off check
├── openssl-ctr/             # OpenSSL 3.0.2 for Rust builds on job nodes
├── cargo-target/            # shared CARGO_TARGET_DIR, so Rust builds are reused
├── uv-python/  cargo/  rustup/    # symlinks to the real locations
└── requirements-common.txt  # copied from the checked-out requirements/
```

Build the CUDA variant the same way, then check it on a job node:

```bash
uv pip install --python "$HOME/vllm-shared/base-venv-cuda/bin/python" torch \
  --index-url https://download.pytorch.org/whl/cu126      # uv caches the wheel
# in a GPU allocation: torch.cuda.is_available(), get_device_name(0), a small matmul
```

Keep both variants: the CPU one keeps the common path cheap, and tests that only
import torch do not need the ~8 GB CUDA tree. Note that the CUDA wheel is ~700 MB
and `uv pip install` reports progress only when it finishes - run it in the
foreground, because a `nohup`-ed install is killed when the SSH session ends.

### Start by checking what already exists

Environment setup is the expensive part, so spend the first commands looking for
work that is already done rather than rebuilding it:

- `ls $HOME/vllm-shared` and read `$HOME/vllm-shared/README.md` - an earlier session
  may already provide the venv, the compiled vllm, the container cache, and the
  model weights.
- `ls $HOME/vllm-tasks` - a task directory may already hold a worktree pinned to the
  commit under test, with its `run.sh` / `run-cuda.sh`.
- `git -C <worktree> rev-parse HEAD` against the target commit - re-pinning is cheap,
  rebuilding is not.
- Only then decide between: reuse the baseline, overlay the change, or build.

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
missing. **Try to reuse a previously compiled vllm before compiling anything.**
Builds from the same period differ little in C/CUDA code, so the artifacts are
usually interchangeable.

Order of preference:

1. **Reuse the shared prebuilt baseline, `$HOME/vllm-shared/prebuilt-venv/`.** It is a
   self-sufficient venv with a torch/CUDA combination plus an already-compiled vllm,
   and it runs models end to end. Two properties make it work, and they are easy to
   break:
   - vllm must be **installed** (package + `*.dist-info` inside the venv's
     `site-packages`). A bare `PYTHONPATH` to a package directory is not enough:
     platform discovery then resolves to `UnspecifiedPlatform`, whose empty
     `device_type` fails with `Device string must not be empty`.
   - The compiled extensions need their CUDA runtime, so the venv's torch CUDA
     version must match what vllm was built against (`libcudart.so.<major>`).
2. **Pure-Python change? Overlay instead of compiling.** Copy the prebuilt package and
   symlink the task's changed Python files over it, then run with the overlay ahead of
   the prebuilt tree. `$HOME/vllm-shared/make-overlay.sh` does this:
   ```bash
   cd "$OVERLAY"                       # NOT the worktree: cwd wins over PYTHONPATH
   export PYTHONPATH="$OVERLAY"
   "$HOME/vllm-shared/prebuilt-venv/bin/python" -m pytest -q --import-mode=importlib <absolute test paths>
   ```
   Without `--import-mode=importlib`, pytest inserts the tests' rootdir and the
   worktree's pure-Python `vllm` shadows the prebuilt package, so `import vllm._C...`
   fails. A plain `PYTHONPATH=<worktree>:<site-packages>` fails the same way.
3. **Only when the compiled surface changed**: build once in a GPU job, then promote
   the result into `$HOME/vllm-shared/prebuilt/` and refresh `prebuilt-venv/` so later
   sessions reuse it instead of rebuilding. `cp -al` copies a venv instantly on the
   same filesystem (hardlinks) - never edit files through the copy.

End-to-end recipe, once the extensions resolve:

```bash
export HF_HOME=$HOME/hf-cache            # shared weights (e.g. Qwen3-0.6B, Llama-3.2-1B)
srun --partition=h20 --gres=gpu:h20:1 --cpus-per-task=8 --mem=64G --time=00:40:00 \
  --container-image='nvcr.io#nvidia/cuda:12.4.1-devel-ubuntu22.04' \
  --container-mounts="$HOME:/host-home" --container-workdir=/host-home \
  bash -lc 'export HF_HOME=/host-home/hf-cache; /host-home/vllm-shared/prebuilt-venv/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(model=\"Qwen/Qwen3-0.6B\", max_model_len=512, gpu_memory_utilization=0.3, enforce_eager=True)
print(llm.generate([\"The capital of France is\"], SamplingParams(max_tokens=8, temperature=0.0))[0].outputs[0].text)"
'
```

Pass criteria for a smoke run: the engine starts, the generation returns text, and the
assertion on that output holds. Budget conservatively - one GPU and under ~40 minutes
for a small model; ask before requesting multi-GPU. Never claim an end-to-end result
without the command output that shows the model actually produced tokens.

Give each unit of work its own tree, and never let a job rewrite someone else's checkout:

```text
$HOME/vllm-tasks/<task>/
├── src/     # its own git worktree pinned to an explicit commit
├── .venv/   # only when extra deps are needed; otherwise use the shared base-venv
└── logs/
```

Rules that keep this working:

- A job script must **not** run `git reset --hard`, `git checkout`, or anything else that rewrites the worktree. Verify the commit and fail loudly instead, so parallel sessions cannot clobber each other. Run this check **before** entering the container: job-node images have no `git` binary.
  ```bash
  expected=<sha>; actual=$(git -C "$SRC" rev-parse HEAD)
  [ "$actual" = "$expected" ] || { echo "HEAD mismatch: $actual != $expected" >&2; exit 2; }
  ```
- Pin a worktree with `git worktree add --detach <dir> <sha>`, and resolve new commits on the head node (fetch, then `git worktree remove` and re-add, or add a fresh directory) instead of mutating a shared checkout in place.
- Install Python deps from the checked-out `requirements/`, so the environment matches the commit under test.
- Prefer `uv` for Python: it provisions the interpreter itself and its wheel cache makes a rebuild cheap. `pip install -e .` is usually unnecessary - putting the checkout on `PYTHONPATH` is enough when the change under test does not need compiled extensions.

## Prepare the remote test

1. On the head node, use the intended vLLM commit or branch under `/home`; inspect the working tree before changing it. Do not overwrite existing remote work.
2. State the test objective, model and data locations, GPU count, time limit, and expected pass condition before submitting work.
3. Inspect capacity with `sinfo` and current jobs with `squeue`. Request only the GPUs, CPU, memory, and time needed.

## Run with Slurm

Every GPU command must explicitly request H20 GPUs with `--partition=h20 --gres=gpu:h20:<count>`. Changing `CUDA_VISIBLE_DEVICES` never grants additional GPU access.

For an interactive investigation, allocate a shell, then confirm the allocation and visible device:

```bash
srun --partition=h20 --gres=gpu:h20:1 --time=01:00:00 --pty bash
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"
```

For reproducible or longer validation, use a batch script with at least the partition, GPU request, time limit, and an output file, then submit it with `sbatch`. Put the actual test command and any environment activation in that script. Use `squeue`, `scontrol show job <job_id>`, and the job's output file to diagnose it; use `scancel <job_id>` only for jobs that belong to the requester.

This cluster runs an older Slurm that rejects long `squeue -o` format strings - use the default output. CPU and memory are scheduled but not hard-enforced, so keep requests modest on the large GPU nodes.

Use `salloc` when multiple `srun` commands must share an allocation. Exit the allocation shell when finished so resources are released.

## Containers

Pyxis/enroot is available on workers. When an isolated CUDA environment is needed, pass an image through `--container-image`. The home directory is mounted by default, which is what makes the shared environment above usable inside the container:

```bash
srun --partition=h20 --gres=gpu:h20:1 --cpus-per-task=8 --mem=64G --pty \
  --container-image='nvcr.io#nvidia/cuda:12.4.1-base-ubuntu22.04' \
  --container-mounts="$HOME:/host-home" \
  --container-workdir=/host-home/vllm-tasks/<task>/src \
  bash -lc '"$HOME/vllm-shared/base-venv/bin/python" -m pytest -q tests/...'
```

Facts that cost time to rediscover:

- `nvcr.io#nvidia/cuda:*-devel` images ship the CUDA toolchain and a C compiler but **no** torch, vllm, pytest, `pkg-config`, `libssl-dev`, `curl`, or even `git`, and their rootfs is **read-only**. Install into a writable mounted directory (a venv under `/host-home`), never into the image, and do any git work outside the container.
- `--container-env` mis-parses values containing `=` or `,` here. Export such variables inside the container's own script and keep only simple path values in `--container-env`.
- For Rust builds, point `OPENSSL_DIR` / `OPENSSL_LIB_DIR` / `OPENSSL_INCLUDE_DIR` at
  `$HOME/vllm-shared/openssl-ctr/usr` (Ubuntu 22.04 OpenSSL 3.0.2, matched to the
  container), because the job-node image has no development headers.
  `$HOME/vllm-tasks/pr55781/{run-rust.sh,rust-inner.sh}` do this and build into the
  shared `CARGO_TARGET_DIR=$HOME/vllm-shared/cargo-target`, so a later session reuses the
  compiled artifacts. Re-creating that directory from Ubuntu debs on the head node needs
  zstd: its `tar` 1.30 cannot read the `data.tar.zst` inside a `.deb`. Flatten
  `include/x86_64-linux-gnu/openssl/*.h` into `include/openssl/` too, or `openssl-sys`
  reports "Failed to find OpenSSL development headers" because its single `-I` misses the
  arch-specific `opensslconf.h`.
- Compile in a Slurm allocation with a limited `CARGO_BUILD_JOBS`; building on the head node competes for its small memory.
- An image that already bundles CUDA torch (an NGC PyTorch image) can host a full end-to-end model run without building a CUDA venv. Create its venv under `/host-home` (or reuse the image interpreter with `--system-site-packages`) so the result persists for later sessions.
- `$HOME/vllm-shared/README.md` records the shared layout, the reuse order, and the measured traps; keep it in sync when the environment changes.

Before running vLLM, verify `nvidia-smi`, the expected CUDA toolchain (`/usr/local/cuda`), and that the code and test dependencies are available in the selected environment.

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

- `PRE_COMMIT_HOME` must be **node-local**. `$HOME/vllm-shared/pre-commit-cache` looks
  right but the pip wheel build of `ruff-pre-commit` dies there with
  `error: [Errno 524] Unknown error 524: 'ruff_pre_commit.egg-info/SOURCES.txt'`; the
  home mount is NFS. `/var/tmp/vllm-shared-precommit` holds the populated cache that
  sessions reuse.
- `rust-cargo-fmt` shells out to `cargo fmt`, so the shared toolchain needs
  `rustup component add rustfmt` (with `RUSTUP_HOME`/`CARGO_HOME` on the shared dirs).
  That hook is marked "only run locally as Buildkite will cover this" - treating it as
  optional because "CI does not run it" just moves the failure to Buildkite.
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
