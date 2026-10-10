---
name: npu-test-runner
description: |
  Connect to the remote Ascend NPU machine (192.168.13.188), enter the
  liurong_vllm Docker container, pull the latest vllm-ascend code, and run
  pytest test cases on real NPU hardware. Also covers session isolation for
  concurrent sessions sharing that container (npu-slot.sh gives each session its
  own git worktree, NPU chips, port and tmpdir), NPU device selection
  (ASCEND_RT_VISIBLE_DEVICES), code sync from local, and common troubleshooting
  for NPU test failures.

  Use this skill whenever the user mentions:
  - Running tests on the NPU machine or remote NPU
  - Connecting to 192.168.13.188 or the liurong_vllm container
  - NPU device visibility, ASCEND_RT_VISIBLE_DEVICES, or switching NPU cards
  - Debugging test failures that need real NPU hardware
  - Syncing vllm-ascend code to the remote test environment
  - pytest commands that need to run on Ascend hardware
  - Two sessions / agents / test runs fighting over the shared container
    (port 8770 already in use, OOM because another run holds the card, edits or
    branch switches appearing from nowhere, a killed vllm server)
---

# NPU Test Runner

Connect to the remote Ascend NPU machine, enter the development container, pull
the latest vllm-ascend code, and run pytest test cases on real NPU hardware
(Ascend 910B/C).

## Quick reference

| Item | Value |
|------|-------|
| Remote host | `192.168.13.188` |
| SSH user | `root` |
| SSH tunnel | `-R 7897:127.0.0.1:7897` (reverse proxy for local services) |
| Container name | `liurong_vllm` (**shared** — never restart it) |
| Work directory | `/home/liurong` |
| Primary repo (in container) | `/home/liurong/vllm-ascend` |
| Session slot allocator | `/home/liurong/tools/npu-slot.sh` |
| Slot worktrees | `/home/liurong/ws/<session-id>` |
| NPU inventory | 8 cards × 2 dies = `/dev/davinci0..15` (16 visible devices) |

> **This container is shared between sessions.** `/home` and `/tmp` are bind
> mounted from the host, the container runs with `--net=host`, all 16 NPU dies
> are exposed, and every `docker exec` shares one pid namespace. That means two
> sessions working in `/home/liurong/vllm-ascend` at the same time will fight
> over the same files, the same cards and the same port 8770.
>
> **Rule of thumb: claim a session slot (Step 3) before you touch code or run
> tests.** It costs one command and removes all five classes of conflict.

---

## Step 1: Connect to the remote machine

From your local terminal, establish the SSH connection with a reverse tunnel:

```bash
ssh -R 7897:127.0.0.1:7897 root@192.168.13.188
```

The `-R 7897:127.0.0.1:7897` flag creates a reverse tunnel so that services
running on your local port 7897 (e.g., a proxy or IDE) are reachable from the
remote machine on `127.0.0.1:7897`.

If the connection times out, check:
- Your network/VPN connection to the 192.168.13.x subnet
- That the machine is powered on and reachable

---

## Step 2: Enter the Docker container

Once logged into the remote machine, enter the development container:

```bash
docker exec -it liurong_vllm bash
```

If the container is not running, start it first:

```bash
docker start liurong_vllm
docker exec -it liurong_vllm bash
```

**Do not `docker restart` or `docker rm` this container.** Another session may be
mid-run; restarting kills its vllm servers and its NPU processes. `docker start`
on a stopped container is fine.

All subsequent steps in this skill run **inside the container** unless noted.

---

## Step 3: Claim a session slot (do this first)

### Why

Five resources are shared host-wide by every session in this container:

| Shared resource | Why it conflicts |
|-----------------|------------------|
| Git working tree | `/home` is bind mounted, so everyone edits `/home/liurong/vllm-ascend`; `git checkout`/`pull`/WIP edits from one session land in another's test run |
| NPU chips | all 16 `/dev/davinci*` are visible (`--privileged`); two sessions default to chip 0 → OOM / device init clashes |
| TCP ports | `--net=host`, and the rlhf e2e conftest defaults to port **8770** → "address already in use", or tests talking to the *other* session's server |
| tmp / cache dirs | `/tmp` is bind mounted from the host, so pytest basetemp / inductor / HCCL temp files collide |
| Process namespace | `docker exec` shares one pid space — one session's `pkill -f python` kills the other's vllm server |

`npu-slot.sh` leases a slot `{worktree, chips, port block, tmpdir}` per session
under `/home/liurong/.npu-slots/`, with an 8h TTL, locked allocation, and
process-group scoped `run`/`kill`.

### Acquire

```bash
# Pick a stable, unique id for your session (branch name, feature, agent name).
eval "$(/home/liurong/tools/npu-slot.sh acquire \
        --session rl-e2e-alice --chips 2 --ref origin/rl-e2e)"
```

This prints (and the `eval` exports) exactly what you need:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1      # your chips, nobody else's
VLLM_RL_TEST_PORT=8800             # your port; the slot patcher makes the conftest read it
NPU_SLOT_WORKTREE=/home/liurong/ws/rl-e2e-alice
TMPDIR=/home/liurong/ws/rl-e2e-alice/.tmp
TORCHINDUCTOR_CACHE_DIR=... VLLM_CACHE_ROOT=...
PYTHONPATH=/home/liurong/ws/rl-e2e-alice   # pins imports to your tree
```

What `acquire` does for you:

- creates `/home/liurong/ws/<session>` with `git worktree add --detach` from
  `--ref` (shares `.git` objects, so it is cheap and has no branch/index races)
- symlinks the prebuilt custom ops into the new worktree, so you do not have to
  rebuild `vllm_ascend/_cann_ops_custom` (`--no-seed-ops` not exposed: it is
  skipped automatically when the artifacts are missing)
- patches the rlhf conftest in **your own worktree** to honour
  `VLLM_RL_TEST_PORT` (falls back to a free port), so port 8770 stops colliding
- reserves chips that are both unleased and physically idle
  (`npu-smi` HBM below `NPU_SLOT_CHIP_BUSY_MB`, default 8192 MiB)

Useful flags: `--chips N` (2 for TP=2 tests), `--ref <git-ref>` (defaults to the
current `HEAD` of the primary repo), `--no-worktree` (work in the shared checkout
— only when you are certain you are alone).

### Use it

```bash
# Preferred: own process group + full log file, and cleanup can never hit another session
/home/liurong/tools/npu-slot.sh run --session rl-e2e-alice -- \
    pytest -sv tests/e2e/pull_request/one_card/rlhf/
```

`run` accepts a single argument as shell syntax, so pipes still work:

```bash
/home/liurong/tools/npu-slot.sh run --session rl-e2e-alice -- \
    'pytest -q tests/e2e/pull_request/one_card/rlhf/ | tail -30'
```

The plain form works too — just `cd "$NPU_SLOT_WORKTREE"` first (the env from
`acquire` already points there):

```bash
cd "$NPU_SLOT_WORKTREE"
pytest -sv tests/e2e/pull_request/one_card/rlhf/
```

### Inspect, extend, release

```bash
npu-slot.sh status                      # who holds which chips/ports + per-die HBM usage
npu-slot.sh renew --session X --hours 4 # extend the lease (default TTL 8h, no auto-renew)
npu-slot.sh kill  --session X           # kill only this session's process group
cd /home/liurong                        # step OUT of the worktree before removing it
npu-slot.sh release --session X --rm-worktree
npu-slot.sh gc                          # drop expired leases + prune worktree metadata
```

`status` lists the lease table and each die's HBM. A die above the busy
threshold that no lease owns means someone started a process outside the
allocator — talk to them instead of guessing.

If the allocator is missing on the machine, redeploy it from this skill:

```bash
scp .agents/skills/npu-test-runner/scripts/npu-slot.sh \
    .agents/skills/npu-test-runner/scripts/patch_rlhf_port.py \
    root@192.168.13.188:/home/liurong/tools/
```

---

## Step 4: Get the code into your slot

A slot worktree is created at exactly the `--ref` you asked for, so no `pull` is
needed to start. To move it to a newer commit:

```bash
cd "$NPU_SLOT_WORKTREE"
git fetch origin
git checkout --detach origin/rl-e2e        # or any branch/tag/commit
git log --oneline -10
git status
```

Commit/push from the worktree as usual (it shares `.git` with the primary repo).
The primary checkout `/home/liurong/vllm-ascend` stays untouched, which is the
point: other sessions can sit on other branches or hold uncommitted work.

### Syncing code from your local machine

**Option A — Push to a remote branch (recommended, works with slots):**

```bash
# On your local machine
git add <files>
git commit -s -m "your message"
git push origin <branch-name>

# Then inside your slot
cd "$NPU_SLOT_WORKTREE" && git fetch origin && git checkout --detach origin/<branch-name>
```

**Option B — Use scp to copy individual files into your slot:**

```bash
# From your local machine (NOT inside the container)
scp tests/e2e/pull_request/one_card/rlhf/conftest.py \
    root@192.168.13.188:/home/liurong/ws/rl-e2e-alice/tests/e2e/pull_request/one_card/rlhf/
```

**Option C — Use rsync for directories:**

```bash
# From your local machine (NOT inside the container)
rsync -avz tests/e2e/pull_request/one_card/rlhf/ \
    root@192.168.13.188:/home/liurong/ws/rl-e2e-alice/tests/e2e/pull_request/one_card/rlhf/
```

Never `pip install -e` in the shared container: vllm/vllm_ascend are installed
editable pointing at `/vllm-workspace/vllm-ascend`, and re-installing rewrites
that mapping for **every** session. Use `PYTHONPATH` (the slot sets it) instead.

---

## Step 5: Run tests

Run from inside your slot, so `PYTHONPATH`/`ASCEND_RT_VISIBLE_DEVICES`/`TMPDIR`
are the slot's:

```bash
cd "$NPU_SLOT_WORKTREE"
pytest -sv <test_path>            # or: npu-slot.sh run --session <id> -- pytest -sv <test_path>
```

### Test command templates

**Run a single test file:**
```bash
pytest -sv tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py
```

**Run a single test class or method:**
```bash
pytest -sv tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py::TestSleepWakeFlags
pytest -sv tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py::TestSleepWakeFlags::test_sleep_sets_is_sleeping_true
```

**Run all rlhf one-card tests:**
```bash
pytest -sv tests/e2e/pull_request/one_card/rlhf/
```

**Run with a specific marker:**
```bash
pytest -sv -m "e2e_model" tests/e2e/pull_request/one_card/rlhf/
```

**Run with verbose output and no capture (for debugging):**
```bash
pytest -svv --tb=long tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py
```

**Run with timeout override:**
```bash
pytest -sv --timeout=600 tests/e2e/pull_request/one_card/rlhf/
```

**Run and stop on first failure:**
```bash
pytest -sv -x tests/e2e/pull_request/one_card/rlhf/
```

Other suites live next to it: `tests/e2e/pull_request/{one_card,two_card,four_card,eight_card}/`
and the units under `tests/ut/` (no NPU needed).

---

## Step 6: NPU device management

### Your slot already owns your chips

`acquire` exported `ASCEND_RT_VISIBLE_DEVICES` — do not override it by hand, or
you will land on another session's die. Verify what you have:

```bash
echo "$ASCEND_RT_VISIBLE_DEVICES"
python -c "import torch; print('NPU count:', torch.npu.device_count())"
```

### Asking for more cards

Multi-card tests need the devices to be visible; re-acquire with the chip count
you need (the slot must be released first):

```bash
npu-slot.sh release --session rl-e2e-alice --rm-worktree
eval "$(npu-slot.sh acquire --session rl-e2e-alice --chips 2 --ref origin/rl-e2e)"
pytest -sv tests/e2e/pull_request/two_card/
```

### Without a slot (single-session, ad hoc)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0      # or 0,1 for two-card tests
npu-smi info                            # show per-die utilization / HBM
```

### When to change visible devices

- **Multi-card tests fail with "not enough NPUs"**: acquire the slot with
  `--chips 2` (or more) instead of hand-picking ids
- **Out of memory errors**: another session may hold that die — check
  `npu-slot.sh status`, then grow the slot or wait; do not kill other sessions'
  processes
- **Default behavior**: if `ASCEND_RT_VISIBLE_DEVICES` is unset, all 16 dies are
  visible and the `torch.npu.device_count()` checks in test files gate multi-card
  tests

---

## Common issues and solutions

### Port 8770 already in use / "vllm server exited during startup"

Two sessions are running the same rlhf tests on the host network. Use a slot
(Step 3) so each session gets its own `VLLM_RL_TEST_PORT`, or patch your own
worktree by hand:

```bash
python3 /home/liurong/tools/patch_rlhf_port.py "$NPU_SLOT_WORKTREE"
```

### My vllm server / pytest process vanished mid-run

Another session ran `pkill -f python`/`pkill pytest` in the shared pid space.
Use `npu-slot.sh run` (own process group) and clean up with
`npu-slot.sh kill --session <id>`; never `pkill` in the shared container.

### My edits disappeared, or the branch changed under me

You are working in the shared checkout `/home/liurong/vllm-ascend`, which other
sessions also use. Move to a slot worktree (Step 3) and re-apply your changes
there.

### "Which vllm_ascend am I actually importing?"

`vllm`/`vllm_ascend` are installed editable pointing at
`/vllm-workspace/vllm-ascend`, which is a **different tree** from your checkout:

```bash
python -c "import vllm_ascend; print(vllm_ascend.__file__)"
```

`pytest` run from the repo root imports your tree (pytest prepends the rootdir),
and `PYTHONPATH=$NPU_SLOT_WORKTREE` pins it everywhere else — the slot sets this.
Running `python some_script.py` from outside the tree silently uses
`/vllm-workspace/vllm-ascend`.

### "NPU not available" or "torch.npu not found"

The CANN environment may not be sourced. Run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Or check that the torch-npu package is installed:

```bash
python -c "import torch_npu; print(torch_npu.__version__)"
```

### "device_count() < 2" — multi-card tests skipped

The slot was acquired with `--chips 1`. Re-acquire with `--chips 2` (see Step 6).

### Out of memory (OOM) on NPU

Reduce memory pressure:
- Lower `gpu_memory_utilization` in test configs (e.g., from 0.95 to 0.6)
- Reduce `max_model_len` (e.g., from 8192 to 1024)
- Check `npu-slot.sh status` for the die's real owner before blaming your code

### Container not running

```bash
docker ps -a | grep liurong_vllm   # check container status
docker start liurong_vllm           # start if stopped (never "docker restart")
docker exec -it liurong_vllm bash   # enter
```

### "Permission denied" on SSH

Check that your SSH key is authorized on the remote machine. If using a password,
ensure you have the correct credentials.

### Test timeouts

RL tests involve server startup and sleep/wake cycles which can be slow on NPU.
Increase the pytest timeout:

```bash
pytest -sv --timeout=1200 tests/e2e/pull_request/one_card/rlhf/
```

### Weight transfer / HCCL tests skipped

These tests require `--weight-transfer-config` which needs NCCL/HCCL setup.
They are skipped by default unless `VLLM_RL_TEST_WEIGHT_TRANSFER=1` is set.

### Model download issues (HuggingFace cache)

The cache at `/root/.cache/huggingface` is shared by every session in this
container and the rlhf harness runs with `HF_HUB_OFFLINE=1`, so pre-download
models instead of relying on a mid-test fetch:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')"
```

---

## One-liners for common workflows

**Quick connection + enter container:**
```bash
ssh -R 7897:127.0.0.1:7897 root@192.168.13.188 -t "docker exec -it liurong_vllm bash"
```

**Claim a slot and run the rlhf tests on it:**
```bash
eval "$(/home/liurong/tools/npu-slot.sh acquire --session my-session --chips 2 --ref origin/rl-e2e)" \
  && /home/liurong/tools/npu-slot.sh run --session my-session -- pytest -sv tests/e2e/pull_request/one_card/rlhf/
```

**Who is using what right now:**
```bash
/home/liurong/tools/npu-slot.sh status
```

**Release when done:**
```bash
cd /home/liurong && /home/liurong/tools/npu-slot.sh release --session my-session --rm-worktree
```

**Single-session only — run one-card rlhf tests in the shared checkout:**
```bash
cd /home/liurong/vllm-ascend && ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv tests/e2e/pull_request/one_card/rlhf/
```

---

## Related files

- `scripts/npu-slot.sh` — per-session slot allocator (worktree/chips/port/tmpdir)
- `scripts/patch_rlhf_port.py` — makes the rlhf conftest read `VLLM_RL_TEST_PORT`
- `tests/e2e/pull_request/one_card/rlhf/` — rlhf e2e suite used in the examples
- `references/troubleshooting.md` — Detailed NPU troubleshooting guide
