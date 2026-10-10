# NPU Troubleshooting Reference

## Environment setup

### CANN toolkit
The Ascend CANN (Compute Architecture for Neural Networks) toolkit must be
sourced before using torch-npu:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Verify the installation:
```bash
python -c "import torch; import torch_npu; print(torch_npu.__version__)"
npu-smi info
```

### Key environment variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `ASCEND_RT_VISIBLE_DEVICES` | Limit visible NPU devices | `0`, `0,1`, `2,3` |
| `VLLM_ASCEND_ENABLE_NZ` | Enable NZ (Neural-Zone) optimization | `0` or `1` (default: `1`) |
| `VLLM_SERVER_DEV_MODE` | Enable dev-mode endpoints (/sleep, /wake_up) | `1` |
| `VLLM_WORKER_MULTIPROC_METHOD` | Worker process start method | `spawn` |
| `VLLM_RL_TEST_MODEL` | Override default test model | `Qwen/Qwen3-0.6B` |
| `VLLM_RL_LOGPROBS_TOLERANCE` | Tolerance for logprobs comparison | `0.01` |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | Allow pickle serialization for IPC | `1` |
| `HF_ENDPOINT` | HuggingFace mirror for model downloads | `https://hf-mirror.com` |

---

## Common error patterns

### Memory errors

**Symptom:** `torch.npu.OutOfMemoryError` or `ACL_ERROR_FAILURE`

**Causes and fixes:**
1. Other processes occupying NPU memory → check with `npu-smi info`, kill or use different card
2. `gpu_memory_utilization` too high → reduce to 0.5-0.6 for tests
3. `max_model_len` too large → reduce to 1024 for quick tests
4. Multiple models loaded simultaneously → ensure cleanup between tests
5. Memory fragmentation → restart the container

### Device visibility

**Symptom:** `RuntimeError: device_count() returned 0` or `NPU device not found`

**Fixes:**
1. Check `ASCEND_RT_VISIBLE_DEVICES` — if set to an empty or invalid value, unset it
2. Verify NPU drivers: `ls /dev/davinci*`
3. Check container has NPU access: `docker inspect liurong_vllm | grep -i device`
4. Source the CANN environment: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

### Sleep/wake test failures

**Symptom:** Tests using `/sleep` or `/wake_up` endpoints return 404

**Fix:** Ensure the server is started with `--enable-sleep-mode` and
`VLLM_SERVER_DEV_MODE=1`. These are automatically set by the `rlhf_server`
fixture in conftest.py.

**Symptom:** `is_sleeping` returns False right after `sleep(level=1)`

**Fix:** sleep operations may be asynchronous. Add a small delay (1-2s) or
poll in a loop:
```python
import time
time.sleep(2)
assert is_sleeping(server)
```

### Weight transfer test failures

**Symptom:** HCCL initialization fails with network errors

**Fix:** HCCL requires proper network configuration between ranks. Check:
- `master_address` and `master_port` are reachable
- No firewall blocking the ports
- `rank_offset` and `world_size` are correctly set

### Model download issues

**Symptom:** `OSError: Can't load config for Qwen/Qwen3-0.6B`

**Fix:**
```bash
# Use HF mirror for faster downloads in China
export HF_ENDPOINT=https://hf-mirror.com

# Or pre-download the model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-0.6B')"
```

### Container issues

**Symptom:** `docker exec` fails with "container not running"

**Fix:**
```bash
docker start liurong_vllm
# Wait a few seconds for initialization
sleep 5
docker exec -it liurong_vllm bash
```

**Never `docker restart`/`docker rm` this container**: it is shared, and a
restart kills every other session's vllm servers and NPU processes.

### Two sessions colliding on the shared container

`liurong_vllm` is shared: `/home` and `/tmp` are host bind mounts, the container
runs with `--net=host`, all 16 NPU dies are visible, and every `docker exec`
shares one pid namespace.

| Symptom | Cause | Fix |
|---------|-------|-----|
| `vllm server exited during startup: ... address already in use` | two sessions bind port 8770 | claim a slot (`npu-slot.sh acquire`; it exports `VLLM_RL_TEST_PORT`), or patch your own worktree: `python3 /home/liurong/tools/patch_rlhf_port.py <worktree>` |
| OOM / `ACL_ERROR_FAILURE` on a die you thought was free | another session's server holds it | `npu-slot.sh status` shows the lease table and per-die HBM; re-`acquire` with more chips instead of killing processes |
| pytest/server process disappeared with no traceback | someone ran `pkill -f python` in the shared pid space | run tests via `npu-slot.sh run` (own process group) and clean up with `npu-slot.sh kill --session <id>`; never `pkill` |
| edits vanished or the branch switched under you | everyone edits the shared checkout `/home/liurong/vllm-ascend` | work in a slot worktree `/home/liurong/ws/<session>` |
| `index.lock` exists / "another git process seems to be running" | two sessions in the same checkout | same as above — `git worktree add` never touches the other session's index |
| `import vllm_ascend` resolves to `/vllm-workspace/vllm-ascend` | the editable install points at a *different* tree than your checkout | run pytest from the repo root, or export `PYTHONPATH=<your worktree>` (slots do this automatically) |

Full recipe: `SKILL.md` → "Step 3: Claim a session slot (do this first)".

---

## Test-specific troubleshooting

### RL test prerequisites

- `--enable-sleep-mode` server flag (set automatically by `rlhf_server` fixture)
- `VLLM_SERVER_DEV_MODE=1` (set automatically by `_build_server_env`)
- Qwen3-0.6B model cached or accessible via HF

### Skipped test markers

Tests marked with the following are intentionally skipped:

| Skip condition | Reason |
|---------------|--------|
| `pytest.mark.skip("Weight checker not yet implemented")` | Upstream feature pending |
| `pytest.mark.skip("Indexer replay not yet implemented")` | Upstream feature pending |
| `pytest.mark.skip("Requires weight transfer configuration")` | Needs `--weight-transfer-config` |
| `pytest.mark.skipif(device_count() < 2, ...)` | Needs ≥ 2 NPUs |
| Auto-skip via `_weight_checker_endpoint_available()` | Endpoint not available |

### Running specific subsets

```bash
# Only one-card tests (skip multi-card)
pytest -sv tests/e2e/pull_request/rlhf/one_card/

# Only state transition tests (exclude weight checker)
pytest -sv tests/e2e/pull_request/rlhf/one_card/state_transitions/ \
    -k "not WeightChecker"

# Only sleep/wake tests
pytest -sv tests/e2e/pull_request/rlhf/one_card/state_transitions/test_sleep_wake.py

# Only tests that don't need multi-NPU
pytest -sv tests/e2e/pull_request/rlhf/ -k "not two_card"
```

---

## Performance notes

- **Server startup**: Qwen3-0.6B takes 30-90s to load on NPU with `--enforce-eager`
- **Sleep/wake roundtrip**: Expected < 60s for 0.6B model on Ascend 910B
- **Memory usage**: Qwen3-0.6B uses ~1.2 GiB NPU memory at BF16
- **Multi-card tests**: Require ≥ 2 NPUs, use `tensor_parallel_size=2`
