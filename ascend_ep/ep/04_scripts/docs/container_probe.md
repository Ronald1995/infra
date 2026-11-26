# Container probe (`root@192.168.13.188`, liurong_vllm)

Recorded on the deployment the benchmark is expected to run on. This is evidence, not a
promise: the catalog's pinned pairing stays what this repository validated, and the
doctor reports the observed triple against it.

## Observed

| Item | Observed value |
|---|---|
| host / container | `node-zj-110-132`, Debian GNU/Linux 10 (buster) |
| `npu-smi` | 25.5.1 (NPU present) |
| CANN install root | `/usr/local/Ascend/cann-9.0.0`, `ascend-toolkit/latest` symlink |
| toolkit env script | `/usr/local/Ascend/ascend-toolkit/set_env.sh` (exists) |
| system python | `/usr/bin/python3.7` only (no 3.10-3.12 in `/usr/bin`) |
| `torch` / `torch_npu` | not importable by the default `python3` |
| `.bashrc` | sources `/usr/local/Ascend/ascend-toolkit/set_env.sh` and reports it missing at login |

## What this changes in the repository

* The container's CANN 9.0.0 is recorded in the `cann-toolkit` entry's notes and in
  this document, **not** as a compatibility row: a row needs concrete torch/torch_npu
  versions, and the probe could not read them. The doctor therefore reports the pairing
  as unlisted (`skip`, never `pass`) until an NPU run confirms it.
* Nothing is re-pinned: the benchmark pins stay the validated ones and the doctor warns
  when the container differs. A mismatch is an image-tag problem.
* Two open items are recorded rather than assumed away:
  1. **No 3.10-3.12 interpreter.** The framework requires `>=3.10,<3.13`; the image
     must provide one (conda/uv/venv or a newer base image) before an NPU run. The
     `python` dependency stays `host`/`container`-provided with this requirement.
  2. **torch/torch_npu not importable by the default interpreter.** The user stated the
     container provides them; the probe could not find them for `python3`, so the run
     must start from the interpreter where they are installed, or the image must expose
     them. The doctor's `torch`/`torch-npu` checks stay verify-only and will report the
     gap instead of installing over the image.
  3. The `.bashrc` login error is an image issue (the file exists, so the sourcing order
     or a mount is wrong); it does not affect `set_env.sh` when sourced explicitly.

## Reproduce

```sh
ssh root@192.168.13.188 'bash -s' < container-probe.sh
```

`container-probe.sh` only reads versions and paths; it installs nothing.
