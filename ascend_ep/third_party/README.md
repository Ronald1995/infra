# third_party: pinned sources that must arrive as bytes

The A5 host cannot reach GitHub or GitCode. Everything this project needs from
upstream therefore has to be brought over and **verified**, never assumed.

## Layout

```
ascend_ep/
├── third_party/
│   ├── sources.yaml          # the catalogue: full commit, licence, operators, commands
│   ├── megatron-lm/          # gitignored checkout at the pinned commit
│   ├── DeepEP/
│   ├── FSDPTurbo/
│   └── …
└── packages/
    ├── envs/                 # the four isolated interpreters (env_manager.py --create)
    ├── wheelhouse/           # TUNA wheels downloaded on a networked machine
    └── native/               # build artefacts collected for transfer
```

Only `sources.yaml`, this README and the `.gitignore` are tracked; the checkouts and
the artefacts are data.

## Populate it

On a machine with network (the Windows workstation is fine):

```bash
python scripts/fetch_third_party.py --fetch            # all pinned sources
python scripts/fetch_third_party.py --fetch --only fsdp-turbo,shmem
```

Then copy `ascend_ep/third_party/` to the target host and verify there:

```bash
python scripts/fetch_third_party.py --check            # exit 1 if anything is off
python scripts/fetch_third_party.py --record-licenses  # read the licence each tree carries
```

## What "verified" means here

* `--check` compares `git rev-parse HEAD` with the catalogue's **full** 40-hex commit.
  A short sha is not accepted anywhere in this system.
* An entry pinned by release tag (`shmem` → `8.2.RC1`) reports the commit the
  checkout actually holds, so the tag is resolved rather than trusted.
* `triton-distributed-ascend` has no pin yet and reports `UNPINNED` instead of
  pretending it matches.
* A missing tree is `MISSING`, a different commit is `COMMIT_MISMATCH`, and neither
  is ever reported as present.

## What must NOT be here

`hccl`, `hcomm`, the driver, CANN and the compiler toolchain come from the
container/CANN. They are listed under `system_provided` in `sources.yaml` so nobody
vendors them, and the project never reinstalls or replaces them.

## Cross-machine rules

* Pure-Python wheels may be downloaded on Windows for the target's CPython 3.12 /
  AArch64 ABI, but a Windows `site-packages` must never be copied to A5.
* Any Ascend native extension must be built on A5. A Windows DLL is not an Ascend
  kernel.
* Ordinary Python dependencies do not travel through Windows: A5 reaches TUNA
  directly.
