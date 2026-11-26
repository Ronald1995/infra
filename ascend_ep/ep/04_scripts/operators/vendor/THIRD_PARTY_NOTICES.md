# Third-party notices

This project contains source code copied from a third-party project. Read this file before
redistributing the repository or shipping a package that includes it.

## Component: ascendep expert-parallel adapters

| | |
|---|---|
| Upstream project | `ascendep` — <https://github.com/penn513/ascendep> |
| Upstream commit | `24ff84d3ab8c01e1e2071802465173e624e1f89d` ("Add Ascend expert-parallel benchmarks and profiling tools", 2026-11-24, author `gp513 <gp513@outlook.com>`) |
| Copied subdirectory | `ep/` |
| Vendor location | `ascend_ep/ep/04_scripts/operators/vendor/ascendep/ep/` |
| Files copied | 23 — see `ascendep/VENDOR.json` for the per-file list and `sha256` |
| Modifications | none; every file is a byte-exact copy |
| Purpose | runtime adapters for the Ascend expert-parallel benchmarks in this repository |

### Copyright holder

**Upstream states no copyright notice.** No file at the pinned commit carries a copyright
line, an author header, or a `@license` / SPDX identifier. The only attributable identity
present in the source is the git metadata of the copied commit — author
`gp513 <gp513@outlook.com>` in the repository `penn513/ascendep`. The repository owner is
not named in any copyright statement, so no copyright holder is asserted here; inventing
one would be inaccurate.

### License status: NO LICENSE DECLARED UPSTREAM

At commit `24ff84d3ab8c01e1e2071802465173e624e1f89d`, the upstream repository contains
**no license file of any kind**. The following were searched for, recursively, over the
whole tree and none exists:

* `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `LICENCE`, `LICENCE.md`
* `COPYING`, `COPYING.md`, `COPYING.LESSER`
* `NOTICE`, `NOTICE.md`, `AUTHORS`, `CONTRIBUTORS`

The upstream `README.md` and `pyproject.toml` declare no `license` field, no license
classifier, and no license text; `grep -i "copyright|license|apache|mit"` matches nothing in
any `.py`, `.md`, `.toml`, `.json`, `.sh` or `.txt` file at that commit.

**No license is granted or implied by this repository for the copied files.** They remain
the property of their authors, with all rights reserved by default.

### Compliance item: owner sign-off required

> **This vendored copy must not be redistributed outside the organisation (and must not be
> published as a package, wheel, container image, or public mirror) until the project owner
> has signed off on the licensing position.**

The concrete items to resolve before wide redistribution:

1. Obtain a written license grant from the upstream author (`gp513 <gp513@outlook.com>`,
   repository owner `penn513`), or
2. obtain the owner's decision that this internal, non-redistributed use is acceptable
   as-is, and record that decision next to this file, or
3. replace the vendored adapters with an independently licensed implementation.

Until one of those is recorded, treat the tree as **internal-use only**. This notice does
not grant any rights and does not change the upstream terms. `VENDOR.json` carries the same
finding in machine-readable form (`license.status: "absent"`).

### What is not covered by this notice

The copied files call into native libraries and external frameworks that are **not**
vendored here and carry their own licenses — DeepEP (`deep_ep`,
`deep_ep_experimental`), SHMEM, `npu_patch`, `fsdp_turbo`, `mega_moe`, `mindspeed`,
`megatron`, `cann_ops_transformer`, and `torch_npu`. Those are installed and licensed
separately through the dependency manifest (`dependencies/manifest.yaml`) and are not part
of this notice.

## How this copy is kept honest

`ascendep/VENDOR.json` records the upstream URL, commit, subdirectory, the `sha256` of every
copied file, the reason each file is needed, and every upstream file that was deliberately
left out with its reason. `scripts/vendor_ascendep.py --check` re-verifies the digests and
fails if the tree, the manifest or the pinned commit in `operators/ascendep.py` drift apart.
