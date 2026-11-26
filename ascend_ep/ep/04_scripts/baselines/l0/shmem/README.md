# SHMEM baselines

`LINK-012` uses the SHMEM MTE engine through `shmem.core.put` (host stream).
`LINK-013` uses the UDMA engine, either from the device kernel in
`native/udma_kernel.cpp` (multi-peer, multi-core, multi-outstanding, quiet policy)
or from the host stream, so the two submission paths of one engine stay distinct.

`HW-007`/`HW-008` keep their original behaviour and share this code.

URMA is rejected on purpose: the pinned SHMEM release exposes
`MTE`, `SDMA`, `ROCE` and `UDMA` data-operation engines only
(`include/host_device/shmem_common_types.h`).  UDMA is a data-movement engine, not
a remote-access mechanism, and no result here is labelled URMA.

Build the kernels:

```sh
bash baselines/l0/shmem/native/build_udma.sh /path/to/shmem-checkout /path/to/build
# then set benchmark.shmem_udma_library to .../libep_l0_udma.so
```

Pinned upstream revision: `f0bcfc0fe51bce1d866ba6935d2adc6e5e5800e5`.
Requirements: `benchmark.shmem_rendezvous` (TCP endpoint, distinct from torchrun),
`shmem` python package, Ascend 950 for UDMA.

Constraints taken from the SHMEM headers, not guessed:

- one UDMA request transfers at most 256 MB,
- `ACLSHMEM_MAX_QP_NUM` is 56 and `aclshmemx_set_qp_num` must agree on every PE,
- `aclshmemx_udma_qp_quiet(pe, qp)` drains one QP; `aclshmemx_udma_quiet(pe)`
  drains the QP 0 group,
- the host-stream Python API has no QP index and no quiet control.

Evidence: `native_descriptor` (engine selected at init + the exact API symbol
invoked).  The physical link (UB vs RoCE) stays `unverified` unless
`aclshmemx_get_qp_net_layer` is wired in by the caller.