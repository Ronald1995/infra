# Native transport baseline

`LINK-001..011` drive HCOMM channel creation and HCOMM data-plane primitives
through one optional shared library, `native/l0_transport_bridge.cpp`.  Without
that library every case reports `dependency_missing`; no case falls back to
another engine or protocol.

## Request table

The request is one `(engine, protocol, endpoint location)` triple per case (see
`common/profiles.py`).  The mapping is derived from the HCOMM public headers, not
from channel class names:

| Case | Engine | Protocol | Channel HCOMM creates |
| --- | --- | --- | --- |
| LINK-001 | `AIV` | `UB_MEM` | `AivUbMemChannel` |
| LINK-002 | `AIV` | `UB_CTP` | `AivUrmaChannel` (URMA family) |
| LINK-003 | `AIV` | `UB_CTP` | same descriptor as LINK-002 |
| LINK-004 | `AIV` | `UB_RTP` | `AivUrmaChannel` |
| LINK-005 | `AICPU_TS` | `UB_CTP` | `AicpuTsUrmaChannel` (URMA family) |
| LINK-006 | `AICPU_TS` | `UB_CTP` | same descriptor as LINK-005 |
| LINK-007 | `AICPU_TS` | `UB_RTP` | `AicpuTsUbRtpChannel` |
| LINK-008 | `CCU` | `UB_CTP` | `CcuUrmaChannel` (CCU requires `UB_CTP`/`UBC_TP`) |
| LINK-009 | `AICPU_TS` | `HCCS` | `AicpuTsHccsChannel` (A2/A3 endpoints only) |
| LINK-010 | `AICPU_TS` | `ROCE` | `AicpuTsRoceHostLocChannel` |
| LINK-011 | `AICPU_TS` | `UBOE` | `AicpuTsUboeChannel` |

Two facts make the URMA cases and the CTP cases share a descriptor:

- `URMA` is not a `CommProtocol`.  It names the channel implementation family.
- HCOMM dispatches on `remoteEndpoint.protocol`; `UB_CTP` is the protocol the
  URMA-family channels are built with.

The URMA case therefore verifies the **channel family**, the CTP case verifies
the **protocol**, and each records the other fact as context.

## Verification

The bridge reports, per peer:

- the protocol the HCCL rank graph advertises (`CommLink.linkAttr.linkProtocol`),
- the protocol and engine handed to `HcclChannelAcquire`,
- whether a channel handle came back and whether remote memory was resolved,
- which submission overload the data path accepted (`plain_write_nbi`,
  `thread_write_nbi`, `thread_write_with_notify_nbi`).

A mismatch between the rank-graph protocol and the requested protocol produces
`fallback`.  A channel created without a rank-graph observation stays
`unverified` for the protocol.  HCOMM exposes no wire-protocol query, so
`observed.physical_link` is always `unverified` unless an HCOMM log says more.

## Build

```sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
bash baselines/l0/transport/native/build.sh
export EP_L0_TRANSPORT_LIBRARY=$PWD/baselines/l0/transport/native/build/libl0_transport_bridge.so
```

The bridge is compiled against the HCOMM 9.2 public API (`include/hcomm_*`,
`include/hccl/*`).  It was **not** compiled or executed while this repository was
prepared: no CANN toolkit was available.  Treat the CHANNEL data path as
`unverified` until it has been built and run on an Ascend 950 host.

## Limits

- `HcclChannelDestroy` currently documents support for CCU-created channels; the
  bridge calls it for every channel and ignores the result.
- The HCCL test and this bridge both need a live communicator; the root
  information blob is exchanged through the existing torch.distributed
  rendezvous.
