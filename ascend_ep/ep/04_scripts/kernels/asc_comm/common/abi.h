/*
 * asc_comm/abi.h -- the ABI between the Ascend EP benchmark and the ASC-COMM
 * communication kernels.
 *
 * This header is the contract.  The Python side mirrors it in
 * `baselines/l0/transport/dataplane.py` (`AscCommTransfer`); the two must be
 * changed together, and `kernels/asc_comm/tests/test_abi.py` fails if the field
 * order or the symbol list drifts.
 *
 * Every entry point is a *host launcher* symbol: it validates the descriptor,
 * binds the transfer list to the requested channel/engine and enqueues the device
 * program.  It must never perform the copy itself -- a host memcpy is not an AIV or
 * CCU data plane.
 */
#ifndef ASC_COMM_ABI_H
#define ASC_COMM_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version.  Bump on any layout change; the loader refuses a mismatch. */
#define ASC_COMM_ABI_VERSION 1u

/* One remote write.  Addresses are device virtual addresses of registered
 * symmetric buffers; `peer` is the global rank id. */
typedef struct {
  uint64_t src;    /* remote-write source, local device address */
  uint64_t dst;    /* remote-write destination on `peer` */
  uint64_t bytes;  /* transfer length in bytes; 0 is a no-op and is rejected */
  uint32_t peer;   /* destination global rank */
  uint32_t flags;  /* ASC_COMM_FLAG_* */
} asc_comm_transfer;

/* Ask the peer to publish an acknowledgement once the payload is visible. */
#define ASC_COMM_FLAG_NOTIFY (1u << 0)
/* Drain the engine's write queue before reporting completion. */
#define ASC_COMM_FLAG_DRAIN (1u << 1)

/* Status codes.  Negative values are failures; the launcher maps them to a
 * BenchmarkUnavailable status in Python. */
#define ASC_COMM_OK 0
#define ASC_COMM_ERR_ARGUMENT (-1)
#define ASC_COMM_ERR_NOT_INITIALIZED (-2)
#define ASC_COMM_ERR_CHANNEL (-3)
#define ASC_COMM_ERR_TIMEOUT (-4)
#define ASC_COMM_ERR_UNSUPPORTED (-5)

/* Which engines were compiled into this library. */
#define ASC_COMM_FLAG_AIV_BUILT (1u << 0)
#define ASC_COMM_FLAG_CCU_BUILT (1u << 1)

/* Filled in by the build; the probe reports them as evidence. */
#ifndef ASC_COMM_SOC_NAME
#define ASC_COMM_SOC_NAME "unknown"
#endif
#ifndef ASC_COMM_BUILD_ID
#define ASC_COMM_BUILD_ID "unversioned"
#endif

/* ------------------------------------------------------------------ AIV ----
 * Submit `count` remote writes on the AIV engine bound to `channel`.
 * Returns ASC_COMM_OK once the descriptors are validated and the kernel is
 * enqueued; completion is established by asc_comm_aiv_drain().
 */
int asc_comm_aiv_write_nbi(uint32_t channel, const asc_comm_transfer* transfers,
                           uint32_t count);

/* Drain the AIV write queue bound to `channel` and wait until every submitted
 * write has retired, with an optional timeout in nanoseconds (0 = no timeout).
 * This is the AIV data-plane completion; it must not be confused with a host
 * channel fence.
 */
int asc_comm_aiv_drain(uint32_t channel, uint64_t timeout_ns);

/* ---------------------------------------------------------------- CCU -----
 * Submit `count` remote writes to the CCU scheduler bound to `channel`, and wait
 * for its completion report.  The CCU data plane has no host-side fence path.
 */
int asc_comm_ccu_submit(uint32_t channel, const asc_comm_transfer* transfers,
                        uint32_t count);
int asc_comm_ccu_wait(uint32_t channel, uint64_t timeout_ns);

/* Introspection used by the capability probe and by the evidence collector. */
typedef struct {
  uint32_t abi_version;
  uint32_t flags;
  char soc[32];
  char build_id[64];
} asc_comm_build_info;

int asc_comm_build_info_get(asc_comm_build_info* out);

#ifdef __cplusplus
}
#endif

#endif /* ASC_COMM_ABI_H */
