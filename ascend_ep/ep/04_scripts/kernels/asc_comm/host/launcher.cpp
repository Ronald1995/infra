// asc_comm/host/launcher.cpp -- host-side entry points of the ASC-COMM kernels.
//
// This file is deliberately small and dependency free: it validates the transfer
// descriptors, resolves the engine/channel binding and forwards to the engine
// implementation.  The device program lives under aiv/ and ccu/ and is compiled by
// the CANN toolchain; this launcher never copies payload bytes.
//
// When an engine is not compiled in, its entry point returns
// ASC_COMM_ERR_UNSUPPORTED.  It must never silently substitute the host copy path,
// because that is exactly the attribution error the benchmark is built to avoid.

#include "common/abi.h"

#include <atomic>
#include <cstring>

namespace {

std::atomic<bool> g_aiv_initialized{false};
std::atomic<bool> g_ccu_initialized{false};

// Channel handles are opaque integers owned by the engine implementation.  The
// launcher only validates that they are in range; the engine decides what a handle
// means.
constexpr uint32_t kMaxChannels = 4096;

bool valid_channel(uint32_t channel) { return channel < kMaxChannels; }

int validate(const asc_comm_transfer* transfers, uint32_t count) {
  if (transfers == nullptr || count == 0) return ASC_COMM_ERR_ARGUMENT;
  for (uint32_t index = 0; index < count; ++index) {
    const asc_comm_transfer& transfer = transfers[index];
    if (transfer.bytes == 0) return ASC_COMM_ERR_ARGUMENT;
    if (transfer.src == 0 || transfer.dst == 0) return ASC_COMM_ERR_ARGUMENT;
    if (transfer.flags & ~(ASC_COMM_FLAG_NOTIFY | ASC_COMM_FLAG_DRAIN)) {
      return ASC_COMM_ERR_UNSUPPORTED;
    }
  }
  return ASC_COMM_OK;
}

}  // namespace

#if defined(ASC_COMM_BUILD_AIV)
// Implemented by aiv/write_nbi/kernel.cpp (device side) plus its host shim.
int asc_comm_aiv_impl_submit(uint32_t channel, const asc_comm_transfer* transfers,
                             uint32_t count);
int asc_comm_aiv_impl_drain(uint32_t channel, uint64_t timeout_ns);
#endif

#if defined(ASC_COMM_BUILD_CCU)
// Implemented by ccu/dsl/kernel.cpp.
int asc_comm_ccu_impl_submit(uint32_t channel, const asc_comm_transfer* transfers,
                             uint32_t count);
int asc_comm_ccu_impl_wait(uint32_t channel, uint64_t timeout_ns);
#endif

extern "C" int asc_comm_aiv_write_nbi(uint32_t channel,
                                      const asc_comm_transfer* transfers,
                                      uint32_t count) {
  const int status = validate(transfers, count);
  if (status != ASC_COMM_OK) return status;
  if (!valid_channel(channel)) return ASC_COMM_ERR_CHANNEL;
#if defined(ASC_COMM_BUILD_AIV)
  g_aiv_initialized.store(true);
  return asc_comm_aiv_impl_submit(channel, transfers, count);
#else
  return ASC_COMM_ERR_UNSUPPORTED;
#endif
}

extern "C" int asc_comm_aiv_drain(uint32_t channel, uint64_t timeout_ns) {
  if (!valid_channel(channel)) return ASC_COMM_ERR_CHANNEL;
#if defined(ASC_COMM_BUILD_AIV)
  if (!g_aiv_initialized.load()) return ASC_COMM_ERR_NOT_INITIALIZED;
  return asc_comm_aiv_impl_drain(channel, timeout_ns);
#else
  (void)timeout_ns;
  return ASC_COMM_ERR_UNSUPPORTED;
#endif
}

extern "C" int asc_comm_ccu_submit(uint32_t channel,
                                   const asc_comm_transfer* transfers,
                                   uint32_t count) {
  const int status = validate(transfers, count);
  if (status != ASC_COMM_OK) return status;
  if (!valid_channel(channel)) return ASC_COMM_ERR_CHANNEL;
#if defined(ASC_COMM_BUILD_CCU)
  g_ccu_initialized.store(true);
  return asc_comm_ccu_impl_submit(channel, transfers, count);
#else
  return ASC_COMM_ERR_UNSUPPORTED;
#endif
}

extern "C" int asc_comm_ccu_wait(uint32_t channel, uint64_t timeout_ns) {
  if (!valid_channel(channel)) return ASC_COMM_ERR_CHANNEL;
#if defined(ASC_COMM_BUILD_CCU)
  if (!g_ccu_initialized.load()) return ASC_COMM_ERR_NOT_INITIALIZED;
  return asc_comm_ccu_impl_wait(channel, timeout_ns);
#else
  (void)timeout_ns;
  return ASC_COMM_ERR_UNSUPPORTED;
#endif
}

extern "C" int asc_comm_build_info_get(asc_comm_build_info* out) {
  if (out == nullptr) return ASC_COMM_ERR_ARGUMENT;
  std::memset(out, 0, sizeof(*out));
  out->abi_version = ASC_COMM_ABI_VERSION;
  out->flags = 0;
#if defined(ASC_COMM_BUILD_AIV)
  out->flags |= ASC_COMM_FLAG_AIV_BUILT;
#endif
#if defined(ASC_COMM_BUILD_CCU)
  out->flags |= ASC_COMM_FLAG_CCU_BUILT;
#endif
  std::strncpy(out->soc, ASC_COMM_SOC_NAME, sizeof(out->soc) - 1);
  std::strncpy(out->build_id, ASC_COMM_BUILD_ID, sizeof(out->build_id) - 1);
  return ASC_COMM_OK;
}
