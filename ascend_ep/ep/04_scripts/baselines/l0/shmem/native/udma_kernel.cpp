/*
 * L0 SHMEM UDMA put kernel.
 *
 * Compared with the original single-request kernel this version supports:
 *   - several outstanding requests per peer (chunked over the message),
 *   - multi-core submission (one core per block, each with its own QP index),
 *   - multi-peer fan-out derived from (rank, world, peer_count),
 *   - a quiet policy: per request or once per peer at the end,
 *   - sends and gets, so the driver can measure both directions.
 *
 * The kernel is launched asynchronously on an ACL stream; the host measures
 * submission (launch) and completion (stream synchronise) separately.
 *
 * The per-core QP index is only valid when the host called
 * aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, cores) before initialisation and
 * the same value on every PE (see aclshmemx_set_qp_num in
 * include/host/init/shmem_host_init.h).  Otherwise every core must use QP 0 and
 * aclshmemx_udma_qp_quiet(peer, 0) == aclshmemx_udma_quiet(peer).
 *
 * Status: compiles only against a CANN SHMEM checkout; see build_udma.sh.
 */
#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

namespace {
constexpr uint32_t kMaxRequestBytes = 256U * 1024U * 1024U;  // documented single-request limit
constexpr uint32_t kPatternModulus = 251U;

__aicore__ inline uint32_t PeerFor(uint32_t rank, uint32_t world, uint32_t index)
{
    return (rank + 1U + index) % world;
}
}  // namespace

/*
 * dst/src are symmetric buffers with the same virtual layout on every PE.
 * Each destination peer writes into its own slot: dst[peer * slot_bytes].
 *
 * quiet_mode: 0 = quiet once per peer after all chunks, 1 = quiet after every request.
 */
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ep_l0_udma_put(
    GM_ADDR destination, GM_ADDR source, uint32_t slot_bytes, uint32_t rank, uint32_t world,
    uint32_t peer_count, uint32_t outstanding, uint32_t quiet_mode, uint32_t qp_per_core)
{
    auto* dst = reinterpret_cast<__gm__ uint8_t*>(destination);
    auto* src = reinterpret_cast<__gm__ uint8_t*>(source);
    const uint32_t cores = AscendC::GetBlockNum();
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t qp = qp_per_core != 0U ? core : 0U;
    const uint32_t chunks = outstanding == 0U ? 1U : outstanding;
    const uint32_t chunk_bytes = (slot_bytes + chunks - 1U) / chunks;

    for (uint32_t index = 0; index < peer_count; ++index) {
        const uint32_t peer = PeerFor(rank, world, index);
        __gm__ uint8_t* remote = dst + static_cast<uint64_t>(peer) * slot_bytes;
        for (uint32_t chunk = core; chunk < chunks; chunk += cores) {
            const uint32_t offset = chunk * chunk_bytes;
            if (offset >= slot_bytes) {
                break;
            }
            const uint32_t remaining = slot_bytes - offset;
            const uint32_t count = remaining > chunk_bytes ? chunk_bytes : remaining;
            const uint32_t limit = count > kMaxRequestBytes ? kMaxRequestBytes : count;
            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3>(
                remote + offset, src + offset, nullptr, limit, static_cast<int>(peer), qp);
            if (quiet_mode == 1U) {
                aclshmemx_udma_qp_quiet(static_cast<int>(peer), qp);
            }
        }
    }
    if (quiet_mode == 0U) {
        for (uint32_t index = 0; index < peer_count; ++index) {
            aclshmemx_udma_qp_quiet(static_cast<int>(PeerFor(rank, world, index)), qp);
        }
    }
}

/* Get variant: the payload is pulled from the peer's symmetric slot into local dst. */
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ep_l0_udma_get(
    GM_ADDR destination, GM_ADDR source, uint32_t slot_bytes, uint32_t rank, uint32_t world,
    uint32_t peer_count, uint32_t outstanding, uint32_t quiet_mode, uint32_t qp_per_core)
{
    auto* dst = reinterpret_cast<__gm__ uint8_t*>(destination);
    auto* src = reinterpret_cast<__gm__ uint8_t*>(source);
    const uint32_t cores = AscendC::GetBlockNum();
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t qp = qp_per_core != 0U ? core : 0U;
    const uint32_t chunks = outstanding == 0U ? 1U : outstanding;
    const uint32_t chunk_bytes = (slot_bytes + chunks - 1U) / chunks;

    for (uint32_t index = 0; index < peer_count; ++index) {
        const uint32_t peer = PeerFor(rank, world, index);
        __gm__ uint8_t* remote = src + static_cast<uint64_t>(peer) * slot_bytes;
        __gm__ uint8_t* local = dst + static_cast<uint64_t>(peer) * slot_bytes;
        for (uint32_t chunk = core; chunk < chunks; chunk += cores) {
            const uint32_t offset = chunk * chunk_bytes;
            if (offset >= slot_bytes) {
                break;
            }
            const uint32_t remaining = slot_bytes - offset;
            const uint32_t count = remaining > chunk_bytes ? chunk_bytes : remaining;
            const uint32_t limit = count > kMaxRequestBytes ? kMaxRequestBytes : count;
            aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3>(
                local + offset, remote + offset, nullptr, limit, static_cast<int>(peer), qp);
            if (quiet_mode == 1U) {
                aclshmemx_udma_qp_quiet(static_cast<int>(peer), qp);
            }
        }
    }
    if (quiet_mode == 0U) {
        for (uint32_t index = 0; index < peer_count; ++index) {
            aclshmemx_udma_qp_quiet(static_cast<int>(PeerFor(rank, world, index)), qp);
        }
    }
}

/*
 * Legacy ABI kept for the original HW-008 case: one request, quiet immediately.
 */
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ep_udma_put(
    GM_ADDR destination, GM_ADDR source, uint32_t bytes, int peer)
{
    auto* dst = reinterpret_cast<__gm__ uint8_t*>(destination);
    auto* src = reinterpret_cast<__gm__ uint8_t*>(source);
    for (uint32_t offset = 0; offset < bytes;) {
        const uint32_t remaining = bytes - offset;
        const uint32_t count = remaining > kMaxRequestBytes ? kMaxRequestBytes : remaining;
        aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(dst + offset, src + offset, nullptr, count, peer);
        aclshmemx_udma_quiet(peer);
        offset += count;
    }
}

extern "C" void ep_l0_udma_submit(void* destination, void* source, uint32_t slot_bytes,
                                  uint32_t rank, uint32_t world, uint32_t peer_count,
                                  uint32_t outstanding, uint32_t quiet_mode, uint32_t cores,
                                  uint32_t qp_per_core, int is_get, void* stream)
{
    const uint32_t blocks = cores == 0U ? 1U : cores;
    if (is_get != 0) {
        ep_l0_udma_get<<<blocks, nullptr, stream>>>(
            static_cast<uint8_t*>(destination), static_cast<uint8_t*>(source), slot_bytes, rank,
            world, peer_count, outstanding, quiet_mode, qp_per_core);
        return;
    }
    ep_l0_udma_put<<<blocks, nullptr, stream>>>(
        static_cast<uint8_t*>(destination), static_cast<uint8_t*>(source), slot_bytes, rank, world,
        peer_count, outstanding, quiet_mode, qp_per_core);
}

extern "C" void ep_benchmark_udma_put(void* destination, void* source, uint32_t bytes, int peer,
                                      void* stream)
{
    ep_udma_put<<<1, nullptr, stream>>>(
        static_cast<uint8_t*>(destination), static_cast<uint8_t*>(source), bytes, peer);
}
