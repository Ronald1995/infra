#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ep_udma_put(
    GM_ADDR destination, GM_ADDR source, uint32_t bytes, int peer)
{
    auto* dst = reinterpret_cast<__gm__ uint8_t*>(destination);
    auto* src = reinterpret_cast<__gm__ uint8_t*>(source);
    constexpr uint32_t max_request = 256U * 1024U * 1024U;
    for (uint32_t offset = 0; offset < bytes;) {
        const uint32_t remaining = bytes - offset;
        const uint32_t count = remaining > max_request ? max_request : remaining;
        aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(dst + offset, src + offset, nullptr, count, peer);
        aclshmemx_udma_quiet(peer);
        offset += count;
    }
}

extern "C" void ep_benchmark_udma_put(void* destination, void* source, uint32_t bytes, int peer, void* stream)
{
    ep_udma_put<<<1, nullptr, stream>>>(
        static_cast<uint8_t*>(destination), static_cast<uint8_t*>(source), bytes, peer);
}
