/*
 * Local memory movement kernels for MEM-002 (MTE) and MEM-003 (local UB copy).
 *
 * mode 0: GM -> UB -> GM through MTE2/MTE3 (what a copy engine does locally)
 * mode 1: UB -> UB inside one core's unified buffer
 *
 * Both paths are deliberately local: no interconnect, no remote peer.  A local
 * copy result must never be reported as UB Memory remote bandwidth.
 *
 * Status: builds only with the Ascend C toolchain; see build_mem.sh.
 */
#include <cstdint>

#include "kernel_operator.h"

namespace {
constexpr uint32_t kTileBytes = 32U * 1024U;
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void l0_mem_copy_kernel(
    GM_ADDR destination, GM_ADDR source, uint32_t bytes, uint32_t mode)
{
    auto* dst = reinterpret_cast<__gm__ uint8_t*>(destination);
    auto* src = reinterpret_cast<__gm__ uint8_t*>(source);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueue, 1, kTileBytes);
    pipe.InitBuffer(outQueue, 1, kTileBytes);
    if (mode == 1U) {
        // UB -> UB: allocate two UB tiles and copy between them, then flush the
        // result back so the host can verify it.
        AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;
        pipe.InitBuffer(scratch, kTileBytes);
        AscendC::LocalTensor<uint8_t> a = scratch.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> b = inQueue.AllocTensor<uint8_t>();
        for (uint32_t offset = 0; offset < bytes; offset += kTileBytes) {
            const uint32_t count = (bytes - offset) > kTileBytes ? kTileBytes : (bytes - offset);
            AscendC::DataCopy(a, src + offset, count);
            AscendC::DataCopy(b, a, count);
            AscendC::DataCopy(dst + offset, b, count);
        }
        inQueue.FreeTensor(b);
        return;
    }
    AscendC::LocalTensor<uint8_t> tile = inQueue.AllocTensor<uint8_t>();
    for (uint32_t offset = 0; offset < bytes; offset += kTileBytes) {
        const uint32_t count = (bytes - offset) > kTileBytes ? kTileBytes : (bytes - offset);
        AscendC::DataCopy(tile, src + offset, count);
        AscendC::DataCopy(dst + offset, tile, count);
    }
    inQueue.FreeTensor(tile);
}

extern "C" int l0_mem_copy(void* destination, void* source, uint32_t bytes, int mode, void* stream)
{
    if (destination == nullptr || source == nullptr || bytes == 0U) {
        return -1;
    }
    l0_mem_copy_kernel<<<1, nullptr, stream>>>(
        static_cast<uint8_t*>(destination), static_cast<uint8_t*>(source), bytes,
        mode == 0 ? 0U : 1U);
    return 0;
}
