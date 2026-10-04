#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#ifndef FP4_BLOCK
#define FP4_BLOCK 16
#endif

// E1M2 FP4 payload and positive normal E4M3 block scales, 16 values/block.
// One SIMT thread produces one output from GM-resident packed data and scale.
__simt_vf__ __launch_bounds__(1024) inline void direct_unpack_vf(
    __gm__ const uint8_t *packed, __gm__ const uint8_t *scale,
    __ubuf__ half *output, uint32_t base)
{
    for (uint32_t i = threadIdx.x; i < FAST_TILE; i += blockDim.x) {
        uint32_t elem = base + i;
        uint8_t byte = packed[elem >> 1];
        uint32_t code = (byte >> ((i & 1) * 4)) & 15;
        uint8_t sb = scale[elem / FP4_BLOCK];
        int32_t exponent = static_cast<int32_t>((sb >> 3) & 15) - 7;
        float base = static_cast<float>(8 + (sb & 7)) * 0.125f;
        float power = 1.0f;
        if (exponent >= 0) {
            power = static_cast<float>(1U << exponent);
        } else {
            power = 1.0f / static_cast<float>(1U << -exponent);
        }
        int32_t magnitude = code & 7;
#ifdef FAST_E2M1
        magnitude = magnitude <= 4 ? 2*magnitude : (magnitude == 5 ? 12 : (magnitude == 6 ? 16 : 24));
#endif
        int32_t signed_code = (code & 8) ? -magnitude : magnitude;
        output[i] = static_cast<half>(static_cast<float>(signed_code) * 0.25f * base * power);
    }
}

extern "C" __global__ __aicore__ void simt_direct_gm_unpack(
    GM_ADDR packed, GM_ADDR scale, GM_ADDR output, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __gm__ const uint32_t *tile = reinterpret_cast<__gm__ const uint32_t *>(tiling);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    pipe.InitBuffer(outputQueue, 1, FAST_TILE * sizeof(half));
    AscendC::GlobalTensor<half> gmOutput;
    gmOutput.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output), tile[0]);
    for (uint32_t t = 0; t < tile[0] / FAST_TILE; ++t) {
        AscendC::LocalTensor<half> ubOutput = outputQueue.AllocTensor<half>();
        asc_vf_call<direct_unpack_vf>(dim3{1024},
            reinterpret_cast<__gm__ const uint8_t *>(packed),
            reinterpret_cast<__gm__ const uint8_t *>(scale),
            reinterpret_cast<__ubuf__ half *>(ubOutput.GetPhyAddr()), t * FAST_TILE);
        outputQueue.EnQue(ubOutput);
        AscendC::LocalTensor<half> ready = outputQueue.DeQue<half>();
        AscendC::DataCopy(gmOutput[t * FAST_TILE], ready, FAST_TILE);
        outputQueue.FreeTensor(ready);
    }
}
