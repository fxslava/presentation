#include "kernel_operator.h"
#include "simt_api/common_functions.h"

// E1M2 FP4 payload and positive normal E4M3 block scales, 16 values/block.
// One SIMT thread produces one output from GM-resident packed data and scale.
__simt_vf__ __launch_bounds__(1024) inline void direct_unpack_vf(
    __gm__ const uint8_t *packed, __gm__ const uint8_t *scale,
    __gm__ half *output, uint32_t elements)
{
    for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x) {
        uint8_t byte = packed[i >> 1];
        uint32_t code = (byte >> ((i & 1) * 4)) & 15;
        uint8_t sb = scale[i >> 4];
        int32_t exponent = static_cast<int32_t>((sb >> 3) & 15) - 7;
        float base = static_cast<float>(8 + (sb & 7)) * 0.125f;
        float power = 1.0f;
        if (exponent >= 0) {
            power = static_cast<float>(1U << exponent);
        } else {
            power = 1.0f / static_cast<float>(1U << -exponent);
        }
        int32_t signed_code = (code & 8) ? -static_cast<int32_t>(code & 7) : static_cast<int32_t>(code & 7);
        output[i] = static_cast<half>(static_cast<float>(signed_code) * 0.25f * base * power);
    }
}

extern "C" __global__ __aicore__ void simt_direct_gm_unpack(
    GM_ADDR packed, GM_ADDR scale, GM_ADDR output, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __gm__ const uint32_t *tile = reinterpret_cast<__gm__ const uint32_t *>(tiling);
    asc_vf_call<direct_unpack_vf>(dim3{1024},
        reinterpret_cast<__gm__ const uint8_t *>(packed),
        reinterpret_cast<__gm__ const uint8_t *>(scale),
        reinterpret_cast<__gm__ half *>(output), tile[0]);
}
