#include "kernel_operator.h"
#include "simt_api/common_functions.h"
__simt_vf__ __launch_bounds__(256) inline void write_one_vf(__gm__ uint16_t *out)
{
    if (threadIdx.x == 0) out[0] = 0;
}
extern "C" __global__ __aicore__ void simt_switch_probe(
    GM_ADDR packed, GM_ADDR scale, GM_ADDR output, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<write_one_vf>(dim3{256}, reinterpret_cast<__gm__ uint16_t *>(output));
}
