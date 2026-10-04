#include "kernel_operator.h"
extern "C" __global__ __aicore__ void scalar_switch_probe(
    GM_ADDR packed, GM_ADDR scale, GM_ADDR output, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    reinterpret_cast<__gm__ uint16_t *>(output)[0] = 0;
}
