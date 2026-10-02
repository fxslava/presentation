/**
 * nvfp4_dequant_main.cpp  --  kernel entry point (single TU, MIX image).
 *
 * ccec compiles this entry into the per-core images `nvfp4_dequant_mix_aic`
 * and `nvfp4_dequant_mix_aiv`; ASCEND_IS_AIC selects which pipeline runs on
 * which image. The Cube pipeline lives in nvfp4_dequant_aic.h, the Vector
 * pipeline in nvfp4_dequant_aiv.h (production arch35 MIX-kernel structure:
 * per-stage implementation units are header-inline because core-type
 * specialization is resolved at entry instantiation time).
 *
 * One block = one Cube core (AIC) + one Vector core (AIV):
 *   KERNEL_TYPE_MIX_AIC_1_1.
 *
 * GM layout (see nvfp4_dequant_common.h):
 *   gmA : numTiles * 256B  L0A fractal, packed (A0,A1) E2M1 scale terms
 *   gmB : numTiles * 256B  L0B fractal, K-replicated E1M2 weights
 *   gmS : numTiles * 32B   16 E4M3 scale bytes + pad
 *   gmC : numTiles * 1024B fp32 Cube staging (AIC->AIV handoff)
 *   gmOut : numTiles * 512B fp16 dequantized [16 groups][16 weights]
 */
#include "nvfp4_dequant_aic.h"
#include "nvfp4_dequant_aiv.h"

extern "C" __global__ __aicore__ void nvfp4_dequant(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmS, GM_ADDR gmC,
                                                    GM_ADDR gmOut, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIC {
        AscendC::TPipe pipe;
        AscendC::Nvfp4CubeStage cube;
        cube.Init(gmA, gmB, gmC, tilingGm, &pipe);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        AscendC::Nvfp4VectorStage vec;
        vec.Init(gmS, gmC, gmOut, tilingGm, &pipe);
        vec.Process();
    }
}
