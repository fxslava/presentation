/**
 * nvfp4_dequant_stages.h
 *
 * Translation-unit separation contract for the NVFP4 dequant kernel.
 *
 *   nvfp4_dequant_aic.cpp : Nvfp4CubeStage  - Cube pipeline (AIC core)
 *       MTE2 GM->UB (double buffered TQue) -> MTE1 UB->L0A/L0B ->
 *       MmadMx (hardware mad_mx, fp4x2_e2m1_t x fp4x2_e1m2_t -> fp32 L0C) ->
 *       Fixpipe L0C -> GM fp32 staging; native HardEvents MTE1_M / M_MTE1 /
 *       M_FIX / FIX_M; CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX> producer signal.
 *
 *   nvfp4_dequant_aiv.cpp : Nvfp4VectorStage - Vector pipeline (AIV core)
 *       CrossCoreWaitFlag -> MTE2 GM->UB -> Cast fp32->fp16 -> pure-integer
 *       exponent correction (vadd on the fp16 bit pattern, delta_e_eff<<10)
 *       -> zero guard -> MTE3 UB->GM fp16; CrossCoreSetFlag<SYNC_MODE2,
 *       PIPE_MTE2> ack.
 *
 *   nvfp4_dequant_main.cpp : kernel entry, MIX_AIC_1_1 dispatch.
 */
#ifndef NVFP4_DEQUANT_STAGES_H
#define NVFP4_DEQUANT_STAGES_H

#include "kernel_operator.h"
#include "nvfp4_dequant_common.h"

namespace AscendC {

// Cross-core flag mode used by production MIX kernels (add_lora et al.) for
// cube<->vector event propagation; the flag fires when the specified pipe has
// drained all previously issued operations.
static constexpr uint64_t NVFP4_SYNC_MODE2 = 2;

// L0A operand: decomposed E4M3 scale terms (A0/A1) as packed FP4 E2M1.
using Nvfp4ScalePkt = fp4x2_e2m1_t;
// L0B operand: NVFP4 weights as packed FP4 E1M2 (target SupportType pairing
// Tuple<float, fp4x2_e2m1_t, fp4x2_e1m2_t> on dav-c310/3510).
using Nvfp4WeightPkt = fp4x2_e1m2_t;

}  // namespace AscendC

#endif  // NVFP4_DEQUANT_STAGES_H
