/**
 * nvfp4_dequant_aiv.h  --  Vector (AIV) pipeline implementation.
 *
 * Per tile t (slot p = t & 1), double buffered:
 *
 *   XCORE  CrossCoreWaitFlag(prod_p)   fp32 Cube result of gmC region t visible
 *   MTE2   GM -> UB                    fp32 C tile (1024B) + 16 E4M3 scale bytes
 *   XCORE  CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(ack_p) -- fires once the
 *          MTE2 reads have drained, releasing gmC region t to the Cube
 *   V      Cast  fp32 -> fp16          (vector convert pipe, no multiply)
 *   V      exponent correction, INTEGER pipe only:
 *            bits16 = fp16_bits(C[g][n]) + (delta_e_eff(g) << 10)
 *          with delta_e_eff(g) = E4M3_exp_field(s_g) - 9  (== E_g - 2),
 *          built as 16 per-group Duplicate rows + one 256-lane Add,
 *          zero-guarded by |bits| != 0 (w == 0 must stay exactly 0).
 *   MTE3  UB -> GM                     fp16 [16][16] tile
 *
 * No vector FPU multiply anywhere on this path; the only arithmetic on the
 * FP16 payload is the integer add on its bit pattern.
 *
 * Implementation note: stage code is header-inline (production arch35 MIX
 * pattern); the AIV translation unit is nvfp4_dequant_aiv.cpp.
 */
#ifndef NVFP4_DEQUANT_AIV_H
#define NVFP4_DEQUANT_AIV_H

#include "nvfp4_dequant_stages.h"

namespace AscendC {

class Nvfp4VectorStage {
public:
    __aicore__ inline void Init(GM_ADDR gmS, GM_ADDR gmC, GM_ADDR gmOut, GM_ADDR gmTiling, TPipe *pipe)
    {
        pipe_ = pipe;
        GlobalTensor<uint32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ uint32_t *)gmTiling, 8);
        numTiles_ = tilingGm.GetValue(0);

        gmS_.SetGlobalBuffer((__gm__ uint8_t *)gmS, static_cast<uint64_t>(numTiles_) * SCALE_BYTES);
        gmC_.SetGlobalBuffer((__gm__ float *)gmC,
                             static_cast<uint64_t>(numTiles_) * (C_F32_BYTES / sizeof(float)));
        gmOut_.SetGlobalBuffer((__gm__ half *)gmOut,
                               static_cast<uint64_t>(numTiles_) * (OUT_F16_BYTES / sizeof(half)));

        pipe_->InitBuffer(queC_, NUM_BUFFERS, C_F32_BYTES);
        pipe_->InitBuffer(queS_, NUM_BUFFERS, SCALE_BYTES);
        pipe_->InitBuffer(queO_, NUM_BUFFERS, OUT_F16_BYTES);

        pipe_->InitBuffer(scratchF16_, OUT_F16_BYTES);
        pipe_->InitBuffer(scratchOff_, OUT_F16_BYTES);
        pipe_->InitBuffer(scratchConstAbs_, OUT_F16_BYTES);
        pipe_->InitBuffer(scratchConstZero_, OUT_F16_BYTES);
        pipe_->InitBuffer(scratchConstOne_, OUT_F16_BYTES);
        pipe_->InitBuffer(scratchAbsC_, OUT_F16_BYTES);
        pipe_->InitBuffer(scratchMask_, OUT_F16_BYTES);
        pipe_->InitBuffer(scratchSum_, OUT_F16_BYTES);

        // Constant lane masks, built once on the vector pipe.
        Duplicate(scratchConstAbs_.Get<int16_t>(), static_cast<int16_t>(0x7FFF), TILE_M * TILE_N);
        Duplicate(scratchConstZero_.Get<int16_t>(), static_cast<int16_t>(0), TILE_M * TILE_N);
        Duplicate(scratchConstOne_.Get<int16_t>(), static_cast<int16_t>(1), TILE_M * TILE_N);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = 0; t < numTiles_; ++t) {
            ConsumeTile(t);
        }
        // Drain outstanding cross-core acks (two ping-pong slots) so no event
        // is left set at kernel exit.
        CrossCoreWaitFlag(EVT_V2C_ACK_EVEN);
        CrossCoreWaitFlag(EVT_V2C_ACK_ODD);
    }

private:
    __aicore__ inline void ConsumeTile(uint32_t tileIdx)
    {
        const uint32_t p = tileIdx % NUM_BUFFERS;

        // fp32 C tile: wait for the Cube's producer signal, then MTE2 GM -> UB.
        auto cUb = queC_.AllocTensor<float>();
        CrossCoreWaitFlag(p == 0 ? EVT_C2V_PROD_EVEN : EVT_C2V_PROD_ODD);
        DataCopy(cUb, gmC_[static_cast<uint64_t>(tileIdx) * (C_F32_BYTES / sizeof(float))],
                 TILE_M * TILE_N);
        queC_.EnQue(cUb);

        // Scale bytes are Cube-independent; prefetch alongside.
        auto sUb = queS_.AllocTensor<uint8_t>();
        DataCopy(sUb, gmS_[static_cast<uint64_t>(tileIdx) * SCALE_BYTES], SCALE_BYTES);
        queS_.EnQue(sUb);

        // Release gmC region t once both MTE2 reads have drained.
        CrossCoreSetFlag<NVFP4_SYNC_MODE2, PIPE_MTE2>(p == 0 ? EVT_V2C_ACK_EVEN : EVT_V2C_ACK_ODD);

        auto c32 = queC_.DeQue<float>();
        auto s8 = queS_.DeQue<uint8_t>();

        // ---- vector stage: fp32 -> fp16 bit pattern -> integer exponent add ----
        auto h16 = scratchF16_.Get<half>();
        Cast(h16, c32, RoundMode::CAST_NONE, TILE_M * TILE_N);

        auto cI16 = h16.ReinterpretCast<int16_t>();

        // Per-group exponent offsets from the raw E4M3 scale bytes (scalar
        // loads + Duplicate fills keep this entirely off the FPU):
        //   delta_e_eff = (exp_field - 7) - 2 = exp_field - 9, <<10 into the
        //   fp16 exponent field.
        for (uint32_t g = 0; g < TILE_M; ++g) {
            const uint8_t scaleByte = s8.GetValue(g);
            const int16_t expAdd = static_cast<int16_t>((((scaleByte >> 3) & 0xF) - 9) << 10);
            Duplicate(scratchOff_.GetWithOffset<int16_t>(TILE_N, g * TILE_N * 2), expAdd, TILE_N);
        }

        // bits + (delta << 10): pure integer lanes.
        auto sumI = scratchSum_.Get<int16_t>();
        Add(sumI, cI16, scratchOff_.Get<int16_t>(), TILE_M * TILE_N);

        // Zero guard: C == +/-0 must remain exactly zero (the exponent add
        // would otherwise synthesize a denormal). Every nonzero C has |bits|
        // >= 0x3800 (|C| >= 1), so Min(|bits|, 1) is an exact 0/1 mask with no
        // compare op: mask = Min(|C|, 1); lanes = 0 - mask; out &= lanes.
        auto absC = scratchAbsC_.Get<int16_t>();
        And(absC, cI16, scratchConstAbs_.Get<int16_t>(), TILE_M * TILE_N);
        auto mask01 = scratchMask_.Get<int16_t>();
        Min(mask01, absC, scratchConstOne_.Get<int16_t>(), TILE_M * TILE_N);
        // absC is dead past the Min; V-pipe ops run in issue order, so its
        // buffer is safely reusable for the 0 -> 0xFFFF negation.
        auto negM = scratchAbsC_.Get<int16_t>();
        Sub(negM, scratchConstZero_.Get<int16_t>(), mask01, TILE_M * TILE_N);
        And(sumI, sumI, negM, TILE_M * TILE_N);

        // Store fp16 tile.
        auto out = queO_.AllocTensor<half>();
        DataCopy(out.ReinterpretCast<int16_t>(), sumI, TILE_M * TILE_N);
        queO_.EnQue(out);
        auto outV = queO_.DeQue<half>();
        DataCopy(gmOut_[static_cast<uint64_t>(tileIdx) * (OUT_F16_BYTES / sizeof(half))], outV,
                 TILE_M * TILE_N);
        queO_.FreeTensor(outV);

        queC_.FreeTensor(c32);
        queS_.FreeTensor(s8);
    }

    TPipe *pipe_ = nullptr;
    uint32_t numTiles_ = 0;

    TQue<QuePosition::VECIN, 1> queC_;   // fp32 Cube result GM->UB (DB=2)
    TQue<QuePosition::VECIN, 1> queS_;   // E4M3 scale bytes GM->UB (DB=2)
    TQue<QuePosition::VECOUT, 1> queO_;  // fp16 result UB->GM (DB=2)

    TBuf<TPosition::VECCALC> scratchF16_;       // Cast destination / int16 view
    TBuf<TPosition::VECCALC> scratchOff_;       // per-group (delta<<10) offset tile
    TBuf<TPosition::VECCALC> scratchConstAbs_;  // constant 0x7FFF lane mask
    TBuf<TPosition::VECCALC> scratchConstZero_; // constant 0 lanes
    TBuf<TPosition::VECCALC> scratchConstOne_;  // constant 1 lanes (zero guard)
    TBuf<TPosition::VECCALC> scratchAbsC_;      // |bits| of C (result)
    TBuf<TPosition::VECCALC> scratchMask_;      // zero-guard mask (result)
    TBuf<TPosition::VECCALC> scratchSum_;       // bits + offset (result)

    GlobalTensor<uint8_t> gmS_;
    GlobalTensor<float> gmC_;
    GlobalTensor<half> gmOut_;
};

}  // namespace AscendC

#endif  // NVFP4_DEQUANT_AIV_H
