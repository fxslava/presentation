/**
 * nvfp4_dequant_aic.h  --  Cube (AIC) pipeline implementation.
 *
 * Pipeline per 16-group tile t (slot p = t & 1), double buffered:
 *
 *   MTE2   GM -> L1 (A1/B1)   DataCopy with Nd2NzParams: the MTE2 engine
 *                            performs the ND -> NZ fractal transform for the
 *                            fp4 payloads (transferred as b8, K even).
 *   MTE1   L1 -> L0A / L0B   LoadData (NZ->ZZ for A, NZ->ZN for B);
 *                            HardEvent MTE2_MTE1 gates, MTE1_M gates Mmad.
 *   M      MmadMx            hardware mad_mx: fp4x2_e2m1_t (A0/A1 scale
 *                            terms) x fp4x2_e1m2_t (weights) -> fp32 L0C
 *                            (CO1). C[g][n] = 4M_g * w_gn  (exact).
 *   FIX    Fixpipe L0C -> GM fp32 staging, NZ2ND row-major [16][16]; the
 *          producer signal CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(prod_p)
 *          fires when the fixpipe has drained.
 *   XCORE  CrossCoreWaitFlag(ack_p) before overwriting gmC region t-2.
 *
 * GM A/B payloads are plain ND: gmA tile = [M=16][K=32] fp4 (row m carries
 * A0_m at k=m, A1_m at k=16+m); gmB tile = [N=16][K=32] fp4 with
 * B[k][n] = w[k mod 16][n] (weight rows replicated along K, prepared by the
 * host). Steady-state overlap: MTE2[t+1] || MTE1[t] || M[t-1] || FIX[t-1].
 */
#ifndef NVFP4_DEQUANT_AIC_H
#define NVFP4_DEQUANT_AIC_H

#include "nvfp4_dequant_stages.h"

namespace AscendC {

class Nvfp4CubeStage {
public:
    __aicore__ inline void Init(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmTiling, TPipe *pipe)
    {
        pipe_ = pipe;
        GlobalTensor<uint32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ uint32_t *)gmTiling, 8);
        numTiles_ = tilingGm.GetValue(0);

        gmA_.SetGlobalBuffer((__gm__ uint8_t *)gmA, static_cast<uint64_t>(numTiles_) * A_FRACTAL_BYTES);
        gmB_.SetGlobalBuffer((__gm__ uint8_t *)gmB, static_cast<uint64_t>(numTiles_) * B_FRACTAL_BYTES);
        gmC_.SetGlobalBuffer((__gm__ float *)gmC,
                             static_cast<uint64_t>(numTiles_) * (C_F32_BYTES / sizeof(float)));

        // L1 staging (A1/B1) and L0A/L0B/L0C slots, ping-pong via offsets.
        pipe_->InitBuffer(a1Buf_, NUM_BUFFERS * A_FRACTAL_BYTES);
        pipe_->InitBuffer(b1Buf_, NUM_BUFFERS * B_FRACTAL_BYTES);
        pipe_->InitBuffer(l0aBuf_, NUM_BUFFERS * A_FRACTAL_BYTES);
        pipe_->InitBuffer(l0bBuf_, NUM_BUFFERS * B_FRACTAL_BYTES);
        pipe_->InitBuffer(l0cBuf_, NUM_BUFFERS * C_F32_BYTES);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = 0; t < numTiles_; ++t) {
            LoadTile(t);      // MTE2 GM -> L1 with ND2NZ fractal transform
            ComputeTile(t);   // MTE1 -> MmadMx -> Fixpipe -> handshake
        }
        // Drain the pipeline: leave no hardware event set at kernel exit (the
        // framework epilogue reuses the event slots).
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(1);
    }

private:
    __aicore__ inline void LoadTile(uint32_t tileIdx)
    {
        const uint32_t p = tileIdx % NUM_BUFFERS;
        SetFlag<HardEvent::S_MTE2>(static_cast<int32_t>(p));
        WaitFlag<HardEvent::S_MTE2>(static_cast<int32_t>(p));

        auto a1U8 = a1Buf_.GetWithOffset<uint8_t>(A_FRACTAL_BYTES, p * A_FRACTAL_BYTES);
        auto b1U8 = b1Buf_.GetWithOffset<uint8_t>(B_FRACTAL_BYTES, p * B_FRACTAL_BYTES);

        // GM ND [16 rows][K=32 fp4] -> L1 NZ; engine transfers fp4 as b8.
        Nd2NzParams nzA;
        nzA.ndNum = 1;
        nzA.nValue = TILE_M;
        nzA.dValue = TILE_K;
        nzA.srcNdMatrixStride = 0;
        nzA.srcDValue = TILE_K;                       // GM row stride (elements)
        nzA.dstNzC0Stride = CUBE_BLK;                 // CeilCubeBlock(16) * 16 / 16
        nzA.dstNzNStride = 1;
        nzA.dstNzMatrixStride = 0;
        DataCopy(a1U8, gmA_[static_cast<uint64_t>(tileIdx) * A_FRACTAL_BYTES], nzA);

        Nd2NzParams nzB;
        nzB.ndNum = 1;
        nzB.nValue = TILE_N;
        nzB.dValue = TILE_K;
        nzB.srcNdMatrixStride = 0;
        nzB.srcDValue = TILE_K;
        nzB.dstNzC0Stride = CUBE_BLK;
        nzB.dstNzNStride = 1;
        nzB.dstNzMatrixStride = 0;
        DataCopy(b1U8, gmB_[static_cast<uint64_t>(tileIdx) * B_FRACTAL_BYTES], nzB);

        SetFlag<HardEvent::MTE2_MTE1>(static_cast<int32_t>(p));
    }

    __aicore__ inline void ComputeTile(uint32_t tileIdx)
    {
        const uint32_t p = tileIdx % NUM_BUFFERS;
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<int32_t>(p));

        auto a1 = a1Buf_.GetWithOffset<Nvfp4ScalePkt>(A_FRACTAL_BYTES, p * A_FRACTAL_BYTES);
        auto b1 = b1Buf_.GetWithOffset<Nvfp4WeightPkt>(B_FRACTAL_BYTES, p * B_FRACTAL_BYTES);
        auto l0aT = l0aBuf_.GetWithOffset<Nvfp4ScalePkt>(A_FRACTAL_BYTES, p * A_FRACTAL_BYTES);
        auto l0bT = l0bBuf_.GetWithOffset<Nvfp4WeightPkt>(B_FRACTAL_BYTES, p * B_FRACTAL_BYTES);
        auto l0cT = l0cBuf_.GetWithOffset<float>(C_F32_BYTES / sizeof(float), p * C_F32_BYTES);

        // L0A/L0B slot p must be free: Mmad of tile t-2 drained.
        if (tileIdx >= NUM_BUFFERS) {
            WaitFlag<HardEvent::M_MTE1>(static_cast<int32_t>(p));
        }

        // L1 NZ -> L0A ZZ / L0B ZN (one 16x32 fp4 fractal unit each: mStep=1,
        // kStep=1; fp4 payloads ride the 2d-v2 s4 load path).
        LoadData2DParamsV2 loadA(0, 0, 1, 1, 0, 1, false, 0);
        LoadData(l0aT, a1, loadA);
        LoadData2DParamsV2 loadB(0, 0, 1, 1, 0, 1, false, 0);
        LoadData(l0bT, b1, loadB);
        PipeBarrier<PIPE_MTE1>();
        SetFlag<HardEvent::MTE1_M>(static_cast<int32_t>(p));

        WaitFlag<HardEvent::MTE1_M>(static_cast<int32_t>(p));

        // L0C slot p must be free: Fixpipe of tile t-2 drained.
        if (tileIdx >= NUM_BUFFERS) {
            WaitFlag<HardEvent::FIX_M>(static_cast<int32_t>(p));
        }

        // ONE hardware mad_mx per tile: m=16 groups, n=16 weights, k=32 fp4
        // elements (two 16-wide replicated K halves carrying A0 and A1).
        MmadParams mmadParams;
        mmadParams.m = TILE_M;
        mmadParams.n = TILE_N;
        mmadParams.k = TILE_K;
        mmadParams.cmatrixInitVal = true;  // overwrite: this Mmad is the full K=32 sum
        MmadMx(l0cT, l0aT, l0bT, mmadParams);
        SetFlag<HardEvent::M_FIX>(static_cast<int32_t>(p));   // L0C[p] holds C
        SetFlag<HardEvent::M_MTE1>(static_cast<int32_t>(p));  // L0A/B[p] reusable

        // Issue the Fixpipe only after the Mmad has fully landed in L0C.
        WaitFlag<HardEvent::M_FIX>(static_cast<int32_t>(p));

        // Vector core must be done reading gmC region (t-2) before overwrite.
        if (tileIdx >= NUM_BUFFERS) {
            CrossCoreWaitFlag(p == 0 ? EVT_V2C_ACK_EVEN : EVT_V2C_ACK_ODD);
        }

        // Fixpipe: L0C NZ -> GM ND row-major fp32 [16][16] (CFG_ROW_MAJOR
        // selects the hardware NZ2ND conversion on 3510).
        FixpipeParamsV220 fixParams;
        fixParams.nSize = TILE_N;
        fixParams.mSize = TILE_M;
        fixParams.srcStride = 0;
        fixParams.dstStride = TILE_N;  // GM row stride (elements)
        fixParams.reluEn = false;
        Fixpipe<float, float, CFG_ROW_MAJOR>(gmC_[static_cast<uint64_t>(tileIdx) * (C_F32_BYTES / sizeof(float))],
                                             l0cT, fixParams);
        SetFlag<HardEvent::FIX_M>(static_cast<int32_t>(p));  // L0C[p] reusable
        // Producer signal: fires when the fixpipe (PIPE_FIX) has drained.
        CrossCoreSetFlag<NVFP4_SYNC_MODE2, PIPE_FIX>(p == 0 ? EVT_C2V_PROD_EVEN : EVT_C2V_PROD_ODD);
    }

    static constexpr uint16_t CUBE_BLK = 16;

    TPipe *pipe_ = nullptr;
    uint32_t numTiles_ = 0;

    TBuf<TPosition::A1> a1Buf_;   // 2 x A_FRACTAL_BYTES (L1 staging)
    TBuf<TPosition::B1> b1Buf_;   // 2 x B_FRACTAL_BYTES (L1 staging)
    TBuf<TPosition::A2> l0aBuf_;  // 2 x A_FRACTAL_BYTES (ping-pong via offset)
    TBuf<TPosition::B2> l0bBuf_;  // 2 x B_FRACTAL_BYTES
    TBuf<TPosition::CO1> l0cBuf_; // 2 x C_F32_BYTES (L0C accumulator slots)

    GlobalTensor<uint8_t> gmA_;
    GlobalTensor<uint8_t> gmB_;
    GlobalTensor<float> gmC_;  // fp32 Cube result staging (GM handoff)
};

}  // namespace AscendC

#endif  // NVFP4_DEQUANT_AIC_H
