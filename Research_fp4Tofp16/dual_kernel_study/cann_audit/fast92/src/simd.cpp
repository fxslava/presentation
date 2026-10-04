/**
 * Native Ascend950PR (dav-3510) SIMD FP4 E1M2 dequantization baseline.
 * Build with FP4_BLOCK=16, 32, or 64; 8,192 outputs per tile.
 * MTE2 stages packed FP4 bytes and positive normal E4M3 scales in UB.
 * Vector instructions split nibbles, broadcast block scales, multiply,
 * and MTE3 writes pair-planar FP16 output. The harness normalizes layout.
 */
#ifndef NAIVE_AIV_FP4_TO_FP16_CPP
#define NAIVE_AIV_FP4_TO_FP16_CPP

#include "kernel_operator.h"
#ifndef FP4_BLOCK
#define FP4_BLOCK 16
#endif

namespace AscendC {

// ------------------------------------------------------------ tile geometry
static constexpr uint32_t AIV_TILE_ELEMS = FAST_TILE;   // outputs per tile
static constexpr uint32_t AIV_TILE_PACKED = AIV_TILE_ELEMS / 2;  // 4096 B
static constexpr uint32_t AIV_TILE_SCALES = AIV_TILE_ELEMS / FP4_BLOCK;
static constexpr uint32_t AIV_SCALE_COPY_BYTES = ((AIV_TILE_SCALES + 31) / 32) * 32;
static constexpr uint32_t AIV_TILE_OUT_BYTES = AIV_TILE_ELEMS * 2;  // 16 KiB
static constexpr uint32_t AIV_QUE_DEPTH = 2;       // MTE2/MTE3 ping-pong

// ------------------------------------------------- skewed scratch geometry
// (offset, bytes) pairs; consecutive allocations padded +32 B so that
// co-operands never share a UB bank phase (delta != 0 mod 256).
static constexpr uint32_t OFF_CODES_LO = 32;          // int8  x 4096 (bank 1)
static constexpr uint32_t OFF_CODES_HI = 4160;        // int8  x 4096 (bank 2)
static constexpr uint32_t OFF_MAG = 8288;             // int8  x 4096 (bank 3)
static constexpr uint32_t OFF_MSK = 12416;            // int8  x 4096 (bank 4)
static constexpr uint32_t OFF_CONST7 = 16544;         // int8  x 4096 (bank 5)
static constexpr uint32_t OFF_CONST15 = 20672;        // int8  x 4096 (bank 6)
static constexpr uint32_t OFF_HF = 24800;             // half  x 4096 (bank 7)
static constexpr uint32_t OFF_TF = 33120;             // half  x 4096 (bank 3)
static constexpr uint32_t OFF_SCALEV = 41504;         // half  x 4096 (bank 1)
static constexpr uint32_t OFF_NEGF = 49952;           // half  x 4096 (bank 1)
static constexpr uint32_t OFF_ZEROF = 58240;          // half  x 4096 (bank 4)
static constexpr uint32_t SCRATCH_BYTES = 66432;

// Layout annotations for the element-indexed scratch views (offsets are
// resolved from the view index; the counts below complete the extent so the
// capacity/alignment/aliasing and bank-conflict checks run on exact bytes).
// @ascend-layout: name=scaleI16 pos=VECCALC dtype=int16_t count=4096
// @ascend-layout: name=codesLo pos=VECCALC dtype=uint8_t count=4096
// @ascend-layout: name=codesHi pos=VECCALC dtype=uint8_t count=4096
// @ascend-layout: name=mag pos=VECCALC dtype=uint8_t count=4096
// @ascend-layout: name=msk pos=VECCALC dtype=uint8_t count=4096
// @ascend-layout: name=const7 pos=VECCALC dtype=uint8_t count=4096
// @ascend-layout: name=const15 pos=VECCALC dtype=uint8_t count=4096
// @ascend-layout: name=h pos=VECCALC dtype=half count=4096
// @ascend-layout: name=t pos=VECCALC dtype=half count=4096
// @ascend-layout: name=negT pos=VECCALC dtype=half count=4096
// @ascend-layout: name=zeroF pos=VECCALC dtype=half count=4096
class NaiveAivFp4Dequant {
public:
    __aicore__ inline void Init(GM_ADDR gmW, GM_ADDR gmS, GM_ADDR gmOut, GM_ADDR gmTiling,
                                TPipe *pipe)
    {
        pipe_ = pipe;
        GlobalTensor<uint32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ uint32_t *)gmTiling, 8);
        totalElems_ = tilingGm.GetValue(0);
        numTiles_ = totalElems_ / AIV_TILE_ELEMS;

        gmW_.SetGlobalBuffer((__gm__ uint8_t *)gmW,
                             static_cast<uint64_t>(totalElems_) / 2);
        gmS_.SetGlobalBuffer((__gm__ uint8_t *)gmS,
                             static_cast<uint64_t>(totalElems_) / FP4_BLOCK);
        gmOut_.SetGlobalBuffer((__gm__ half *)gmOut, static_cast<uint64_t>(totalElems_));

        pipe_->InitBuffer(quePacked_, AIV_QUE_DEPTH, AIV_TILE_PACKED);
        pipe_->InitBuffer(queScales_, AIV_QUE_DEPTH, AIV_SCALE_COPY_BYTES);
        pipe_->InitBuffer(queOut_, AIV_QUE_DEPTH, AIV_TILE_OUT_BYTES);
        pipe_->InitBuffer(scratch_, SCRATCH_BYTES);

        // Constant lanes, built once on the vector pipe (element-indexed
        // views of the skewed scratch pool).
        LocalTensor<uint8_t> s8pool = scratch_.Get<uint8_t>();
        LocalTensor<half> sFpool = scratch_.Get<half>();
        LocalTensor<uint8_t> const7 = s8pool[OFF_CONST7];
        LocalTensor<uint8_t> const15 = s8pool[OFF_CONST15];
        LocalTensor<half> zeroF = sFpool[OFF_ZEROF / 2];
        Duplicate(const7.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x0707), AIV_TILE_PACKED / 2);
        Duplicate(const15.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x0F0F), AIV_TILE_PACKED / 2);
        Duplicate(zeroF, static_cast<half>(0.0f), AIV_TILE_PACKED);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = 0; t < numTiles_; ++t) {
            ConsumeTile(t);
        }
    }

private:
    __aicore__ inline void ConsumeTile(uint32_t tileIdx)
    {
        // ---- MTE2: packed nibbles + block scales --------------------------
        LocalTensor<uint8_t> wUb = quePacked_.AllocTensor<uint8_t>();
        DataCopy(wUb, gmW_[static_cast<uint64_t>(tileIdx) * AIV_TILE_PACKED],
                 AIV_TILE_PACKED);
        quePacked_.EnQue(wUb);

        LocalTensor<uint8_t> sUb = queScales_.AllocTensor<uint8_t>();
        DataCopy(sUb, gmS_[static_cast<uint64_t>(tileIdx) * AIV_TILE_SCALES],
                 AIV_SCALE_COPY_BYTES);
        queScales_.EnQue(sUb);

        LocalTensor<uint8_t> w8 = quePacked_.DeQue<uint8_t>();
        LocalTensor<uint8_t> s8 = queScales_.DeQue<uint8_t>();

        // Whole-pool typed bases; every scratch tensor below is an element-
        // indexed view at the skewed constants above (uint8 index == byte
        // offset; 2-byte types index OFF/2), which keeps the exact byte
        // offsets statically resolvable for the layout checker.
        LocalTensor<uint8_t> s8pool = scratch_.Get<uint8_t>();
        LocalTensor<int16_t> s16pool = scratch_.Get<int16_t>();
        LocalTensor<half> sFpool = scratch_.Get<half>();

        // ---- scale vector: 512 blocks x (8 lanes) of bits16(s * 0.25) -----
        // Built through the int16 view; the half view feeds the Mul below.
        constexpr uint32_t SCALE_RUN_LANES = AIV_TILE_PACKED / AIV_TILE_SCALES;  // 8
        LocalTensor<int16_t> scaleI16 = s16pool[OFF_SCALEV / 2];
        for (uint32_t b = 0; b < AIV_TILE_SCALES; ++b) {
            const uint8_t sb = s8.GetValue(b);
            const uint16_t bits = static_cast<uint16_t>((((sb & 0x7F) << 7) + 0x1800));
            // Both 8-lane halves share a 32-byte aligned destination.
            // The 950 register store requires this alignment even for 8 lanes.
            if constexpr (FP4_BLOCK == 16) {
                uint64_t laneMask[2] = {(b & 1) ? 0xFF00ULL : 0xFFULL, 0};
                Duplicate(s16pool[OFF_SCALEV / 2 + (b / 2) * 16],
                          static_cast<int16_t>(bits), laneMask, 1, 1, 8);
            } else {
                Duplicate(s16pool[OFF_SCALEV / 2 + b * (FP4_BLOCK / 2)],
                          static_cast<int16_t>(bits), FP4_BLOCK / 2);
            }
        }

        // ---- nibble split -------------------------------------------------
        LocalTensor<uint8_t> codesLo = s8pool[OFF_CODES_LO];
        LocalTensor<uint8_t> codesHi = s8pool[OFF_CODES_HI];
        LocalTensor<uint8_t> const15 = s8pool[OFF_CONST15];
        And(codesLo, w8, const15, AIV_TILE_PACKED);
        ShiftRight(codesHi.ReinterpretCast<uint16_t>(), w8.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(4), AIV_TILE_PACKED / 2);
        And(codesHi, codesHi, const15, AIV_TILE_PACKED);

        // ---- output tile, plane 0 (even) then plane 1 (odd) ----------------
        LocalTensor<half> outUb = queOut_.AllocTensor<half>();
        DequantPlane(codesLo, s8pool, sFpool, scaleI16, outUb, 0);
        DequantPlane(codesHi, s8pool, sFpool, scaleI16, outUb, AIV_TILE_ELEMS / 2);

        quePacked_.FreeTensor(w8);
        queScales_.FreeTensor(s8);

        queOut_.EnQue(outUb);
        LocalTensor<half> outV = queOut_.DeQue<half>();
        DataCopy(gmOut_[static_cast<uint64_t>(tileIdx) * AIV_TILE_ELEMS], outV,
                 AIV_TILE_ELEMS);
        queOut_.FreeTensor(outV);
    }

    __aicore__ inline void DequantPlane(const LocalTensor<uint8_t> &codes,
                                        const LocalTensor<uint8_t> &s8pool,
                                        const LocalTensor<half> &sFpool,
                                        const LocalTensor<int16_t> &scaleI16,
                                        LocalTensor<half> &outUb, uint32_t laneOffset)
    {
        const uint32_t n = AIV_TILE_PACKED;  // lanes in one plane
        LocalTensor<uint8_t> mag = s8pool[OFF_MAG];
        LocalTensor<uint8_t> msk = s8pool[OFF_MSK];
        LocalTensor<uint8_t> const7 = s8pool[OFF_CONST7];
        LocalTensor<half> h = sFpool[OFF_HF / 2];
        LocalTensor<half> t = sFpool[OFF_TF / 2];
        LocalTensor<half> negT = sFpool[OFF_NEGF / 2];
        LocalTensor<half> zeroF = sFpool[OFF_ZEROF / 2];

        And(mag, codes, const7, n);
        Cast(h, codes, RoundMode::CAST_NONE, n);
        CompareScalar(msk, h, static_cast<half>(7), CMPMODE::GT, n);
        Cast(h, mag, RoundMode::CAST_NONE, n);
#ifdef FAST_E2M1
        // Produce 4*q = {0,2,4,6,8,12,16,24}, then reuse scale s/4.
        Muls(h, h, static_cast<half>(2), n);
        CompareScalar(msk, h, static_cast<half>(8), CMPMODE::GT, n);
        Muls(t, h, static_cast<half>(2), n);
        Adds(t, t, static_cast<half>(-8), n);
        Select(h, msk, t, h, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        CompareScalar(msk, h, static_cast<half>(16), CMPMODE::GT, n);
        Duplicate(t, static_cast<half>(24), n);
        Select(h, msk, t, h, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        Cast(t, codes, RoundMode::CAST_NONE, n);
        CompareScalar(msk, t, static_cast<half>(7), CMPMODE::GT, n);
#endif
        Mul(t, h, scaleI16.ReinterpretCast<half>(), n);     // x s*0.25
        Sub(negT, zeroF, t, n);
        // Negative-zero codes canonicalize to +0 because 0 - (0*s) = +0.
        Select(outUb[laneOffset], msk, negT, t, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    }

    TPipe *pipe_ = nullptr;
    uint32_t totalElems_ = 0;
    uint32_t numTiles_ = 0;

    TQue<TPosition::VECIN, 2> quePacked_;   // packed nibbles GM->UB
    TQue<TPosition::VECIN, 2> queScales_;   // E4M3 scale bytes
    TQue<TPosition::VECOUT, 2> queOut_;     // fp16 result UB->GM
    TBuf<TPosition::VECCALC> scratch_;                  // skewed scratch pool

    GlobalTensor<uint8_t> gmW_;
    GlobalTensor<uint8_t> gmS_;
    GlobalTensor<half> gmOut_;
};

}  // namespace AscendC

/*
 * Host-side tiling:
 *   uint32_t tiling[8] = { totalElems, 0, ... };   totalElems % 8192 == 0
 *
 * Grid: one AIV core per block (KERNEL_TYPE_AIV_ONLY); partition the tile
 * list across blocks host-side (tile i -> block i % blockDim on the AIV
 * queue).
 */
extern "C" __global__ __aicore__ void naive_aiv_fp4_to_fp16(GM_ADDR gmW, GM_ADDR gmS,
                                                            GM_ADDR gmOut,
                                                            GM_ADDR gmTiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::NaiveAivFp4Dequant kernel;
    kernel.Init(gmW, gmS, gmOut, gmTiling, &pipe);
    kernel.Process();
}

#endif  // NAIVE_AIV_FP4_TO_FP16_CPP

