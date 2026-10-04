/**
 * Native Ascend950PR (dav-3510) FP4 E1M2 dequantization hybrid.
 * Entry: simt_cube_hybrid. Build with FP4_BLOCK=16, 32, or 64.
 *
 * SIMT VF #1 reads packed FP4 and E4M3 scales directly from GM, forming
 * int8 ND Cube operands in GM staging. The AIC uses MTE2/MTE1, one INT8
 * Mmad per 16x16 output tile, and DEQF16 Fixpipe into an FP16 GM buffer.
 * SIMT VF #2 folds the scale exponent into FP16 bits and writes output GM.
 * Two cross-core flags order these three phases. No vector Duplicate runs.
 *
 * Scale contract: positive normal E4M3, NaN code excluded.
 * Output contract: sequential FP16, 256 values per Cube tile.
 */
#ifndef SIMT_CUBE_HYBRID_CPP
#define SIMT_CUBE_HYBRID_CPP

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#ifndef FP4_BLOCK
#define FP4_BLOCK 16
#endif

// SIMT threads read packed FP4 and block scales from GM into scalar registers.
// The Cube core consumes the resulting ND int8 operands from a GM workspace.
__simt_vf__ __launch_bounds__(512) inline void hybrid_prepare_vf(
    __gm__ const uint8_t *packed, __gm__ const uint8_t *scales,
    __gm__ int8_t *operand_a, __gm__ int8_t *operand_b, uint32_t tiles)
{
    uint32_t x = threadIdx.x;
    for (uint32_t t = 0; t < tiles; ++t) {
        uint32_t row = x >> 5;
        uint32_t col = x & 31;
        uint32_t sb = scales[(t * 256 + row * 16) / FP4_BLOCK];
        uint32_t mant = sb & 7;
        int32_t av0 = mant == 0 ? 8 : (mant < 5 ? 16 : 24);
        int32_t av1 = 16 + 2 * static_cast<int32_t>(mant) - av0;
        operand_a[t * 512 + x] = (col == row) ? av0 : ((col == row + 16) ? av1 : 0);

        uint32_t group = col & 15;
        uint32_t weight = t * 256 + group * 16 + row;
        uint8_t byte = packed[weight >> 1];
        uint32_t code = (byte >> ((weight & 1) * 4)) & 15;
        int32_t val = (code & 8) ? -static_cast<int32_t>(code & 7) : static_cast<int32_t>(code & 7);
        operand_b[t * 512 + x] = val;
    }
}

// Fold the E4M3 exponent into Fixpipe's exact FP16 result, per SIMT thread.
__simt_vf__ __launch_bounds__(1024) inline void hybrid_epilogue_vf(
    __gm__ const uint16_t *cube_bits, __gm__ const uint8_t *scales,
    __gm__ uint16_t *output, uint32_t elements)
{
    for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x) {
        uint16_t bits = cube_bits[i];
        uint32_t exp = (scales[i / FP4_BLOCK] >> 3) & 15;
        int32_t delta = static_cast<int32_t>(exp) - 13;
        output[i] = (bits & 0x7fff) ? static_cast<uint16_t>(static_cast<int32_t>(bits) + delta * 1024) : 0;
    }
}

namespace AscendC {

// ------------------------------------------------------------ tile geometry
static constexpr uint16_t CT_TILE_M = 16;  // scale groups per tile (M)
static constexpr uint16_t CT_TILE_N = 16;  // weights per group    (N)
static constexpr uint16_t CT_TILE_K = 32;  // 2 x 16 replicated K (int8 k0=32)
static constexpr uint16_t CT_NUM_BUFFERS = 2;

static constexpr uint16_t CT_A_BYTES = CT_TILE_M * CT_TILE_K;        // 512
static constexpr uint16_t CT_B_BYTES = CT_TILE_K * CT_TILE_N;        // 512
static constexpr uint16_t CT_C_ELEMS = CT_TILE_M * CT_TILE_N;        // 256
static constexpr uint16_t CT_L0C_BYTES = CT_C_ELEMS * 4;             // 1024 int32
static constexpr uint16_t CT_C_F16_BYTES = CT_C_ELEMS * 2;           // 512
static constexpr uint16_t CT_S_BYTES = 32;                           // 16 + pad
static constexpr uint16_t CT_OUT_BYTES = CT_C_F16_BYTES;

// NZ C0 stride for int8 fractals (32 elements per C0 unit).
static constexpr uint16_t CT_NZ_C0_STRIDE = 32;

// Host tiling block; [0] is the number of 256-output Cube tiles.
struct CubeTrickTiling {
    uint32_t numTiles;
    uint32_t reserved[7];
};

// ===========================================================================
// Cube stage (AIC)
// ===========================================================================
// Layout annotations for the element-indexed slot/scratch views (counts in
// elements; offsets come from the view index into the ping-pong pools).
// @ascend-layout: name=a1 pos=A1 dtype=int8_t count=512
// @ascend-layout: name=b1 pos=B1 dtype=int8_t count=512
// @ascend-layout: name=l0aT pos=A2 dtype=int8_t count=512
// @ascend-layout: name=l0bT pos=B2 dtype=int8_t count=512
// @ascend-layout: name=l0cT pos=CO1 dtype=int32_t count=256
class CubeTrickCubeStage {
public:
    __aicore__ inline void Init(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmTiling,
                                TPipe *pipe)
    {
        pipe_ = pipe;
        __gm__ CubeTrickTiling *tiling = (__gm__ CubeTrickTiling *)gmTiling;
        numTiles_ = tiling->numTiles;

        gmA_.SetGlobalBuffer((__gm__ int8_t *)gmA,
                             static_cast<uint64_t>(numTiles_) * CT_A_BYTES);
        gmB_.SetGlobalBuffer((__gm__ int8_t *)gmB,
                             static_cast<uint64_t>(numTiles_) * CT_B_BYTES);
        gmC_.SetGlobalBuffer((__gm__ half *)gmC,
                             static_cast<uint64_t>(numTiles_) * CT_C_ELEMS);

        pipe_->InitBuffer(a1Buf_, CT_NUM_BUFFERS * CT_A_BYTES);
        pipe_->InitBuffer(b1Buf_, CT_NUM_BUFFERS * CT_B_BYTES);
        pipe_->InitBuffer(l0aBuf_, CT_NUM_BUFFERS * CT_A_BYTES);
        pipe_->InitBuffer(l0bBuf_, CT_NUM_BUFFERS * CT_B_BYTES);
        pipe_->InitBuffer(l0cBuf_, CT_NUM_BUFFERS * CT_L0C_BYTES);
    }

    __aicore__ inline void Process()
    {
        // Prime the slot-free tokens for tiles 0/1: both ping-pong slots of
        // L0A/L0B, L0C and L1 start free, and priming lets every in-loop
        // WaitFlag issue unconditionally, which keeps the set/wait ledger
        // exactly balanced (per id: 1 prime + N sets == N waits + 1 drain).
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        for (uint32_t t = 0; t < numTiles_; ++t) {
            LoadTile(t);
            ComputeTile(t);
        }
        // Drain: exactly one token per id survives the 2-slot pipeline.
        // Consuming them leaves no hardware event set at kernel exit and
        // closes the ledger against the priming sets above.
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    }

private:
    __aicore__ inline void LoadTile(uint32_t tileIdx)
    {
        if (tileIdx % CT_NUM_BUFFERS == 0) {
            LoadTileSlotEven(tileIdx);
        } else {
            LoadTileSlotOdd(tileIdx);
        }
    }

    // Slot bodies are deliberately NOT templated: a template parameter is
    // opaque to static analysis, so each slot gets a fully literal body
    // (event ids, pool offsets) selected by the runtime parity dispatch,
    // which folds per loop iteration.

    __aicore__ inline void LoadTileSlotEven(uint32_t tileIdx)
    {
        // Scalar params for the DataCopy below must be visible to MTE2.
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        // L1 slot 0 is reusable only after LoadData of tile t-2 drained
        // (the process prologue primes this token for tiles 0/1).
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        // Slot views of the ping-pong pools (element index == byte offset
        // for int8), statically resolvable by the layout checker.
        LocalTensor<int8_t> a1 = a1Buf_.Get<int8_t>()[0];
        LocalTensor<int8_t> b1 = b1Buf_.Get<int8_t>()[0];

        // GM ND [16 rows][32 k] int8 -> L1 NZ, one 512-B GM burst per copy.
        Nd2NzParams nzA;
        nzA.ndNum = 1;
        nzA.nValue = CT_TILE_M;
        nzA.dValue = CT_TILE_K;
        nzA.srcNdMatrixStride = 0;
        nzA.srcDValue = CT_TILE_K;  // GM row stride (elements)
        nzA.dstNzC0Stride = CT_NZ_C0_STRIDE;
        nzA.dstNzNStride = 1;
        nzA.dstNzMatrixStride = 0;
        DataCopy(a1, gmA_[static_cast<uint64_t>(tileIdx) * CT_A_BYTES], nzA);

        Nd2NzParams nzB;
        nzB.ndNum = 1;
        nzB.nValue = CT_TILE_N;
        nzB.dValue = CT_TILE_K;
        nzB.srcNdMatrixStride = 0;
        nzB.srcDValue = CT_TILE_K;
        nzB.dstNzC0Stride = CT_NZ_C0_STRIDE;
        nzB.dstNzNStride = 1;
        nzB.dstNzMatrixStride = 0;
        DataCopy(b1, gmB_[static_cast<uint64_t>(tileIdx) * CT_B_BYTES], nzB);

        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void LoadTileSlotOdd(uint32_t tileIdx)
    {
        SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);

        LocalTensor<int8_t> a1 = a1Buf_.Get<int8_t>()[CT_A_BYTES];
        LocalTensor<int8_t> b1 = b1Buf_.Get<int8_t>()[CT_B_BYTES];

        Nd2NzParams nzA;
        nzA.ndNum = 1;
        nzA.nValue = CT_TILE_M;
        nzA.dValue = CT_TILE_K;
        nzA.srcNdMatrixStride = 0;
        nzA.srcDValue = CT_TILE_K;
        nzA.dstNzC0Stride = CT_NZ_C0_STRIDE;
        nzA.dstNzNStride = 1;
        nzA.dstNzMatrixStride = 0;
        DataCopy(a1, gmA_[static_cast<uint64_t>(tileIdx) * CT_A_BYTES], nzA);

        Nd2NzParams nzB;
        nzB.ndNum = 1;
        nzB.nValue = CT_TILE_N;
        nzB.dValue = CT_TILE_K;
        nzB.srcNdMatrixStride = 0;
        nzB.srcDValue = CT_TILE_K;
        nzB.dstNzC0Stride = CT_NZ_C0_STRIDE;
        nzB.dstNzNStride = 1;
        nzB.dstNzMatrixStride = 0;
        DataCopy(b1, gmB_[static_cast<uint64_t>(tileIdx) * CT_B_BYTES], nzB);

        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void ComputeTile(uint32_t tileIdx)
    {
        if (tileIdx % CT_NUM_BUFFERS == 0) {
            ComputeTileSlotEven(tileIdx);
        } else {
            ComputeTileSlotOdd(tileIdx);
        }
    }

    __aicore__ inline void ComputeTileSlotEven(uint32_t tileIdx)
    {
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        LocalTensor<int8_t> a1 = a1Buf_.Get<int8_t>()[0];
        LocalTensor<int8_t> b1 = b1Buf_.Get<int8_t>()[0];
        LocalTensor<int8_t> l0aT = l0aBuf_.Get<int8_t>()[0];
        LocalTensor<int8_t> l0bT = l0bBuf_.Get<int8_t>()[0];
        LocalTensor<int32_t> l0cT = l0cBuf_.Get<int32_t>()[0];

        // L0A/L0B slot P must be free: Mmad of tile t-2 drained (primed for
        // tiles 0/1 by the process prologue).
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);

        // L1 NZ -> L0A ZZ / L0B ZN: one LoadData2D per operand covering the
        // full 16x32 int8 fractal (mStep=1, kStep=1; k=32 is a single int8
        // k0 step).  LoadData2DParamsV2 is the dav-c310/CANN 9.x spelling;
        // CANN 7/8.x for 910B uses LoadData2DParams with the same fields.
        LoadData2DParamsV2 loadA(0, 0, 1, 1, 0, 1, false, 0);
        LoadData(l0aT, a1, loadA);
        LoadData2DParamsV2 loadB(0, 0, 1, 1, 0, 1, false, 0);
        LoadData(l0bT, b1, loadB);
        PipeBarrier<PIPE_MTE1>();
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);  // L1 slot free for t+2
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

        // L0C slot P must be free: Fixpipe of tile t-2 drained (primed for
        // tiles 0/1 by the process prologue).
                    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

        // ONE int8 Mmad per tile: m=16 groups, n=16 weights, k=32 (the two
        // replicated K halves carry 4*A0 and 4*A1).  cmatrixInitVal=true:
        // this Mmad is the full K=32 sum (overwrites, no accumulate).
        MmadParams mmadParams;
        mmadParams.m = CT_TILE_M;
        mmadParams.n = CT_TILE_N;
        mmadParams.k = CT_TILE_K;
        mmadParams.cmatrixInitVal = true;
        Mmad(l0cT, l0aT, l0bT, mmadParams);
                    SetFlag<HardEvent::M_FIX>(EVENT_ID0);   // L0C[0] holds C'
            SetFlag<HardEvent::M_MTE1>(EVENT_ID3);  // L0A/B[0] reusable
            // Fixpipe only after the Mmad has fully landed in L0C.
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

        // Fixpipe: L0C int32 -> this tile's GM FP16 slot, NZ2ND [16][16]; the
        // int32->half conversion is exact (|C'| <= 210, 8 significand bits).
        FixpipeParamsV220 fixParams;
        fixParams.nSize = CT_TILE_N;
        fixParams.mSize = CT_TILE_M;
        fixParams.srcStride = 0;
        fixParams.dstStride = CT_TILE_N;  // GM row stride (fp16 elements)
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::DEQF16;
        fixParams.deqScalar = 0x3F800000ULL; // fp32 unity dequantization scale
        Fixpipe<half, int32_t, CFG_ROW_MAJOR>(
            gmC_[static_cast<uint64_t>(tileIdx) * CT_C_ELEMS], l0cT, fixParams);
                    SetFlag<HardEvent::FIX_M>(EVENT_ID0);  // L0C[0] reusable
    }
    __aicore__ inline void ComputeTileSlotOdd(uint32_t tileIdx)
    {
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        LocalTensor<int8_t> a1 = a1Buf_.Get<int8_t>()[CT_A_BYTES];
        LocalTensor<int8_t> b1 = b1Buf_.Get<int8_t>()[CT_A_BYTES];
        LocalTensor<int8_t> l0aT = l0aBuf_.Get<int8_t>()[CT_A_BYTES];
        LocalTensor<int8_t> l0bT = l0bBuf_.Get<int8_t>()[CT_A_BYTES];
        LocalTensor<int32_t> l0cT = l0cBuf_.Get<int32_t>()[CT_L0C_BYTES / 4];

        // L0A/L0B slot P must be free: Mmad of tile t-2 drained (primed for
        // tiles 0/1 by the process prologue).
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);

        // L1 NZ -> L0A ZZ / L0B ZN: one LoadData2D per operand covering the
        // full 16x32 int8 fractal (mStep=1, kStep=1; k=32 is a single int8
        // k0 step).  LoadData2DParamsV2 is the dav-c310/CANN 9.x spelling;
        // CANN 7/8.x for 910B uses LoadData2DParams with the same fields.
        LoadData2DParamsV2 loadA(0, 0, 1, 1, 0, 1, false, 0);
        LoadData(l0aT, a1, loadA);
        LoadData2DParamsV2 loadB(0, 0, 1, 1, 0, 1, false, 0);
        LoadData(l0bT, b1, loadB);
        PipeBarrier<PIPE_MTE1>();
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        // L0C slot P must be free: Fixpipe of tile t-2 drained (primed for
        // tiles 0/1 by the process prologue).
                    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);

        // ONE int8 Mmad per tile: m=16 groups, n=16 weights, k=32 (the two
        // replicated K halves carry 4*A0 and 4*A1).  cmatrixInitVal=true:
        // this Mmad is the full K=32 sum (overwrites, no accumulate).
        MmadParams mmadParams;
        mmadParams.m = CT_TILE_M;
        mmadParams.n = CT_TILE_N;
        mmadParams.k = CT_TILE_K;
        mmadParams.cmatrixInitVal = true;
        Mmad(l0cT, l0aT, l0bT, mmadParams);
                    SetFlag<HardEvent::M_FIX>(EVENT_ID1);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID1);

        // Fixpipe: L0C int32 -> this tile's GM FP16 slot, NZ2ND [16][16]; the
        // int32->half conversion is exact (|C'| <= 210, 8 significand bits).
        FixpipeParamsV220 fixParams;
        fixParams.nSize = CT_TILE_N;
        fixParams.mSize = CT_TILE_M;
        fixParams.srcStride = 0;
        fixParams.dstStride = CT_TILE_N;  // GM row stride (fp16 elements)
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::DEQF16;
        fixParams.deqScalar = 0x3F800000ULL;
        Fixpipe<half, int32_t, CFG_ROW_MAJOR>(
            gmC_[static_cast<uint64_t>(tileIdx) * CT_C_ELEMS], l0cT, fixParams);
                    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    }

    TPipe *pipe_ = nullptr;
    uint32_t numTiles_ = 0;

    TBuf<TPosition::A1> a1Buf_;    // 2 x 512 B L1 staging (ping-pong)
    TBuf<TPosition::B1> b1Buf_;    // 2 x 512 B
    TBuf<TPosition::A2> l0aBuf_;   // 2 x 512 B L0A slots
    TBuf<TPosition::B2> l0bBuf_;   // 2 x 512 B L0B slots
    TBuf<TPosition::CO1> l0cBuf_;  // 2 x 1024 B L0C accumulator slots

    GlobalTensor<int8_t> gmA_;
    GlobalTensor<int8_t> gmB_;
    GlobalTensor<half> gmC_;  // one FP16 staging slot per output tile
};

}  // namespace AscendC

extern "C" __global__ __aicore__ void simt_cube_hybrid(
    GM_ADDR gmPacked, GM_ADDR gmScale, GM_ADDR gmA, GM_ADDR gmB,
    GM_ADDR gmC, GM_ADDR gmOut, GM_ADDR gmTiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIV {
        __gm__ const uint32_t *tile = reinterpret_cast<__gm__ const uint32_t *>(gmTiling);
        asc_vf_call<hybrid_prepare_vf>(dim3{512},
            reinterpret_cast<__gm__ const uint8_t *>(gmPacked),
            reinterpret_cast<__gm__ const uint8_t *>(gmScale),
            reinterpret_cast<__gm__ int8_t *>(gmA),
            reinterpret_cast<__gm__ int8_t *>(gmB), tile[0]);
        AscendC::CrossCoreSetFlag<2, PIPE_V>(5);
        AscendC::CrossCoreWaitFlag(4);
        asc_vf_call<hybrid_epilogue_vf>(dim3{1024},
            reinterpret_cast<__gm__ const uint16_t *>(gmC),
            reinterpret_cast<__gm__ const uint8_t *>(gmScale),
            reinterpret_cast<__gm__ uint16_t *>(gmOut), tile[0] * 256);
    }
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag(5);
        AscendC::TPipe pipe;
        AscendC::CubeTrickCubeStage cube;
        cube.Init(gmA, gmB, gmC, gmTiling, &pipe);
        cube.Process();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(4);
    }
}

#endif  // SIMT_CUBE_HYBRID_CPP


