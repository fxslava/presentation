/**
 * cube_trick_fp4_to_fp16.cpp -- Kernel B: Cube/Fixpipe-accelerated NVFP4 ->
 * FP16 dequantization (pseudo-GEMM trick from the research repo, portable
 * INT8-contraction form for Ascend 910B / DaVinci v2).
 *
 * One block = KERNEL_TYPE_MIX_AIC_1_1: the AIC reconstructs the mantissa
 * product on the systolic array (one Mmad per 16x16 tile), the Fixpipe
 * converts L0C -> fp16, and a thin AIV epilogue folds the block-scale
 * exponent with a pure INTEGER add on the fp16 bit pattern.  No vector FPU
 * multiply exists anywhere after the Cube.
 *
 * ---------------------------------------------------------------------------
 * Mathematical core (additive scale decomposition, K=2 replicated contraction)
 * ---------------------------------------------------------------------------
 * NVFP4 block = 16 weights w (FP4-E1M2 grid) + 1 FP8-E4M3 scale s, bias 7:
 *
 *     s = 2^E * (1 + m/8)  = 2^(E-2) * 4M,   4M = 4 + m/2 in {4.0 ... 7.5}
 *
 * 4M is partitioned additively into two FP4-E2M1 grid points (repo LUT):
 *
 *     4M = A0 + A1,   A0 in {2,4,6},  A1 in {0.5,1,1.5,2}
 *
 * INT8 contraction form (native on 910B -- the FP4 mad_mx SupportType tuple
 * fp4x2_e2m1_t x fp4x2_e1m2_t only exists on DaVinci v3 / ascend351x, where
 * the identical kernel runs with A' = A0/A1 and B' = w directly in fp4):
 *
 *     A'[g][k] = 4*A0_g at k = g,   4*A1_g at k = 16+g     in {2..24}, int8
 *     B'[k][n] = 4 * w[k mod 16][n]                        in [-7,7], int8
 *
 *     C'[g][n] = A' . B' = 16 * (A0+A1) * w = 16 * (4M * w)   (exact int32)
 *
 * ONE int8 Mmad (m=16, n=16, k=32 = a single k0=32 fractal step) per tile
 * produces C' exactly; k=32 exploits the int8 systolic depth perfectly.
 *
 * Fixpipe drains L0C int32 -> fp16 (|C'| <= 16*7.5*1.75 = 210, exact in
 * fp16) into a 2-slot GM staging ring.
 *
 * AIV epilogue (pure integer, no FPU):
 *
 *     out_bits = fp16_bits(C') + ((E - 6) << 10)
 *              = fp16_bits(C') + ((e_field - 13) << 10)
 *
 * with E = e_field - 7.  Non-zero C' has fp16 exponent field 19..23, so the
 * add stays inside the normal range for every legal scale; w == 0 keeps
 * C' == 0 and is preserved by a Min(|bits|,1) zero guard.
 *
 * ---------------------------------------------------------------------------
 * Pipeline (per tile t, slot p = t & 1, double-buffered A1/B1/L0A/L0B/L0C)
 * ---------------------------------------------------------------------------
 *   AIC:  S_MTE2 barrier -> MTE2 (Nd2Nz GM->L1, 512 B per operand, one full
 *         GM burst per copy) -> MTE1 (LoadData L1->L0A/L0B) -> M (one Mmad)
 *         -> FIX (Fixpipe L0C -> gmC ring, fp16) -> CrossCoreSetFlag(prod_p)
 *   AIV:  CrossCoreWaitFlag(prod_p) -> MTE2 (gmC ring + scale bytes) ->
 *         CrossCoreSetFlag(ack_p) -> 16 row Duplicates ((delta)<<10 offset
 *         tile) -> integer Add -> zero guard -> MTE3 fp16 tile.
 *
 *   Steady state overlap: MTE2[t+1] || MTE1[t] || M[t-1] || FIX[t-1] on the
 *   AIC, AIV epilogue of tile t-1 in parallel on the vector core.
 *
 * Hazard chain (all Set/Wait pairs use LITERAL event ids; parity selects the
 * id via a compile-time branch, never a runtime value):
 *   MTE2_MTE1  L1 fractal ready          -> MTE1 LoadData
 *   MTE1_M     L0A/L0B loaded            -> Mmad
 *   M_MTE1     Mmad drained              -> L0A/L0B slot reuse (t+2)
 *   M_FIX      L0C holds C'              -> Fixpipe
 *   FIX_M      Fixpipe drained           -> L0C slot reuse (t+2)
 *   MTE1_MTE2  LoadData drained          -> L1 slot reuse (t+2)
 *   prod_p     Fixpipe drained (AIC->AIV) -> AIV MTE2 of gmC slot p
 *   ack_p      AIV MTE2 drained (AIV->AIC) -> Fixpipe overwrites slot p (t+2)
 *
 * ---------------------------------------------------------------------------
 * GM contract
 * ---------------------------------------------------------------------------
 *   gmA     : numTiles * 512 B, ND [16 rows g][32 k] int8, row g carries
 *             4*A0_g at byte g and 4*A1_g at byte 16+g (host-prepared).
 *   gmB     : numTiles * 512 B, ND n-outer [16 rows n][32 k] int8,
 *             B'[k][n] = 4 * w[k mod 16][n] (host-prepared).
 *   gmC     : workspace ring, NUM_BUFFERS * 512 B fp16 (2 slots only).
 *   gmS     : numTiles * 32 B (16 E4M3 bytes + 16 B pad).
 *   gmOut   : numTiles * 512 B fp16, sequential [16 groups][16 weights].
 *   gmTiling: uint32_t[8]; [0] = numTiles.
 *
 * Target: Ascend 910B (DaVinci v2, INT8 Mmad k0=32) / "Ascend 950PR" label.
 * For ascend351x silicon the same kernel maps to MmadMx with fp4x2_e2m1_t /
 * fp4x2_e1m2_t operands (delta becomes (e_field - 9) << 10) and half the
 * MTE2 operand bytes; ascend_analyzer gates mad_mx behind the mx_cube
 * feature which only ascend351x implements.
 */
#ifndef CUBE_TRICK_FP4_TO_FP16_CPP
#define CUBE_TRICK_FP4_TO_FP16_CPP

#include "kernel_operator.h"

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

// Cross-core event ids (literal constants; SYNC_MODE2 = fire when the
// named pipe has drained all previously issued operations).
static constexpr uint64_t CT_SYNC_MODE2 = 2;
enum CtEventId : uint16_t {
    EVT_PROD_EVEN = 0,
    EVT_PROD_ODD = 1,
    EVT_ACK_EVEN = 2,
    EVT_ACK_ODD = 3,
};

// ------------------------------------------------- AIV skewed scratch pool
// 512-B tensors at offsets 32,576,1120,... -> pairwise deltas of 32/64/.../
// 224 mod 256: no two co-operands of one vector op share a UB bank phase.
static constexpr uint32_t CT_OFF_EXP = 32;       // int16 x 256 (delta<<10 tile)
static constexpr uint32_t CT_OFF_ABSMASK = 576;  // int16 x 256 (0x7FFF)
static constexpr uint32_t CT_OFF_ONE = 1120;     // int16 x 256 (1)
static constexpr uint32_t CT_OFF_ZERO = 1664;    // int16 x 256 (0)
static constexpr uint32_t CT_OFF_ABSC = 2208;    // int16 x 256 (|bits|)
static constexpr uint32_t CT_OFF_MASK01 = 2752;  // int16 x 256 (zero mask)
static constexpr uint32_t CT_OFF_SUM = 3296;     // int16 x 256 (bits+delta)
static constexpr uint32_t CT_SCRATCH_BYTES = 3808;

// Host tiling block (canonical CANN layout; field values bind from the
// analyzer's --tiling-data manifest, and the host writes the same struct).
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
                             static_cast<uint64_t>(CT_NUM_BUFFERS) * CT_C_ELEMS);

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
        CrossCoreWaitFlag(EVT_ACK_EVEN);
        CrossCoreWaitFlag(EVT_ACK_ODD);
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

        // gmC ring slot P is reusable only after the vector core finished
        // reading tile t-2 (the vector stage primes both ack tokens at its
        // start, encoding that both ring slots are initially free).
                    CrossCoreWaitFlag(EVT_ACK_EVEN);

        // Fixpipe: L0C int32 -> GM fp16, NZ2ND row-major [16][16]; the
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
            gmC_[0], l0cT, fixParams);
                    SetFlag<HardEvent::FIX_M>(EVENT_ID0);  // L0C[0] reusable
            // Producer signal: fires when the FIX pipe has drained the store.
            CrossCoreSetFlag<CT_SYNC_MODE2, PIPE_FIX>(EVT_PROD_EVEN);
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

        // gmC ring slot P is reusable only after the vector core finished
        // reading tile t-2 (the vector stage primes both ack tokens at its
        // start, encoding that both ring slots are initially free).
                    CrossCoreWaitFlag(EVT_ACK_ODD);

        // Fixpipe: L0C int32 -> GM fp16, NZ2ND row-major [16][16]; the
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
            gmC_[static_cast<uint64_t>(CT_C_ELEMS)], l0cT, fixParams);
                    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
            CrossCoreSetFlag<CT_SYNC_MODE2, PIPE_FIX>(EVT_PROD_ODD);
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
    GlobalTensor<half> gmC_;  // fp32/int32-free: fp16 staging ring, 2 slots
};

// ===========================================================================
// Vector stage (AIV)
// ===========================================================================
// @ascend-layout: name=expOff pos=VECCALC dtype=int16_t count=256
// @ascend-layout: name=sumI pos=VECCALC dtype=int16_t count=256
// @ascend-layout: name=absC pos=VECCALC dtype=int16_t count=256
// @ascend-layout: name=mask01 pos=VECCALC dtype=int16_t count=256
// @ascend-layout: name=cI pos=VECCALC dtype=int16_t count=256
class CubeTrickVectorStage {
public:
    __aicore__ inline void Init(GM_ADDR gmS, GM_ADDR gmC, GM_ADDR gmOut, GM_ADDR gmTiling,
                                TPipe *pipe)
    {
        pipe_ = pipe;
        __gm__ CubeTrickTiling *tiling = (__gm__ CubeTrickTiling *)gmTiling;
        numTiles_ = tiling->numTiles;

        gmS_.SetGlobalBuffer((__gm__ uint8_t *)gmS,
                             static_cast<uint64_t>(numTiles_) * CT_S_BYTES);
        gmC_.SetGlobalBuffer((__gm__ half *)gmC,
                             static_cast<uint64_t>(CT_NUM_BUFFERS) * CT_C_ELEMS);
        gmOut_.SetGlobalBuffer((__gm__ half *)gmOut,
                               static_cast<uint64_t>(numTiles_) * CT_C_ELEMS);

        pipe_->InitBuffer(queC_, CT_NUM_BUFFERS, CT_C_F16_BYTES);
        pipe_->InitBuffer(queS_, CT_NUM_BUFFERS, CT_S_BYTES);
        pipe_->InitBuffer(queOut_, CT_NUM_BUFFERS, CT_OUT_BYTES);
        pipe_->InitBuffer(scratch_, CT_SCRATCH_BYTES);

        // Constant lanes, built once (element-indexed views of the pool).
        LocalTensor<int16_t> s16pool = scratch_.Get<int16_t>();
        Duplicate(s16pool[CT_OFF_ABSMASK / 2], static_cast<int16_t>(0x7FFF), CT_C_ELEMS);
        Duplicate(s16pool[CT_OFF_ONE / 2], static_cast<int16_t>(1), CT_C_ELEMS);
        Duplicate(s16pool[CT_OFF_ZERO / 2], static_cast<int16_t>(0), CT_C_ELEMS);
    }

    __aicore__ inline void Process()
    {
        // Prime the cross-core ack tokens: both gmC ring slots are initially
        // free for the cube's Fixpipe writes.  Priming keeps the AIC's ack
        // waits unconditional and the cross-core ledger balanced (per id:
        // 1 prime + N sets == N AIC waits + 1 AIC drain).
        CrossCoreSetFlag<CT_SYNC_MODE2, PIPE_MTE2>(EVT_ACK_EVEN);
        CrossCoreSetFlag<CT_SYNC_MODE2, PIPE_MTE2>(EVT_ACK_ODD);
        for (uint32_t t = 0; t < numTiles_; ++t) {
            ConsumeTile(t);
        }
        // All prod tokens were consumed in-loop; the ack ledger is closed by
        // the cube stage's epilogue drain.  Nothing to wait on exit.
    }

private:
    __aicore__ inline void ConsumeTile(uint32_t tileIdx)
    {
        const bool even = (tileIdx % CT_NUM_BUFFERS) == 0;

        // ---- fp16 C' tile: wait for the Cube producer, then MTE2 GM->UB ---
        LocalTensor<half> cUb = queC_.AllocTensor<half>();
        if (even) {
            CrossCoreWaitFlag(EVT_PROD_EVEN);
        } else {
            CrossCoreWaitFlag(EVT_PROD_ODD);
        }
        DataCopy(cUb, gmC_[static_cast<uint64_t>(tileIdx % CT_NUM_BUFFERS) * CT_C_ELEMS],
                 CT_C_ELEMS);
        queC_.EnQue(cUb);

        // Scale bytes are Cube-independent; prefetch alongside.
        LocalTensor<uint8_t> sUb = queS_.AllocTensor<uint8_t>();
        DataCopy(sUb, gmS_[static_cast<uint64_t>(tileIdx) * CT_S_BYTES], CT_S_BYTES);
        queS_.EnQue(sUb);

        // Release gmC ring slot once BOTH MTE2 reads have drained.
        if (even) {
            CrossCoreSetFlag<CT_SYNC_MODE2, PIPE_MTE2>(EVT_ACK_EVEN);
        } else {
            CrossCoreSetFlag<CT_SYNC_MODE2, PIPE_MTE2>(EVT_ACK_ODD);
        }

        LocalTensor<half> c16 = queC_.DeQue<half>();
        LocalTensor<uint8_t> s8 = queS_.DeQue<uint8_t>();
        LocalTensor<int16_t> cI = c16.ReinterpretCast<int16_t>();

        // ---- integer exponent fold ---------------------------------------
        // Whole-pool int16 base; scratch tensors are element-indexed views
        // (index == OFF/2) so byte offsets stay statically resolvable.
        LocalTensor<int16_t> s16pool = scratch_.Get<int16_t>();

        // delta = (E - 6) = (exp_field - 13), <<10 into the fp16 exponent
        // field; one 16-lane Duplicate row per scale group (16 rows).
        LocalTensor<int16_t> expOff = s16pool[CT_OFF_EXP / 2];
        for (uint32_t g = 0; g < CT_TILE_M; ++g) {
            const uint8_t scaleByte = s8.GetValue(g);
            const int16_t expAdd =
                static_cast<int16_t>((((scaleByte >> 3) & 0xF) - 13) << 10);
            Duplicate(s16pool[(CT_OFF_EXP + g * CT_TILE_N * 2) / 2], expAdd, CT_TILE_N);
        }

        // bits + (delta << 10): pure integer lanes.
        LocalTensor<int16_t> sumI = s16pool[CT_OFF_SUM / 2];
        Add(sumI, cI, expOff, CT_C_ELEMS);

        // Zero guard: C' == 0 must remain exactly 0 (the exponent add would
        // otherwise synthesize a bogus normal).  Every non-zero C' has
        // |bits| >= 0x4C00, so Min(|bits|, 1) is an exact 0/1 mask built
        // without a compare op:  mask = Min(|C'|, 1); lanes = 0 - mask.
        LocalTensor<int16_t> absC = s16pool[CT_OFF_ABSC / 2];
        And(absC, cI, s16pool[CT_OFF_ABSMASK / 2], CT_C_ELEMS);
        LocalTensor<int16_t> mask01 = s16pool[CT_OFF_MASK01 / 2];
        Min(mask01, absC, s16pool[CT_OFF_ONE / 2], CT_C_ELEMS);
        // absC is dead past the Min; V-pipe ops issue in order, so the buffer
        // is safely reused for the 0 -> 0xFFFF negation.
        Sub(absC, s16pool[CT_OFF_ZERO / 2], mask01, CT_C_ELEMS);

        // ---- store fp16 tile (sequential [16][16] layout) ------------------
        LocalTensor<half> outUb = queOut_.AllocTensor<half>();
        And(outUb.ReinterpretCast<int16_t>(), sumI, absC, CT_C_ELEMS);
        queOut_.EnQue(outUb);
        LocalTensor<half> outV = queOut_.DeQue<half>();
        DataCopy(gmOut_[static_cast<uint64_t>(tileIdx) * CT_C_ELEMS], outV, CT_C_ELEMS);
        queOut_.FreeTensor(outV);

        queC_.FreeTensor(c16);
        queS_.FreeTensor(s8);
    }

    TPipe *pipe_ = nullptr;
    uint32_t numTiles_ = 0;

    TQue<TPosition::VECIN, 2> queC_;   // fp16 C' ring GM->UB
    TQue<TPosition::VECIN, 2> queS_;   // E4M3 scale bytes
    TQue<TPosition::VECOUT, 2> queOut_;  // fp16 tile UB->GM
    TBuf<TPosition::VECCALC> scratch_;              // skewed scratch pool

    GlobalTensor<uint8_t> gmS_;
    GlobalTensor<half> gmC_;
    GlobalTensor<half> gmOut_;
};

}  // namespace AscendC

/*
 * Host-side tiling:
 *   uint32_t tiling[8] = { numTiles, 0, ... };
 *   gmC workspace = 2 * 512 B (ping-pong ring, NOT numTiles slots).
 */
extern "C" __global__ __aicore__ void cube_trick_fp4_to_fp16(GM_ADDR gmA, GM_ADDR gmB,
                                                             GM_ADDR gmS, GM_ADDR gmC,
                                                             GM_ADDR gmOut,
                                                             GM_ADDR gmTiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIC {
        AscendC::TPipe pipe;
        AscendC::CubeTrickCubeStage cube;
        cube.Init(gmA, gmB, gmC, gmTiling, &pipe);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        AscendC::CubeTrickVectorStage vec;
        vec.Init(gmS, gmC, gmOut, gmTiling, &pipe);
        vec.Process();
    }
}

#endif  // CUBE_TRICK_FP4_TO_FP16_CPP
