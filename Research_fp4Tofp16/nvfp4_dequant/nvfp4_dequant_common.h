/**
 * nvfp4_dequant_common.h
 *
 * Shared constants, layouts and the E4M3->2xE2M1 scale decomposition LUT for the
 * Cube-accelerated NVFP4 -> FP16 dequantization pipeline on Ascend 950PR
 * (dav-c310, __NPU_ARCH__ 3510, CANN 9.x).
 *
 * ---------------------------------------------------------------------------
 * Mathematical core
 * ---------------------------------------------------------------------------
 * An NVFP4 block carries 16 weights w (FP4) plus one FP8-E4M3 block scale s.
 * With E4M3 bias 7:
 *
 *      s = 2^E * (1 + m/8),  E in [-6, 8],  m in [0,7]
 *        = 2^(E-2) * 4M,     4M = 4 + m/2 in {4.0, 4.5, ..., 7.5}   (16 values)
 *
 * 4M is partitioned additively into two native E2M1 grid points:
 *
 *      4M = A0 + A1,  A0 in {2,3,4,6},  A1 in {0.5,1,1.5,2,3}
 *
 * (see k4mDecompLUT below; A0 restricted to {2,4,6} and A1 to {0.5..2} is a
 *  sufficient monotone subcover of all 16 mantissas).
 *
 * Per weight:   s * w = 2^(E-2) * (A0*w + A1*w)
 *
 * (A0*w + A1*w) is a K=2 dot product -> computed by ONE fp4 `mad_mx` systolic
 * tile with:
 *      L0A (A-side, fp4x2_e2m1_t): row g = scale group g, A[g][g]     = A0_g
 *                                                      A[g][16+g]  = A1_g
 *      L0B (B-side, fp4x2_e1m2_t): B[k][n] = w[(k mod 16)][n]  (weight row
 *                                   replication along K, [W;W] tiling)
 *      C[g][n] (fp32, L0C)       = 4M_g * w_gn          (exact in fp32)
 *
 * The effective exponent shift applied POST-Cube is therefore
 *
 *      delta_e_eff = E - 2 = (E4M3_exp_field - 7) - 2 = exp_field - 9
 *
 * which the AIV stage folds into the FP16 bit pattern with a pure INTEGER add:
 *
 *      fp16_bits(out) = fp16_bits(C) + (delta_e_eff << 10)   [when C != 0]
 *
 * No vector FPU multiply is performed anywhere after the Cube.
 *
 * ---------------------------------------------------------------------------
 * Tile / buffer geometry (one minimum 16x16 systolic tile, FP4 => K = 32)
 * ---------------------------------------------------------------------------
 *   gmA      : numTiles * A_FRACTAL_BYTES   L0A-fractal scale-term pairs
 *   gmB      : numTiles * B_FRACTAL_BYTES   L0B-fractal k-replicated weights
 *   gmS      : numTiles * SCALE_BYTES       16 E4M3 scale bytes + 16B pad
 *   gmC      : numTiles * C_F32_BYTES       fp32 Cube accumulators (staging,
 *                                            Fixpipe L0C->UB/GM handoff)
 *   gmOut    : numTiles * OUT_F16_BYTES     fp16 dequantized weights [g][n]
 */
#ifndef NVFP4_DEQUANT_COMMON_H
#define NVFP4_DEQUANT_COMMON_H

#include <cstdint>

// ---------------------------------------------------------------- tile shape
static constexpr uint16_t NVFP4_BLK   = 16;  // NVFP4 block size (weights per scale)
static constexpr uint16_t TILE_M      = 16;  // scale groups per tile (M dim)
static constexpr uint16_t TILE_N      = 16;  // weights per group   (N dim)
static constexpr uint16_t TILE_K      = 32;  // 2 x 16 replicated K (fp4 k-unit)
static constexpr uint16_t NUM_BUFFERS = 2;   // double buffering depth

// ------------------------------------------------------------- byte counts
static constexpr uint16_t A_FRACTAL_BYTES = TILE_M * TILE_K / 2;  // 256
static constexpr uint16_t B_FRACTAL_BYTES = TILE_K * TILE_N / 2;  // 256
static constexpr uint16_t SCALE_BYTES     = 32;  // 16 E4M3 bytes padded to one 32B MTE block
static constexpr uint16_t C_F32_BYTES     = TILE_M * TILE_N * 4;  // 1024
static constexpr uint16_t OUT_F16_BYTES   = TILE_M * TILE_N * 2;  // 512

// ------------------------------------------------------------ E2M1 encodings
// OCP MX E2M1 (1S/2E/1M, bias 1): 0,0.5,1,1.5,2,3,4,6  -> codes 0x0..0x7
enum Fp4E2M1Code : uint8_t {
    FP4_E2M1_P0_0 = 0x0,
    FP4_E2M1_P0_5 = 0x1,
    FP4_E2M1_P1_0 = 0x2,
    FP4_E2M1_P1_5 = 0x3,
    FP4_E2M1_P2_0 = 0x4,
    FP4_E2M1_P3_0 = 0x5,
    FP4_E2M1_P4_0 = 0x6,
    FP4_E2M1_P6_0 = 0x7,
};

/**
 * 4M decomposition LUT, indexed by mantissa half-steps:
 *   idx = 2*(4M) - 8  in [0,7]   <=>   4M = 4.0 + 0.5*idx
 * Each entry: {A0 code, A1 code}, both E2M1 grid points, A0 + A1 == 4M.
 */
struct Fp4ScalePair {
    uint8_t a0;
    uint8_t a1;
};

static constexpr Fp4ScalePair K4M_DECOMP_LUT[8] = {
    {FP4_E2M1_P2_0, FP4_E2M1_P2_0},  // 4.0 = 2.0 + 2.0
    {FP4_E2M1_P4_0, FP4_E2M1_P0_5},  // 4.5 = 4.0 + 0.5
    {FP4_E2M1_P4_0, FP4_E2M1_P1_0},  // 5.0 = 4.0 + 1.0
    {FP4_E2M1_P4_0, FP4_E2M1_P1_5},  // 5.5 = 4.0 + 1.5
    {FP4_E2M1_P4_0, FP4_E2M1_P2_0},  // 6.0 = 4.0 + 2.0
    {FP4_E2M1_P6_0, FP4_E2M1_P0_5},  // 6.5 = 6.0 + 0.5
    {FP4_E2M1_P6_0, FP4_E2M1_P1_0},  // 7.0 = 6.0 + 1.0
    {FP4_E2M1_P6_0, FP4_E2M1_P1_5},  // 7.5 = 6.0 + 1.5
};

// E4M3 helper: unbiased exponent of a (positive, normal) E4M3 scale byte.
static inline uint8_t E4M3ExpField(uint8_t scaleByte)
{
    return static_cast<uint8_t>((scaleByte >> 3) & 0xF);
}

// delta_e_eff = E - 2 = (exp_field - 7) - 2, folded as int16 fp16 exponent add.
static inline int16_t DeltaEEffAsFp16ExpAdd(uint8_t scaleByte)
{
    int32_t delta = static_cast<int32_t>(E4M3ExpField(scaleByte)) - 9;
    return static_cast<int16_t>(delta << 10);
}

// --------------------------------------------------------------- tiling data
struct Nvfp4TilingData {
    uint32_t numTiles;  // number of 16-group (16x16 weight) tiles to dequantize
    uint32_t reserved0;
};

// Cross-core event ids for the AIC->AIV producer/consumer handshake
// (ping-pong, 2 producers + 2 acks; ids chosen inside the 8-bit id space).
enum Nvfp4EventId : uint16_t {
    EVT_C2V_PROD_EVEN = 0,
    EVT_C2V_PROD_ODD  = 1,
    EVT_V2C_ACK_EVEN  = 2,
    EVT_V2C_ACK_ODD   = 3,
};

#endif  // NVFP4_DEQUANT_COMMON_H
