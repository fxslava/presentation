/**
 * nvfp4_pack.h  --  host-side NVFP4 packing + golden reference dequantizer.
 *
 * Mirrors exactly the layouts consumed by the kernel (nvfp4_dequant_common.h):
 *
 *  PackAFractal : 16x32 fp4 nibbles, row g = scale group; A[g][g] = A0_g
 *                 (K-half 0), A[g][16+g] = A1_g (K-half 1). All other
 *                 nibbles are zero. 256 B per tile.
 *  PackBFractal : 32x16 fp4 nibbles stored n-outer (row n = 32 K nibbles):
 *                 B[k][n] = w[k mod 16][n]  (weight rows replicated [W;W]).
 *                 256 B per tile.
 *  PackScales   : 16 E4M3 scale bytes + 16 B zero pad (32 B MTE block).
 *
 * Nibble order inside a byte: low nibble = even k, high nibble = odd k.
 *
 * Weight format is E1M2 by default to match the target's mad_mx SupportType
 * pairing (fp4x2_e2m1_t x fp4x2_e1m2_t -> float); set NVFP4_WEIGHTS_E2M1 to
 * pack standard-NVFP4 (OCP MX E2M1) weight grids instead -- the decomposition
 * and the kernel are bit-identical for both.
 */
#ifndef NVFP4_PACK_H
#define NVFP4_PACK_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "nvfp4_dequant_common.h"

// ---------------------------------------------------------------- FP4 decode
static inline float Fp4E1M2Value(uint8_t code)
{
    const uint8_t sign = (code >> 3) & 0x1;
    const uint8_t e = (code >> 2) & 0x1;
    const uint8_t m = code & 0x3;
    float v = (e == 0) ? 0.25f * static_cast<float>(m)          // denorm: 0..0.75
                       : (1.0f + 0.25f * static_cast<float>(m)); // normal: 1..1.75
    return sign ? -v : v;
}

static inline float Fp4E2M1Value(uint8_t code)
{
    const uint8_t sign = (code >> 3) & 0x1;
    const uint8_t e = (code >> 1) & 0x3;
    const uint8_t m = code & 0x1;
    float v = (e == 0) ? 0.5f * static_cast<float>(m)
                       : std::ldexp(1.0f + 0.5f * static_cast<float>(m), static_cast<int>(e) - 1);
    return sign ? -v : v;
}

static inline float Fp4WeightValue(uint8_t code)
{
#if defined(NVFP4_WEIGHTS_E2M1)
    return Fp4E2M1Value(code);
#else
    return Fp4E1M2Value(code);
#endif
}

// --------------------------------------------------------------- E4M3 decode
// Positive, normal scales only (denorm/NaN/0 excluded by generation).
static inline float E4M3Value(uint8_t scaleByte)
{
    const uint8_t e = E4M3ExpField(scaleByte);
    const uint8_t m = scaleByte & 0x7;
    return std::ldexp(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
}

// ------------------------------------------------------------ fp16 bit packer
static inline uint16_t Fp16BitsFromFloat(float v)
{
    if (v == 0.0f) {
        return 0x0000;  // +0 / -0 preserved by sign check below
    }
    const uint32_t sign = (std::signbit(v) ? 1U : 0U) << 15;
    const float a = std::fabs(v);
    int e = 0;
    float frac = std::frexp(a, &e);  // frac in [0.5, 1), a = frac * 2^e
    // fp16 normal: a = 1.f * 2^(E - 15); all values here are normal.
    const int E = e + 14;
    const uint32_t mantissa = static_cast<uint32_t>((frac * 2.0f - 1.0f) * 1024.0f + 0.5f);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(E) << 10) |
                                 (mantissa & 0x3FF));
}

// fp16 bits -> float (for -0 == +0 tolerant comparison only).
static inline float Fp16BitsToFloatForCmp(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>((bits >> 15) & 0x1);
    const uint32_t e = static_cast<uint32_t>((bits >> 10) & 0x1F);
    const uint32_t m = static_cast<uint32_t>(bits & 0x3FF);
    float v;
    if (e == 0) {
        v = std::ldexp(static_cast<float>(m), -24);  // denormal
    } else {
        v = std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, static_cast<int>(e) - 15);
    }
    return sign ? -v : v;
}

// ------------------------------------------------------------------- packers
static inline void PutNibble(uint8_t *bytes, uint32_t nibbleIdx, uint8_t code)
{
    uint8_t &b = bytes[nibbleIdx >> 1];
    if ((nibbleIdx & 1U) == 0U) {
        b = static_cast<uint8_t>((b & 0xF0) | (code & 0x0F));
    } else {
        b = static_cast<uint8_t>((b & 0x0F) | ((code & 0x0F) << 4));
    }
}

struct TileInputs {
    std::vector<uint8_t> aCodes[TILE_M];  // per group: {A0, A1} E2M1 codes
    std::vector<std::vector<uint8_t>> wCodes;  // [group][weight] fp4 codes
    std::vector<uint8_t> scaleBytes;           // 16 E4M3 bytes
};

// Decompose the tile's E4M3 scales into (A0, A1) E2M1 codes.
static inline void DecomposeScales(const std::vector<uint8_t> &scaleBytes,
                                   uint8_t (&a0Codes)[TILE_M], uint8_t (&a1Codes)[TILE_M])
{
    for (uint32_t g = 0; g < TILE_M; ++g) {
        const uint8_t sb = scaleBytes[g];
        const uint8_t m = sb & 0x7;
        const Fp4ScalePair pair = K4M_DECOMP_LUT[m];
        a0Codes[g] = pair.a0;
        a1Codes[g] = pair.a1;
    }
}

static inline void PackAFractal(const uint8_t (&a0Codes)[TILE_M], const uint8_t (&a1Codes)[TILE_M],
                                uint8_t (&out)[A_FRACTAL_BYTES])
{
    std::memset(out, 0, A_FRACTAL_BYTES);
    for (uint32_t g = 0; g < TILE_M; ++g) {
        // row g spans bytes [16g, 16g+16): nibble k = g (K-half 0) and 16+g (K-half 1)
        PutNibble(out + static_cast<uint32_t>(TILE_K) / 2 * g, g, a0Codes[g]);
        PutNibble(out + static_cast<uint32_t>(TILE_K) / 2 * g, TILE_K / 2 + g, a1Codes[g]);
    }
}

static inline void PackBFractal(const std::vector<std::vector<uint8_t>> &wCodes,
                                uint8_t (&out)[B_FRACTAL_BYTES])
{
    std::memset(out, 0, B_FRACTAL_BYTES);
    for (uint32_t n = 0; n < TILE_N; ++n) {
        uint8_t *row = out + static_cast<uint32_t>(TILE_K) / 2 * n;  // 16 B per n row
        for (uint32_t k = 0; k < TILE_K; ++k) {
            PutNibble(row, k, wCodes[k % TILE_M][n]);  // [W;W] K replication
        }
    }
}

// Golden reference: out16[g][n] = fp16(w_gn * s_g), bit-exact.
static inline void ReferenceDequant(const TileInputs &in, uint16_t (&out16)[TILE_M][TILE_N])
{
    for (uint32_t g = 0; g < TILE_M; ++g) {
        const float s = E4M3Value(in.scaleBytes[g]);
        for (uint32_t n = 0; n < TILE_N; ++n) {
            out16[g][n] = Fp16BitsFromFloat(Fp4WeightValue(in.wCodes[g][n]) * s);
        }
    }
}

#endif  // NVFP4_PACK_H
