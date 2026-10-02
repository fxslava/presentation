/**
 * host/main.cpp  --  CAModel (npusim / dav_3510 / Ascend950PR) driver harness.
 *
 * Flow:
 *   1. Generate random NVFP4 tiles (FP4 weights + E4M3 block scales), pack the
 *      L0A/L0B fractals and scale blocks, and compute the golden FP16 output.
 *   2. Load the ccec-built device object (nvfp4_dequant_kernel.o) via
 *      aclrtCreateBinary / aclrtBinaryLoad / aclrtBinaryGetFunction.
 *   3. Launch with aclrtLaunchKernel (1 block = 1 Cube + 1 Vector core,
 *      KERNEL_TYPE_MIX_AIC_1_1), synchronize, copy back, verify bit-exactly.
 *
 * Usage: ./nvfp4_host [kernel.o] [numTiles] [iters] [seed]
 */
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "nvfp4_pack.h"

#define ACL_CALL(expr)                                                              \
    do {                                                                            \
        const aclError __err = (expr);                                              \
        if (__err != ACL_SUCCESS) {                                                 \
            fprintf(stderr, "ACL error %d at %s:%d (%s)\n", static_cast<int>(__err),\
                    __FILE__, __LINE__, #expr);                                     \
            return 1;                                                               \
        }                                                                           \
    } while (0)

namespace {

std::vector<uint8_t> ReadFile(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        fprintf(stderr, "cannot open %s\n", path);
        return {};
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (sz > 0 && fread(buf.data(), 1, static_cast<size_t>(sz), f) != static_cast<size_t>(sz)) {
        fclose(f);
        return {};
    }
    fclose(f);
    return buf;
}

struct HostBuffers {
    std::vector<uint8_t> gmA;
    std::vector<uint8_t> gmB;
    std::vector<uint8_t> gmS;
    std::vector<uint16_t> refOut;  // numTiles * 256
    std::vector<TileInputs> tiles;
};

// Random E4M3 scale byte: positive, normal, non-NaN (skip 0x7F).
uint8_t RandomScaleByte(std::mt19937 &rng)
{
    static std::uniform_int_distribution<int> expDist(1, 15);
    static std::uniform_int_distribution<int> manDist(0, 7);
    for (;;) {
        const uint8_t sb = static_cast<uint8_t>((expDist(rng) << 3) | manDist(rng));
        if (sb != 0x7F) {  // E4M3 NaN encoding
            return sb;
        }
    }
}

uint8_t RandomWeightCode(std::mt19937 &rng)
{
    static std::uniform_int_distribution<int> dist(0, 15);
    return static_cast<uint8_t>(dist(rng));
}

void GenerateInputs(uint32_t numTiles, uint32_t seed, HostBuffers &hb)
{
    std::mt19937 rng(seed);
    const bool probeIdentity = getenv("NVFP4_PROBE_IDENTITY") != nullptr;
    hb.gmA.resize(static_cast<size_t>(numTiles) * A_FRACTAL_BYTES);
    hb.gmB.resize(static_cast<size_t>(numTiles) * B_FRACTAL_BYTES);
    hb.gmS.resize(static_cast<size_t>(numTiles) * SCALE_BYTES);
    hb.refOut.resize(static_cast<size_t>(numTiles) * TILE_M * TILE_N);
    hb.tiles.resize(numTiles);

    for (uint32_t t = 0; t < numTiles; ++t) {
        TileInputs &in = hb.tiles[t];
        in.wCodes.assign(TILE_M, std::vector<uint8_t>(TILE_N));
        in.scaleBytes.resize(TILE_M);
        if (probeIdentity) {
            // B = identity => C reveals the hardware's A-fractal interpretation.
            for (uint32_t g = 0; g < TILE_M; ++g) {
                for (uint32_t n = 0; n < TILE_N; ++n) {
                    in.wCodes[g][n] = (g == n) ? 0x2 : 0x0;  // E1M2 code 1.0 / 0
                }
                in.scaleBytes[g] = 0x40;  // exp field 8 -> E=1, 4M=4 => A0=2,A1=2
            }
        } else {
            for (uint32_t g = 0; g < TILE_M; ++g) {
                in.scaleBytes[g] = RandomScaleByte(rng);
                for (uint32_t n = 0; n < TILE_N; ++n) {
                    in.wCodes[g][n] = RandomWeightCode(rng);
                }
            }
        }

        uint8_t a0[TILE_M];
        uint8_t a1[TILE_M];
        DecomposeScales(in.scaleBytes, a0, a1);
        if (probeIdentity) {
            // Overwrite A codes with a positional probe pattern: nibble (m,k)
            // carries code (m*16+k) & 0xF, packed in our assumed layout.
            for (uint32_t m = 0; m < TILE_M; ++m) {
                a0[m] = static_cast<uint8_t>((m * 16 + m) & 0xF);
                a1[m] = static_cast<uint8_t>((m * 16 + 16 + m) & 0xF);
            }
        }

        auto *aFr = reinterpret_cast<uint8_t (*)[A_FRACTAL_BYTES]>(
            hb.gmA.data() + static_cast<size_t>(t) * A_FRACTAL_BYTES);
        PackAFractal(a0, a1, *aFr);

        auto *bFr = reinterpret_cast<uint8_t (*)[B_FRACTAL_BYTES]>(
            hb.gmB.data() + static_cast<size_t>(t) * B_FRACTAL_BYTES);
        PackBFractal(in.wCodes, *bFr);

        uint8_t *s = hb.gmS.data() + static_cast<size_t>(t) * SCALE_BYTES;
        std::memset(s, 0, SCALE_BYTES);
        std::memcpy(s, in.scaleBytes.data(), TILE_M);

        uint16_t (*ref)[TILE_M][TILE_N] = reinterpret_cast<uint16_t (*)[TILE_M][TILE_N]>(
            hb.refOut.data() + static_cast<size_t>(t) * TILE_M * TILE_N);
        ReferenceDequant(in, *ref);
    }
}

}  // namespace

int main(int argc, char **argv)
{
    const char *kernelPath = (argc > 1) ? argv[1] : "nvfp4_dequant_kernel.o";
    const uint32_t numTiles = (argc > 2) ? static_cast<uint32_t>(atoi(argv[2])) : 64;
    const uint32_t iters = (argc > 3) ? static_cast<uint32_t>(atoi(argv[3])) : 3;
    const uint32_t seed = (argc > 4) ? static_cast<uint32_t>(atoi(argv[4])) : 2026u;

    printf("[host] kernel=%s tiles=%u iters=%u seed=%u\n", kernelPath, numTiles, iters, seed);

    HostBuffers hb;
    GenerateInputs(numTiles, seed, hb);

    std::vector<uint8_t> kernelBin = ReadFile(kernelPath);
    if (kernelBin.empty()) {
        fprintf(stderr, "failed to read kernel object %s\n", kernelPath);
        return 1;
    }
    printf("[host] kernel object: %zu bytes\n", kernelBin.size());

    ACL_CALL(aclInit(nullptr));
    ACL_CALL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CALL(aclrtCreateStream(&stream));

    // ---- device allocations -------------------------------------------------
    const size_t bytesA = static_cast<size_t>(numTiles) * A_FRACTAL_BYTES;
    const size_t bytesB = static_cast<size_t>(numTiles) * B_FRACTAL_BYTES;
    const size_t bytesS = static_cast<size_t>(numTiles) * SCALE_BYTES;
    const size_t bytesC = static_cast<size_t>(numTiles) * C_F32_BYTES;
    const size_t bytesOut = static_cast<size_t>(numTiles) * OUT_F16_BYTES;

    void *devA = nullptr;
    void *devB = nullptr;
    void *devS = nullptr;
    void *devC = nullptr;
    void *devOut = nullptr;
    void *devTiling = nullptr;
    ACL_CALL(aclrtMalloc(&devA, bytesA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMalloc(&devB, bytesB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMalloc(&devS, bytesS, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMalloc(&devC, bytesC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMalloc(&devOut, bytesOut, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMalloc(&devTiling, 64, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CALL(aclrtMemcpy(devA, bytesA, hb.gmA.data(), bytesA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CALL(aclrtMemcpy(devB, bytesB, hb.gmB.data(), bytesB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CALL(aclrtMemcpy(devS, bytesS, hb.gmS.data(), bytesS, ACL_MEMCPY_HOST_TO_DEVICE));

    Nvfp4TilingData tiling;
    tiling.numTiles = numTiles;
    tiling.reserved0 = 0;
    ACL_CALL(aclrtMemcpy(devTiling, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- kernel binary load -------------------------------------------------
    aclrtBinary binary = aclrtCreateBinary(kernelBin.data(), kernelBin.size());
    if (binary == nullptr) {
        fprintf(stderr, "aclrtCreateBinary failed\n");
        return 1;
    }
    aclrtBinHandle binHandle = nullptr;
    ACL_CALL(aclrtBinaryLoad(binary, &binHandle));
    aclrtFuncHandle func = nullptr;
    ACL_CALL(aclrtBinaryGetFunction(binHandle, "nvfp4_dequant", &func));
    printf("[host] kernel loaded: nvfp4_dequant\n");

    // ---- launch: args = (a, b, s, c, out, workspace, tiling) -----------------
    // NOTE: the CAModel runtime forwards argsData to the device address window
    // verbatim, so the args block must live in device memory.
    struct KernelArgs {
        void *gmA;
        void *gmB;
        void *gmS;
        void *gmC;
        void *gmOut;
        void *workspace;
        void *tiling;
    } args;
    args.gmA = devA;
    args.gmB = devB;
    args.gmS = devS;
    args.gmC = devC;
    args.gmOut = devOut;
    args.workspace = nullptr;
    args.tiling = devTiling;

    void *devArgs = nullptr;
    ACL_CALL(aclrtMalloc(&devArgs, sizeof(args), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CALL(aclrtMemcpy(devArgs, sizeof(args), &args, sizeof(args), ACL_MEMCPY_HOST_TO_DEVICE));

    for (uint32_t it = 0; it < iters; ++it) {
        ACL_CALL(aclrtMemset(devOut, bytesOut, 0, bytesOut));
        const auto t0 = std::chrono::steady_clock::now();
        ACL_CALL(aclrtLaunchKernel(func, 1, devArgs, sizeof(args), stream));
        ACL_CALL(aclrtSynchronizeStream(stream));
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[host] iter %u: launch+sync wall %.3f ms (simulator wall time, NOT cycles)\n", it, ms);
    }

    // ---- verify --------------------------------------------------------------
    std::vector<uint16_t> got(static_cast<size_t>(numTiles) * TILE_M * TILE_N);
    ACL_CALL(aclrtMemcpy(got.data(), bytesOut, devOut, bytesOut, ACL_MEMCPY_DEVICE_TO_HOST));

    // Layout probe: dump the fp32 Cube staging tile 0 (see docs/cycle_model.md).
    if (getenv("NVFP4_DUMP_C") != nullptr) {
        std::vector<float> c32(static_cast<size_t>(numTiles) * TILE_M * TILE_N);
        ACL_CALL(aclrtMemcpy(c32.data(), bytesC, devC, bytesC, ACL_MEMCPY_DEVICE_TO_HOST));
        printf("[probe] gmC tile 0 (fp32, hardware mad_mx interpretation of the fractals):\n");
        for (uint32_t g = 0; g < TILE_M; ++g) {
            printf("  m%02u:", g);
            for (uint32_t n = 0; n < TILE_N; ++n) {
                printf(" %10.4f", c32[g * TILE_N + n]);
            }
            printf("\n");
        }
        printf("[probe] gmA tile 0 bytes:");
        for (uint32_t i = 0; i < 64; ++i) {
            printf(" %02x", hb.gmA[i]);
        }
        printf(" ...\n[probe] gmB tile 0 bytes:");
        for (uint32_t i = 0; i < 64; ++i) {
            printf(" %02x", hb.gmB[i]);
        }
        printf(" ...\n");
    }

    size_t mismatches = 0;
    size_t zeros = 0;
    double maxRelDiff = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const uint16_t r = hb.refOut[i];
        const uint16_t g = got[i];
        if (r == g) {
            if ((r & 0x7FFF) == 0) {
                ++zeros;
            }
            continue;
        }
        // -0 == +0 compare (kernel zero-guard canonicalizes -0 to +0)
        const float rf = Fp16BitsToFloatForCmp(r);
        const float gf = Fp16BitsToFloatForCmp(g);
        if (rf == gf && rf == 0.0f) {
            continue;
        }
        if (mismatches < 8) {
            fprintf(stderr, "  mismatch idx=%zu tile=%zu elem=%zu ref=0x%04x got=0x%04x (ref=%g got=%g)\n",
                    i, i / (TILE_M * TILE_N), i % (TILE_M * TILE_N), r, g, rf, gf);
        }
        ++mismatches;
        const float denom = (rf != 0.0f) ? fabsf(rf) : 1.0f;
        const double rel = fabsf(gf - rf) / denom;
        if (rel > maxRelDiff) {
            maxRelDiff = rel;
        }
    }

    const size_t total = got.size();
    printf("[host] elements=%zu exact=%zu zero=%zu MISMATCH=%zu (%.4f%%)\n", total,
           total - mismatches, zeros, mismatches, 100.0 * static_cast<double>(mismatches) /
                                                  static_cast<double>(total));
    printf("[host] RESULT: %s\n", mismatches == 0 ? "PASS (bit-exact)" : "FAIL");

    (void)aclrtBinaryUnLoad(binHandle);
    (void)aclrtDestroyBinary(binary);  // best-effort cleanup
    ACL_CALL(aclrtFree(devArgs));
    ACL_CALL(aclrtFree(devA));
    ACL_CALL(aclrtFree(devB));
    ACL_CALL(aclrtFree(devS));
    ACL_CALL(aclrtFree(devC));
    ACL_CALL(aclrtFree(devOut));
    ACL_CALL(aclrtFree(devTiling));
    ACL_CALL(aclrtDestroyStream(stream));
    ACL_CALL(aclrtResetDevice(0));
    ACL_CALL(aclFinalize());
    return mismatches == 0 ? 0 : 2;
}
