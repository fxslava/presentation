#!/usr/bin/env python3
"""
emulate_pipeline.py -- bit-exact CPU emulation of the NVFP4->FP16 Cube pipeline.

Validates, independently of any Ascend toolchain:
  1. The 4M = A0 + A1 E2M1 decomposition LUT covers all 16 E4M3 mantissas.
  2. The L0A/L0B fractal byte layouts round-trip (packing <-> decode).
  3. The K=32 replicated Mmad computes C[g][n] = 4M_g * w_gn exactly.
  4. The AIV integer exponent add on the FP16 bit pattern (delta << 10) with
     zero guard reproduces the golden w * s FP16 for every element.
  5. Range safety: no overflow to inf, no denormal C, no fp16 exponent-field
     overflow for the full E4M3/E1M2 (and E2M1) input domains.

Run:  python sim/emulate_pipeline.py [--tiles 64] [--seed 1]
Exit code 0 = PASS.
"""
import argparse
import struct
import sys

NVFP4_BLK = 16
TILE_M, TILE_N, TILE_K = 16, 16, 32

# ---------------------------------------------------------------- fp4 decode
def fp4_e2m1(code):
    sign = (code >> 3) & 1
    e = (code >> 1) & 3
    m = code & 1
    v = (m * 0.5) if e == 0 else (2.0 ** (e - 1)) * (1.0 + 0.5 * m)
    return -v if sign else v

def fp4_e1m2(code):
    sign = (code >> 3) & 1
    e = (code >> 2) & 1
    m = code & 3
    v = (m * 0.25) if e == 0 else 1.0 + 0.25 * m
    return -v if sign else v

FP4_E2M1_GRID = [fp4_e2m1(c) for c in range(16)]
FP4_E1M2_GRID = [fp4_e1m2(c) for c in range(16)]

# E2M1 encode (positive values)
E2M1_CODE = {v: c for c, v in enumerate(FP4_E2M1_GRID[:8])}

# ------------------------------------------------- decomposition LUT (mirror)
LUT = [
    (E2M1_CODE[2.0], E2M1_CODE[2.0]),  # 4.0
    (E2M1_CODE[4.0], E2M1_CODE[0.5]),  # 4.5
    (E2M1_CODE[4.0], E2M1_CODE[1.0]),  # 5.0
    (E2M1_CODE[4.0], E2M1_CODE[1.5]),  # 5.5
    (E2M1_CODE[4.0], E2M1_CODE[2.0]),  # 6.0
    (E2M1_CODE[6.0], E2M1_CODE[0.5]),  # 6.5
    (E2M1_CODE[6.0], E2M1_CODE[1.0]),  # 7.0
    (E2M1_CODE[6.0], E2M1_CODE[1.5]),  # 7.5
]

def e4m3_value(sb):
    e = (sb >> 3) & 0xF
    m = sb & 7
    if e == 0:
        return (m / 8.0) * 2.0 ** -6  # denormal (excluded in generation, but decode anyway)
    return (2.0 ** (e - 7)) * (1.0 + m / 8.0)

def fp16_bits(v):
    return struct.unpack("<H", struct.pack("<e", v))[0]

def fp16_from_bits(bits):
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]

def i16(x):
    return ((x + 0x8000) & 0xFFFF) - 0x8000

# ------------------------------------------------------------------- packing
def put_nibble(buf, idx, code):
    if idx % 2 == 0:
        buf[idx // 2] = (buf[idx // 2] & 0xF0) | (code & 0xF)
    else:
        buf[idx // 2] = (buf[idx // 2] & 0x0F) | ((code & 0xF) << 4)

def get_nibble(buf, idx):
    return (buf[idx // 2] >> (4 * (idx % 2))) & 0xF

def pack_a_fractal(a0, a1):
    """L0A: row g (16B = 32 k-nibbles): A0_g at k=g, A1_g at k=16+g."""
    out = bytearray(TILE_M * TILE_K // 2)
    for g in range(TILE_M):
        row_nib = g * TILE_K  # each row spans TILE_K nibbles (TILE_K/2 bytes)
        put_nibble(out, row_nib + g, a0[g])
        put_nibble(out, row_nib + TILE_K // 2 + g, a1[g])
    return out

def pack_b_fractal(w):
    """L0B: n-outer rows; row n (16B): k=0..31 nibbles of B[k][n] = w[k%16][n]."""
    out = bytearray(TILE_K * TILE_N // 2)
    for n in range(TILE_N):
        row_nib = n * TILE_K  # each row spans TILE_K nibbles (TILE_K/2 bytes)
        for k in range(TILE_K):
            put_nibble(out, row_nib + k, w[k % TILE_M][n])
    return out

# ---------------------------------------------------------------- emulation
def emulate_tile(w_codes, scale_bytes, weight_fmt="e1m2"):
    decode = fp4_e1m2 if weight_fmt == "e1m2" else fp4_e2m1
    # decomposition
    a0 = [0] * TILE_M
    a1 = [0] * TILE_M
    for g in range(TILE_M):
        a0[g], a1[g] = LUT[scale_bytes[g] & 7]
    a_fr = pack_a_fractal(a0, a1)
    b_fr = pack_b_fractal(w_codes)

    # Mmad emulation: C[g][n] = sum_k A[g][k] * B[k][n], decoded from the
    # packed fractal bytes (validates the layouts, not just the math).
    C = [[0.0] * TILE_N for _ in range(TILE_M)]
    for g in range(TILE_M):
        arow = a_fr[g * (TILE_K // 2):(g + 1) * (TILE_K // 2)]
        for n in range(TILE_N):
            brow = b_fr[n * (TILE_K // 2):(n + 1) * (TILE_K // 2)]
            acc = 0.0
            for k in range(TILE_K):
                av = fp4_e2m1(get_nibble(arow, k))
                bv = decode(get_nibble(brow, k))
                acc += av * bv
            C[g][n] = acc  # fp32 accumulation: all partials are exact in fp32
    return C

def aiv_stage(C, scale_bytes):
    """Cast fp32->fp16, integer exponent add, zero guard. Returns bit matrix."""
    out = [[0] * TILE_N for _ in range(TILE_M)]
    for g in range(TILE_M):
        delta = (((scale_bytes[g] >> 3) & 0xF) - 9) << 10
        for n in range(TILE_N):
            cval = C[g][n]
            if abs(cval) > 65504.0:
                raise ValueError(f"C overflow to fp16 inf: {cval}")
            bits = fp16_bits(cval)
            if bits & 0x7FFF == 0:
                out[g][n] = 0
                continue
            s = i16(bits + delta)
            # post-add sanity: must remain normal fp16 (exp field 1..30, not inf)
            if not (1 <= ((s >> 10) & 0x1F) <= 30):
                raise ValueError(
                    f"exponent add produced non-normal fp16: bits={bits:#06x} "
                    f"delta={delta:#06x} -> {s:#06x} (C={cval})")
            out[g][n] = s & 0xFFFF
    return out

def golden(w_codes, scale_bytes, weight_fmt="e1m2"):
    decode = fp4_e1m2 if weight_fmt == "e1m2" else fp4_e2m1
    ref = [[0] * TILE_N for _ in range(TILE_M)]
    for g in range(TILE_M):
        s = e4m3_value(scale_bytes[g])
        for n in range(TILE_N):
            v = decode(w_codes[g][n]) * s
            ref[g][n] = 0 if v == 0 else fp16_bits(v)
    return ref

# ------------------------------------------------------------------- checks
def check_lut():
    for m in range(8):
        a0v = FP4_E2M1_GRID[LUT[m][0]]
        a1v = FP4_E2M1_GRID[LUT[m][1]]
        want = 4.0 + 0.5 * m
        assert a0v + a1v == want, f"LUT[{m}]: {a0v}+{a1v} != {want}"
        assert LUT[m][0] in (E2M1_CODE[2.0], E2M1_CODE[3.0], E2M1_CODE[4.0], E2M1_CODE[6.0]), "A0 outside {2,3,4,6}"
        assert LUT[m][1] in (E2M1_CODE[0.5], E2M1_CODE[1.0], E2M1_CODE[1.5], E2M1_CODE[2.0], E2M1_CODE[3.0]), "A1 outside {0.5,1,1.5,2,3}"
    print("[check] LUT: 8/8 mantissas decompose exactly; A0 in {2,3,4,6}, A1 in {0.5,1,1.5,2,3}")

def exhaustive_sweep(weight_fmt):
    """All 16 E4M3 mantissas x all valid scale exponents x all nonzero weights."""
    n_cases = 0
    for exp_field in range(1, 15):  # skip denorm exp; 15 reserved (NaN at man=7)
        for m in range(8):
            sb = (exp_field << 3) | m
            if sb == 0x7F:
                continue
            for wc in range(16):
                w_codes = [[wc] * TILE_N for _ in range(TILE_M)]
                scale_bytes = [sb] * TILE_M
                C = emulate_tile(w_codes, scale_bytes, weight_fmt)
                got = aiv_stage(C, scale_bytes)
                ref = golden(w_codes, scale_bytes, weight_fmt)
                for g in range(TILE_M):
                    for n in range(TILE_N):
                        gv, rv = got[g][n], ref[g][n]
                        if gv != rv and not (fp16_from_bits(gv) == fp16_from_bits(rv) == 0.0):
                            raise AssertionError(
                                f"sweep fail fmt={weight_fmt} sb={sb:#04x} wc={wc}: "
                                f"got {gv:#06x} ({fp16_from_bits(gv)}) ref {rv:#06x} ({fp16_from_bits(rv)})")
                        n_cases += 1
    print(f"[check] exhaustive sweep ({weight_fmt}): {n_cases} element cases bit-exact")

def random_tiles(n_tiles, seed, weight_fmt):
    import random
    rng = random.Random(seed)
    total = 0
    for t in range(n_tiles):
        w_codes = [[rng.randrange(16) for _ in range(TILE_N)] for _ in range(TILE_M)]
        scale_bytes = []
        while len(scale_bytes) < TILE_M:
            sb = (rng.randrange(1, 15) << 3) | rng.randrange(8)
            if sb != 0x7F:
                scale_bytes.append(sb)
        C = emulate_tile(w_codes, scale_bytes, weight_fmt)
        got = aiv_stage(C, scale_bytes)
        ref = golden(w_codes, scale_bytes, weight_fmt)
        for g in range(TILE_M):
            for n in range(TILE_N):
                gv, rv = got[g][n], ref[g][n]
                if gv != rv and not (fp16_from_bits(gv) == fp16_from_bits(rv) == 0.0):
                    raise AssertionError(
                        f"tile {t} [{g}][{n}]: got {gv:#06x} ref {rv:#06x} "
                        f"(w={w_codes[g][n]:#x} s={scale_bytes[g]:#x})")
                total += 1
    print(f"[check] random tiles ({weight_fmt}): {n_tiles} tiles x 256 = {total} elements bit-exact")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print(f"NVFP4 Cube-dequant pipeline emulation (tiles={args.tiles}, seed={args.seed})")
    check_lut()
    for fmt in ("e1m2", "e2m1"):  # e1m2: target SupportType; e2m1: standard NVFP4
        exhaustive_sweep(fmt)
        random_tiles(args.tiles, args.seed + (0 if fmt == "e1m2" else 1000), fmt)
    print("PASS: decomposition, fractal layouts, K=32 replicated Mmad, and the")
    print("integer exponent-add AIV stage are all bit-exact vs the golden model.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
