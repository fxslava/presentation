# Dual Kernel Design & Audit — FP4→FP16 Dequantization on Ascend (Naive AIV vs. Cube Trick)

**Scope:** design, model, implement and statically verify two kernels for the same
NVFP4→FP16 dequantization problem on Ascend 910B-class DaVinci silicon
("Ascend 950PR" commercial label → `ascend910b` in both PipeSim and
`ascend_analyzer`; the MX-native `mad_mx` instruction set maps to
`ascend351x` only).

**Benchmark problem (identical for both kernels):** dequantize
`W16 ∈ R^{2048×4096}` fp16 (16 MiB) from NVFP4 storage — 4 MiB packed FP4
weights + 512 KiB E4M3 block scales. That is 8,388,608 outputs, 524,288
scale blocks, 32,768 tiles of `[16 groups × 16 weights]`.

**Deliverables in this directory**

| Artifact | Description |
|---|---|
| `naive_aiv_fp4_to_fp16.cpp` | Kernel A — pure-Vector (AIV) baseline, audits 0/0/0 |
| `cube_trick_fp4_to_fp16.cpp` | Kernel B — Cube+Fixpipe trick (MIX_AIC_1_1), audits 0 fatal / 0 warning / 4 info |
| `model_pipesim.py`, `pipesim_results.json` | PipeSim spatial-temporal modeling + autotiling sweeps |
| `autotile_consumer_gemm.json` | Consumer-GEMM fusion context (2048×4096×4096) |
| `kernel_a_audit.txt`, `kernel_b_audit.txt` | Full `ascend_analyzer` CI-gate outputs |
| `tiling.json` | Tiling manifest (optional `--tiling-data` binding for Kernel B) |
| `REPORT.md` | This report |

Toolchain fixes made along the way (required to audit a real MIX kernel at
all): four parser defects fixed in `D:\Projects\ascend-kernel-analyzer`
(+13 new tests, full suite **742 passed**); see §5.

---

## 1. Research extraction: the Cube dequantization trick

Source: `Research_fp4Tofp16` repo (`nvfp4_dequant/`, `host/nvfp4_pack.h`,
`sim/emulate_pipeline.py`).

### 1.1 FP4 variant and bit layout

* **Variant:** NVFP4-style microscaling blocks — **16 weights + 1 FP8-E4M3
  block scale** (bias 7, positive normals only; `0x7F` NaN excluded).
* **Weights:** FP4 **E1M2 grid** by default (`value = magnitude_code × 0.25`
  exactly — the identity that makes both kernels' vector decodes exact),
  matching the target's `mad_mx` SupportType pairing
  `Tuple<float, fp4x2_e2m1_t, fp4x2_e1m2_t>`. Standard OCP E2M1 weight grids
  are supported as a host packing option; the kernel math is identical.
* **Packing:** 2 nibbles/byte, **low nibble = even element, high nibble = odd**
  (little-endian nibble order). Sign is the nibble's bit 3
  (signed-magnitude).
* **Layouts (per 16×16 tile, host-prepared fractals):**
  - `gmA`: L0A scale-term matrix `[16 groups][32 k]`, row *g* carries the
    two additive scale terms at k = *g* and k = 16+*g*.
  - `gmB`: L0B weight matrix, n-outer rows, `B[k][n] = w[k mod 16][n]`
    (weight rows replicated along K — the `[W;W]` tiling).
  - `gmS`: 16 E4M3 bytes + 16 B pad (one 32-B MTE block).

### 1.2 The mathematical core

With E4M3 bias 7, every block scale factors as

```
s = 2^E · (1 + m/8) = 2^(E−2) · 4M,   4M = 4 + m/2 ∈ {4.0 … 7.5}
```

`4M` decomposes **additively** into two native E2M1 grid points
(repo LUT): `4M = A0 + A1`, `A0 ∈ {2,4,6}`, `A1 ∈ {0.5,1,1.5,2}`. Hence

```
s·w = 2^(E−2) · (A0·w + A1·w)
```

and the parenthesized term is a **K=2 dot product** — executed as ONE
systolic `mad` per 16×16 tile with the K-replicated `[W;W]` B-operand. The
residual power-of-two `2^(E−2)` folds into the fp16 **exponent field with a
pure integer add** on the vector core:

```
bits16(out) = bits16(C) + (δ << 10),   δ = exp_field − 9      (fp4-native form)
```

**No vector FPU multiply exists anywhere downstream of the Cube.** Zero
inputs are preserved by a compare-free guard `out &= (0 − Min(|bits|, 1))`
(nonzero results have |bits| ≥ 0x3800, so `Min(|bits|,1)` is an exact 0/1
mask).

### 1.3 The portable INT8 contraction (Kernel B as shipped)

`mad_mx` is a DaVinci-v3 (`ascend351x`) instruction — `ascend_analyzer`
gates it behind the `mx_cube` feature and flags it **AKA1011 FATAL on
910B**. Scaling the decomposition by 4 moves the whole trick onto the
standard 910B INT8 Mmad with **bit-identical results**:

| Quantity | fp4-native (351x) | INT8 contraction (910B) |
|---|---|---|
| A-side | A0, A1 as E2M1 | `A′ = 4·A0 ∈ {8,16,24}`, `4·A1 ∈ {2,4,6,8}` |
| B-side | w as E1M2 | `B′ = 4·w ∈ [−7,7]` (E1M2 ×4 is integral) |
| Contraction | `mad_mx`, K=32 | `Mmad` int8, **k=32 = exactly one int8 k0 fractal** |
| Accumulator | C = 4M·w (fp32) | `C′ = 16·C` (int32, exact, \|C′\| ≤ 210) |
| Fixpipe | fp32→fp16 | **int32→fp16 inline** (exact, ≤ 8 significand bits) |
| Exponent fold | δ = exp_field − 9 | δ = exp_field − 13 (C′ carries 2^4) |
| Operand bytes/tile | 256 + 256 | 512 + 512 (both full 512-B GM bursts) |

Verified end-to-end by the repo's exhaustive CPU emulation (all 16 mantissas
× all exponents × all weight codes, both weight grids, bit-exact vs golden).

---

## 2. Spatial-temporal modeling (PipeSim, `ascend910b`)

Chip model: 1.8 GHz, MTE2/MTE3 64 B/cyc, Vector 256 B/cyc (4-cyc issue),
Cube 4096 fp16-MACs/cyc (8192 int8), UB 192 KiB, L1 512 KiB, L0A/L0B 64 KiB,
L0C 128 KiB. FP4 operands modeled as int8 (sub-byte proxy; the shipped
Kernel B *is* int8, so its MTE2 numbers are exact).

### 2.1 Kernel A — AIV tile sweep (ELEMWISE flow, 7.4 ops/element)

| Tile (elements) | MTE2 | VECTOR | MTE3 | Limiting stage | Total |
|---|---|---|---|---|---|
| 4096 | 64 | 355.2 | 128 | **VECTOR_BOUND** | 727,800 cyc |
| 8192 | 128 | 710.4 | 256 | **VECTOR_BOUND** | 727,800 cyc |
| 16384 | 256 | 1420.8 | 512 | **VECTOR_BOUND** | 727,800 cyc |
| 24576 | 383 | 2127.1 | 767 | **VECTOR_BOUND** | 727,800 cyc |

(cycles per step; steady state is tile-size invariant — the UB budget
decides, and 8192 elements is the shipped choice.)

### 2.2 Kernel B — Cube panel-batch sweep (pseudo-GEMM m=16P, n=16, k=32P)

| P (panels/mad) | MTE2 | MTE1 | CUBE | FIXPIPE | cyc / 256 outputs |
|---|---|---|---|---|---|
| **1** | 24 | 16 | 2 | 16 | **24.0** |
| 2 | 64 | 48 | 5 | 32 | 32.0 |
| 4 | 192 | 160 | 17 | 64 | 48.0 |
| 8 | 640 | 576 | 65 | 128 | 80.0 |

**P=1 is optimal** — the minimal 16×16×32 tile. Panel batching inflates the
sparse A-fractal traffic ∝P (2 nonzero nibbles per row regardless of K) and
loses to the fixed Fixpipe cost. The Cube itself runs at **~4% of peak
MACs** — the pseudo-GEMM is a staging-bound "free multiplier", not a
throughput GEMM. PipeSim flagged per-tile GM bursts < 512 B on the fp4
variant; the int8 form's 512-B operand tiles are exactly one burst each.

### 2.3 Issue-overhead adjustment and composed walls

PipeSim's elementwise model prices vector *work bytes* but not instruction
*issue*. Both designs must materialize a per-16-element scale broadcast on
the vector pipe (Kernel A: one shared 8-lane-run scale vector per 512-block
— the even/odd planes share values; Kernel B: one 16-lane exponent-offset
row per group) — 1 Duplicate issue per 16 outputs ≈ **0.25 cycles/element**
on both paths. This "broadcast tax" is structural on DaVinci v2/v3 vector
pipes (no lane-doubling unpack, no per-lane gather) and hits the naive AIV
and the Cube-trick epilogue equally.

| Design (single core-pair, whole problem) | Cycles | Time | Limiter |
|---|---|---|---|
| Kernel A (naive AIV, pair-planar out) | 2,824,604 | 1569.2 µs | Vector work 0.087 + tax 0.25 cyc/el |
| Kernel B, fp32 GM staging (repo-faithful) | 3,288,594 | 1827.0 µs | AIV epilogue 100.4 cyc/tile |
| **Kernel B, fp16 Fixpipe staging (shipped)** | **2,760,372** | **1533.5 µs** | AIV 84.2 cyc/tile; AIC 16 cyc/tile |

**Speedup of the Cube trick vs naive AIV: 1.02× per core-pair** (1.78× over
Kernel A per pair against the repo-faithful fp32 staging) — a photo finish,
because both are dominated by the same broadcast tax. Chip-level on 910B
(24 AIC + 48 AIV): Kernel A on 48 AIVs = 58,846 cyc/core vs Kernel B on 24
MIX pairs = 115,016 cyc/pair — standalone throughput favors the AIV fleet
1.96× (it has 2× the cores), but Kernel B leaves 24 AIVs and all of MTE3
completely free.

### 2.4 SRAM budgets (from the audited footprints)

* Kernel A: UB high-water **108,416 B (55.1%)** — 2×4 KiB packed in, 2×16 KiB
  fp16 out, 512-B scale tiles, 66.4-KiB bank-skewed scratch pool.
* Kernel B: AIC — L1 2 KiB (A1+B1 ping-pong), L0A/L0B 1 KiB each, L0C 2 KiB
  (all ≪ capacity); AIV — UB ≈ 6 KiB. gmC staging is a **2-slot ring
  (1 KiB)**, not a full-size buffer — saves 16 MiB of GM workspace vs the
  repo layout.
* Double buffering (depth 2) everywhere; consumer-GEMM autotile
  (2048×4096×4096, tile 128×256×256) peaks L1/L0A/L0B/L0C at 100% — the
  fusion target Kernel B's fp4 operands ultimately feed.

---

## 3. The two kernels

### Kernel A — `naive_aiv_fp4_to_fp16.cpp` (pure PIPE_MTE2/V/MTE3)

Per 8192-element tile: MTE2 loads 4096 B packed + 512 B scales → nibble
split (`And` 0x0F / `ShiftRight` 4) → per plane: `And` 0x07 magnitude,
`Compare>7` sign mask, `Cast` int8→half (0..7 exact), `Mul` by the
shift-add-decoded fp16 scale `bits(s·0.25) = ((sb&0x7F)<<7) + 0x1800`,
`Sub`-negate + `Select` sign fold → MTE3 16 KiB fp16. Output is
**pair-planar per tile** (evens then odds): the vector unit has no
lane-doubling unpack, so a strictly sequential layout would force 2-B-granular
MTE3 strided stores (modeled ≈16× MTE3 waste, ≈2.1 M cycles); the planar
layout is the honest baseline contract. TQue with `TPosition::VECIN/VECOUT`,
literal depth 2; all scratch is one explicitly **bank-skewed** pool —
every co-operand pair of every vector op sits at byte offsets differing
mod 256 B (AKA3006-orthogonal by construction).

### Kernel B — `cube_trick_fp4_to_fp16.cpp` (MIX_AIC_1_1)

AIC per tile: `S_MTE2` scalar barrier → MTE2 `Nd2Nz` GM→L1 (2×512 B) →
`MTE2_MTE1` → MTE1 `LoadData` L1→L0A/L0B → `MTE1_M` → one int8 `Mmad`
(16×16×32) → `M_FIX` → `Fixpipe` L0C→GM fp16 ring (int32→fp16 inline) →
`CrossCoreSetFlag(prod_p)`. AIV per tile: `CrossCoreWaitFlag(prod_p)` →
MTE2 (fp16 C′ + scales) → `CrossCoreSetFlag(ack_p)` → 16 exponent-offset
`Duplicate` rows → integer `Add` on fp16 bits → zero guard
(`And`/`Min`/`Sub`/`And`) → MTE3. Engineering deltas vs the research repo:

1. **Primed ping-pong**: slot-free tokens (`M_MTE1`, `FIX_M`, `MTE1_MTE2`,
   cross-core acks) are seeded before the loops so every in-loop `WaitFlag`
   is unconditional and the set/wait ledger closes exactly (1 prime + N
   sets = N waits + 1 drain per id). Also removes the repo's npusim
   `execute_set_flag already has same set_flag!` double-set exposure.
2. **MTE1→MTE2 L1-reuse backedge** (`MTE1_MTE2`), missing in the repo —
   without it MTE2 can overwrite an L1 slot MTE1 is still reading.
3. **2-slot gmC ring** (vs full-size staging) — makes the ack handshake
   load-bearing and cuts GM workspace 16 MiB → 1 KiB.
4. **fp16 Fixpipe staging** (vs fp32) — halves staging traffic and removes
   the AIV Cast; modeled 1.19× end-to-end win.
5. **Literal EVENT_ID0..3 everywhere** (branch-selected, never computed) —
   required for exact static pairing (AKA3003-class hazards).

---

## 4. Static verification (ascend_analyzer, `--chip ascend910b`)

Both kernels pass the CI gate:

```
python -m ascend_analyzer <kernel>.cpp --chip ascend910b --warnings-as-errors
```

| Gate | Kernel A | Kernel B |
|---|---|---|
| FATAL / WARNING / INFO | **0 / 0 / 0** — ACCEPTED, exit 0 | **0 / 0 / 4** — ACCEPTED, exit 0 |
| AKA1001 overflow / AKA1002 32-B alignment | clean (footprints fully resolved) | clean |
| AKA2010 happens-before | queue-managed; graph acyclic | **42 flag ops, 20 matched set/wait, 6 loop-carried, dependency graph acyclic** |
| AKA3006 UB bank conflicts | 0 (skewed-pool construction verified) | 0 |

Kernel B's four INFOs are AKA3004 "fragmented SRAM layout" advisories on
L1/L0A/L0B/L0C — an artifact of the checker seeing one ping-pong slot's
tensor views statically claimed while the other slot's are live at a
different pipeline phase; a 2-slot ring is inherently 50%-unclaimed from a
single static snapshot. Full outputs: `kernel_a_audit.txt`,
`kernel_b_audit.txt` (includes SRAM maps and per-pipe utilization).

---

## 5. Analyzer tooling fixes (prerequisite for §4)

Auditing a real heterogeneous MIX kernel initially collapsed to
`ops: 0`. Root causes fixed in `ascend_analyzer/parsing/ast_visitor.py`
(+13 tests in `tests/test_parser.py`, suite **729 → 742 passed**):

1. **`auto` tensor inference** — `auto x = que.AllocTensor<T>()` /
   `DeQue` / `Get` / `GetWithOffset` initializers now declare fully typed
   tensors (element type from the call's template argument).
2. **AST template-argument parsing for `TQue`/`TBuf`** — position and depth
   read from the `template_argument_list` (named depth constants resolve
   through the unit's constant environment; `QuePosition::` spelling
   accepted), replacing the literal-digit-only regex.
3. **`GetWithOffset` CANN signature** — `(elementCount, byteOffset)`:
   count sizes the tensor, offset is pool-relative (placed at
   buffer-base + offset); `GetBufferByByte`/`GetBufferAddr` keep
   byte-first semantics.
4. **MIX symbol-scope collision** — member calls through a class-typed
   local (`cube.Init(...)`/`vec.Init(...)` with colliding method names)
   now resolve via the receiver's static class, so both `ASCEND_IS_AIC` and
   `ASCEND_IS_AIV` stage classes inline into one entry walk.

---

## 6. When does the Cube trick win? (LLM inference guidance)

* **Standalone dequant of checkpoint weights (one-off, batch=1 warmup,
  or CPU-offload style):** the naive AIV kernel is the right default —
  simpler, no MIX orchestration, and chip-level it exploits 2× more cores
  (48 AIV vs 24 AIC+AIV pairs). The Cube trick merely reaches parity per
  pair (1.02×) because both paths pay the same per-16-block broadcast tax.
* **Prefill / large-batch serving:** dequant should not exist as a kernel
  at all. The consumer GEMM (2048×4096×4096 autotiles to 128×256×256,
  MTE2-starved at 66.7% cube utilization) is bandwidth-bound feeding fp16
  weights; keeping weights FP4 in L0B and contracting with `mad_mx`
  (351x) or the int8 contraction (910B) **halves the weight bytes on the
  exact pipe that binds** and eliminates the fp16 materialization (≈8.1 B
  of GM traffic per output element in Kernel B, 2.6 B in Kernel A — both
  zero when fused). This is the trick's real payoff and why the hardware
  MX SupportType tuple exists.
* **Decode / small-batch:** weights are read once per token — dequant is
  strictly waste; fusion is mandatory. Where a discrete dequant pass is
  still needed (e.g. mixed-precision LoRA paths), the Cube trick is
  preferable when the AIVs are the bottleneck (activation processing,
  sampling, KV-cache ops): it moves the multiply into the mad/Fixpipe path
  and leaves ≈40% of the paired AIV idle as headroom.
* **Silicon note:** on deployed 910B fleets use the int8 contraction as
  shipped; on v3 parts (351x) switch `Mmad`→`mad_mx`, A′/B′→fp4 operands,
  δ = exp_field − 9, halving MTE2 operand bytes (modeled: AIC per-tile MTE2
  24→12 cyc, leaving the Fixpipe 16 cyc/tile as the AIC limiter).

## 7. Reproducing

```bash
# modeling
PYTHONPATH=/d/Projects/PipeSim/src python model_pipesim.py
# audit (both exit 0)
PYTHONPATH=/d/Projects/ascend-kernel-analyzer python -m ascend_analyzer \
    naive_aiv_fp4_to_fp16.cpp   --chip ascend910b --warnings-as-errors
PYTHONPATH=/d/Projects/ascend-kernel-analyzer python -m ascend_analyzer \
    cube_trick_fp4_to_fp16.cpp  --chip ascend910b --warnings-as-errors
# optional tiling binding for Kernel B (binds numTiles for trip analysis)
python -m ascend_analyzer cube_trick_fp4_to_fp16.cpp --chip ascend910b \
    --tiling-data tiling.json --warnings-as-errors
# analyzer test suite
cd /d/Projects/ascend-kernel-analyzer && python -m pytest tests/   # 742 passed
```
