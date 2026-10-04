# FP4 unpacking on Ascend 950PR: SIMD, direct SIMT, and SIMT + Cube

Date: 2026-10-04. Toolchain: CANN 9.2.0-beta.2 toolkit and 950 ops in the `fp4-cann92-audit` container. All nine measured runs compiled for `dav-3510`, used `npusim record -s Ascend950` (resolved profile `Ascend950PR_9589`, CA mode), and passed exact FP16 checks for 8,192 outputs. The model used one logical kernel block with its installed default configuration.

## Implementations and tested contract

| Path | Source | Input and output path |
|---|---|---|
| A, SIMD | `simd_naive_unpack.cpp` | MTE2 copies packed bytes and scales directly from GM to UB. Vector operations decode nibbles, broadcast scales using `Duplicate`, multiply, and MTE3 stores pair-planar FP16 output. |
| B, direct SIMT | `simt_direct_gm_unpack.cpp` | A `__simt_vf__` loads packed bytes and scales from GM into thread registers. Threads decode and scale one FP16 value each, write the 16 KiB output tile to UB, and MTE3 drains it to GM in sequential order. |
| C, SIMT + Cube | `simt_cube_hybrid.cpp`, entrypoint `simt_cube_hybrid` | A SIMT VF reads packed bytes/scales from GM and materializes INT8 Cube operands into a GM workspace. AIC uses MTE2/MTE1, INT8 Mmad, and Fixpipe. A second SIMT VF reads the FP16 result and scale exponents, folds the exponent, and stores sequential FP16 output. |

The test format is the project's FP4 E1M2 nibble grid (value = signed magnitude × 0.25) with positive normal E4M3 scales. The harness covers all 16 nibble codes and normal scale exponent fields 1–15, excluding the NaN scale code. It normalizes A's pair-planar output layout before comparing exact FP16 bits and canonicalizes signed zero. This is a dequantization test, not a full matrix multiplication. Block size is the number of FP4 values sharing a scale: 16, 32, or 64.

SIMT VF pointer parameters support GM and UB memory. The available interface does not provide a direct SIMT pointer to Cube L0A/L0B; C must stage operands for MTE2/MTE1 before contraction. [Official Ascend C SIMT syntax](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/language_extension/simd_and_simt_hybrid_programming_builtin_keyword.md) documents the memory qualifiers and `asc_vf_call` interface.

## Cycle comparison: 8,192-output tile

| Scale block | A: pure SIMD | B: direct GM → registers → UB | C: SIMT + Cube | B speedup vs A | C vs A |
|---:|---:|---:|---:|---:|---:|
| 16 | 42,326 | 8,546 | 39,646 | 4.95× | 1.07× faster |
| 32 | 15,199 | 8,577 | 38,489 | 1.77× | 2.53× slower |
| 64 | 11,206 | 8,479 | 37,944 | 1.32× | 3.39× slower |

Every row is a completed CA run with 8,192/8,192 exact outputs. Path B saves 79.8%, 43.6%, and 24.3% of A's cycles at blocks 16, 32, and 64 respectively. The direct-GM-to-GM variant in `simt_direct_gm_only.cpp` also passed at block 16 and measured 7,706 cycles. Its 840-cycle advantage over B includes a different output path, so it is supplementary rather than the main UB-output comparison.

At block 16, the comparable profiler fields are:

| Metric | A: SIMD | B: direct SIMT | C: SIMT + Cube |
|---|---:|---:|---:|
| Simulated cycles per 8,192-output tile | 42,326 | 8,546 | 39,646 |
| Broadcast instructions / vector slot cycles | 515 / 3,090 | 0 / 0 | 0 / 0 |
| Explicit UB allocation | 105.875 KiB | 16 KiB | 0 KiB |
| Additional GM workspace | 0 | 0 | 48 KiB |
| Input-memory evidence | AIV MTE2 utilization 1.99% | 512 SIMT register loads; AIV MTE2 0.06% | 1,536 SIMT register loads; AIC MTE2 13.68% |
| Compute utilization | AIV SIMD 30.32% | AIV SIMT 68.20% | AIV SIMT 67.94%; Cube 5.49% |
| Direct GM load latency / bank conflicts | Not exported | Not exported | Not exported |

This is a utilization and instruction breakdown, not an additive decomposition of total cycles. The kernels perform different mixes of half, integer, and INT8 matrix arithmetic, so a common FLOPs/cycle number would be misleading; the table uses the reported compute-pipeline utilization instead.

For a 4096 × 4096 weight matrix, this tile represents 1/2048 of the elements. A *linear single-core extrapolation* of the measured 8,192-output results gives approximately 31.13M / 17.57M / 78.83M cycles for A/B/C at block 32, or 22.95M / 17.36M / 77.71M at block 64. These are arithmetic projections, **not full-matrix simulator results**. They omit changes in cache behavior, memory contention, host preparation, and multi-core scheduling. The 20-core configuration requested earlier was not imposed or verified.

## Broadcasts, memory traffic, and utilization

| Scale block | A `RV_VDUPS` count | A broadcast pipe-slot cycles | B `RV_VDUPS` | C `RV_VDUPS` | Explicit UB allocation: A / B / C |
|---:|---:|---:|---:|---:|---:|
| 16 | 515 | 3,090 | 0 | 0 | 105.875 / 16 / 0 KiB |
| 32 | 259 | 1,554 | 0 | 0 | 105.375 / 16 / 0 KiB |
| 64 | 131 | 786 | 0 | 0 | 105.125 / 16 / 0 KiB |

Each traced `RV_VDUPS` occupied a vector execution slot for 6 cycles. The slot-cycle totals overlap other pipelines and are **not exclusive wall-cycle costs**. Per output they are 0.377, 0.190, and 0.096 cycles at blocks 16, 32, and 64. The often quoted 0.25 cycles/output broadcast tax is therefore not an isolated constant of these kernels. Reducing block-16 to block-32 broadcasts also removes 11,515 scalar instructions (17,008 → 5,493), explaining much of A's large cycle drop. A's MTE2 utilization is only 1.99% at block 16; scalar execution is its busiest pipeline at 53.48%. The data do not support MTE2 congestion as the measured baseline bottleneck.

At block 16, B's measured AIV SIMT utilization is 68.20%, AIV MTE3 is 6.93%, and AIV MTE2 is 0.06%. The profiler records 512 SIMT register-load instructions and no vector broadcasts. C uses no explicit AIV UB allocation in its selected entrypoint, but it needs **48 KiB of GM workspace** per 8,192 outputs: 32 KiB of dense INT8 operands plus 16 KiB of Fixpipe FP16 results, excluding final output. That would scale to 96 MiB of workspace for 4096 × 4096 weights. Its AIC MTE2/MTE1/Cube/Fixpipe utilizations are 13.68% / 2.67% / 5.49% / 22.24%; AIV SIMT is 67.94%. C's operand-preparation VF occupies 24,081 cycles and its exponent-fold VF 3,034 cycles. Cube utilization is low because the SIMT preparation and GM staging dominate this tile.

The 16 KiB and zero-UB figures are explicit source allocations. The report does not expose actual peak UB occupancy, VF register spills, a per-instruction direct-GM load latency, or an eight-bank conflict count. Pipeline utilization cannot be converted into exclusive memory/arithmetic/stall cycles. Path B structurally removes the dual-source vector multiply and scale broadcast, but a numeric bank-conflict reduction cannot be certified from the available counters.

## VF transition control

The minimal probes execute one GM store. The scalar kernel measured **1,328 cycles**; the kernel with one `asc_vf_call` to a 256-thread SIMT VF measured **1,704 cycles**, a **376-cycle observed premium** for this specific control. The VF's reported execution interval is 712 cycles. This comparison includes differing generated instructions, synchronization, and VF work; it is not a pure hardware switch-latency measurement.

B makes one SIMT VF call per 8,192-output tile; its block-16 VF interval is 6,013 of 8,546 total cycles. C makes two VF calls, whose intervals are 24,081 and 3,034 of 39,646 total cycles. The remaining cycles include kernel setup, Cube/MTE/Fixpipe work, and handoff waits. The 376-cycle probe premium is much smaller than B's 33,780-cycle saving over A at block 16. C's poorer result is driven mainly by dense operand staging and preparation, not by merely making two VF calls.

## Analyzer and validity limits

`ascend_analyzer` is absent from PATH and the installed CANN tree; `native950/analyzer_discovery.log` records the search. No AKA3005 result or zero-hazard certification is claimed. All nine device builds emitted empty compiler diagnostic logs, and every completed CA run passed exact outputs without a reported event-flag error. This validates the tested shapes at runtime; it does not prove every boundary case. The CA report did not export an UB bank-conflict counter or complete exclusive pipe-stall breakdown.

Manual memory-domain review: B's SIMT thread `i` reads GM inputs and writes UB output `i`; no two threads write the same element. Queue enqueue/dequeue orders the UB tile before MTE3. C's preparation thread `(tile, x)` writes unique GM operand offsets, then cross-core event 5 releases the AIC. The AIC writes disjoint GM FP16 tile slots and signals event 4 only after the Fixpipe pipe drains; the exponent-fold VF then writes unique GM output indices. Both VFs pass only GM/UB pointers. These are source and tested-shape checks, not a substitute for the missing analyzer.

The installed model uses its default 950PR configuration and a single logical kernel block. The full 4096 × 4096 matrix was not run cycle-accurately. Input packing, scale creation, GM workspace allocation, and transfers are excluded from kernel cycles; C's GM operand materialization inside the kernel is included. A emits pair-planar output while B and C emit sequential output, so any downstream layout conversion would affect an end-to-end comparison.

## Reproduce and inspect

Inside `fp4-cann92-audit`, `/work` maps to this project:

```bash
bash /work/dual_kernel_study/cann_audit/build_three_paths.sh
bash /work/dual_kernel_study/cann_audit/sweep_three_paths.sh
bash /work/dual_kernel_study/cann_audit/report_three_paths.sh
bash /work/dual_kernel_study/cann_audit/build_switch_probes.sh
bash /work/dual_kernel_study/cann_audit/simulate_switch_probes.sh
bash /work/dual_kernel_study/cann_audit/report_switch_probes.sh
bash /work/dual_kernel_study/cann_audit/check_analyzer.sh
python3 /work/dual_kernel_study/cann_audit/extract_simt_study.py
```

The machine-readable results are `native950/simt_study_results.json`. Individual report directories are `native950/report_simd_b16`, `report_simd_b32`, `report_simd_b64`, `report_simt_direct_ub_b16`, `report_simt_direct_ub_b32`, `report_simt_direct_ub_b64`, and `report_simt_hybrid_clean_b16/b32/b64`. The `sim_*` archives and launcher logs preserve correctness output and raw instruction traces. `npusim report -e <archive> -o <report-dir> -n all` regenerates a report; adding `-f` failed instruction-log mapping in this installed toolchain.
