# CANN 9.2 and native 950PR audit

Date: 2026-10-04. CANN **9.2.0-beta.2 toolkit + 950 ops** installed successfully; this is a beta release, not a claimed 9.2.0 GA installation.

## Environment and installation

Ubuntu-22.04 runs as WSL 2, x86_64. Docker initially contained only ascend-suites with CANN 8.5, not 9.2. An attempted host 8.5 installation failed because pip3 was missing. The successful 9.2 installation uses isolated container `fp4-cann92-audit`, preserving the existing container. Its installed `/opt/Ascend92` tree was also copied to WSL. Simulation/report dependencies are verified in the container.

Packages are in `/home/ladislav/cann-build`. Both installer integrity checks and installations passed:

| Package | Bytes | MD5 matching server ETag |
|---|---:|---|
| Ascend-cann-toolkit_9.2.0-beta.2_linux-x86_64.run | 1,398,240,001 | 6b5a5dc6d3cf9e9fcf680bf4684906cf |
| Ascend-cann-950-ops_9.2.0-beta.2_linux-x86_64.run | 2,850,241,840 | f141319e4e672b5484f5b758213464ee |

[Official package source](https://github.com/Ascend/cann-container-image/blob/main/cann/9.2.0-beta.2-950-ubuntu22.04-py3.12/Dockerfile).

Source `/opt/Ascend92/cann/set_env.sh`. Compiler: `/opt/Ascend92/cann-9.2.0-beta.2/bin/bisheng` (clang 15.0.5). Simulator CLI: `/opt/Ascend92/cann-9.2.0-beta.2/bin/npusim`. No standalone camodel/npumodel executable exists. Implementations under `/opt/Ascend92/cann-9.2.0-beta.2/tools/simulator/dav_3510/` include `camodel/libruntime_camodel.so` (CA), `lib/libruntime_cmodel.so` and `lib/libnpu_drv_pvmodel.so` (PV).

## Target and scope

Native compilation uses `--npu-arch=dav-3510`; `npusim record -s Ascend950` resolves to Ascend950PR_9589. Logs identify Ascend950PR, CA mode, core version 5. [Official documentation](https://asc.gitcode.com/api/SIMD-API/c_api/cube_data_move/asc_copy_l12l0b/asc_copy_l12l0b_3d_arch_3510.html) maps 950PR to dav-3510. The original requested 910B invariant does not match this native target.

Measurements use the installed default configuration, blockDim=1 and 8,192 output elements. The requested 20-core/256 KiB UB/512 KiB L1/64 B per cycle assumptions were not verified or imposed. This is kernel execution, excluding host packing, allocation and transfers. A uses one 8,192-element tile; B uses 32 tiles of 256 elements. It is not a full-device throughput measurement.

The independent harness exercises all 16 nibble codes and positive normal E4M3 scale exponents 1–15, excluding the NaN code. It normalizes A's pair-planar layout before comparing exact FP16 bits and canonicalizes signed zero. General support for negative/subnormal/NaN scales and arbitrary tails is not established.

## Source corrections

Original kernel files remain unchanged; corrected copies are in this audit directory.

* A: supported word operations replace unsupported byte operations; explicit Select mode; aligned masked 8-lane scale broadcasts avoid 32-byte store alignment assertions.
* B: L0C offsets corrected to 0/1024; M→MTE1 flags use 3/4 to avoid TPipe reserved flags; Fixpipe explicitly uses DEQF16 with unity dequantization to convert INT32 into FP16.

Failed intermediate runs are excluded from benchmark results, with archives retained for diagnosis.

## Verified results

| Kernel | Outputs | Bit mismatches | Core cycles | Cycles/output |
|---|---:|---:|---:|---:|
| A: vector pair-planar | 8,192 | 0 | 42,327 | 5.166870 |
| B: INT8 Cube + FP16 Fixpipe staging | 8,192 | 0 | 33,248 | 4.058594 |
| B control: INT32 staging + AIV conversion | 8,192 | 0 | 34,335 | 4.191284 |

B saves 9,079 cycles (21.45%), a 1.273× speedup for this test. Clean archives: `native950/sim_A/npusim_20261004083724_raw_harness` and `native950/sim_B/npusim_20261004083120_raw_harness`. Reports: `native950/report_A` and `native950/report_B_nomap`. Machine-readable results: `native950/simulation_results.json`.

## Pipeline and synchronization evidence

| Pipeline utilization | A | B |
|---|---:|---:|
| AIV scalar | 53.50% | 84.56% |
| AIV SIMD | 30.32% | 41.73% |
| AIV MTE2 | 2.00% | 31.04% |
| AIV MTE3 | 1.09% | 21.91% |
| AIC MTE2 | — | 46.43% |
| AIC MTE1 | — | 3.18% |
| Cube | — | 6.60% |
| Fixpipe | — | 18.15% |

Utilization is not exclusive stall cycles. The reports expose neither complete MTE/Vector/Cube stall accounting nor an 8-bank conflict counter; those requested metrics remain unavailable.

Extracted trace token spans in B: Cube→Fixpipe event 0 is 1 cycle for all 16 pairs; event 1 is 1–4 cycles. MTE1→Cube event 0 is 1 cycle; event 1 is 1–12. Buffer-reuse spans are longer, e.g. Cube→MTE1 event 3 is 514–2,786 cycles. These are elapsed SetFlag-to-WaitFlag flow spans, not exclusive stall costs, and overlapping spans cannot be summed as overhead. Reserved TPipe flags held until teardown do not imply kernel stalls. Extraction sorts trace events by timestamp and converts microseconds using the simulator's 1650 MHz clock.

The claimed ~0.25 cycles/output Duplicate tax is **not established**. Native 950 uses generated vector functions and masked broadcasts; scalar execution is the busiest reported pipeline. Isolating Duplicate requires a controlled broadcast-only comparison.

The controlled staging comparison supports a **1,087-cycle (3.17%) reduction** from FP16 Fixpipe staging versus INT32 staging followed by AIV conversion. Both variants use the same INT8 contraction and pass exact checks. The control doubles the intermediate GM slot width and adds a conversion buffer, so this measures the complete staging-path change rather than isolated Fixpipe instruction latency. It does not compare against an FP32 contraction kernel. Control archive: `native950/sim_stage32/npusim_20261004084052_raw_harness_stage32`; report: `native950/report_stage32`.

## Functional-model limitation

CA executes compiled kernels and verifies exact output bits. The separate PV-library launch failed during aclInit because halGetSocVersion returned an empty SoC string (runtime 507008 / aclInit 500000). No separate NPUModel/PV pass is claimed.

The mathematical CPU emulator also passed its sweep; this is separate from compiled-kernel verification. The older analytical estimates in `dual_kernel_study/REPORT.md` are not CA measurements.

## Reproduction

Inside `fp4-cann92-audit`, `/work` maps to this project:

```bash
bash /work/dual_kernel_study/cann_audit/build950.sh
bash /work/dual_kernel_study/cann_audit/simulate950.sh A
bash /work/dual_kernel_study/cann_audit/simulate950.sh B
bash /work/dual_kernel_study/cann_audit/build_stage32.sh
bash /work/dual_kernel_study/cann_audit/control950.sh
source /opt/Ascend92/cann/set_env.sh
npusim report -e <archive> -o <report-directory> -n all
python3 /work/dual_kernel_study/cann_audit/extract_metrics.py
```

Report generation without binary mapping succeeds; adding `-f` to the report command failed instruction-log discovery. Installation logs, compile logs and launcher logs are retained beside the artifacts.
