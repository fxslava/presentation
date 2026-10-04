#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
bash "$root/build.sh" dav-3510 512 32
bash "$root/run.sh" dav-3510 Ascend950 512 32 simd pv e2m1 padded
bash "$root/run.sh" dav-3510 Ascend950 512 32 simd ca e2m1 padded
bash "$root/report.sh" dav-3510_e2m1_b32_e512_simd_ca_padded
