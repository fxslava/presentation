#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
while pgrep -f '^bash /work/dual_kernel_study/cann_audit/fast92/sweep.sh$' >/dev/null; do sleep 5; done
bash "$root/build.sh" dav-3510 512 32
bash "$root/run_isolated.sh" 512 32 simd padded
bash "$root/run_isolated.sh" 1024 16 cube stable
bash "$root/run_isolated.sh" 2048 16 simd stable
bash "$root/run_isolated.sh" 2048 16 cube stable
bash "$root/final_verify.sh"
