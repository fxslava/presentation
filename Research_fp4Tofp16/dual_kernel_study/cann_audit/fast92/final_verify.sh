#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
# Run after the CA build/sweep has finished; no binaries are rewritten live.
for e in 512 1024 2048; do
  for b in 16 32; do
    for v in simd cube simt; do
      bash "$root/run.sh" dav-3510 Ascend950 "$e" "$b" "$v" pv e2m1 final
    done
    bash "$root/run.sh" dav-2201 Ascend910B "$e" "$b" simd pv e2m1 final
  done
done
