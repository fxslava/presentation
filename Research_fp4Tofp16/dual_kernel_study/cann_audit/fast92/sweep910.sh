#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
for e in 512 1024 2048; do
  for b in 16 32; do
    bash "$root/build.sh" dav-2201 "$e" "$b"
    for v in simd cube; do
      bash "$root/run.sh" dav-2201 Ascend910B1 "$e" "$b" "$v" pv
    done
  done
done
