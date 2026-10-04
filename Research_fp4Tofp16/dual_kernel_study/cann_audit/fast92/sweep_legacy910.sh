#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
for e in 512 1024 2048; do
  for b in 16 32; do
    [ "$e:$b" != 512:16 ] || continue
    bash "$root/legacy910_run.sh" "$e" "$b" simd
  done
done
