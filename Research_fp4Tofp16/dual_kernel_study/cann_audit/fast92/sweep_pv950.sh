#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
for e in 512 1024 2048; do
  for b in 16 32; do
    build="$root/build/dav-3510_e2m1_b${b}_e${e}"
    for v in simd cube simt; do
      while [ ! -f "$build/${v}.ready" ]; do sleep 2; done
      bash "$root/run.sh" dav-3510 Ascend950 "$e" "$b" "$v" pv
    done
  done
done
