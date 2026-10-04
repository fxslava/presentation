#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
for e in 512 1024 2048; do
  for b in 16 32; do
    bash "$root/build.sh" dav-3510 "$e" "$b"
    for v in simd cube simt; do
      # The 512/16 SIMT control has already completed; preserve its archive.
      if [ "$e:$b:$v" != 512:16:simt ]; then
        bash "$root/run.sh" dav-3510 Ascend950 "$e" "$b" "$v" ca
      fi
      bash "$root/report.sh" "dav-3510_e2m1_b${b}_e${e}_${v}_ca"
    done
  done
done
