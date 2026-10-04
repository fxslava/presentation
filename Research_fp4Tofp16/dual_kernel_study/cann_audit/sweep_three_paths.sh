#!/bin/bash
out=/work/dual_kernel_study/cann_audit/native950
log="$out/sweep_progress.log"
: > "$log"
for pair in simd_naive_unpack:16 simd_naive_unpack:32 simt_direct_gm_unpack:32 simt_cube_hybrid:32 simd_naive_unpack:64 simt_direct_gm_unpack:64 simt_cube_hybrid:64; do
  path=${pair%:*}
  block=${pair#*:}
  echo "START $path block=$block elements=8192" | tee -a "$log"
  bash /work/dual_kernel_study/cann_audit/simulate_three_paths.sh "$path" "$block" 8192 >> "$log" 2>&1
  status=$?
  echo "END $path block=$block exit=$status" | tee -a "$log"
done
