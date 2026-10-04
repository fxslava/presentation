#!/bin/bash
out=/work/dual_kernel_study/cann_audit/native950
log="$out/rerun_hybrid_clean_progress.log"
: > "$log"
for block in 16 32 64; do
  echo "START clean hybrid block=$block" | tee -a "$log"
  bash /work/dual_kernel_study/cann_audit/simulate_three_paths.sh simt_cube_hybrid "$block" 8192 >> "$log" 2>&1
  status=$?
  echo "END clean hybrid block=$block exit=$status" | tee -a "$log"
done
