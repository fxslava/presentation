#!/bin/bash
set -eo pipefail
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
selected=${1:-all}
for block in 16 32 64; do
  for path in simd_naive_unpack simt_direct_gm_unpack simt_cube_hybrid; do
    key="${path}:$block"
    if [ "$selected" != all ] && [ "$selected" != "$key" ]; then continue; fi
    case "$path" in
      simd_naive_unpack) report="report_simd_b$block";;
      simt_direct_gm_unpack) report="report_simt_direct_ub_b$block";;
      simt_cube_hybrid) report="report_simt_hybrid_clean_b$block";;
    esac
    archive=$(find "$out/sim_${path}_b${block}_n8192" -maxdepth 1 -type d -name 'npusim_*' | sort | tail -1)
    if [ -z "$archive" ]; then echo "Missing archive: $key" >&2; exit 1; fi
    npusim report -e "$archive" -o "$out/$report" -n all > "$out/$report.log" 2>&1 || { tail -30 "$out/$report.log"; exit 1; }
    echo "Reported $key from $archive"
  done
done
