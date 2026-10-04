#!/bin/bash
set -eo pipefail
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
for block in 16 32 64; do
  archive=$(find "$out/sim_simt_cube_hybrid_b${block}_n8192" -maxdepth 1 -type d -name 'npusim_*' | sort | tail -1)
  npusim report -e "$archive" -o "$out/report_simt_hybrid_clean_b${block}" -n all > "$out/report_simt_hybrid_clean_b${block}.log" 2>&1
  python3 -c 'import json,sys; print(sys.argv[1],json.load(open(sys.argv[2]))["kernel_info"]["kernel_total_clocks"])' "$block" "$out/report_simt_hybrid_clean_b${block}/results/kernel_0_reports/summary.json"
done
