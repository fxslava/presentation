#!/bin/bash
set -eo pipefail
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
for kind in scalar simt; do
  archive=$(find "$out/sim_${kind}_switch" -maxdepth 1 -type d -name 'npusim_*' | sort | tail -1)
  npusim report -e "$archive" -o "$out/report_${kind}_switch" -n all > "$out/report_${kind}_switch.log" 2>&1
  python3 -c 'import json,sys; print(sys.argv[1],json.load(open(sys.argv[2]))["kernel_info"]["kernel_total_clocks"])' "$kind" "$out/report_${kind}_switch/results/kernel_0_reports/summary.json"
done
