#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
mkdir -p /tmp/fp4-simt-hybrid
cd /tmp/fp4-simt-hybrid
timeout -k 10 300 npusim record -s Ascend950 -o "$out/sim_simt_hybrid" -f "$out/simt_cube_hybrid_kernel.o" -u "B $out/simt_cube_hybrid_kernel.o" "$out/raw_harness_simt_hybrid" > "$out/sim_simt_hybrid_launcher.log" 2>&1
status=$?
tail -35 "$out/sim_simt_hybrid_launcher.log"
exit "$status"
