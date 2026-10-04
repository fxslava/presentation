#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
mkdir -p /tmp/fp4-simt-direct
cd /tmp/fp4-simt-direct
timeout -k 10 300 npusim record -s Ascend950 -o "$out/sim_simt_direct" -f "$out/simt_direct_kernel.o" -u "A $out/simt_direct_kernel.o" "$out/raw_harness_simt_direct" > "$out/sim_simt_direct_launcher.log" 2>&1
status=$?
tail -30 "$out/sim_simt_direct_launcher.log"
exit "$status"
