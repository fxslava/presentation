#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
mkdir -p /tmp/fp4-stage32
cd /tmp/fp4-stage32
export CONTROL_STAGE32=1
timeout -k 10 300 npusim record -s Ascend950 -o "$out/sim_stage32" -f "$out/stage32_kernel.o" -u "B $out/stage32_kernel.o" "$out/raw_harness_stage32" > "$out/sim_stage32_launcher.log" 2>&1
status=$?
tail -30 "$out/sim_stage32_launcher.log"
exit "$status"
