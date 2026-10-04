#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
mkdir -p /tmp/fp4-hybrid-probe
cd /tmp/fp4-hybrid-probe
export FP4_ELEMENTS=${FP4_ELEMENTS:-256}
timeout -k 10 180 npusim record -s Ascend950 -o "$out/sim_hybrid_probe_$FP4_ELEMENTS" -f "$out/simt_cube_hybrid_kernel.o" -u "B $out/simt_cube_hybrid_kernel.o" "$out/raw_harness_simt_hybrid" > "$out/sim_hybrid_probe_${FP4_ELEMENTS}.log" 2>&1
status=$?
tail -30 "$out/sim_hybrid_probe_${FP4_ELEMENTS}.log"
exit "$status"
