#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
mkdir -p /tmp/fp4-native950
cd /tmp/fp4-native950
k=${1:-A}
if [ "$k" = A ]; then binary=naive_aiv_kernel.o; else binary=cube_trick_kernel.o; fi
timeout -k 10 300 npusim record -s Ascend950 -o "$out/sim_$k" -f "$out/$binary" -u "$k $out/$binary" "$out/raw_harness" > "$out/sim_${k}_launcher.log" 2>&1
status=$?
echo "simulation $k exit=$status"
tail -30 "$out/sim_${k}_launcher.log"
exit "$status"
