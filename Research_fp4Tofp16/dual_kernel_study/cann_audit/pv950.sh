#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
sim=$ASCEND_HOME_PATH/tools/simulator/dav_3510/lib
out=/work/dual_kernel_study/cann_audit/native950
mkdir -p /tmp/fp4-pv/lib /tmp/fp4-pv/run
ln -sf "$sim/libruntime_cmodel.so" /tmp/fp4-pv/lib/libruntime.so
export LD_LIBRARY_PATH="/tmp/fp4-pv/lib:$sim:$LD_LIBRARY_PATH"
cd /tmp/fp4-pv/run
k=${1:-A}
if [ "$k" = A ]; then binary=naive_aiv_kernel.o; else binary=cube_trick_kernel.o; fi
timeout -k 10 60 "$out/raw_harness" "$k" "$out/$binary" > "$out/pv_$k.log" 2>&1
status=$?
echo "PV $k exit=$status"
tail -25 "$out/pv_$k.log"
exit "$status"
