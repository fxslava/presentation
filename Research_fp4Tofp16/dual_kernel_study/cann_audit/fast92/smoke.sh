#!/bin/bash
set -eo pipefail
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
mkdir -p "$root/logs" /tmp/fast92-pv/lib /tmp/fast92-pv/run
g++ -O2 -I"$ASCEND_HOME_PATH/include" "$root/smoke.cpp" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -o "$root/build/smoke"
for arch in dav_2201 dav_3510; do
  sim="$ASCEND_HOME_PATH/tools/simulator/$arch/lib"
  [ -d "$sim" ] || sim="$ASCEND_HOME_PATH/x86_64-linux/simulator/$arch/lib"
  ln -sf "$sim/libruntime_cmodel.so" /tmp/fast92-pv/lib/libruntime.so
  ln -sf "$sim/libnpu_drv_pvmodel.so" /tmp/fast92-pv/lib/libnpu_drv.so
  cd /tmp/fast92-pv/run
  set +e
  LD_LIBRARY_PATH="/tmp/fast92-pv/lib:$sim:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH" timeout -k 3 20 "$root/build/smoke" > "$root/logs/smoke_${arch}.log" 2>&1
  status=$?
  set -e
  echo "$arch PV smoke exit=$status"
  tail -20 "$root/logs/smoke_${arch}.log"
done
