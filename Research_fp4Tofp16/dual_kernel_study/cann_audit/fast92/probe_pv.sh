#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
export ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1
for arch in dav_2201 dav_3510; do
  sim="$ASCEND_HOME_PATH/tools/simulator/$arch/lib"
  mkdir -p "/tmp/fast92-pv2/$arch"
  cd "/tmp/fast92-pv2/$arch"
  mkdir -p lib
  ln -sf "$sim/libruntime_cmodel.so" lib/libruntime.so
  LD_LIBRARY_PATH="$PWD/lib:$sim:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH" timeout -k 3 20 "$root/build/smoke" > "$root/logs/smoke_${arch}_native_driver.log" 2>&1
  echo "native driver $arch exit=$?"
  tail -35 "$root/logs/smoke_${arch}_native_driver.log"
done
