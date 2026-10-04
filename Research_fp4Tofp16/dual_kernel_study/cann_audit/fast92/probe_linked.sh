#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
for arch in dav_2201 dav_3510; do
  sim="$ASCEND_HOME_PATH/tools/simulator/$arch/lib"
  mkdir -p "/tmp/fast92-linked/$arch" "$root/logs"
  cd "/tmp/fast92-linked/$arch"
  for model in pv ca; do
    if [ "$model" = pv ]; then drv=pvmodel; else drv=camodel; fi
    g++ -O2 -I"$ASCEND_HOME_PATH/include" "$root/smoke.cpp" -L"$sim" -L"$ASCEND_HOME_PATH/lib64" -Wl,--no-as-needed -lruntime_cmodel -lnpu_drv_$drv -lnpu_drv -lascendcl -Wl,--disable-new-dtags -Wl,-rpath,"$sim:$ASCEND_HOME_PATH/lib64" -o "$root/build/smoke_${arch}_$model" > "$root/logs/link_${arch}_$model.log" 2>&1
    LD_LIBRARY_PATH="$sim:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH" timeout -k 3 25 "$root/build/smoke_${arch}_$model" > "$root/logs/smoke_${arch}_linked_$model.log" 2>&1
    echo "linked $arch $model exit=$?"
    tail -12 "$root/logs/smoke_${arch}_linked_$model.log"
  done
done
